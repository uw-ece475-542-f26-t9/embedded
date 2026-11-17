"""Reader for the team firmware's capture stream (embedded/fatigue_sensor/capture.h, version 1).

Written from the format spec in capture.h, not from the team's parser.

Frame, little-endian:
    A5 5A | type u8 | length u16 | payload | crc u16
crc is CRC-16/CCITT-FALSE over type, length and payload.

Frames used here:
    0x01 STATUS  u8 version, u16 emg_hz, u16 adc_hz, u16 imu_hz, u16 emg_per_mv,
                 u32 emg_index, u32 millis, u32 adc_overflows, u32 frames_dropped
    0x02 EMG     u32 first sample index, u8 n, u8 clipped, n x u16 sample (1/emg_per_mv mV)
    0x03 IMU     u32 emg_index, i16 ax ay az (milli-g), i16 gx gy gz (0.1 deg/s), f32 angle, u8 still
TURN, REP and FEAT frames come from the team's rep counter and model. They are counted and
ignored: this pipeline does its own gating and features.

Everything is stamped with the EMG sample index, which is the master clock. Its real rate is
measured against the board's millis() from the STATUS frames.

The writers at the bottom build frames, for the tests and the simulator.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

from .session import Health, Session, longest_nan_run, longest_spacing

SYNC = b"\xa5\x5a"
STATUS, EMG, IMU, TURN, REP, FEAT = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06
FRAME_NAMES = {STATUS: "status", EMG: "emg", IMU: "imu", TURN: "turn", REP: "rep", FEAT: "feat"}
MAX_PAYLOAD = 1024

STATUS_FMT = struct.Struct("<BHHHHIIII")  # 25 bytes
EMG_HEAD = struct.Struct("<IBB")          # 6 bytes, then n x u16
IMU_FMT = struct.Struct("<I6hfB")         # 21 bytes


def _crc_table() -> list[int]:
    table = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) & 0xFFFF if c & 0x8000 else (c << 1) & 0xFFFF
        table.append(c)
    return table


_TABLE = _crc_table()


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF), as in capture.h."""
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


class FrameReader:
    """Finds good frames in a byte stream that arrives in arbitrary chunks."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.crc_errors = 0
        self.skipped_bytes = 0

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buf += data
        buf, n, pos, out = self.buf, len(self.buf), 0, []
        while True:
            j = buf.find(SYNC, pos)
            if j < 0:
                # Keep a trailing A5: it may be the first half of the next sync.
                end = n - 1 if n > pos and buf[-1] == SYNC[0] else n
                self.skipped_bytes += end - pos
                pos = end
                break
            self.skipped_bytes += j - pos
            pos = j
            if n - pos < 5:
                break
            length = buf[pos + 3] | buf[pos + 4] << 8
            if length > MAX_PAYLOAD:  # not a real header
                self.skipped_bytes += 1
                pos += 1
                continue
            total = 7 + length
            if n - pos < total:
                break
            body = bytes(buf[pos + 2 : pos + 5 + length])
            if crc16(body) != (buf[pos + 5 + length] | buf[pos + 6 + length] << 8):
                self.crc_errors += 1
                self.skipped_bytes += 1
                pos += 1
                continue
            out.append((body[0], body[3:]))
            pos += total
        del buf[:pos]
        return out

    def finish(self) -> list[tuple[int, bytes]]:
        """End of stream: what's left can't complete, so rescan it a byte on until it's used up.
        (A corrupted length can make the reader wait for bytes that never come.)"""
        out = []
        while len(self.buf) >= 7:
            del self.buf[:1]
            self.skipped_bytes += 1
            out += self.feed(b"")
        self.skipped_bytes += len(self.buf)
        self.buf.clear()
        return out


def _split_boots(frames: list[tuple[int, bytes]], jump: int) -> list[int]:
    """Boot number of each frame. A reboot shows as millis() or the EMG index going backwards."""
    boot, last_ms, last_end, boots = 0, None, None, []
    for ftype, p in frames:
        if ftype == STATUS and len(p) >= STATUS_FMT.size:
            ms = STATUS_FMT.unpack_from(p)[6]
            if last_ms is not None and ms < last_ms:
                boot, last_end = boot + 1, None
            last_ms = ms
        elif ftype == EMG and len(p) >= EMG_HEAD.size:
            first, n, _ = EMG_HEAD.unpack_from(p)
            if last_end is not None and first + jump < last_end:
                boot, last_ms, last_end = boot + 1, None, None
            last_end = first + n if last_end is None else max(last_end, first + n)
        boots.append(boot)
    return boots


def parse(data: bytes, cfg: dict, name: str = "capture") -> Session:
    nominal_hz = cfg["input"]["emg_hz"]
    reader = FrameReader()
    frames = reader.feed(data) + reader.finish()
    boots = _split_boots(frames, int(cfg["input"]["reboot_jump_s"] * nominal_hz))

    # Keep the boot with the most EMG.
    emg_per_boot = Counter()
    for (ftype, p), b in zip(frames, boots):
        if ftype == EMG and len(p) >= EMG_HEAD.size:
            emg_per_boot[b] += p[4]
    keep = emg_per_boot.most_common(1)[0][0] if emg_per_boot else 0

    statuses, blocks, imu, kinds = [], [], [], Counter()
    clipped = 0
    for (ftype, p), b in zip(frames, boots):
        kinds[FRAME_NAMES.get(ftype, f"unknown_{ftype:#x}")] += 1
        if b != keep:
            continue
        if ftype == STATUS and len(p) >= STATUS_FMT.size:
            statuses.append(STATUS_FMT.unpack_from(p))
        elif ftype == EMG and len(p) >= EMG_HEAD.size:
            first, n, clip = EMG_HEAD.unpack_from(p)
            if len(p) < EMG_HEAD.size + 2 * n:
                continue
            blocks.append((first, np.frombuffer(p, dtype="<u2", count=n, offset=EMG_HEAD.size)))
            clipped += clip
        elif ftype == IMU and len(p) >= IMU_FMT.size:
            imu.append(IMU_FMT.unpack_from(p))

    per_mv = statuses[-1][4] if statuses else 10
    adc_hz = statuses[-1][2] if statuses else 8000
    health = Health(crc_errors=reader.crc_errors, skipped_bytes=reader.skipped_bytes,
                    reboots=max(boots, default=0), clipped=clipped)

    # EMG on one array indexed by sample number; NaN where nothing arrived.
    if blocks:
        first = min(f for f, _ in blocks)
        end = max(f + len(v) for f, v in blocks)
        emg = np.full(end - first, np.nan)
        for f, v in blocks:
            emg[f - first : f - first + len(v)] = v / per_mv
    else:
        first, emg = 0, np.zeros(0)

    # Real EMG rate from (index, millis) pairs; fall back to nominal over short spans.
    rate = float(statuses[-1][1]) if statuses else float(nominal_hz)
    if len(statuses) >= 2 and statuses[-1][6] - statuses[0][6] >= 1000:
        idx = np.array([s[5] for s in statuses], float)
        ms = np.array([s[6] for s in statuses], float)
        rate = float(np.polyfit(ms, idx, 1)[0] * 1000)

    # Board counters are totals since boot; count only what happened during this recording.
    if len(statuses) >= 2:
        health.adc_overflows = statuses[-1][7] - statuses[0][7]
        health.frames_dropped = statuses[-1][8] - statuses[0][8]
    elif statuses:
        health.adc_overflows, health.frames_dropped = statuses[0][7], statuses[0][8]

    health.emg_readings = len(emg)
    health.missing = int(np.isnan(emg).sum())
    health.longest_gap = longest_nan_run(emg)
    health.raw_readings = int(round((len(emg) - health.missing) * adc_hz / rate)) if rate else 0

    if imu:
        a = np.array(sorted(imu, key=lambda r: r[0]), float)
        imu_t, accel, gyro = (a[:, 0] - first) / rate, a[:, 1:4] / 1000, a[:, 4:7] / 10
    else:
        imu_t, accel, gyro = np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3))
    health.imu_longest_gap_s = longest_spacing(imu_t)

    meta = {"frames": dict(kinds), "first_emg_index": int(first),
            "nominal_emg_hz": int(statuses[-1][1]) if statuses else nominal_hz, "adc_hz": int(adc_hz)}
    return Session(name=name, source="capture", emg=emg, emg_hz=rate, imu_t=imu_t, accel=accel,
                   gyro=gyro, markers=[], meta=meta, health=health)


def read_capture(path: str | Path, cfg: dict) -> Session:
    """A .bin from the recorder, plus its .json sidecar (labels, who, which set) if there is one."""
    path = Path(path)
    session = parse(path.read_bytes(), cfg, name=path.stem)
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        info = json.loads(sidecar.read_text())
        session.markers = [(float(t), str(label)) for t, label in info.pop("markers", [])]
        session.meta.update(info)
    return session


# ------------------------------------------------------------------ writing


def frame(ftype: int, payload: bytes) -> bytes:
    body = bytes([ftype]) + struct.pack("<H", len(payload)) + payload
    return SYNC + body + struct.pack("<H", crc16(body))


def status_payload(emg_index: int, millis: int, overflows: int = 0, dropped: int = 0,
                   emg_hz: int = 2000, adc_hz: int = 8000, imu_hz: int = 50, per_mv: int = 10) -> bytes:
    return STATUS_FMT.pack(1, emg_hz, adc_hz, imu_hz, per_mv, emg_index, millis, overflows, dropped)


def emg_payload(first: int, mv: np.ndarray, clipped: int = 0, per_mv: int = 10) -> bytes:
    vals = np.clip(np.round(np.asarray(mv, float) * per_mv), 0, 65535).astype("<u2")
    return EMG_HEAD.pack(first, len(vals), min(clipped, 255)) + vals.tobytes()


def imu_payload(stamp: int, accel_g, gyro_dps, angle: float = 0.0, still: bool = False) -> bytes:
    a = [int(np.clip(round(v * 1000), -32768, 32767)) for v in accel_g]
    g = [int(np.clip(round(v * 10), -32768, 32767)) for v in gyro_dps]
    return IMU_FMT.pack(stamp, *a, *g, angle, int(still))
