"""Reads the framed binary stream the firmware sends in capture mode.

See fatigue_sensor/capture.h for the format. Everything is stamped with the
EMG sample index (2 kHz from boot), which is what lines the streams up.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

SYNC = b"\xa5\x5a"
STATUS, EMG, IMU, TURN, REP = 0x01, 0x02, 0x03, 0x04, 0x05
MAX_PAYLOAD = 1024


def crc16(data: bytes, crc: int = 0xFFFF) -> int:
    """CRC-16/CCITT-FALSE, as in capture.h."""
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


class FrameReader:
    """Finds frames in a byte stream fed in arbitrary chunks."""

    def __init__(self) -> None:
        self.buf = bytearray()
        self.crc_errors = 0
        self.skipped_bytes = 0  # bytes that weren't part of any frame

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.buf += data
        frames = []
        while True:
            start = self.buf.find(SYNC)
            if start < 0:
                keep = 1 if self.buf.endswith(SYNC[:1]) else 0
                self.skipped_bytes += len(self.buf) - keep
                del self.buf[: len(self.buf) - keep]
                return frames
            if start:
                self.skipped_bytes += start
                del self.buf[:start]
            if len(self.buf) < 5:
                return frames
            ftype = self.buf[2]
            length = self.buf[3] | self.buf[4] << 8
            if length > MAX_PAYLOAD:
                # Not a real header; resync one byte on.
                self.skipped_bytes += 1
                del self.buf[:1]
                continue
            total = 5 + length + 2
            if len(self.buf) < total:
                return frames
            body = bytes(self.buf[2 : 5 + length])
            crc = self.buf[5 + length] | self.buf[6 + length] << 8
            if crc16(body) != crc:
                self.crc_errors += 1
                self.skipped_bytes += 1
                del self.buf[:1]
                continue
            frames.append((ftype, bytes(self.buf[5 : 5 + length])))
            del self.buf[:total]


@dataclass
class Recording:
    """One capture session, decoded."""

    emg_hz: int = 2000
    adc_hz: int = 8000
    imu_hz: int = 50
    emg_per_mv: int = 10
    # EMG: one value per sample index from first_index on; NaN where a frame
    # was lost.
    first_index: int = 0
    emg_mv: np.ndarray = field(default_factory=lambda: np.zeros(0))
    clipped: int = 0
    # IMU at 50 Hz: stamp, accel (g), gyro (deg/s), arm angle (deg), still.
    imu_stamp: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    imu_accel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    imu_gyro: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    imu_angle: np.ndarray = field(default_factory=lambda: np.zeros(0))
    imu_still: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    # Turning points: stamp, angle, +1 top / -1 bottom.
    turn_stamp: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    turn_angle: np.ndarray = field(default_factory=lambda: np.zeros(0))
    turn_top: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    # Reps: stamp when counted, count since boot, range of motion (deg).
    rep_stamp: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    rep_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    rep_rom: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # STATUS frames: (emg_index, millis), and stream health.
    status: list[tuple[int, int]] = field(default_factory=list)
    adc_overflows: int = 0
    frames_dropped: int = 0
    crc_errors: int = 0
    skipped_bytes: int = 0
    missing_samples: int = 0

    @property
    def n(self) -> int:
        return len(self.emg_mv)

    def effective_emg_hz(self) -> float | None:
        """EMG rate measured against the ESP32's millis() clock."""
        if len(self.status) < 2:
            return None
        (i0, t0), (i1, t1) = self.status[0], self.status[-1]
        return (i1 - i0) / ((t1 - t0) / 1000) if t1 > t0 else None


def parse(data: bytes) -> Recording:
    reader = FrameReader()
    frames = reader.feed(data)
    rec = Recording(crc_errors=reader.crc_errors, skipped_bytes=reader.skipped_bytes)
    emg_blocks: list[tuple[int, np.ndarray]] = []
    imu, turns, reps = [], [], []
    for ftype, p in frames:
        if ftype == STATUS and len(p) >= 25:
            (_, emg_hz, adc_hz, imu_hz, per_mv, idx, ms, ovf, dropped) = struct.unpack_from("<BHHHHIIII", p)
            rec.emg_hz, rec.adc_hz, rec.imu_hz, rec.emg_per_mv = emg_hz, adc_hz, imu_hz, per_mv
            rec.status.append((idx, ms))
            rec.adc_overflows, rec.frames_dropped = ovf, dropped
        elif ftype == EMG and len(p) >= 6:
            first, n, clipped = struct.unpack_from("<IBB", p)
            vals = np.frombuffer(p, dtype="<u2", count=n, offset=6).astype(float)
            emg_blocks.append((first, vals))
            rec.clipped += clipped
        elif ftype == IMU and len(p) >= 21:
            imu.append(struct.unpack_from("<I6hfB", p))
        elif ftype == TURN and len(p) >= 9:
            turns.append(struct.unpack_from("<Ifb", p))
        elif ftype == REP and len(p) >= 8:
            reps.append(struct.unpack_from("<IHH", p))

    if emg_blocks:
        emg_blocks.sort(key=lambda b: b[0])
        first = emg_blocks[0][0]
        last = max(b[0] + len(b[1]) for b in emg_blocks)
        emg = np.full(last - first, np.nan)
        for start, vals in emg_blocks:
            emg[start - first : start - first + len(vals)] = vals / rec.emg_per_mv
        rec.first_index, rec.emg_mv = first, emg
        rec.missing_samples = int(np.isnan(emg).sum())
    if imu:
        a = np.array(imu, dtype=float)
        rec.imu_stamp = a[:, 0].astype(np.int64)
        rec.imu_accel = a[:, 1:4] / 1000
        rec.imu_gyro = a[:, 4:7] / 10
        rec.imu_angle = a[:, 7]
        rec.imu_still = a[:, 8].astype(bool)
    if turns:
        t = np.array(turns, dtype=float)
        rec.turn_stamp, rec.turn_angle, rec.turn_top = t[:, 0].astype(np.int64), t[:, 1], t[:, 2].astype(np.int8)
    if reps:
        r = np.array(reps, dtype=float)
        rec.rep_stamp, rec.rep_count, rec.rep_rom = r[:, 0].astype(np.int64), r[:, 1].astype(np.int64), r[:, 2] / 10
    return rec


def load(path: str) -> Recording:
    with open(path, "rb") as f:
        return parse(f.read())


# ------------------------------------------------------------ writing
# Used by simulate_recording.py so the analysis can be tested end to end
# without hardware.


def frame(ftype: int, payload: bytes) -> bytes:
    body = bytes([ftype]) + struct.pack("<H", len(payload)) + payload
    return SYNC + body + struct.pack("<H", crc16(body))
