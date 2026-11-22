"""The one object every stage of the pipeline works on.

Whatever produced a recording (the team firmware's capture stream, the sensor-testing
logger), a reader turns it into a Session: EMG in mV at about 2 kHz, IMU on the same clock,
markers, metadata, and a Health record of everything that went wrong while recording.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Health:
    emg_readings: int = 0     # EMG samples the session spans, missing ones included
    missing: int = 0          # EMG samples that never arrived (NaN in Session.emg)
    longest_gap: int = 0      # longest run of missing EMG samples
    raw_readings: int = 0     # raw ADC readings behind the EMG (the board reads faster than it reports)
    clipped: int = 0          # raw ADC readings at 0 or full scale
    adc_overflows: int = 0    # times the board's ADC buffer filled: readings lost with no gap in the index
    frames_dropped: int = 0   # frames the board couldn't send (they show up as missing samples)
    crc_errors: int = 0       # frames that failed their checksum
    skipped_bytes: int = 0    # bytes that weren't part of any good frame
    reboots: int = 0          # board restarts seen in the stream; the longest stretch is kept
    imu_longest_gap_s: float = 0.0  # longest time between IMU samples (normal: 0.02 s)


@dataclass
class Session:
    name: str
    source: str               # "capture" or "legacy"
    emg: np.ndarray           # mV at the ADC pin, NaN where missing
    emg_hz: float             # measured rate, not the nominal one
    imu_t: np.ndarray = field(default_factory=lambda: np.zeros(0))       # s, on the EMG clock
    accel: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # g
    gyro: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # deg/s, bias removed by the board
    markers: list[tuple[float, str]] = field(default_factory=list)       # (s, label)
    meta: dict = field(default_factory=dict)
    health: Health = field(default_factory=Health)

    @property
    def emg_t(self) -> np.ndarray:
        """Time of each EMG sample in s. EMG sample 0 is t = 0."""
        return np.arange(len(self.emg)) / self.emg_hz

    @property
    def duration_s(self) -> float:
        return len(self.emg) / self.emg_hz

    @property
    def imu_hz(self) -> float:
        """Average IMU rate. (Not the median spacing: on the board each IMU reading is stamped with the
        EMG sample count, which moves in 8 ms chunks, so spacings alternate 16 / 24 ms.)"""
        if len(self.imu_t) < 2:
            return 0.0
        return float((len(self.imu_t) - 1) / (self.imu_t[-1] - self.imu_t[0]))

    def problems(self, cfg: dict) -> list[str]:
        """Reasons to reject this session. Empty means usable."""
        c, h = cfg["input"], self.health
        if h.emg_readings == 0:
            return ["no EMG at all"]
        out = []
        share = h.missing / h.emg_readings
        if share > c["max_missing_share"]:
            out.append(f"{share:.2%} of EMG is missing (limit {c['max_missing_share']:.2%})")
        gap_s = h.longest_gap / self.emg_hz
        if gap_s > c["max_gap_s"]:
            out.append(f"an EMG gap of {gap_s:.2f} s (limit {c['max_gap_s']} s)")
        if h.raw_readings and h.clipped / h.raw_readings > c["max_clipped_share"]:
            out.append(f"{h.clipped / h.raw_readings:.2%} of ADC readings clipped "
                       f"(limit {c['max_clipped_share']:.2%}): check the RAW wire and the gain")
        if h.adc_overflows:
            out.append(f"the ADC buffer overflowed {h.adc_overflows} times: readings were lost "
                       "without a gap in the index, so EMG timing is wrong")
        if self.source == "capture":
            if len(self.imu_t) == 0:
                out.append("no IMU data: the angle gate can't run")
            elif h.imu_longest_gap_s > c["max_imu_gap_s"]:
                out.append(f"an IMU gap of {h.imu_longest_gap_s:.2f} s (limit {c['max_imu_gap_s']} s)")
        return out

    def notes(self) -> list[str]:
        """Things worth knowing that don't make the session unusable."""
        h, out = self.health, []
        if h.reboots:
            out.append(f"board rebooted {h.reboots} time(s) during recording; kept the longest stretch")
        if h.crc_errors:
            out.append(f"{h.crc_errors} frames failed their checksum (their samples count as missing)")
        if h.frames_dropped:
            out.append(f"board dropped {h.frames_dropped} frames (USB couldn't keep up)")
        if h.missing:
            out.append(f"{h.missing} EMG samples missing, longest gap {h.longest_gap / self.emg_hz * 1000:.0f} ms")
        if self.source == "legacy":
            out.append("sensor-testing recording: no IMU")
        return out

    def summary(self) -> str:
        imu = f"IMU {len(self.imu_t)} samples at {self.imu_hz:.1f} Hz" if len(self.imu_t) else "no IMU"
        level = np.nanmedian(self.emg) if len(self.emg) else float("nan")
        return (f"{self.name} [{self.source}]: {self.duration_s:.1f} s, EMG at {self.emg_hz:.1f} Hz "
                f"(median {level:.0f} mV), {imu}, {len(self.markers)} markers")


def longest_nan_run(x: np.ndarray) -> int:
    """Length of the longest run of NaNs in x."""
    bad = np.isnan(x)
    if not bad.any():
        return 0
    edges = np.diff(np.r_[0, bad.astype(np.int8), 0])
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    return int((ends - starts).max())


def longest_spacing(t: np.ndarray) -> float:
    """Longest time between consecutive samples, in s (0 with fewer than 2 samples)."""
    return float(np.diff(t).max()) if len(t) >= 2 else 0.0
