"""Step 5a: the forearm angle from the wrist IMU, in real time.

A port of RepSense's angle (src/labels.py: track_down, hang_axis, segment_angle), made one-way:
  1. Track "down" as the sensor sees it: each IMU sample, turn the estimate by the gyro, then nudge
     it toward the accelerometer by dt / (tau + dt). With tau = 1 s at 50 Hz that is 0.0196: the
     98% gyro / 2% accelerometer blend. The gyro follows fast moves; the accelerometer stops drift.
  2. "Hanging" is the median of "down" over the stillest hang_s before the first lift (sets start
     with the arm hanging), fixed once the forearm turns freeze_deg away from it. The angle from
     then back to that still moment is filled in when it's fixed (the board knows it from then).
  3. The angle is between the hanging direction and the current one: 0 hanging, ~135 at the top.
     Twisting the forearm (the sync flicks, palm turning) turns about that direction, so it doesn't
     change the angle; how the strap sits doesn't matter.
  4. Direction: lifting when the angle rises faster than move_dps, lowering when it falls faster.

What changed from RepSense: its angle is then smoothed with a 3 Hz forward-backward filter, which
needs the future. This one isn't smoothed: the gyro-tracked angle is already smooth. Its speed is a
backward difference (past samples only).

One honest limit: with only a wrist IMU, body swing adds to the angle. This is the forearm's angle
from hanging, not the elbow's bend.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .session import Session


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def track_down(t: np.ndarray, gyro_dps: np.ndarray, accel: np.ndarray, tau_s: float) -> np.ndarray:
    """RepSense track_down, unchanged: "down" in sensor coordinates, one unit vector per sample."""
    felt = unit(-accel)  # at rest the accelerometer reads "up"
    rate = np.radians(gyro_dps)
    dt = np.diff(t, prepend=t[0])
    pull = dt / (tau_s + dt)
    down = np.empty_like(felt)
    down[0] = felt[0]
    for i in range(1, len(t)):
        d = down[i - 1] - np.cross(rate[i], down[i - 1]) * dt[i]  # the room turns the other way
        d = (1 - pull[i]) * d + pull[i] * felt[i]
        down[i] = d / np.linalg.norm(d)
    return down


@dataclass
class AngleTrack:
    t: np.ndarray            # IMU times (s, EMG clock)
    angle_deg: np.ndarray    # forearm angle from hanging; NaN until hanging is known
    rate_dps: np.ndarray     # how fast it changes (+ = rising)
    direction: np.ndarray    # +1 lifting, -1 lowering, 0 neither
    hang_known_s: float      # when hanging was fixed: no angle is known before this (NaN if never)

    def at(self, t_emg: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Angle and direction at each EMG sample. Interpolating needs the next IMU sample, so this
        runs 20 ms behind; that fits inside the cleaning latency."""
        ang = np.interp(t_emg, self.t, self.angle_deg, left=np.nan, right=np.nan)
        rate = np.interp(t_emg, self.t, self.rate_dps, left=np.nan, right=np.nan)
        i = np.clip(np.searchsorted(self.t, t_emg), 1, len(self.t) - 1)
        gap = (self.t[i] - self.t[i - 1]) > max_gap_s
        ang[gap] = np.nan
        rate[gap] = np.nan
        return ang, rate


def forearm_angle(s: Session, cfg: dict) -> AngleTrack:
    c = cfg["angle"]
    t, gyro, accel = s.imu_t, s.gyro, s.accel
    if len(t) < 3:
        raise ValueError(f"{s.name}: no IMU data")
    down = track_down(t, gyro, accel, c["track_tau_s"])

    # Hanging: the stillest hang_s seen so far (RepSense's stillness score: turning speed plus
    # shove), fixed once the forearm has turned freeze_deg away from it. A real hanging arm is never
    # perfectly still (hand tremor reached 11-22 deg/s in RepSense's recordings), so there's no
    # absolute threshold. Twisting doesn't change the angle, so the sync flicks can't fix it early.
    need = max(1, int(round(c["hang_s"] / np.median(np.diff(t)))))
    score = np.linalg.norm(gyro, axis=1) + c["shove_dps_per_g"] * np.abs(np.linalg.norm(accel, axis=1) - 1)
    rolling = np.convolve(score, np.ones(need) / need, mode="valid")  # rolling[k] covers samples k..k+need-1
    best, up, known = None, None, None
    for i in range(need - 1, len(t)):
        k = i - need + 1
        if best is None or rolling[k] < rolling[best]:
            best = k
            up = unit(-np.median(down[best:best + need], axis=0))  # RepSense hang_axis
        if np.degrees(np.arccos(np.clip(-down[i] @ up, -1, 1))) > c["freeze_deg"]:
            known = i
            break
    angle = np.full(len(t), np.nan)
    if known is not None:
        cosang = np.clip(np.einsum("ij,j->i", -down[best:], up), -1, 1)  # RepSense segment_angle
        angle[best:] = np.degrees(np.arccos(cosang))

    rate = np.zeros(len(t))
    rate[1:] = np.diff(angle) / np.diff(t)  # backward difference: past samples only
    rate = np.where(np.isnan(rate), 0.0, rate)
    direction = np.where(rate > c["move_dps"], 1, np.where(rate < -c["move_dps"], -1, 0))
    return AngleTrack(t=t, angle_deg=angle, rate_dps=rate, direction=direction,
                      hang_known_s=float(t[known]) if known is not None else float("nan"))
