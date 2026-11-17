"""Step 5b: soft angle bins. Every EMG sample is tagged with the forearm angle and direction, and
each sweep of the arm through a bin becomes one pass.

Bins: centres every 15 deg (15-135), each reaching 15 deg either side with a weight that fades
linearly from 1 at the centre to 0 at the edge, so neighbouring bins overlap and a sample near a
boundary counts partly in both. A small angle error then nudges a weight instead of flipping the
sample into the other bin. Lifting and lowering have separate bins.

A pass is one continuous run of samples inside a bin, moving the right way (a short stall inside
the bin, as in a grinding last rep, doesn't split it; the stalled samples get weight 0). It carries its samples'
weights (which also taper the spectrum, like a window) and when the board would know it was done.
Skipped samples (cleaning: jolts, gaps) keep weight 0. Nothing here counts reps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .angle import AngleTrack
from .clean import Cleaned


@dataclass
class Pass:
    bin_deg: float
    direction: int          # +1 lifting, -1 lowering
    start: int              # first EMG sample index
    stop: int               # one past the last
    weights: np.ndarray     # per sample, 0..1
    t_mid_s: float          # weighted centre time
    t_known_s: float        # when the board knows the pass is done (end + latency)
    weight_sum: float       # weighted samples
    speed_dps: float        # mean turning speed through the bin

    @property
    def n(self) -> int:
        return self.stop - self.start


def bin_weights(angle: np.ndarray, centre: float, halfwidth: float) -> np.ndarray:
    return np.clip(1 - np.abs(angle - centre) / halfwidth, 0, None)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _merge(runs, inside, reversed_, gap):
    """Join runs separated by a short stall that stayed inside the bin without reversing."""
    out = []
    for i, j in runs:
        if out:
            pi, pj = out[-1]
            if i - pj <= gap and inside[pj:i].all() and not reversed_[pj:i].any():
                out[-1] = (pi, j)
                continue
        out.append((i, j))
    return out


def find_passes(cl: Cleaned, track: AngleTrack, cfg: dict, cut_s: float | None = None) -> list[Pass]:
    """Every pass in the recording, in the order the board would finish them. With cut_s, only
    what the board would know by then (the real-time check)."""
    b, a = cfg["bins"], cfg["angle"]
    t = np.arange(len(cl.emg)) / cl.fs
    angle, rate = track.at(t, a["max_imu_gap_s"])
    direction = np.where(rate > a["move_dps"], 1, np.where(rate < -a["move_dps"], -1, 0))
    usable = ~cl.skip & ~np.isnan(angle)
    imu_step = float(np.median(np.diff(track.t))) if len(track.t) > 1 else 0.0
    latency = cl.latency_s + imu_step  # cleaning latency + waiting for the next IMU sample
    dirs = (1, -1) if b["lowering"] else (1,)
    out = []
    for d in dirs:
        moving = direction == d
        for c in b["centres_deg"]:
            w_all = bin_weights(np.nan_to_num(angle, nan=-1e9), c, b["halfwidth_deg"])
            inside = w_all > 0
            for i, j in _merge(_runs(inside & moving), inside, direction == -d, int(b["merge_gap_s"] * cl.fs)):
                w = w_all[i:j] * usable[i:j] * moving[i:j]
                total = float(w.sum())
                if total < b["min_weight"] or j - i < b["min_span"]:
                    continue
                p = Pass(bin_deg=float(c), direction=d, start=int(i), stop=int(j), weights=w,
                         t_mid_s=float(np.sum(w * t[i:j]) / total),
                         t_known_s=float(max(t[j - 1], track.hang_known_s) + latency),
                         weight_sum=total, speed_dps=float(np.mean(np.abs(rate[i:j]))))
                if cut_s is None or p.t_known_s <= cut_s:
                    out.append(p)
    return sorted(out, key=lambda p: (p.t_known_s, p.bin_deg))


def pass_spectrum(x: np.ndarray, w: np.ndarray, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density (mV^2/Hz) of one pass, its bin weights as the taper. A pass longer
    than nfft (slow arm at the ends of the curl) is split into chunks whose power is added, so every
    spectrum lands on the same nfft frequency bins as the noise floor it's cleaned against."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    xm = x - np.sum(w * x) / max(np.sum(w), 1e-12)
    power = np.zeros(nfft // 2 + 1)
    for k in range(0, len(x), nfft):
        power += np.abs(np.fft.rfft(xm[k:k + nfft] * w[k:k + nfft], nfft)) ** 2
    p = power / (fs * max(np.sum(w**2), 1e-12))
    p[1:-1] *= 2
    return np.fft.rfftfreq(nfft, 1 / fs), p
