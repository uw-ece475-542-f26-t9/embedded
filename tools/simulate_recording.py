#!/usr/bin/env python3
"""Makes fake recordings in the exact capture format, to test the analysis.

The EMG is band-limited noise whose spectrum compresses toward lower
frequencies as a fatiguing set goes on (and doesn't in a control set), with
60 and 120 Hz mains pickup on top. Arm motion, turning points and reps follow
a 2 s up / 2 s down tempo. This only proves the pipeline works; it says
nothing about real muscle.

    python tools/simulate_recording.py --out /tmp/sim
    python tools/fatigue_features.py /tmp/sim
"""

from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np
from scipy import signal

import capture_format as cf

FS = 2000
IMU_HZ = 50
UP_S, DOWN_S, PAUSE_S, REST_S = 2.0, 2.0, 0.4, 3.0
ROM_DEG = 130.0
COUNT_DEG = 45.0  # the rep counter counts a rep this far down from the top


def emg_segment(rng, n, scale, amp):
    """n samples of muscle-like noise; scale < 1 compresses the spectrum."""
    sos = signal.butter(2, [40 * scale, 220 * scale], btype="bandpass", fs=FS, output="sos")
    return amp * signal.sosfilt(sos, rng.standard_normal(n + 400))[400:] / 0.25


def simulate(rng, reps: int, fatigue: bool, rest_only=False):
    """Returns EMG (mV), arm angle per EMG sample, turning points, rep stamps."""
    emg, angle, turns, rep_stamps = [], [], [], []
    pos = 0

    def rest(seconds):
        nonlocal pos
        n = int(seconds * FS)
        emg.append(emg_segment(rng, n, 1.0, 2.0))
        angle.append(np.full(n, angle[-1][-1] if angle else 0.0))
        pos += n

    rest(REST_S if not rest_only else 20)
    if rest_only:
        return np.concatenate(emg), np.concatenate(angle), turns, rep_stamps
    for k in range(reps):
        p = k / max(reps - 1, 1) if fatigue else 0.0
        # Fatigue: spectrum compresses ~25% and amplitude rises ~30%;
        # every rep also scatters a little.
        scale = (1 - 0.25 * p) * (1 + 0.02 * rng.standard_normal())
        amp = 40 * (1 + 0.3 * p)
        turns.append((pos, 0.0, -1))
        n_up, n_down = int(UP_S * FS * (1 + 0.15 * p)), int(DOWN_S * FS)
        t = np.arange(n_up) / n_up
        env = 0.3 + 0.7 * np.sin(np.pi * t)  # effort peaks mid-lift
        emg.append(emg_segment(rng, n_up, scale, amp) * env)
        angle.append(ROM_DEG * (1 - np.cos(np.pi * t)) / 2)
        pos += n_up
        turns.append((pos, ROM_DEG, +1))
        t = np.arange(n_down) / n_down
        down = ROM_DEG - ROM_DEG * (1 - np.cos(np.pi * t)) / 2
        emg.append(emg_segment(rng, n_down, scale, amp * 0.45))
        angle.append(down)
        rep_stamps.append(pos + int(np.argmax(down <= ROM_DEG - COUNT_DEG)))
        pos += n_down
        rest(PAUSE_S)
    rest(REST_S)
    return np.concatenate(emg), np.concatenate(angle), turns, rep_stamps


def write(path_base, emg, angle, turns, rep_stamps, meta, rng, boot_reps=0, rate_error=0.003):
    t = np.arange(len(emg)) / FS
    mains = 6.0 * np.sin(2 * np.pi * 60 * t + 0.3) + 1.0 * np.sin(2 * np.pi * 120 * t)
    mv = 1650 + emg + mains
    codes = np.clip(np.round(mv * 10), 0, 65535).astype("<u2")
    out = bytearray()
    first = 20000  # pretend the ESP32 has been on for 10 s
    events = []
    for i in range(0, len(codes) - 31, 32):
        events.append((first + i, cf.frame(cf.EMG, struct.pack("<IBB", first + i, 32, 0) + codes[i : i + 32].tobytes())))
    for i in range(0, len(emg), FS // IMU_HZ):
        a = angle[i]
        w = (angle[min(i + 1, len(angle) - 1)] - angle[i]) * FS
        still = abs(w) < 2
        events.append((first + i, cf.frame(cf.IMU, struct.pack(
            "<I6hfB", first + i, 0, int(1000 * np.sin(np.radians(a))), int(1000 * np.cos(np.radians(a))),
            int(w * 10), 0, 0, float(a), int(still)))))
    for stamp, ang, top in turns:
        events.append((first + stamp + 1, cf.frame(cf.TURN, struct.pack("<Ifb", first + stamp, ang, top))))
    for k, stamp in enumerate(rep_stamps, 1):
        events.append((first + stamp + 2, cf.frame(cf.REP, struct.pack("<IHH", first + stamp, boot_reps + k, int(ROM_DEG * 10)))))
    # STATUS once a second; millis runs slightly off the nominal EMG rate,
    # like a real ADC clock.
    for i in range(0, len(emg), FS):
        ms = int((first + i) / (FS * (1 + rate_error)) * 1000)
        events.append((first + i, cf.frame(cf.STATUS, struct.pack("<BHHHHIIII", 1, FS, 8000, IMU_HZ, 10, first + i, ms, 0, 0))))
    events.sort(key=lambda e: e[0])
    for _, f in events:
        out += f
    with open(path_base + ".bin", "wb") as f:
        f.write(out)
    with open(path_base + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for subject in ("S01", "S02"):
        for session in (1, 2):
            base = os.path.join(args.out, f"{subject}_s{session}")
            sets = [("rest", 0, False), ("control", 12, False), ("fatigue", int(rng.integers(11, 15)), True),
                    ("fatigue", int(rng.integers(10, 14)), True)]
            for i, (kind, reps, fat) in enumerate(sets):
                emg, angle, turns, rep_stamps = simulate(rng, reps, fat, rest_only=kind == "rest")
                meta = {"subject": subject, "session": session, "set_type": kind,
                        "failure_rep": reps if kind == "fatigue" else None, "rpe": 10 if fat else 4,
                        "notes": "simulated"}
                write(f"{base}_{kind}{i}", emg, angle, turns, rep_stamps, meta, rng, boot_reps=10 * i)
    print(f"Wrote simulated recordings to {args.out}")


if __name__ == "__main__":
    main()
