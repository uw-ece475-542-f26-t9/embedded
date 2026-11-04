#!/usr/bin/env python3
"""Write simulated recordings with known answers, in the same format the logger saves.

"Good" recordings contain what each test should detect; "bad" ones are built to fail:
  rest / rest_10500     clean rest at two rates; bad_* adds a 300 Hz tone at 10,000/s that moves to
                        800 Hz at 10,500/s (folded-down interference)
  squeeze               3 squeezes of 3 s
  light, heavy1-3       holds: light stays flat; heavy middle pitch falls and loudness rises
  bad_heavy1-2          heavy holds with no fatigue at all
  drop_heavy1           heavy1 with the arm giving out at 40 s
  set1-2, clight        curls: middle pitch falls ~28% across reps; clight stays flat. (With one 400 ms
                        window per rep, a 16-rep set with an ~18% drop is NOT reliably detected -- per-rep
                        middle pitch is that noisy -- so these sets use a clear drop.)

Usage: python tests/make_sim.py OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

FS = 10000
rng = np.random.default_rng(0)


def emg(n, fs, fc, amp):
    """Band-limited noise centred near fc Hz, built block by block so fc and amp can drift."""
    fc, amp = np.broadcast_to(fc, n), np.broadcast_to(amp, n)
    out, blk = np.zeros(n), int(0.25 * fs)
    for i in range(0, n, blk):
        j, c = min(n, i + blk), fc[i]
        sos = signal.butter(2, [c / 2, c * 2], btype="bandpass", fs=fs, output="sos")
        x = signal.sosfilt(sos, rng.standard_normal(j - i + 2000))[2000:]
        out[i:j] = amp[i] * x / x.std()
    return out


def save(out, name, raw, env, fs, markers=(), tone=None):
    t = np.arange(len(raw)) / fs
    raw = 2048 + raw + 4 * rng.standard_normal(len(raw)) + 3 * np.sin(2 * np.pi * 60 * t)
    if tone:
        raw += 20 * np.sin(2 * np.pi * tone * t)
    data = np.column_stack([np.clip(np.round(raw), 0, 4095), np.clip(np.round(env), 0, 4095)]).astype(int)
    np.savetxt(out / f"{name}.csv", data, fmt="%d", delimiter=",", header="raw,env", comments="")
    (out / f"{name}.json").write_text(json.dumps({
        "name": name, "rate_hz": fs, "board_overflows": 0, "resyncs": 0, "board_reboots": 0,
        "markers": [[int(m * fs), label] for m, label in markers], "ok": True}))


def hold(out, name, secs, f0, f1, a0, a1):
    n = (secs + 2) * FS
    k = np.clip((np.arange(n) / FS - 1) / secs, 0, 1)
    save(out, name, emg(n, FS, f0 + (f1 - f0) * k, a0 + (a1 - a0) * k), np.full(n, 800.0), FS,
         markers=[(1, "start"), (secs + 1, "end")])


def curls(out, name, reps, f0, f1, amp):
    n = (3 + 2 * reps + 3) * FS
    fc, gate = np.full(n, float(f0)), np.zeros(n)
    for r in range(reps):
        s = int((3 + 2 * r) * FS)
        gate[s:s + FS] = np.hanning(FS)
        fc[s:s + 2 * FS] = f0 + (f1 - f0) * r / max(1, reps - 1)
    save(out, name, emg(n, FS, fc, amp) * gate, 200 + 1500 * gate, FS, markers=[(2.9, "start"), (3 + 2 * reps, "end")])


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for prefix, tone in (("good", None), ("bad", 300)):
        save(out, f"{prefix}_rest", np.zeros(10 * FS), np.full(10 * FS, 200.0), FS, tone=tone)
        save(out, f"{prefix}_rest_10500", np.zeros(10 * 10500), np.full(10 * 10500, 200.0), 10500,
             tone=800 if tone else None)
    n, gate = 40 * FS, np.zeros(40 * FS)
    for s in (5, 18, 31):
        gate[s * FS:(s + 3) * FS] = 1
    save(out, "good_squeeze", emg(n, FS, 85, 150) * gate, 200 + 1500 * gate, FS)

    hold(out, "good_light", 90, 85, 85, 30, 30)
    hold(out, "good_heavy1", 60, 90, 68, 120, 170)
    hold(out, "good_heavy2", 55, 89, 69, 120, 170)
    hold(out, "good_heavy3", 50, 88, 70, 120, 165)
    hold(out, "bad_heavy1", 60, 90, 90, 120, 120)
    hold(out, "bad_heavy2", 60, 90, 90, 120, 120)

    d = np.loadtxt(out / "good_heavy1.csv", delimiter=",", skiprows=1)
    d[41 * FS:, 0] = np.round(2048 + 4 * rng.standard_normal(len(d) - 41 * FS))  # arm gives out at 40 s
    np.savetxt(out / "drop_heavy1.csv", d, fmt="%d", delimiter=",", header="raw,env", comments="")
    (out / "drop_heavy1.json").write_text((out / "good_heavy1.json").read_text())

    curls(out, "good_set1", 20, 90, 65, 150)
    curls(out, "good_set2", 16, 89, 64, 150)
    curls(out, "good_clight", 20, 85, 85, 60)
    print(f"Simulated recordings written to {out}")


if __name__ == "__main__":
    main()
