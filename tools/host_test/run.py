#!/usr/bin/env python3
"""Tests the firmware's fatigue features and model on a computer.

Builds replay.cpp (the firmware's rep counter, fatigue_features.h and
fatigue_model.h, no Arduino), feeds it a simulated fatiguing set and a
control set from simulate_recording.py, and checks:

  - the model gives the same answers as tools/train_model.py (self-test)
  - every rep is counted and its lift found, including the first, with the
    board mounted straight, tilted, or upside down (which flips which way
    the rep counter's angle goes on the way up)
  - EMG RMS and median frequency match numpy/scipy doing the same maths on
    the same samples, so training and the device can't drift apart
  - the features move the way the simulation was built to: in the fatigued
    set speed drops, EMG rises and its median frequency falls; in the
    control set they stay put

    python tools/host_test/run.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import simulate_recording as sim  # noqa: E402

FS = 2000
FFT_N, FFT_HOP = 512, 256
# fatigue_features.h's filter, rebuilt the same way.
SOS = np.vstack([signal.butter(2, [20, 400], btype="bandpass", fs=FS, output="sos"),
                 np.concatenate(signal.iirnotch(60, 30, fs=FS))])
COLUMNS = ["rep", "lift_from", "lift_to", "emg_ok", "has_baseline", "peak_dps",
           "moving_s", "rom_deg", "emg_rms_mv", "mdf_hz", "velocity_loss", "duration_ratio",
           "rom_ratio", "emg_ratio", "mdf_ratio", "rir"]


def build(out_dir: str) -> str:
    exe = os.path.join(out_dir, "replay")
    subprocess.run(["clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-o", exe,
                    os.path.join(HERE, "replay.cpp")], check=True)
    return exe


def rotation(axis, degrees):
    """Rotation matrix about `axis` (Rodrigues)."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = np.radians(degrees)
    return np.eye(3) + np.sin(a) * kx + (1 - np.cos(a)) * kx @ kx


MOUNTS = {
    "straight": np.eye(3),
    "tilted 40°": rotation([1, 1, 0], 40),
    "upside down": rotation([0, 1, 0], 180),
}


def make_set(rng, reps: int, fatigue: bool, mount=np.eye(3)):
    """Simulated EMG at the ADC pin (mV) and the IMU lines for one set, with
    the IMU turned by `mount` relative to the forearm."""
    emg, angle, _, _ = sim.simulate(rng, reps, fatigue)
    t = np.arange(len(emg)) / FS
    mv = 1650 + emg + 6 * np.sin(2 * np.pi * 60 * t + 0.3) + np.sin(2 * np.pi * 120 * t)
    lines, step = [], FS // 50
    bias = np.array([1.5, -0.8, 0.6])  # deg/s, like a real MPU-6050
    for i in range(len(emg)):
        if i % step == 0 and i + 1 < len(emg):
            a = np.radians(angle[i])
            w = (angle[i + 1] - angle[i]) * FS
            acc = mount @ np.array([0, np.sin(a), np.cos(a)]) + 0.003 * rng.standard_normal(3)
            gyro = mount @ np.array([w, 0, 0]) + bias + 0.5 * rng.standard_normal(3)
            # The firmware stamps IMU reads with the EMG samples so far.
            lines.append(f"I {i} {acc[0]:.5f} {acc[1]:.5f} {acc[2]:.5f} "
                         f"{gyro[0]:.4f} {gyro[1]:.4f} {gyro[2]:.4f} 0.02")
        lines.append(f"E {i} {mv[i]:.3f}")
    return mv, "\n".join(lines) + "\n"


def run(exe: str, stdin: str):
    out = subprocess.run([exe], input=stdin, capture_output=True, text=True, check=True).stdout
    selftest, reps = None, []
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "selftest":
            selftest = float(parts[1])
        elif parts[0] == "rep":
            reps.append(dict(zip(COLUMNS, map(float, parts[1:]))))
    return selftest, reps


def reference(mv: np.ndarray, rep: dict):
    """EMG RMS and MDF the way fatigue_features.h computes them, in numpy."""
    filtered = np.round(signal.sosfilt(SOS, mv) * 10) / 10  # stored in 0.1 mV
    a, b = int(rep["lift_from"]), int(rep["lift_to"])
    rms = np.sqrt(np.mean(filtered[a:b] ** 2))
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(FFT_N) / FFT_N)
    p = sum(np.abs(np.fft.rfft(filtered[s : s + FFT_N] * hann)) ** 2
            for s in range(a, b - FFT_N + 1, FFT_HOP))
    bin_hz = FS / FFT_N
    lo, hi = int(np.ceil(20 / bin_hz)), int(np.floor(400 / bin_hz))
    cum = np.cumsum(p[lo : hi + 1])
    k = int(np.searchsorted(cum, cum[-1] / 2))
    before = cum[k - 1] if k else 0.0
    mdf = (lo + k - 0.5 + (cum[-1] / 2 - before) / p[lo + k]) * bin_hz
    return rms, mdf


def main() -> int:
    rng = np.random.default_rng(7)
    failures = []

    def check(ok, what):
        print(f"  [{'pass' if ok else 'FAIL'}] {what}")
        if not ok:
            failures.append(what)

    with tempfile.TemporaryDirectory() as tmp:
        exe = build(tmp)
        sets = [("fatiguing set", 12, True, "straight"), ("control set", 12, False, "straight"),
                ("fatiguing set", 12, True, "tilted 40°"), ("fatiguing set", 12, True, "upside down")]
        for name, reps, fatigue, mount in sets:
            mv, stdin = make_set(rng, reps, fatigue, MOUNTS[mount])
            selftest, out = run(exe, stdin)
            print(f"\n{name}, {reps} simulated reps, board {mount}:")
            print("  rep  peak°/s  moving s  rom°   emg mV  mdf Hz   speed-  emg x  mdf x   RIR")
            for r in out:
                rir = f"{r['rir']:5.1f}" if r["rir"] >= 0 else "    -"
                print(f"  {int(r['rep']):3d}  {r['peak_dps']:7.1f}  {r['moving_s']:8.2f}  {r['rom_deg']:4.0f}"
                      f"  {r['emg_rms_mv']:7.1f}  {r['mdf_hz']:6.1f}  {r['velocity_loss']:6.1%}"
                      f"  {r['emg_ratio']:5.2f}  {r['mdf_ratio']:5.2f}  {rir}")

            check(selftest is not None and selftest < 0.01, f"model self-test ({selftest:.6f} reps)")
            check(len(out) == reps and [int(r["rep"]) for r in out] == list(range(1, reps + 1)),
                  f"all {reps} lifts measured, from the first ({len(out)})")
            check(all(abs(r["rom_deg"] - sim.ROM_DEG) < 10 for r in out), "lift found (rom ~130°)")
            err_rms = max(abs(r["emg_rms_mv"] / reference(mv, r)[0] - 1) for r in out)
            err_mdf = max(abs(r["mdf_hz"] - reference(mv, r)[1]) for r in out)
            check(err_rms < 0.002, f"EMG RMS matches numpy (worst {err_rms:.4%})")
            check(err_mdf < 0.2, f"EMG MDF matches numpy (worst {err_mdf:.3f} Hz)")

            # The end of the set: the last 3 reps, since each rep scatters.
            last = {k: np.mean([r[k] for r in out[-3:]]) for k in ("velocity_loss", "emg_ratio", "mdf_ratio")}
            if fatigue:
                check(last["velocity_loss"] > 0.05, f"speed dropped ({last['velocity_loss']:.0%})")
                # The simulator's EMG gets 30% louder but its narrowing
                # band loses power, so in the 20-400 Hz band it's ~x1.1.
                check(last["emg_ratio"] > 1.05, f"EMG rose (x{last['emg_ratio']:.2f})")
                check(last["mdf_ratio"] < 0.88, f"MDF fell (x{last['mdf_ratio']:.2f})")
            else:
                after = out[3:]  # after the baseline reps
                worst = max(after, key=lambda r: abs(r["mdf_ratio"] - 1))
                check(all(abs(r["velocity_loss"]) < 0.05 for r in after), "speed steady")
                check(all(0.85 < r["emg_ratio"] < 1.15 for r in after), "EMG steady")
                check(abs(worst["mdf_ratio"] - 1) < 0.08, f"MDF steady (worst x{worst['mdf_ratio']:.2f})")

    print("\nAll checks passed." if not failures else f"\n{len(failures)} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
