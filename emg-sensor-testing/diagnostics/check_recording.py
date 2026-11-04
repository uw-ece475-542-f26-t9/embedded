#!/usr/bin/env python3
"""Chunk-by-chunk health check of one recording: is RAW centred, clipping, or acting like a broken input?

Healthy RAW sits near 1,900 at rest and swings both ways when the muscle works. Two failure
signatures seen during testing:
  - B_check: RAW sits at 0 at rest and rises in the *shape of ENV* -> the firmware was reading an
    unconnected pin (GPIO 1 while RAW was wired to GPIO 4), which picks up a ghost of the previous
    channel (ENV).
  - B_light: RAW slams between 0 and 4095 with a constant ~297 Hz middle pitch while ENV stays normal
    -> the RAW jumper came loose; the sensor and electrodes were fine.

Usage: python diagnostics/check_recording.py B_check [--compare A_squeeze] [--chunk 1.0]
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import FULL_SCALE, RESULTS_DIR, ROOT, load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--compare", help="a healthy recording to plot alongside")
    ap.add_argument("--chunk", type=float, default=1.0, help="seconds per row")
    args = ap.parse_args()

    rec = load(args.name)
    raw, env, fs = rec.raw, rec.env, rec.fs
    rails = (raw <= 5) | (raw >= FULL_SCALE - 5)
    print(f"{rec.name}: RAW {raw.min():.0f}-{raw.max():.0f}, median {np.median(raw):.0f}, "
          f"at rails {rails.mean():.2%}; ENV {env.min():.0f}-{env.max():.0f}")
    n = int(args.chunk * fs)
    for i in range(0, len(raw), n):
        r, e = raw[i:i + n], env[i:i + n]
        print(f"  {i / fs:5.1f}s  RAW median {np.median(r):5.0f}  spread {r.std():6.1f}  min {r.min():5.0f}"
              f"  max {r.max():5.0f}  at rails {np.mean((r <= 5) | (r >= FULL_SCALE - 5)):6.1%}"
              f"  ENV median {np.median(e):5.0f}")

    rows = 2 if args.compare else 1
    fig, axes = plt.subplots(rows, 1, figsize=(11, 3.2 * rows), squeeze=False)
    t = np.arange(len(raw)) / fs
    axes[0, 0].plot(t, raw, lw=0.4, label="RAW")
    axes[0, 0].plot(t, env, lw=0.6, label="ENV")
    axes[0, 0].set(title=f"{rec.name}: RAW and ENV", xlabel="seconds", ylabel="ADC reading", ylim=(0, FULL_SCALE))
    axes[0, 0].legend(loc="upper left")
    if args.compare:
        ref = load(args.compare)
        tr = np.arange(len(ref.raw)) / ref.fs
        loud = int(np.argmax(np.abs(ref.raw - np.median(ref.raw))))  # show 1 s around its biggest swing
        z = (tr > tr[loud] - 0.5) & (tr < tr[loud] + 0.5)
        axes[1, 0].plot(tr[z], ref.raw[z], lw=0.5)
        axes[1, 0].set(title=f"For comparison: healthy RAW in {ref.name}", xlabel="seconds", ylabel="ADC reading")
    out = RESULTS_DIR / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"check_{rec.name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Plot: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


if __name__ == "__main__":
    main()
