#!/usr/bin/env python3
"""How noisy is one half-second middle-pitch reading, and how big is the real drift per slice?

Both are measured around each hold's straight-line trend. They are the two numbers a Kalman filter
needs: the scatter is its measurement noise (measured, not tuned), and the drift bounds how fast the
true value can move. Result from testing: ~10% scatter per slice in every hold, against a real drift
of only 0.02-0.12% per slice -- one reading is almost pure noise, so any tracker needs ~25 slices
(~12 s) of memory for +-2% precision.

Usage: python diagnostics/slice_noise.py A_light B_light2 A_heavy1 A_heavy2 A_heavy3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import hold_series, load, trend  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="hold recordings")
    args = ap.parse_args()

    print(f"{'hold':10s}{'scatter per slice':>20}{'real drift per slice':>23}")
    for name in args.names:
        t, pitch, _ = hold_series(load(name))
        pct, fit, _, start = trend(t, pitch)
        line = 100 * (fit.intercept + fit.slope * t) / start
        print(f"{name:10s}{(pct - line).std():>19.1f}%{100 * fit.slope * 0.5 / start:>+22.3f}%")


if __name__ == "__main__":
    main()
