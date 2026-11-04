#!/usr/bin/env python3
"""One table for every static hold: effort, length, middle-pitch trend and loudness trend.

Effort = the hold's starting loudness as a % of that person's strongest squeeze (from their Test 1
squeeze recording). "Chance random" is the p-value of the straight-line trend.

Usage: python diagnostics/hold_summary.py A_squeeze:A_light,A_heavy1,A_heavy2,A_heavy3 \\
                                          B_squeeze2:B_light2,B_try1,B_try2,B_try3
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import hold_series, load, max_squeeze_level, trend  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", help="SQUEEZE:HOLD1,HOLD2,... one group per person")
    args = ap.parse_args()

    print(f"{'hold':10s}{'effort':>8}{'secs':>6}{'pitch change':>14}{'chance random':>15}"
          f"{'loudness change':>17}{'chance random':>15}")
    for group in args.groups:
        squeeze, holds = group.split(":")
        mvc = max_squeeze_level(squeeze)
        for name in holds.split(","):
            t, pitch, level = hold_series(load(name))
            _, pitch_fit, pitch_change, _ = trend(t, pitch)
            _, level_fit, level_change, _ = trend(t, level)
            print(f"{name:10s}{level[:10].mean() / mvc:>8.0%}{t[-1] + 0.5:>6.0f}"
                  f"{pitch_change:>+13.1f}%{pitch_fit.pvalue:>15.1e}"
                  f"{level_change:>+16.1f}%{level_fit.pvalue:>15.1e}")


if __name__ == "__main__":
    main()
