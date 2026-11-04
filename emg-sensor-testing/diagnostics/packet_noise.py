#!/usr/bin/env python3
"""Look for a glitch that repeats once per USB packet.

Stacks the RAW signal one packet period at a time and averages: random noise averages away, anything
tied to sending a packet survives. The firmware first sent 128 readings per packet (78 packets/s at
10,000 readings/s), and in Person B's wiring that glitch put spikes at 78, 156, 234 Hz... inside the
muscle band. The current firmware sends 10 readings per packet, so the glitch repeats at 1,000 Hz and
every echo of it lands outside 20-450 Hz.

Usage: python diagnostics/packet_noise.py A_rest B_rest --period 128     (old firmware)
       python diagnostics/packet_noise.py B_rest2 --period 10            (current firmware)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="relaxed-arm recordings work best")
    ap.add_argument("--period", type=int, default=10, help="readings per packet in the firmware that recorded it")
    args = ap.parse_args()

    for name in args.names:
        rec = load(name)
        fs = int(round(rec.fs))
        x = rec.raw[fs:-fs] - rec.raw.mean()  # skip the first and last second
        folded = x[:len(x) // args.period * args.period].reshape(-1, args.period).mean(axis=0)
        print(f"{rec.name}: pattern repeating every {args.period} readings ({rec.fs / args.period:,.1f}/s) is "
              f"{np.ptp(folded):.1f} counts peak-to-peak; random noise level {x.std():.1f} counts")


if __name__ == "__main__":
    main()
