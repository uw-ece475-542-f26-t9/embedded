#!/usr/bin/env python3
"""Where does a recording's background noise come from? Best run on relaxed-arm rest recordings.

For each recording it reports:
  - loudness after the standard 20-450 Hz band-pass and hum filters
  - share of that power below 150 Hz: muscle activity puts ~80%+ there, flat electrical noise ~30-40%
    (Person B's first rest showed 90%: the arm wasn't relaxed, not a wiring problem)
  - 60 Hz hum as a share of the band power, measured before the hum filter
  - broadband hiss above 1,000 Hz, in ADC counts RMS (the converter's own background noise)
  - spikes standing 10 dB above their neighbours between 20 and 450 Hz

Usage: python diagnostics/noise_sources.py A_rest B_rest2 [--start 1 --end -1]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import BAND, anti_alias, band_power, clean, find_spikes, load, rms  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--start", type=float, default=1, help="seconds to skip at the start")
    ap.add_argument("--end", type=float, default=-1, help="stop this many seconds before the end (negative)")
    args = ap.parse_args()

    for name in args.names:
        rec = load(name)
        x, fs = clean(rec.raw, rec.fs)
        seg = x[int(args.start * fs):len(x) + int(args.end * fs)]
        unfiltered, fs2 = anti_alias(rec.raw, rec.fs)
        unfiltered = signal.sosfiltfilt(signal.butter(4, BAND, btype="bandpass", fs=fs2, output="sos"), unfiltered)
        useg = unfiltered[int(args.start * fs2):len(unfiltered) + int(args.end * fs2)]
        centred = rec.raw - rec.raw.mean()
        print(f"{rec.name}:")
        print(f"  loudness (filtered)          {rms(seg):6.1f} counts")
        print(f"  share below 150 Hz           {band_power(seg, fs, 20, 150) / band_power(seg, fs, *BAND):6.0%}"
              "   (muscle ~80%+, electrical ~30-40%)")
        print(f"  60 Hz hum share              {band_power(useg, fs2, 55, 65) / band_power(useg, fs2, *BAND):6.0%}")
        print(f"  hiss above 1,000 Hz          {np.sqrt(band_power(centred, rec.fs, 1000, rec.fs / 2)):6.1f} counts RMS")
        print(f"  spikes (20-450 Hz)           {np.round(find_spikes(rec.raw, rec.fs)).tolist()}")


if __name__ == "__main__":
    main()
