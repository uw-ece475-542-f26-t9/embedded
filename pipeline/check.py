"""Check recordings before using them: what's in each one and whether it's usable. Rest
recordings also get the hum and interference check (step 4).

    python -m pipeline.check recordings/P01_set1.bin A_squeeze B_rest2

Exit code 1 if any recording should be rejected, so a lab script can stop on it.
"""

from __future__ import annotations

import argparse
import sys

from .clean import rest_check
from .config import load_config
from .load import load_session


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--config", help="config.toml to use (default: the repo's)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bad = 0
    for name in args.recordings:
        s = load_session(name, cfg)
        problems = s.problems(cfg)
        rc = rest_check(s, cfg) if s.meta.get("recording") == "rest" or "rest" in s.name else None
        if rc:
            problems += rc.problems
        bad += bool(problems)
        print(("REJECT  " if problems else "OK      ") + s.summary())
        if rc:
            peaks = ", ".join(f"{hz:g} Hz {kind}" for hz, _, kind in rc.lines) or "none"
            print(f"    rest:    hum {rc.hum_mv} mV, noise {rc.noise_mv} mV, peaks: {peaks}")
        for p in problems:
            print(f"    problem: {p}")
        for n in s.notes():
            print(f"    note:    {n}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
