"""Script 1 of 2: the 3 cold max pulls, recorded as soon as the participant arrives (electrodes on,
before any warm-up). Everything else is script 2, `python -m pipeline.lab`.

    python -m pipeline.cold P01
"""

from .lab import COLD_STEPS, main

if __name__ == "__main__":
    main(COLD_STEPS, "Cold pulls")
