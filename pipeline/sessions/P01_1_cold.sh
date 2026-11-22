#!/bin/zsh
# P01, script 1 of 2: the 3 cold max pulls. Run as soon as P01 arrives: electrodes on, meter check done, no warm-up yet.
cd "${0:A:h}/.." && .venv/bin/python -m pipeline.cold P01
