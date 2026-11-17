"""Reader for the sensor-testing recordings (emg-sensor-testing/data: NAME.csv.gz + NAME.json).

Those came from the dev-kit logger: RAW and ENV as 12-bit ADC counts at about 10,000 readings/s.
This turns RAW into mV and brings it down to about 2,000/s, like the team firmware, so every
later stage can be built and checked on real EMG before the wearable is back.

The rate change uses a linear-phase FIR (scipy's resample_poly) over the whole file. That looks
ahead, which is fine here: it's a one-off format conversion of old data, not part of the live
pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from .session import Health, Session

FULL_SCALE_COUNTS = 4095


def read_legacy(path: str | Path, cfg: dict) -> Session:
    c = cfg["input"]
    path = Path(path)
    stem = path.name.removesuffix(".gz").removesuffix(".csv")
    sidecar = path.with_name(stem + ".json")
    info = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    raw = pd.read_csv(path, usecols=["raw"])["raw"].to_numpy(float)
    rate = float(info.get("rate_hz") or info.get("requested_rate_hz") or 10000)
    q = max(1, round(rate / c["emg_hz"]))
    # padtype="line": the default pads the ends with zeros, so the output ramps from ~1,440 mV
    # down to 0 at each end of the file, a huge false swing.
    emg = resample_poly(raw * c["legacy_full_scale_mv"] / FULL_SCALE_COUNTS, 1, q, padtype="line")

    rail = c["legacy_rail_counts"]
    health = Health(
        emg_readings=len(emg),
        raw_readings=len(raw),
        clipped=int(np.sum((raw <= rail) | (raw >= FULL_SCALE_COUNTS - rail))),
        adc_overflows=int(info.get("board_overflows", 0)),
        crc_errors=int(info.get("resyncs", 0)),
        reboots=int(info.get("board_reboots", 0)),
    )
    markers = [(float(i) / rate, str(label)) for i, label in info.get("markers", [])]
    meta = {k: v for k, v in info.items() if k != "markers"}
    meta.update(original_rate_hz=rate, decimation=q)
    return Session(name=stem, source="legacy", emg=emg, emg_hz=rate / q, markers=markers,
                   meta=meta, health=health)
