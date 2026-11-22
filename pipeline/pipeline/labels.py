"""Step 10, item 2: video labels into the recordings (DATA_COLLECTION_PROTOCOL.md §5).

Two raters watch each set's video independently and type times *as the video shows them*:
  video_sync_s       the first of the three sync flicks
  set_start_video_s  the start of the first lift
  stop_point_video_s the start of the first bad rep (torso swing, upper-arm drift, partial, missed beat)
  failure_video_s    set 3 only: the rep the participant couldn't complete

The flicks line the video up with the recording: they are sharp twists of the forearm, which the wrist
IMU sees as spikes in rotation *about the forearm axis* (a twist doesn't change the forearm angle, so
it can't be mistaken for a lift). recording time = video time - video_sync_s + the flick time found in
the IMU.

Agreement (reported, as the protocol asks): same rep if the two stop points are less than half a
metronome cycle apart, within one rep if within 1.5 cycles. Disagreements beyond one rep are settled by
discussion: add a row with rater "agreed". The label written is the "agreed" row if there is one,
otherwise the mean of two raters within one rep; anything else is left unlabelled and listed.

    python -m pipeline.labels template recordings/          writes recordings/labels.csv to fill in
    python -m pipeline.labels apply recordings/              checks agreement, writes labels into the .json files
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .angle import forearm_angle
from .config import load_config
from .load import load_session

COLUMNS = ["recording", "rater", "video_sync_s", "set_start_video_s", "stop_point_video_s", "failure_video_s", "notes"]


def find_flicks(rec_path: Path, cfg: dict, min_dps: float | None = None, window_s: float = 2.5) -> float | None:
    """Time (recording clock) of the first sync flick: the start of the first cluster of 3+ sharp
    rotations (any axis) within window_s, before the first lift. Each flick (out and back) gives two
    rotation peaks. Any axis, because a real flick is often a wrist flex rather than a pure twist
    (found on the board 2026-11-19: 240 deg/s of rotation, only 58 deg/s about the forearm).
    Searched from the "flick" cue (sync_cue marker) on, where nothing else moves, so the threshold can
    be low: flick_min_dps, or twice the fastest rotation while the arm hung still before the cue if
    that is higher. With a dumbbell in hand flicks were only 70-130 deg/s (pilot 2026-11-20), so a
    fixed 90 deg/s missed set 1's."""
    from scipy.signal import find_peaks

    from .features import find_lifts

    s = load_session(rec_path, cfg)
    if not len(s.imu_t):
        return None
    tr = forearm_angle(s, cfg)
    lifts = find_lifts(tr, s.gyro, cfg)
    before = lifts[0].t_start if lifts else s.imu_t[-1]
    spin = np.linalg.norm(s.gyro, axis=1)
    hz = 1 / np.median(np.diff(s.imu_t)) if len(s.imu_t) > 1 else 50.0
    cue = dict((label, t) for t, label in s.markers).get("sync_cue")
    if min_dps is None:
        min_dps = cfg["lab"]["flick_min_dps"]
        if cue is not None and (s.imu_t < cue).any():
            min_dps = max(min_dps, 2 * float(spin[s.imu_t < cue].max()))
        elif cue is None:
            min_dps = max(min_dps, 90.0)   # no cue to search from: keep the strict threshold
    peaks, _ = find_peaks(spin, height=min_dps, distance=max(1, int(0.15 * hz)))
    after = (cue - 0.3) if cue is not None else -np.inf
    times = [float(s.imu_t[k]) for k in peaks if after <= s.imu_t[k] < before]
    for i, t0 in enumerate(times):
        if sum(t0 <= t <= t0 + window_s for t in times[i:]) >= 3:
            return t0
    return None


def template(data_dir: Path, raters=("rater1", "rater2")) -> Path:
    sets = sorted(p.stem for p in Path(data_dir).glob("*_set[0-9].bin"))
    rows = [{c: "" for c in COLUMNS} | {"recording": r, "rater": who} for r in sets for who in raters]
    out = Path(data_dir) / "labels.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    return out


def apply(data_dir: Path, cfg: dict, sheet: Path | None = None) -> pd.DataFrame:
    """Write agreed labels into each set's .json; return one row per set with the agreement."""
    data_dir = Path(data_dir)
    df = pd.read_csv(sheet or data_dir / "labels.csv")
    cycle = cfg["lab"]["cadence_s"]
    report = []
    for rec, g in df.groupby("recording"):
        bin_path = data_dir / f"{rec}.bin"
        flick = find_flicks(bin_path, cfg)
        row = {"recording": rec, "flick_s": flick, "status": "", "stop_diff_s": np.nan}
        if flick is None:
            row["status"] = "no sync flicks found in the IMU"
            report.append(row)
            continue

        def to_rec(r, col):
            v = r[col]
            return None if pd.isna(v) or pd.isna(r["video_sync_s"]) else float(v) - float(r["video_sync_s"]) + flick

        raters = g[g.rater != "agreed"]
        agreed = g[g.rater == "agreed"]
        stops = [to_rec(r, "stop_point_video_s") for _, r in raters.iterrows()]
        stops = [x for x in stops if x is not None]
        if len(stops) >= 2:
            d = abs(stops[0] - stops[1])
            row["stop_diff_s"] = d
            row["same_rep"] = d < cycle / 2
            row["within_one_rep"] = d <= 1.5 * cycle
        if len(agreed):
            src = [agreed.iloc[-1]]
            row["status"] = "agreed row"
        elif len(stops) >= 2 and row.get("within_one_rep"):
            src = [r for _, r in raters.iterrows()]
            row["status"] = "mean of raters"
        elif len(stops) == 1:
            src = [raters.iloc[0]]
            row["status"] = "one rater only"
        else:
            row["status"] = "raters disagree by more than one rep: settle it and add an 'agreed' row"
            report.append(row)
            continue
        labels = {}
        for col, key in (("set_start_video_s", "set_start_s"), ("stop_point_video_s", "stop_point_s"),
                         ("failure_video_s", "failure_s")):
            vals = [to_rec(r, col) for r in src]
            vals = [v for v in vals if v is not None]
            labels[key] = round(float(np.mean(vals)), 3) if vals else None
        side = bin_path.with_suffix(".json")
        meta = json.loads(side.read_text())
        meta["labels"] = labels
        meta["label_check"] = {"flick_s": round(flick, 3), "stop_diff_s": None if np.isnan(row["stop_diff_s"])
                               else round(float(row["stop_diff_s"]), 3), "status": row["status"]}
        side.write_text(json.dumps(meta, indent=1))
        report.append(row)
    return pd.DataFrame(report)


def agreement(report: pd.DataFrame) -> str:
    two = report.dropna(subset=["stop_diff_s"])
    if not len(two):
        return "no sets with two raters yet"
    return (f"{len(two)} sets with two raters: same rep {two.same_rep.mean():.0%}, within one rep "
            f"{two.within_one_rep.mean():.0%}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["template", "apply"])
    ap.add_argument("data_dir")
    ap.add_argument("--sheet", help="labels CSV (default <data_dir>/labels.csv)")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.action == "template":
        print(f"Fill in: {template(Path(args.data_dir))}")
        return
    report = apply(Path(args.data_dir), cfg, Path(args.sheet) if args.sheet else None)
    print(report.to_string(index=False))
    print(agreement(report))


if __name__ == "__main__":
    main()
