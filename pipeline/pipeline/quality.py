"""Data-quality checks for collected sessions: is each recording usable, and does it match the protocol?

    python -m pipeline.quality [recordings] [--people P01 P02] [--csv out.csv]

One row per check: OK, WARN (usable, but know this), FAIL (don't use it as is) or EXCLUDED (the
operator's annotation in the .json says to drop it). Checks:

  per person:  every protocol file is there; the real rest between sets matches the rest order.
  every file:  stream health (missing / clipped EMG, IMU gaps), the operator's exclusions.
  rest:        hum and interference; aliasing (rest vs rest_alt).
  pull:        loud enough to be a real pull (3x the rest noise); when the burst started after "Pull!"
               (listening lag); whether it ran past the scored window; a scale reading was typed.
  set:         sync flicks found; lifts found and how late the first one came; pace vs the
               metronome (first reps and overall); half reps; the kit's end / stop marks; how many
               lifts have usable EMG; curls louder than the max pull (the max pull wasn't maximal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .clean import alias_check, clean_emg, rest_check
from .config import ROOT, load_config
from .lab import COLD_STEPS, MAIN_STEPS, NO_FILE
from .load import load_session

FILE_STEPS = [s for s in COLD_STEPS + MAIN_STEPS if s not in NO_FILE]


def _row(person, rec, check, status, detail):
    return {"person": person, "recording": rec, "check": check, "status": status, "detail": detail}


def meta_of(path: Path) -> dict:
    side = path.with_suffix(".json")
    return json.loads(side.read_text()) if side.exists() else {}


def person_checks(folder: Path, p: str, cfg: dict) -> list[dict]:
    rows = []
    have = {f.stem[len(p) + 1:] for f in folder.glob(f"{p}_*.bin")}
    missing = [s for s in FILE_STEPS if s not in have]
    cold = [s for s in missing if s in COLD_STEPS]
    scale_only = [s for s in cold if (folder / f"{p}_{s}.json").exists()]
    if scale_only:
        rows.append(_row(p, "-", "files", "WARN", f"cold pulls are scale readings only, no recording "
                         f"({', '.join(scale_only)}): force 100% and warm-up effect OK, EMG 100% is warm pulls only"))
    if len(scale_only) < len(cold):
        rows.append(_row(p, "-", "files", "WARN", f"no cold pulls ({', '.join(s for s in cold if s not in scale_only)})"
                         ": 100% is best of the 3 warm pulls only, and the warm-up effect can't be measured"))
    other = [s for s in missing if s not in COLD_STEPS]
    if other:
        rows.append(_row(p, "-", "files", "FAIL", f"missing {', '.join(other)}"))
    if not missing:
        rows.append(_row(p, "-", "files", "OK", "all protocol files"))
    # real rest = start of set n - end of set n-1 (host clock), against the ordered rest
    for n in (2, 3):
        a, b = folder / f"{p}_set{n - 1}.json", folder / f"{p}_set{n}.json"
        if not (a.exists() and b.exists()):
            continue
        ma, mb = json.loads(a.read_text()), json.loads(b.read_text())
        real = mb["host_start"] - (ma["host_start"] + ma["duration_s"])
        want = mb.get("rest_before_s")
        if want is None:
            continue
        off = real - want
        st = "OK" if abs(off) <= max(15, 0.15 * want) else "WARN"
        rows.append(_row(p, f"set{n}", "rest", st, f"rest {real:.0f} s, ordered {want} s ({off:+.0f} s)"))
    return rows


def burst(pull, cfg: dict, frac: float = 0.5) -> tuple[float, float, float]:
    """(onset, offset, peak mV) of the pull's main burst: where the 100 ms RMS envelope is above
    frac x its peak, the longest such run."""
    cl = clean_emg(pull, cfg)
    x = np.where(cl.skip, np.nan, cl.emg)
    n = max(1, int(0.1 * cl.fs))
    env = np.sqrt(pd.Series(x ** 2).rolling(n, center=True, min_periods=n // 2).mean().to_numpy())
    skip0 = int(0.3 * cl.fs)                       # filter warm-up
    env[:skip0] = np.nan
    peak = np.nanmax(env)
    above = np.nan_to_num(env, nan=0) > frac * peak
    edges = np.flatnonzero(np.diff(np.r_[0, above.astype(int), 0]))
    runs = edges.reshape(-1, 2)
    a, b = max(runs, key=lambda r: r[1] - r[0])
    return a / cl.fs, b / cl.fs, float(peak)


def pull_checks(path: Path, s, meta: dict, cfg: dict, floor_mv: float | None) -> list[dict]:
    from .features import pull_loudness, pull_window
    p, rec, lab = meta.get("participant", path.stem[:3]), path.stem[4:], cfg["lab"]
    rows = []
    loud = pull_loudness(s, cfg)
    if floor_mv:
        st = "OK" if loud >= lab["pull_min_ratio"] * floor_mv else "FAIL"
        rows.append(_row(p, rec, "pull loudness", st, f"{loud:.1f} mV = {loud / floor_mv:.0f}x rest noise"))
    marks = {label: t for t, label in s.markers}
    if "pull_start" in marks:
        on, off, _ = burst(s, cfg)
        lag = on - marks["pull_start"]
        _, wb = pull_window(s, cfg)
        win_end = wb / s.emg_hz
        st = "OK" if -0.3 <= lag <= 1.5 else "WARN"
        rows.append(_row(p, rec, "pull timing", st, f"burst {on:.1f}-{off:.1f} s, starts {lag:+.1f} s after "
                         f"'Pull!'; window ends {win_end:.1f} s"))
        if off > win_end + 0.2 or off >= s.duration_s - 0.15:
            cut = "recording" if off >= s.duration_s - 0.15 else "window"
            rows.append(_row(p, rec, "pull cut off", "WARN", f"burst still going at the end of the {cut} "
                             f"({off:.1f} s): a longer pull interval would catch it"))
    if meta.get("force_kg") is None:
        rows.append(_row(p, rec, "scale reading", "WARN", "no luggage-scale reading typed"))
    return rows


def dead_lifts(df: pd.DataFrame, floor_mv: float | None, cfg: dict) -> list[int]:
    """Lift numbers (1-based) whose EMG is under pull_min_ratio x the rest noise: no muscle signal.
    A curl is a real contraction, so it should never be that quiet (P01 set 3: 1 mV from lift 23)."""
    if not floor_mv:
        return []
    low = df.rms_mv < cfg["lab"]["pull_min_ratio"] * floor_mv
    return [int(i) + 1 for i in np.flatnonzero(low.to_numpy())]


def set_checks(path: Path, s, meta: dict, cfg: dict, floor_mv: float | None = None) -> list[dict]:
    from .angle import forearm_angle
    from .features import find_lifts, set_features
    from .labels import find_flicks
    p, rec = meta.get("participant", path.stem[:3]), path.stem[4:]
    cad = cfg["lab"]["cadence_s"]
    rows = []
    marks = {label: t for t, label in s.markers}
    flick = find_flicks(path, cfg)
    rows.append(_row(p, rec, "sync flicks", "OK" if flick is not None else "WARN",
                     f"at {flick:.1f} s" if flick is not None else "none found: video sync needs another cue"))
    lifts = find_lifts(forearm_angle(s, cfg), s.gyro, cfg)
    if not lifts:
        return rows + [_row(p, rec, "lifts", "FAIL", "no lifts found")]
    cue = marks.get("sync_cue", 0.0)
    first = lifts[0].t_start - cue
    rows.append(_row(p, rec, "first lift", "OK" if first <= 6 else "WARN",
                     f"{len(lifts)} lifts; first {first:.1f} s after the flick cue"))
    starts = np.array([lf.t_start for lf in lifts])
    gaps = np.diff(starts)
    if len(gaps) >= 4:
        early = gaps[:4]   # off the beat: any gap 15% off, or the average 7.5% off (rushing / dragging)
        bad = np.any(np.abs(early - cad) > 0.15 * cad) or abs(early.mean() - cad) > 0.075 * cad
        rows.append(_row(p, rec, "pace, first reps", "WARN" if bad else "OK",
                         f"first 4 rep gaps {', '.join(f'{g:.1f}' for g in early)} s (beat {cad:g} s)"))
        med = float(np.median(gaps))
        rows.append(_row(p, rec, "pace, whole set", "OK" if abs(med - cad) <= 0.2 else "WARN",
                         f"median rep gap {med:.2f} s"))
    tops = np.array([lf.top_deg for lf in lifts])
    ref = np.median(tops)
    half = [i + 1 for i, t in enumerate(tops[:-1]) if t < 0.7 * ref]   # the last (failure) rep may be partial
    nb = cfg["features"]["baseline_lifts"]
    if half:
        base = [i for i in half if i <= nb]
        rows.append(_row(p, rec, "half reps", "WARN", f"lift(s) {half} reached under 70% of the usual "
                         f"top ({ref:.0f} deg)" + (f"; lift {base} is in the {nb}-lift baseline" if base else "")))
    if "end" not in marks:
        rows.append(_row(p, rec, "end mark", "FAIL", "no end mark from the kit"))
    if meta.get("set") == 3:
        st = "OK" if "stop_called" in marks and marks["stop_called"] < marks.get("end", np.inf) else "WARN"
        rows.append(_row(p, rec, "stop point", st, f"stop {marks.get('stop_called', float('nan')):.1f} s, "
                         f"end {marks.get('end', float('nan')):.1f} s"))
    df = set_features(path, cfg)
    dead = dead_lifts(df, floor_mv, cfg)
    if dead:
        t = df.t_start.iloc[dead[0] - 1]
        rows.append(_row(p, rec, "EMG dropout", "FAIL", f"lifts {dead[0]}-{dead[-1]} ({len(dead)} of {len(df)}, "
                         f"from {t:.0f} s) are under {cfg['lab']['pull_min_ratio']:g}x the rest noise: the sensor "
                         "stopped reading the muscle (pad lifting?). Use only the lifts before it"))
    # A curl doesn't get much quieter as it fatigues; a lift under 35% of the baseline lifts' loudness
    # (not the last, partial rep) is a pad losing contact for a while (P03 set 1, lifts 10-11).
    dips = [int(i) + 1 for i in np.flatnonzero((df.loud_ratio < 0.35).to_numpy()[:-1]) if i + 1 not in dead]
    if dips:
        rows.append(_row(p, rec, "EMG dip", "WARN", f"lift(s) {dips} under 35% of the baseline loudness, "
                         "then back: intermittent pad contact? Treat their EMG as suspect"))
    later = df[~df.is_baseline]
    share =later.emg_ok.mean() if len(later) else 0.0
    rows.append(_row(p, rec, "EMG per lift", "OK" if share >= 0.8 else "WARN",
                     f"{share:.0%} of non-baseline lifts have usable EMG"))
    lm = df.loud_max.dropna()
    if len(lm) and lm.max() > 1.0:
        rows.append(_row(p, rec, "max pull", "WARN", f"curls reach {lm.max():.2f}x the best pull "
                         f"({(lm > 1).mean():.0%} of lifts above it): the pull wasn't maximal, so loud_max is off"))
    return rows


def check_folder(folder: Path, cfg: dict, people: list[str] | None = None) -> pd.DataFrame:
    people = people or sorted({f.stem[:3] for f in folder.glob("P[0-9][0-9]_*.bin")})
    rows = []
    for p in people:
        rows += person_checks(folder, p, cfg)
        rest = folder / f"{p}_rest.bin"
        floor = rest_check(load_session(rest, cfg), cfg).noise_mv if rest.exists() else None
        for path in sorted(folder.glob(f"{p}_*.bin")):
            rec, meta = path.stem[4:], meta_of(path)
            ann = meta.get("annotations", {})
            if ann.get("exclude"):
                rows.append(_row(p, rec, "annotation", "EXCLUDED", ann.get("exclude_reason", "")))
            s = load_session(path, cfg)
            probs = s.problems(cfg)
            rows.append(_row(p, rec, "stream", "FAIL" if probs else "OK", "; ".join(probs) or
                             f"{s.duration_s:.1f} s, clipped {s.health.clipped}, missing {s.health.missing}"))
            kind = meta.get("recording", "")
            if kind == "rest":
                rc = rest_check(s, cfg)
                pr = list(rc.problems)
                if rec == "rest_alt" and rest.exists():
                    pr += alias_check(load_session(rest, cfg), s, cfg)["problems"]
                rows.append(_row(p, rec, "rest", "WARN" if pr else "OK",
                                 "; ".join(pr) or f"hum {rc.hum_mv} mV, noise {rc.noise_mv} mV"))
            elif kind == "pull":
                rows += pull_checks(path, s, meta, cfg, floor)
            elif kind == "set":
                rows += set_checks(path, s, meta, cfg, floor)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=str(ROOT / "recordings"))
    ap.add_argument("--people", nargs="*")
    ap.add_argument("--csv")
    args = ap.parse_args()
    df = check_folder(Path(args.folder), load_config(), args.people)
    if args.csv:
        df.to_csv(args.csv, index=False)
    flagged = df[df.status != "OK"]
    with pd.option_context("display.max_colwidth", 140, "display.width", 250):
        print(flagged.to_string(index=False))
    print("\n" + df.groupby(["person", "status"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
