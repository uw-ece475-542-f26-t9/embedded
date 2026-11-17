"""Step 3: the scoring harness. Every model is judged here, the same way.

A model has two methods:

    fit(train: list[SetData])        learn from the training people's sets
    predict(s: SetData) -> (t, left) times in s (recording clock) and predicted seconds until
                                     the stop point at each of them, using only data up to t

Scoring:
  - Train on 7 people, test on the 8th, for every person (leave one person out). A model never
    sees the person it's tested on.
  - Each model is read on the same grid (every grid_s from the set's first lift) using its latest
    prediction, so models that update at different rates are judged alike.
  - Truth is the video stop-point label (rater error included). In set 3, after the stop point
    the target is 0.
  - Warnings: "2 reps" fires the first time the prediction is <= warn_s[0], "1 rep" at <= warn_s[1].
      miss          the stop point arrives with no warning
      false alarm   the warning fires with more than false_alarm_s left (any warning, in a set
                    with no stop point)
      lead          how many seconds before the stop point it fired
      error         mean |predicted - true seconds left| before the stop point

    python -m pipeline.evaluate sim/normal
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from .config import load_config
from .load import load_session
from .session import Session


@dataclass
class SetData:
    """One set and what's known about it. Models read the session; scoring reads the labels."""
    name: str
    path: Path
    participant: str
    set_no: int
    rest_before_s: float | None     # None for set 1
    set_start_s: float              # first lift (video)
    stop_s: float | None            # stop point (video); None if the set had none
    failure_s: float | None
    meta: dict = field(default_factory=dict)

    @cached_property
    def session(self) -> Session:
        return load_session(self.path)

    @cached_property
    def truth(self) -> dict:
        """Simulator ground truth (absent for real data). Only the oracle may read it."""
        f = self.path.with_name(self.name + "_truth.npz")
        if not f.exists():
            return {}
        d = dict(np.load(f))
        d.update(json.loads(self.path.with_name(self.name + "_truth.json").read_text()))
        return d

    def pulls(self) -> list[Session]:
        """This person's baseline max pulls (for loudness as a share of max)."""
        return [load_session(p) for p in sorted(self.path.parent.glob(f"{self.participant}_pull_base*.bin"))]


def load_sets(folder: str | Path) -> list[SetData]:
    folder, sets = Path(folder), []
    for side in sorted(folder.glob("*_set[0-9].json")):
        info = json.loads(side.read_text())
        labels = info.get("labels", {})
        sets.append(SetData(name=side.stem, path=side.with_suffix(".bin"), participant=info["participant"],
                            set_no=int(info["set"]), rest_before_s=info.get("rest_before_s"),
                            set_start_s=float(labels["set_start_s"]), stop_s=labels.get("stop_point_s"),
                            failure_s=labels.get("failure_s"), meta=info))
    if not sets:
        raise FileNotFoundError(f"no sets (NAME_setN.json) in {folder}")
    return sets


class Model(Protocol):
    name: str

    def fit(self, train: list[SetData]) -> None: ...

    def predict(self, s: SetData) -> tuple[np.ndarray, np.ndarray]: ...


# ------------------------------------------------------------------ scoring one set


def grid_for(s: SetData, cfg: dict) -> np.ndarray:
    end = s.failure_s if s.failure_s is not None else (s.stop_s if s.stop_s is not None else None)
    if end is None:  # no stop point: score the whole recording
        end = s.meta.get("duration_s") or s.session.duration_s
    return np.arange(s.set_start_s, end + 1e-9, cfg["eval"]["grid_s"])


def hold_latest(t: np.ndarray, left: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The latest prediction at or before each grid time (NaN before the first one)."""
    order = np.argsort(t, kind="stable")
    t, left = np.asarray(t, float)[order], np.asarray(left, float)[order]
    i = np.searchsorted(t, grid + 1e-9, side="right") - 1  # tolerance: 26.0 may be stored as 26.0000001
    out = np.full(len(grid), np.nan)
    ok = i >= 0
    out[ok] = left[i[ok]]
    return out


def score_set(s: SetData, t: np.ndarray, left: np.ndarray, cfg: dict,
              fired: tuple[float | None, float | None] | None = None) -> dict:
    """fired: the model's own warning times (2-rep, 1-rep), from a step 9 policy. Without it a warning
    fires when the prediction first reaches warn_s, read on the grid."""
    e = cfg["eval"]
    grid = grid_for(s, cfg)
    pred = hold_latest(t, left, grid)
    row = {"set": s.name, "participant": s.participant, "set_no": s.set_no,
           "has_stop": s.stop_s is not None}
    if s.stop_s is not None:
        before = grid <= s.stop_s
        true_left = np.maximum(s.stop_s - grid, 0)
        ok = before & ~np.isnan(pred)
        row["error_s"] = float(np.mean(np.abs(pred[ok] - true_left[ok]))) if ok.any() else np.nan
    for level, (warn, false_s) in enumerate(zip(e["warn_s"], e["false_alarm_s"]), start=1):
        if fired is not None:
            when = fired[level - 1]
        else:
            below = np.nonzero(pred <= warn + 1e-9)[0]
            when = float(grid[below[0]]) if len(below) else None
        key = f"w{len(e['warn_s']) - level + 1}"  # w2 = the "2 reps" warning, w1 = "1 rep"
        if s.stop_s is None:
            row[f"{key}_false"] = when is not None
            row[f"{key}_miss"] = False
            row[f"{key}_lead_s"] = np.nan
            continue
        on_time = when is not None and when <= s.stop_s
        row[f"{key}_miss"] = not on_time
        row[f"{key}_false"] = when is not None and s.stop_s - when > false_s
        row[f"{key}_lead_s"] = s.stop_s - when if on_time else np.nan
    return row


# ------------------------------------------------------------------ leave one person out


def evaluate(make_model: Callable[[], Model], sets: list[SetData], cfg: dict) -> pd.DataFrame:
    rows = []
    for person in sorted({s.participant for s in sets}):
        model = make_model()
        model.fit([s for s in sets if s.participant != person])
        for s in (s for s in sets if s.participant == person):
            t, left = model.predict(s)
            fired = model.warnings(s) if hasattr(model, "warnings") else None
            rows.append(score_set(s, t, left, cfg, fired))
    return pd.DataFrame(rows)


def summarize(name: str, df: pd.DataFrame) -> dict:
    stop = df[df["has_stop"]]
    per_person = stop.groupby("participant")["error_s"].mean() if len(stop) else pd.Series(dtype=float)
    out = {"model": name, "sets": len(df)}
    for key in ("w2", "w1"):
        out[f"{key} misses"] = f"{int(stop[f'{key}_miss'].sum())}/{len(stop)}" if len(stop) else "-"
        out[f"{key} false alarms"] = f"{int(df[f'{key}_false'].sum())}/{len(df)}"
        out[f"{key} lead s (median)"] = round(float(stop[f"{key}_lead_s"].median()), 1) if len(stop) else np.nan
    out["error s"] = round(float(stop["error_s"].mean()), 2) if len(stop) else np.nan
    out["error s, range over people"] = (f"{per_person.min():.1f}-{per_person.max():.1f}"
                                         if len(per_person) else "-")
    return out


def report(results: dict[str, pd.DataFrame], title: str) -> str:
    table = pd.DataFrame([summarize(n, df) for n, df in results.items()])
    lines = [f"# {title}", "", "w2 = the 2-rep warning, w1 = the 1-rep warning. Lead is seconds before "
             "the stop point; error is mean |predicted - true seconds left| before it.", ""]
    cols = list(table.columns)
    lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def causal_check(model: Model, s: SetData, cut_s: float, tol: float = 1e-6) -> bool:
    """A real-time model's predictions up to cut_s mustn't change if the data after it is removed.
    Models that read the session must honour a `cut_s` attribute on the set for this to work."""
    t_full, full = model.predict(s)
    s.meta["cut_s"] = cut_s
    try:
        t_cut, cut = model.predict(s)
    finally:
        s.meta.pop("cut_s")
    keep = t_full <= cut_s
    return bool(len(t_cut) == keep.sum() and np.allclose(full[keep], cut, atol=tol, equal_nan=True))


def main() -> None:
    from .models import BASELINES, Oracle

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="a folder of sets, e.g. sim/normal")
    ap.add_argument("--oracle", action="store_true", help="include the cheating oracle (simulated data only)")
    ap.add_argument("--out", help="write the report here (markdown)")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sets = load_sets(args.folder)
    models = dict(BASELINES)
    if args.oracle:
        models["oracle (cheats)"] = lambda: Oracle(cfg)
        models["oracle, tiredness only"] = lambda: Oracle(cfg, own_level=False)
    results = {name: evaluate(make, sets, cfg) for name, make in models.items()}
    text = report(results, f"Scores: {args.folder}")
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
