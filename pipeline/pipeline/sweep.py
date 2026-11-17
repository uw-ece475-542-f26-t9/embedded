"""Step 10.5: the knob sweep, nested by person, real data only.

    python -m pipeline.sweep recordings/ --out results/sweep.md

The settings are the grid in config.toml [sweep] (Adi, 2026-11-17: bin width 15/20 deg, window
256/384/512 samples, baseline lifts 3/4 = 12 settings). Each setting is judged by the core-v1 straight
line's error in seconds (the fixed reference model; judging by the search would rerun it hundreds of
times).

For each held-out person:
  1. every setting is scored on the other people, leaving one of them out at a time,
  2. the best setting is chosen,
  3. a model with that setting is fitted on all the other people and scored once on the held-out one.
That held-out score is an honest score for "sweep, then fit". The default setting is scored the same
way beside it, so the report shows whether sweeping helps at all, and which setting each fold chose
(if folds disagree, the knob doesn't matter much and the default stays).

Signal-processing knobs change the features, so each setting has its own feature cache. Fitted
models are reused when the same training people come up again (fold pairs are symmetric).
"""

from __future__ import annotations

import argparse
import copy
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .evaluate import SetData, load_sets, md_table, score_set
from .model import CORE_V1, FatigueModel


def settings(cfg: dict) -> list[dict]:
    """Every combination of the [sweep] grid, as {'section.knob': value}."""
    keys = list(cfg["sweep"])
    return [dict(zip(keys, vals)) for vals in itertools.product(*(cfg["sweep"][k] for k in keys))]


def apply(cfg: dict, setting: dict) -> dict:
    out = copy.deepcopy(cfg)
    for key, value in setting.items():
        section, knob = key.split(".", 1)
        out[section][knob] = value
    return out


def label(setting: dict) -> str:
    return ", ".join(f"{k.split('.')[-1]} {v}" for k, v in setting.items())


class Scorer:
    """Mean error (s) of the core-v1 line for a setting, trained on some people, scored on others."""

    def __init__(self, cfg: dict, sets: list[SetData], make=None):
        self.cfg, self.sets = cfg, sets
        self.make = make or (lambda c: FatigueModel(c, CORE_V1, "ridge"))
        self.fitted = {}

    def model(self, k: int, setting: dict, train_people: frozenset):
        key = (k, train_people)
        if key not in self.fitted:
            m = self.make(apply(self.cfg, setting))
            m.fit([s for s in self.sets if s.participant in train_people])
            self.fitted[key] = m
        return self.fitted[key]

    def error(self, k: int, setting: dict, train_people: frozenset, test_people: set) -> float:
        m = self.model(k, setting, train_people)
        errs = []
        for s in (s for s in self.sets if s.participant in test_people):
            t, p = m.predict(s)
            errs.append(score_set(s, t, p, self.cfg).get("error_s", np.nan))
        return float(np.nanmean(errs)) if errs else np.nan


def sweep(cfg: dict, sets: list[SetData], scorer: Scorer | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = settings(cfg)
    default = {k: _get(cfg, k) for k in cfg["sweep"]}
    scorer = scorer or Scorer(cfg, sets)
    people = sorted({s.participant for s in sets})
    folds, inner_rows = [], []
    for out in people:
        rest = [p for p in people if p != out]
        inner = []
        for k, setting in enumerate(grid):
            errs = [scorer.error(k, setting, frozenset(set(rest) - {i}), {i}) for i in rest]
            inner.append(float(np.nanmean(errs)))
            inner_rows.append({"held out": out, "setting": label(setting), "inner error s": round(inner[-1], 2)})
        best = int(np.nanargmin(inner))
        k_def = grid.index(default) if default in grid else None
        folds.append({"held out": out, "chosen": label(grid[best]),
                      "held-out error s (swept)": scorer.error(best, grid[best], frozenset(rest), {out}),
                      "held-out error s (default)": scorer.error(k_def, default, frozenset(rest), {out})
                      if k_def is not None else np.nan})
    return pd.DataFrame(folds), pd.DataFrame(inner_rows)


def _get(cfg: dict, key: str):
    section, knob = key.split(".", 1)
    return cfg[section][knob]


def report(folds: pd.DataFrame, inner: pd.DataFrame, simulated: bool) -> str:
    lines = (["> **CODE CHECK on simulated data. Not a result.**", ""] if simulated else [])
    lines += ["# Knob sweep", "", "Each fold chose a setting using only its training people; the held-out person "
              "was scored once.", "", md_table(folds.round(2)), "",
              f"Mean held-out error: swept {folds['held-out error s (swept)'].mean():.2f} s, default "
              f"{folds['held-out error s (default)'].mean():.2f} s.", "",
              "How often each setting was chosen:", ""]
    lines += [f"- {s}: {n} of {len(folds)}" for s, n in Counter(folds.chosen).most_common()]
    lines += ["", "## Inner scores", "", md_table(inner.pivot(index="setting", columns="held out",
                                                           values="inner error s"), index=True), ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out")
    ap.add_argument("--sim", action="store_true", help="allow simulated data, as a code check")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    folder = Path(args.folder)
    simulated = (folder / "index.json").exists()
    if simulated and not args.sim:
        raise SystemExit(f"{folder} is simulated data; pass --sim to run it as a code check.")
    folds, inner = sweep(cfg, load_sets(folder))
    text = report(folds, inner, simulated)
    out = Path(args.out or ROOT / "results" / f"sweep_{folder.name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
