"""Step 10.4: one command from a folder of recordings to a markdown report.

    python -m pipeline.report recordings/ --out results/report.md

1. Every recording's health (and the rest checks), problems first.
2. Labels: which sets are labelled, rater agreement (same rep / within one rep).
3. Every model, scored leaving one person out, each with warnings tuned by Adi's rule (step 9):
   the clock baselines, core-v1 (straight line and small net), the physics comparison, and the
   feature search + straight line (fully nested). Misses and false alarms are shown weighted by set
   (later sets count more) and unweighted.
4. Which features the search kept, and in how many folds.
The header records the date, git commit and config, so a result can be traced back.

Simulated folders (they have the simulator's index.json) only run with --sim, and the report then
says CODE CHECK at the top: simulated data is for testing the code, never a result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from collections import Counter
from pathlib import Path

import json

import numpy as np
import pandas as pd

from .clean import rest_check
from .config import ROOT, load_config
from .evaluate import evaluate, load_sets, md_table, summarize
from .load import load_session
from .model import CORE_V1, FatigueModel, PhysicsModel, config_key
from .models import BASELINES
from .search import SearchedModel
from .warn import Warned


def provenance(cfg: dict) -> str:
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True,
                               text=True).stdout.strip()
        commit = commit + (" (with uncommitted changes)" if dirty else "")
    except OSError:
        commit = "unknown"
    return f"{dt.datetime.now():%Y-%m-%d %H:%M} · commit {commit} · feature config {config_key(cfg)}"


def health_table(folder: Path, cfg: dict) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.bin")):
        s = load_session(p, cfg)
        problems = s.problems(cfg)
        if p.stem.endswith("_rest"):
            problems += rest_check(s, cfg).problems
        rows.append({"recording": p.stem, "seconds": round(s.duration_s, 1), "ok": not problems,
                     "problems": "; ".join(problems)})
    return pd.DataFrame(rows)


def label_table(folder: Path, cfg: dict) -> tuple[pd.DataFrame, str]:
    rows, cycle = [], cfg["lab"]["cadence_s"]
    for side in sorted(folder.glob("*_set[0-9].json")):
        meta = json.loads(side.read_text())
        lab, chk = meta.get("labels", {}), meta.get("label_check", {})
        d = chk.get("stop_diff_s")
        rows.append({"set": side.stem, "labelled": lab.get("set_start_s") is not None,
                     "status": chk.get("status", "simulator labels" if "world" in meta else ""),
                     "stop_diff_s": d, "same_rep": None if d is None else d < cycle / 2,
                     "within_one_rep": None if d is None else d <= 1.5 * cycle})
    df = pd.DataFrame(rows)
    two = df.dropna(subset=["stop_diff_s"]) if len(df) else df
    note = (f"{len(two)} sets rated twice: same rep {two.same_rep.mean():.0%}, within one rep "
            f"{two.within_one_rep.mean():.0%}" if len(two) else "no sets rated twice yet")
    return df, note


def weighted_rates(df: pd.DataFrame, sets_by_name: dict, cfg: dict) -> dict:
    w = np.array([cfg["warn"]["set_weights"][min(sets_by_name[n].set_no, 3) - 1] for n in df["set"]])
    out = {}
    for key in ("w2", "w1"):
        out[f"{key} false alarms, weighted"] = f"{np.sum(w * df[f'{key}_false']) / np.sum(w):.0%}"
        out[f"{key} misses, weighted"] = f"{np.sum(w * df[f'{key}_miss']) / np.sum(w):.0%}"
    return out


def models(cfg: dict, fast: bool = False) -> dict:
    out = {name: (lambda make=make: Warned(make, cfg)) for name, make in BASELINES.items()}
    out["core-v1 line"] = lambda: Warned(lambda: FatigueModel(cfg, CORE_V1, "ridge"), cfg)
    if not fast:
        out["core-v1 small net"] = lambda: Warned(lambda: FatigueModel(cfg, CORE_V1, "net"), cfg)
        out["physics (#1 level / slope)"] = lambda: Warned(lambda: PhysicsModel(cfg), cfg)
    out["search + line"] = lambda: Warned(lambda: SearchedModel(cfg), cfg)
    return out


def build(folder: Path, cfg: dict, sim: bool = False, fast: bool = False) -> str:
    folder = Path(folder)
    simulated = (folder / "index.json").exists()
    if simulated and not sim:
        raise SystemExit(f"{folder} is simulated data (it has the simulator's index.json). Simulated data is for "
                         "checking code, not results; pass --sim to run it as a code check.")
    lines = []
    if simulated:
        lines += ["> **CODE CHECK on simulated data. Not a result.**", ""]
    lines += [f"# Fatigue pipeline report: {folder.name}", "", provenance(cfg), ""]

    health = health_table(folder, cfg)
    bad = health[~health.ok]
    lines += ["## 1. Recordings", f"{len(health)} recordings, {len(bad)} with problems.", ""]
    if len(bad):
        lines += [md_table(bad), ""]

    labels, note = label_table(folder, cfg)
    lines += ["## 2. Labels", f"{int(labels.labelled.sum()) if len(labels) else 0} of {len(labels)} sets labelled. {note}.", ""]

    sets = load_sets(folder)
    by_name = {s.name: s for s in sets}
    people = sorted({s.participant for s in sets})
    lines += ["## 3. Scores", f"{len(people)} people, {len(sets)} sets, leaving one person out each time. Every model's "
              "warnings are tuned on its training people only (false alarms in at most 1 in 3 sets, later sets "
              "weighted more, then fewest misses).", ""]
    table, kept = [], Counter()
    for name, make in models(cfg, fast).items():
        made = []

        def tracked(make=make):
            m = make()
            made.append(m)
            return m
        df = evaluate(tracked, sets, cfg)
        row = summarize(name, df)
        row.update(weighted_rates(df, by_name, cfg))
        table.append(row)
        if name == "search + line":
            for m in made:
                kept.update(getattr(m.model, "kept", []))
    lines += [md_table(pd.DataFrame(table)), ""]

    lines += ["## 4. Features the search kept", f"Out of {len(people)} folds (each fold's search ran on its training "
              "people only):", ""]
    if kept:
        lines += [md_table(pd.DataFrame([{"feature": f"#{f}", "folds kept": n} for f, n in kept.most_common()])), ""]
    else:
        lines += ["None beat shuffled luck in any fold.", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default=None, help="markdown file (default results/report_<folder>.md)")
    ap.add_argument("--sim", action="store_true", help="allow simulated data, as a code check")
    ap.add_argument("--fast", action="store_true", help="skip the small net and the physics comparison")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    text = build(Path(args.folder), cfg, args.sim, args.fast)
    out = Path(args.out or ROOT / "results" / f"report_{Path(args.folder).name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
