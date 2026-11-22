"""Preliminary analysis of collected sessions: do the fatigue signals move the way they should?

    python -m pipeline.prelim [recordings] --people P01 P02 P03 P04 --out ../analysis_2026-11-21

Per set: reps and time to failure, and how each per-lift signal trends from the start of the set to
failure (Spearman rho against "fraction of the set done", and the mean of the last 3 lifts vs the
baseline). Signals: EMG median frequency (should fall), EMG loudness (should rise), lift speed loss
(should rise), range of motion (should shrink). Plus the strength pulls (force and EMG, after each set
as % of the best warm pull). Recordings the operator annotated as excluded are left out.
Writes sets.csv, lifts.csv, pulls.csv and trends.png into --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import ROOT, load_config
from .features import pull_loudness, set_features
from .load import load_session
from .clean import rest_check
from .quality import dead_lifts

EMG_COLS = ["mdf_hz", "rms_mv", "loud_ratio", "mdf_ratio", "mdf_session", "loud_max", "mnf_ratio",
            "dimitrov_ratio", "lowhigh_ratio", "fatigue_index", "effort_index", "emg_cost", "mdf_lower_ratio"]
SIGNALS = {"mdf_ratio": "EMG median freq (x baseline)", "loud_ratio": "EMG loudness (x baseline)",
           "speed_loss": "lift speed loss", "top_change": "top angle change (deg)"}


def meta(path: Path) -> dict:
    return json.loads(path.with_suffix(".json").read_text())


def excluded(path: Path) -> bool:
    return bool(meta(path).get("annotations", {}).get("exclude"))


def set_rows(folder: Path, people: list[str], cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    sets, lifts = [], []
    for p in people:
        for n in (1, 2, 3):
            path = folder / f"{p}_set{n}.bin"
            if not path.exists() or excluded(path):
                continue
            m = meta(path)
            marks = {label: t for t, label in m["markers"]}
            df = set_features(path, cfg)
            rest = folder / f"{p}_rest.bin"
            floor = rest_check(load_session(rest, cfg), cfg).noise_mv if rest.exists() else None
            dead = dead_lifts(df, floor, cfg)
            if dead:   # no muscle signal from here on: keep the IMU signals, drop the EMG ones
                df.loc[dead[0] - 1:, EMG_COLS] = np.nan
            t0, end = df.t_start.iloc[0], marks.get("end", df.t_end.iloc[-1])
            df["frac"] = (df.t_start - t0) / (end - t0)
            df["person"], df["set"], df["lift"] = p, n, np.arange(1, len(df) + 1)
            prev = folder / f"{p}_set{n - 1}.json"
            real = None
            if prev.exists():
                pm = json.loads(prev.read_text())
                real = round(m["host_start"] - pm["host_start"] - pm["duration_s"])
            row = {"person": p, "set": n, "rest_ordered_s": m.get("rest_before_s"), "rest_real_s": real,
                   "emg_dead_from_lift": dead[0] if dead else None,
                   "weight_lb": m.get("weight_lb"), "reps": len(df), "time_to_failure_s": round(end - t0, 1)}
            if "stop_called" in marks:
                row["stop_frac"] = round((marks["stop_called"] - t0) / (end - t0), 2)
            later = df[~df.is_baseline]
            for col in SIGNALS:
                x = later[["frac", col]].dropna()
                row[f"{col}_rho"] = round(spearmanr(x.frac, x[col]).statistic, 2) if len(x) >= 5 else np.nan
                row[f"{col}_last3"] = round(x[col].tail(3).mean(), 2) if len(x) else np.nan
            row["mdf_hz_first"] = round(df.mdf_hz.head(3).mean(), 1)
            row["mdf_session_last3"] = round(df.mdf_session.dropna().tail(3).mean(), 2)
            sets.append(row)
            lifts.append(df)
    return pd.DataFrame(sets), pd.concat(lifts, ignore_index=True)


def pull_rows(folder: Path, people: list[str], cfg: dict) -> pd.DataFrame:
    rows = []
    for p in people:
        vals = {}
        for path in sorted(folder.glob(f"{p}_pull*.json")):   # .json: cold pulls may be scale-only
            step = path.stem[len(p) + 1:]
            if excluded(path):
                continue
            kg = meta(path).get("force_kg")
            rec = path.with_suffix(".bin")
            mv = pull_loudness(load_session(rec, cfg), cfg) if rec.exists() else np.nan
            vals[step] = (np.nan if kg is None else kg, mv)
        pre = [v for k, v in vals.items() if k.startswith(("pull_base", "pull_warm"))]
        top_f = np.nanmax([v[0] for v in pre]) if pre else np.nan
        top_e = np.nanmax([v[1] for v in pre]) if pre else np.nan
        for step, (kg, mv) in vals.items():
            rows.append({"person": p, "pull": step, "force_kg": kg, "force_pct": round(100 * kg / top_f, 0),
                         "emg_mv": round(mv, 1), "emg_pct": round(100 * mv / top_e, 0)})
    return pd.DataFrame(rows)


def plot(lifts: pd.DataFrame, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(4.2 * len(SIGNALS), 3.6))
    colors = {"P01": "#1f77b4", "P02": "#ff7f0e", "P03": "#2ca02c", "P04": "#d62728"}
    styles = {1: "-", 2: "--", 3: ":"}
    for ax, (col, title) in zip(axes, SIGNALS.items()):
        for (p, n), g in lifts[~lifts.is_baseline].groupby(["person", "set"]):
            ax.plot(g.frac, g[col], styles[n], color=colors.get(p, "k"), lw=1.2, alpha=.8, label=f"{p} set{n}")
        ax.axhline(1 if "ratio" in col else 0, color="grey", lw=.6)
        ax.set_xlabel("fraction of set (0 = first lift, 1 = failure)")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out / "trends.png", dpi=110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", default=str(ROOT / "recordings"))
    ap.add_argument("--people", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg, folder, out = load_config(), Path(args.folder), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sets, lifts = set_rows(folder, args.people, cfg)
    pulls = pull_rows(folder, args.people, cfg)
    sets.to_csv(out / "sets.csv", index=False)
    lifts.to_csv(out / "lifts.csv", index=False)
    pulls.to_csv(out / "pulls.csv", index=False)
    plot(lifts, out)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(sets.to_string(index=False))
        print()
        print(pulls.pivot(index="person", columns="pull", values="force_pct").to_string())
        print()
        print(pulls.pivot(index="person", columns="pull", values="emg_pct").to_string())


if __name__ == "__main__":
    main()
