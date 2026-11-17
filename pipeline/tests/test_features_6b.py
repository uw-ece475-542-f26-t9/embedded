"""Step 6b: the 12 remaining candidate features respond the way they should. Simulated worlds check
the EMG and timing features; the real RepSense wrist recordings check the form features the
simulator doesn't model (bottom shortfall, sticking points, wobble). Thresholds come from a 12-person
measurement on 2026-11-17 with margins for 6 people."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import load_config
from pipeline.features import lift_features, set_features
from pipeline.session import Session
from pipeline.simulate import simulate

CFG = load_config()
REPSENSE = Path.home() / "Documents" / "repsense"


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    out = {}
    for w in ("normal", "emg_blind", "effort"):
        d = simulate(CFG, w, root / w, people=6, seed=70)
        frames = []
        for i in range(1, 7):
            for k in (1, 2, 3):
                name = f"P0{i}_set{k}"
                df = set_features(d / f"{name}.bin", CFG)
                tr = np.load(d / f"{name}_truth.npz")
                reps = json.loads((d / f"{name}_truth.json").read_text())["reps"]
                near = [min(reps, key=lambda r: abs(r["start_s"] - t)) for t in df.t_start]
                df["F"] = np.interp((df.t_start + df.t_end) / 2, tr["t"], tr["F"])
                df["kind"] = [r["kind"] for r in near]
                df["effort"] = [r["effort"] for r in near]
                frames.append(df)
        out[w] = pd.concat(frames, ignore_index=True)
    return out


def slope(df, col):
    g = df[(df.kind == "good") & df[col].notna() & ~df.is_baseline]
    return np.polyfit(g.F, g[col], 1)[0]


@pytest.mark.parametrize("col, sign, size", [("mnf_ratio", -1, 0.08), ("dimitrov_ratio", 1, 0.5),
                                             ("lowhigh_ratio", 1, 0.5), ("mdf_lower_ratio", -1, 0.08)])
def test_spectral_features_move_with_fatigue_only_when_the_emg_does(worlds, col, sign, size):
    normal, blind = slope(worlds["normal"], col), slope(worlds["emg_blind"], col)
    assert sign * normal > size
    assert sign * (normal - blind) > 0.75 * size


def test_dimitrov_index_moves_the_opposite_way_for_effort(worlds):
    e = worlds["effort"]
    e = e[e.dimitrov_ratio.notna() & ~e.is_baseline]
    assert e[e.effort > 0.2].dimitrov_ratio.mean() < e[e.effort == 0].dimitrov_ratio.mean()


def test_loudness_as_share_of_max_pull_rises_with_fatigue(worlds):
    assert slope(worlds["normal"], "loud_max") > 0.1 and abs(slope(worlds["emg_blind"], "loud_max")) < 0.08
    assert 0.2 < worlds["normal"].loud_max.median() < 1.2  # curls are a fraction of an all-out pull


def test_fatigue_index_and_emg_cost_rise(worlds):
    assert slope(worlds["normal"], "fatigue_index") > 0.15
    assert slope(worlds["normal"], "emg_cost") > slope(worlds["emg_blind"], "emg_cost") + 0.2


def test_duration_and_lateness_follow_slowing_and_bad_reps(worlds):
    n = worlds["normal"]
    assert slope(n, "duration_ratio") > 0.15
    late = n[n.lateness.notna() & ~n.is_baseline]
    assert late[late.kind != "good"].lateness.mean() > late[late.kind == "good"].lateness.mean() + 0.2


@pytest.mark.skipif(not (REPSENSE / "data" / "raw" / "p6_s1.csv").exists(), reason="RepSense data not here")
def test_real_wrist_form_features_by_condition():
    sys.path.insert(0, str(REPSENSE))
    try:
        from src.dataio import ACCEL, GYRO, load_annotations, slice_set
        from src.dataio import load_session as rs_load
    finally:
        sys.path.pop(0)
    rows = []
    for sess in ("p2_s1", "p3_s1", "p4_s1", "p5_s1", "p6_s1", "p7_s1"):
        try:
            df, ann = rs_load(sess), load_annotations(sess)
        except FileNotFoundError:
            continue
        for _, row in ann[ann.exercise == "curl"].iterrows():
            seg = slice_set(df, row)
            if len(seg) < 300:
                continue
            t = seg["t"].to_numpy() - seg["t"].iloc[0]
            s = Session(name="x", source="capture", emg=np.full(int(t[-1] * 2000) + 2000, 1440.0), emg_hz=2000,
                        imu_t=t, accel=seg[ACCEL].to_numpy(), gyro=seg[GYRO].to_numpy())
            late = lift_features(s, CFG)
            late = late[~late.is_baseline]
            if len(late) >= 2:
                rows.append((row["condition"], late.duration_ratio.median(), late.bottom_change.max(),
                             late.lateness.max(), late.sticking_dip.max()))
    r = pd.DataFrame(rows, columns=["condition", "duration", "bottom", "lateness", "dip"]).groupby("condition").median()
    assert r.loc["exhaustion", "duration"] > r.loc["normal", "duration"] + 0.1        # 1.18 vs 1.00
    assert r.loc["partial", "bottom"] > r.loc["normal", "bottom"] + 20                # 49 vs 4 deg
    assert r.loc["exhaustion", "dip"] > r.loc["normal", "dip"] + 0.15                 # 0.38 vs 0.02
    assert r.loc["partial", "lateness"] > r.loc["normal", "lateness"] + 0.5           # 1.29 vs 0.13 s
