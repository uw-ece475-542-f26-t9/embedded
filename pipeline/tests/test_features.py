"""Step 6: per-lift features follow the truth, ignore what they should ignore, and run in real time."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import load_config, load_session
from pipeline.features import cleaner_for, lift_features, truncated
from pipeline.session import Session
from pipeline.simulate import simulate

CFG = load_config()
REPSENSE = Path.home() / "Documents" / "repsense"


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    still = dict(CFG)
    still["sim"] = {**CFG["sim"], "tremor_rise": 0.0}
    out = {w: simulate(CFG, w, root / w, people=3, seed=9) for w in ("normal", "emg_blind", "effort")}
    out["no_tremor_rise"] = simulate(still, "normal", root / "nt", people=3, seed=9)
    return out


def features_with_truth(d, name):
    pid = name.split("_")[0]
    df = lift_features(load_session(d / f"{name}.bin", CFG), CFG, cleaner_for(d / f"{name}.bin", pid, CFG))
    tr = np.load(d / f"{name}_truth.npz")
    reps = json.loads((d / f"{name}_truth.json").read_text())["reps"]
    mid = (df.t_start + df.t_end) / 2
    near = [min(reps, key=lambda r: abs(r["start_s"] - t)) for t in df.t_start]
    df["true_mdf"] = np.interp(mid, tr["t"], tr["mdf_hz"])
    df["true_loud"] = np.interp(mid, tr["t"], tr["loudness"])
    df["true_F"] = np.interp(mid, tr["t"], tr["F"])
    df["kind"] = [r["kind"] for r in near]
    df["effort"] = [r["effort"] for r in near]
    df["lift_s"] = [r["lift_s"] for r in near]
    base = df.is_baseline
    df["true_mdf_ratio"] = df.true_mdf / df.true_mdf[base].mean()
    df["true_loud_ratio"] = df.true_loud / df.true_loud[base].mean()
    return df


def all_sets(d, people=3):
    return pd.concat([features_with_truth(d, f"P0{i}_set{k}") for i in range(1, people + 1) for k in (1, 2, 3)],
                     ignore_index=True)


def test_middle_pitch_and_loudness_follow_the_truth(worlds):
    df = all_sets(worlds["normal"])
    good = df[df.emg_ok & (df.kind == "good")]
    late = good.groupby((good.true_F > good.true_F.quantile(0.75))).mdf_ratio.mean()
    assert late[True] < late[False] - 0.07                      # tired lifts read clearly lower
    assert np.corrcoef(good.mdf_ratio, good.true_mdf_ratio)[0, 1] > 0.5
    assert abs((good.mdf_ratio - good.true_mdf_ratio).mean()) < 0.04   # no bias: -0.6% mean, 1.6% sd per seed (7 seeds)
    assert (good.mdf_ratio - good.true_mdf_ratio).std() < 0.11         # per-lift scatter: 8.4% over 6 seeds (7-10%)
    assert np.corrcoef(good.loud_ratio, good.true_loud_ratio)[0, 1] > 0.75


def test_middle_pitch_slope_is_there_only_when_fatigue_shows_in_emg(worlds):
    """Slope of middle-pitch change against true tiredness: clearly below the EMG-blind world's.
    Judging the EMG-blind slope on its own needs many simulated people: with 3 its noise is about
    +-0.06, which once looked like a confound (it wasn't: 8 seeds gave +1.1% +- 1.8% by the stop point)."""
    slope = {}
    for w in ("normal", "emg_blind"):
        df = all_sets(worlds[w])
        good = df[df.emg_ok & (df.kind == "good")]
        slope[w] = np.polyfit(good.true_F, good.mdf_ratio, 1)[0]
    assert slope["normal"] < -0.12                    # truth: -0.15 per stop_level (0.8) = -0.19 per unit
    assert slope["normal"] < slope["emg_blind"] - 0.08
    assert abs(slope["emg_blind"]) < 0.15


def test_effort_moves_pitch_and_loudness_up_together(worlds):
    df = all_sets(worlds["effort"])
    ok = df[df.emg_ok]
    surge, calm = ok[ok.effort > 0.2], ok[ok.effort == 0]
    assert surge.loud_ratio.mean() > calm.loud_ratio.mean() + 0.15
    assert surge.mdf_ratio.mean() > calm.mdf_ratio.mean()      # up, not the fatigue direction


def test_speed_loss_follows_slowing(worlds):
    df = all_sets(worlds["normal"]).dropna(subset=["speed_loss"])
    assert np.corrcoef(df.speed_loss, 1 - 1 / df.lift_s)[0, 1] > 0.9


def test_top_angle_separates_partial_and_drift_reps(worlds):
    df = all_sets(worlds["normal"]).dropna(subset=["top_change"])
    good = df[df.kind == "good"].top_change
    assert df[df.kind == "partial"].top_change.max() < -25
    assert df[df.kind == "drift"].top_change.min() > good.quantile(0.9)
    assert good.abs().median() < 6


def test_tremor_tracks_tiredness_and_not_slowing(worlds):
    rising = all_sets(worlds["normal"]).dropna(subset=["tremor_ratio"])
    assert np.corrcoef(rising.tremor_ratio, rising.true_F)[0, 1] > 0.5
    flat = all_sets(worlds["no_tremor_rise"]).dropna(subset=["tremor_ratio"])
    # With tremor not growing, slower tired lifts mustn't fake it. Bad reps (swing, partial, failed)
    # are jerky and do raise the band (~+25%), so this is checked on good lifts.
    good = flat[flat.kind == "good"]
    assert abs(np.corrcoef(good.tremor_ratio, good.true_F)[0, 1]) < 0.3
    assert flat[flat.kind != "good"].tremor_ratio.mean() > good.tremor_ratio.mean()


def test_features_are_real_time(worlds):
    d = worlds["normal"]
    s = load_session(d / "P01_set1.bin", CFG)
    cln = cleaner_for(d / "P01_set1.bin", "P01", CFG)
    full = lift_features(s, CFG, cln)
    cut = float(full.t_known.iloc[len(full) // 2])
    part = lift_features(truncated(s, cut + 0.01), CFG, cln)
    a = full[full.t_known <= cut].reset_index(drop=True)
    b = part[part.t_known <= cut].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.skipif(not (REPSENSE / "data" / "raw" / "p6_s1.csv").exists(), reason="RepSense data not here")
def test_real_wrist_data_drift_partial_and_exhaustion():
    sys.path.insert(0, str(REPSENSE))
    try:
        from src.dataio import ACCEL, GYRO, load_annotations, slice_set
        from src.dataio import load_session as rs_load
    finally:
        sys.path.pop(0)
    rows = []
    for sess in ("p6_s1", "p7_s1"):
        df, ann = rs_load(sess), load_annotations(sess)
        for _, row in ann[ann.exercise == "curl"].iterrows():
            seg = slice_set(df, row)
            t = seg["t"].to_numpy() - seg["t"].iloc[0]
            s = Session(name="x", source="capture", emg=np.full(int(t[-1] * 2000) + 2000, 1440.0), emg_hz=2000,
                        imu_t=t, accel=seg[ACCEL].to_numpy(), gyro=seg[GYRO].to_numpy())
            late = lift_features(s, CFG)
            late = late[~late.is_baseline]
            if len(late) >= 2:
                rows.append((row["condition"], late.top_change.max(), late.top_change.min(), late.speed_loss.median()))
    r = pd.DataFrame(rows, columns=["condition", "top_max", "top_min", "speed_loss"]).groupby("condition").median()
    assert r.loc["drift", "top_max"] > r.loc["normal", "top_max"] + 5
    assert r.loc["partial", "top_min"] < r.loc["normal", "top_min"] - 15
    assert r.loc["exhaustion", "speed_loss"] > r.loc["normal", "speed_loss"]
