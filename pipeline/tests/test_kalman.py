"""Step 7: the Kalman filters are correct, real time, tunable on training people, and honest."""

import json

import numpy as np
import pandas as pd
import pytest

from pipeline import load_config
from pipeline.features import set_features
from pipeline.kalman import FilterSpec, filter_rows, fit, innovation_check, run_filter, specs_from
from pipeline.simulate import simulate

CFG = load_config()


def model_data(rng, n=40, q=1e-5, r=0.006, dt=2.0):
    """Data drawn from the filter's own model: a level drifting with a wandering slope, plus noise."""
    t = np.arange(n) * dt
    level, slope, xs = 1.0, -0.005, []
    for _ in range(n):
        xs.append(level)
        slope += rng.normal(0, np.sqrt(q * dt))
        level += slope * dt
    return t, np.array(xs), np.array(xs) + rng.normal(0, np.sqrt(r), n)


def spec(q=1e-5, r=0.006):
    return FilterSpec("x", "trend", q, r, level0=1.0, level_sd0=0.05, slope_sd0=0.01)


def test_filter_beats_raw_noise_and_its_innovations_are_plain_noise():
    rng = np.random.default_rng(0)
    errs, e_all = [], []
    for _ in range(200):
        t, truth, z = model_data(rng)
        out = run_filter(t, z, spec(), t0=-2.0)
        errs.append((np.std(z - truth), np.std(out.level - truth)))
        e_all.append((out.innov / out.innov_sd).to_numpy())
    raw, filt = np.mean(errs, axis=0)
    assert filt < 0.6 * raw
    e = np.concatenate(e_all)
    assert 0.9 < e.std() < 1.1


def test_fit_recovers_the_measurement_noise():
    rng = np.random.default_rng(1)
    sets = []
    for _ in range(30):
        t, _, z = model_data(rng, n=20, r=0.006)
        sets.append(pd.DataFrame({"t_known": t, "x": z, "is_baseline": False}))
    q, r = fit(sets, spec(q=1e-6, r=0.02))
    assert r == pytest.approx(0.006, rel=0.3)


def test_missing_lifts_widen_the_uncertainty_instead_of_breaking():
    t = np.arange(10) * 2.0
    z = np.full(10, 0.95)
    z[4:7] = np.nan
    out = run_filter(t, z, spec(), t0=-2.0)
    assert out.level_sd[6] > out.level_sd[3] and out.level_sd[7] < out.level_sd[6]
    assert not out.level.isna().any()


def test_filter_is_real_time():
    rng = np.random.default_rng(2)
    t, _, z = model_data(rng, n=30)
    full = run_filter(t, z, spec(), t0=-2.0)
    part = run_filter(t[:12], z[:12], spec(), t0=-2.0)
    pd.testing.assert_frame_equal(full.iloc[:12].reset_index(drop=True), part)


def test_no_fixed_start_begins_at_the_first_value():
    out = run_filter(np.arange(5.0), np.array([np.nan, 0.9, 0.9, 0.9, 0.9]),
                     FilterSpec("x", "trend", 1e-6, 0.006, None, 0.1, 0.01))
    assert np.isnan(out.level[0]) and out.level[1] == pytest.approx(0.9)


@pytest.fixture(scope="module")
def sims(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    out = {}
    for w in ("normal", "emg_blind"):
        d = simulate(CFG, w, root / w, people=6, seed=12)
        sets = []
        for i in range(1, 7):
            for k in (1, 2, 3):
                name = f"P0{i}_set{k}"
                df = set_features(d / f"{name}.bin", CFG)
                tr = np.load(d / f"{name}_truth.npz")
                v = np.interp((df.t_start + df.t_end) / 2, tr["t"], tr["mdf_hz"])
                df["true_ratio"] = v / v[df.is_baseline].mean()
                side = json.loads((d / f"{name}.json").read_text())
                df["pid"], df["set"], df["rest"] = side["participant"], k, side.get("rest_before_s")
                sets.append(df)
        out[w] = sets
    return out


def held_out(sets, feature="mdf_ratio"):
    train = [s for s in sets if s.pid.iloc[0] in ("P01", "P02", "P03")]
    test = [s for s in sets if s.pid.iloc[0] not in ("P01", "P02", "P03")]
    q, r = fit(train, {s.feature: s for s in specs_from(CFG)}[feature], CFG["kalman"]["q_floor"])
    return [filter_rows(s, specs_from(CFG, {feature: (q, r)})) for s in test], (q, r)


def test_filtered_middle_pitch_beats_the_raw_value_on_held_out_people(sims):
    filt, (q, r) = held_out(sims["normal"])
    assert q >= CFG["kalman"]["q_floor"] and 0.003 < r < 0.015        # per-lift sd 5.5-12%
    raw, lvl = [], []
    for f in filt:
        g = f[~f.is_baseline & f.emg_ok]
        raw += list(g.mdf_ratio - g.true_ratio)
        lvl += list(g.mdf_ratio_level - g.true_ratio)
    assert np.std(lvl) < 0.8 * np.std(raw)
    ic = innovation_check(filt, "mdf_ratio")
    assert 0.8 < ic["spread"] < 1.35 and abs(ic["lag1"]) < 0.3


def test_slope_is_negative_only_when_fatigue_shows_in_emg(sims):
    end = {}
    for w in ("normal", "emg_blind"):
        filt, _ = held_out(sims[w])
        end[w] = np.mean([f.mdf_ratio_slope.dropna().iloc[-1] for f in filt if f.mdf_ratio_slope.notna().any()])
    assert end["normal"] < -0.002 and end["normal"] < end["emg_blind"] - 0.002


def test_session_relative_pitch_shows_carried_over_tiredness(tmp_path):
    """#26 at the start of set 2: below 1.0 after a 60 s rest, ~1.0 after 180 s, and flat when the EMG
    carries no fatigue. Needs 12 people per world to see it (one lift scatters ~10%); measured
    2026-11-17: 0.943 +- 0.022 vs 1.005 +- 0.019."""
    means = {}
    for w in ("normal", "emg_blind"):
        by = {60: [], 180: []}
        for seed in (12, 13):
            d = simulate(CFG, w, tmp_path / f"{w}{seed}", people=6, seed=seed)
            for i in range(1, 7):
                df = set_features(d / f"P0{i}_set2.bin", CFG)
                rest = json.loads((d / f"P0{i}_set2.json").read_text())["rest_before_s"]
                by[rest].append(df.mdf_session.dropna().iloc[:5].mean())
        means[w] = {k: np.mean(v) for k, v in by.items()}
    assert means["normal"][60] < 0.97 and means["normal"][60] < means["normal"][180] - 0.02
    assert all(abs(v - 1) < 0.05 for v in means["emg_blind"].values())
