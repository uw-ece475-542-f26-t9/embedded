"""Step 8: code checks for the models and the feature search. Simulated data checks the code only:
nothing fitted here is used on real data."""

import numpy as np
import pytest

from pipeline import load_config
from pipeline.evaluate import causal_check, evaluate, load_sets
from pipeline.model import CORE_V1, FatigueModel, PhysicsModel, Ridge, SmallNet, cached_rows, config_key
from pipeline.search import SearchedModel, choose_subsets, rank, subset_scores
from pipeline.simulate import simulate

CFG = load_config()
FAST = {**CFG, "search": {**CFG["search"], "subsets": 200, "shuffles": 5, "bootstrap": 5},
        "model": {**CFG["model"], "net_epochs": 150, "net_ensemble": 2}}


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    return {w: load_sets(simulate(CFG, w, root / w, people=4, seed=60)) for w in ("normal", "emg_blind")}


def test_batched_subset_lines_match_a_plain_ridge_fit():
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(120, 5)), rng.normal(size=120)
    groups = np.repeat(np.arange(4), 30)
    subsets = np.array([[0, 2, 4], [1, 3, 4]])
    batched = subset_scores(X, y, groups, subsets, lam=1.0)
    for s, score in zip(subsets, batched):
        err = []
        for g in range(4):
            tr, te = groups != g, groups == g
            m = Ridge(1.0).fit(X[tr][:, s], y[tr])
            err.append(np.mean(np.abs(np.clip(m.predict(X[te][:, s]), 0, None) - y[te])))
        assert score == pytest.approx(-np.mean(err))


def test_planted_feature_is_kept_and_noise_is_dropped():
    """RepSense's planted test: one deciding feature among nine noise features."""
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(8), 40)
    X = rng.normal(size=(320, 10))
    y = 10 + 3 * X[:, 0] + rng.normal(0, 1, 320)
    cfg = {**CFG, "search": {**CFG["search"], "subsets": 252, "size": 5, "shuffles": 10, "bootstrap": 10}}
    r = rank(X, y, groups, [f"f{i}" for i in range(10)], cfg)
    assert list(r[r.kept].feature) == ["f0"]


def test_subsets_are_all_combinations_when_few_and_random_when_many():
    rng = np.random.default_rng(2)
    assert len(choose_subsets(8, 6, 10000, rng)) == 28
    many = choose_subsets(25, 6, 500, rng)
    assert len(many) == 500 and len({tuple(s) for s in many}) == 500


def test_net_obeys_the_size_rule():
    net = SmallNet(CFG)
    assert net.hidden_for(n_rows=290, n_in=6) * 8 + 1 <= 0.5 * 290
    with pytest.raises(ValueError):
        SmallNet(CFG).fit(np.zeros((10, 6)), np.zeros(10))


def test_feature_rows_are_cached_per_config(worlds):
    s = worlds["normal"][0]
    a = cached_rows(s, CFG)
    assert (s.path.parent / ".feature_cache" / f"{s.name}_{config_key(CFG)}.pkl").exists()
    b = cached_rows(s, CFG)
    assert a.equals(b)
    other = {**CFG, "bins": {**CFG["bins"], "halfwidth_deg": 20}}
    assert config_key(other) != config_key(CFG)


@pytest.mark.parametrize("make", [lambda: FatigueModel(FAST, CORE_V1, "ridge"), lambda: FatigueModel(FAST, CORE_V1, "net"),
                                  lambda: PhysicsModel(FAST), lambda: SearchedModel(FAST)])
def test_models_run_through_the_harness_and_are_real_time(worlds, make):
    sets = worlds["normal"]
    df = evaluate(make, sets, FAST)
    assert len(df) == len(sets) and df.error_s.notna().all()
    m = make()
    m.fit([s for s in sets if s.participant != sets[0].participant])
    assert causal_check(m, sets[0], sets[0].set_start_s + 15)


def test_emg_features_carry_signal_only_when_the_emg_does(worlds):
    """Wiring check: EMG-only features beat a no-feature line in the normal world, not in the EMG-blind one."""
    gain = {}
    for w in ("normal", "emg_blind"):
        emg = evaluate(lambda: FatigueModel(FAST, ["1", "26"], "ridge"), worlds[w], FAST).error_s.mean()
        none = evaluate(lambda: FatigueModel(FAST, [], "ridge"), worlds[w], FAST).error_s.mean()
        gain[w] = none - emg
    assert gain["normal"] > 0.3 and gain["normal"] > gain["emg_blind"] + 0.2


def test_a_person_fingerprint_feature_is_never_kept():
    """A feature that only identifies the person (constant within each one) can only help a model that
    has seen that person. Leave-one-person-out, including in the bootstrap, must give it no credit."""
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(8), 30)
    offset = rng.normal(0, 5, 8)
    X = rng.normal(size=(240, 6))
    X[:, 0] = np.repeat(rng.normal(size=8), 30)          # fingerprint: one value per person
    X[:, 1] = rng.normal(size=240)
    y = 20 + offset[groups] + 3 * X[:, 1] + rng.normal(0, 1, 240)
    cfg = {**CFG, "search": {**CFG["search"], "subsets": 20, "size": 3, "shuffles": 10, "bootstrap": 20}}
    r = rank(X, y, groups, [f"f{i}" for i in range(6)], cfg).set_index("feature")
    assert not r.loc["f0", "kept"] and r.loc["f1", "kept"]
    # the old bootstrap (a person drawn twice treated as two people) passed it in 60% of resamples
    assert r.loc["f0", "bootstrap_share"] < 0.3
