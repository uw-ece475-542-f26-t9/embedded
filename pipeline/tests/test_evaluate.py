"""Step 3: the referee scores correctly, never leaks the test person, and passes its world checks."""

from pathlib import Path

import numpy as np
import pytest

from pipeline import load_config
from pipeline.evaluate import SetData, causal_check, evaluate, hold_latest, load_sets, score_set
from pipeline.models import BASELINES, Oracle
from pipeline.simulate import simulate

CFG = load_config()


def fake_set(stop=30.0, start=10.0, failure=None, set_no=1, person="P01"):
    return SetData(name=f"{person}_set{set_no}", path=Path("x.bin"), participant=person, set_no=set_no,
                   rest_before_s=None, set_start_s=start, stop_s=stop, failure_s=failure,
                   meta={"duration_s": 60.0})


def run(s, left_fn):
    t = s.set_start_s + np.arange(0, 50, 0.5)
    return score_set(s, t, left_fn(t), CFG)


def test_hold_latest_uses_the_last_prediction_so_far():
    out = hold_latest(np.array([1.0, 2.0]), np.array([10.0, 20.0]), np.array([0.5, 1.0, 1.7, 2.5]))
    np.testing.assert_array_equal(out, [np.nan, 10, 10, 20])


def test_perfect_prediction_warns_on_time():
    s = fake_set()
    r = run(s, lambda t: np.maximum(s.stop_s - t, 0))
    assert r["error_s"] == pytest.approx(0, abs=1e-9)
    assert not r["w2_miss"] and not r["w2_false"] and r["w2_lead_s"] == pytest.approx(4)
    assert not r["w1_miss"] and r["w1_lead_s"] == pytest.approx(2)


def test_late_prediction_is_a_miss():
    s = fake_set()
    r = run(s, lambda t: np.maximum(s.stop_s + 10 - t, 0))  # thinks there's 10 s more than there is
    assert r["w2_miss"] and r["w1_miss"] and not r["w2_false"]
    assert r["error_s"] == pytest.approx(10, abs=0.01)


def test_early_warning_is_a_false_alarm():
    s = fake_set()
    r = run(s, lambda t: np.maximum(s.stop_s - 10 - t, 0))  # warns ~10 s too early
    assert r["w2_false"] and r["w1_false"] and not r["w2_miss"]


def test_any_warning_in_a_set_without_a_stop_point_is_false():
    s = fake_set(stop=None)
    assert run(s, lambda t: np.full(len(t), 3.0))["w2_false"]
    assert not run(s, lambda t: np.full(len(t), np.inf))["w2_false"]


class Spy:
    """Records who it was trained on."""
    name = "spy"
    seen: list = []

    def fit(self, train):
        Spy.seen.append({s.participant for s in train})

    def predict(self, s):
        assert s.participant not in Spy.seen[-1], "trained on the person it's tested on"
        return np.array([s.set_start_s]), np.array([np.inf])


def test_leave_one_person_out_never_trains_on_the_test_person():
    sets = [fake_set(person=p, set_no=k) for p in ("P01", "P02", "P03") for k in (1, 2)]
    Spy.seen = []
    df = evaluate(Spy, sets, CFG)
    assert len(Spy.seen) == 3 and all(len(seen) == 2 for seen in Spy.seen)
    assert len(df) == 6


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    return {w: load_sets(simulate(CFG, w, root / w, people=6, seed=4)) for w in ("normal", "shortcut", "effort")}


def mean_error(df):
    return df.loc[df["has_stop"], "error_s"].mean()


def test_models_are_causal(worlds):
    s = worlds["normal"][0]
    for make in [*BASELINES.values(), lambda: Oracle(CFG)]:
        m = make()
        m.fit(worlds["normal"][3:])
        assert causal_check(m, s, cut_s=s.set_start_s + 8)


def test_shortcut_world_baselines_are_nearly_perfect(worlds):
    for make in BASELINES.values():
        assert mean_error(evaluate(make, worlds["shortcut"], CFG)) < 0.5


def test_normal_world_oracle_beats_every_baseline(worlds):
    oracle = evaluate(lambda: Oracle(CFG), worlds["normal"], CFG)
    for make in BASELINES.values():
        assert mean_error(oracle) < mean_error(evaluate(make, worlds["normal"], CFG))
    assert oracle["w2_miss"].sum() == 0


def test_effort_world_oracle_never_warns(worlds):
    oracle = evaluate(lambda: Oracle(CFG), worlds["effort"], CFG)
    assert not oracle["w2_false"].any() and not oracle["w1_false"].any()
