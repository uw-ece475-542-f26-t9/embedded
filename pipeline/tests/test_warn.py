"""Step 9: warnings fire, latch and persist as designed, and tuning follows Adi's rule: weighted false
alarms in at most 1 in 3 sets (later sets weigh more), then fewest misses. Code checks only."""

from pathlib import Path

import numpy as np
import pytest

from pipeline import load_config
from pipeline.evaluate import SetData, evaluate, load_sets
from pipeline.model import FatigueModel
from pipeline.simulate import simulate
from pipeline.warn import Policy, Warned, outcome, tune

CFG = load_config()


def fake(stop=30.0, set_no=1, person="P01", k=0):
    return SetData(name=f"{person}_set{set_no}_{k}", path=Path("x.bin"), participant=person, set_no=set_no,
                   rest_before_s=None, set_start_s=5.0, stop_s=stop, failure_s=None, meta={})


def test_warning_needs_its_persistence_then_latches():
    t = np.arange(6) * 2.0
    p = np.array([10, 3, 5, 3, 3, 9])
    assert Policy((4, 4), (1, 1)).fire_times(t, p) == (2.0, 2.0)
    assert Policy((4, 4), (2, 2)).fire_times(t, p) == (8.0, 8.0)     # two in a row at 6 s and 8 s
    assert Policy((4, 2), (1, 1)).fire_times(t, p) == (2.0, None)


def predictions(bias_by_set, n_per_set=4, seed=0):
    """Fake out-of-person predictions: the true seconds left plus a per-set bias and noise."""
    rng = np.random.default_rng(seed)
    out = []
    for set_no, bias in bias_by_set.items():
        for k in range(n_per_set):
            s = fake(set_no=set_no, person=f"P{k}", k=k)
            t = np.arange(5.0, 31.0, 2.0)
            out.append((s, t, np.maximum(s.stop_s - t, 0) + bias + rng.normal(0, 0.5, len(t))))
    return out


def weighted(preds, pol, cfg, level=0):
    w = np.array([cfg["warn"]["set_weights"][s.set_no - 1] for s, _, _ in preds])
    res = [outcome(s, pol.fire_times(t, p)[level], cfg["eval"]["false_alarm_s"][level]) for s, t, p in preds]
    miss, fa = np.array(res).T
    return np.sum(w * fa) / np.sum(w), np.sum(w * miss)


def test_tuning_keeps_false_alarms_under_the_cap_then_minimises_misses():
    preds = predictions({1: +3.0, 2: -2.0, 3: +1.0})   # set 1 reads late, set 2 early
    pol = tune(preds, CFG)
    fa, miss = weighted(preds, pol, CFG)
    assert fa <= CFG["warn"]["max_false_alarm_share"]
    # brute force: no allowed threshold has fewer weighted misses
    lo, hi, step = CFG["warn"]["threshold_grid_s"]
    for th in np.arange(lo, hi + 1e-9, step):
        for n in CFG["warn"]["persistence"]:
            f2, m2 = weighted(preds, Policy((th, th), (n, n)), CFG)
            if f2 <= CFG["warn"]["max_false_alarm_share"]:
                assert miss <= m2 + 1e-9


def test_later_sets_weigh_more():
    """Set 1 reads late (a high threshold avoids its misses); set 3 reads early (a high threshold gives it
    false alarms). Weighting set 3 more must pull the chosen threshold down: measured 7.5 s with equal
    weights vs 6.0 s with set 3 x4."""
    preds = predictions({1: +6.0, 3: -3.0}, n_per_set=6)
    equal = {**CFG, "warn": {**CFG["warn"], "set_weights": [1.0, 1.0, 1.0], "max_false_alarm_share": 0.6}}
    heavy = {**CFG, "warn": {**CFG["warn"], "set_weights": [1.0, 1.0, 4.0], "max_false_alarm_share": 0.6}}
    assert tune(preds, heavy).threshold_s[0] < tune(preds, equal).threshold_s[0]


class Spy:
    name = "spy"
    log = []

    def fit(self, train):
        self.people = {s.participant for s in train}
        Spy.log.append(self.people)

    def predict(self, s):
        assert s.participant not in self.people or len(self.people) > 2, "predicted a person it was fitted on"
        t = np.arange(5.0, 31.0, 2.0)
        return t, np.maximum(s.stop_s - t, 0)


def test_thresholds_are_tuned_on_out_of_person_predictions():
    sets = [fake(person=p, set_no=k) for p in ("P1", "P2", "P3", "P4") for k in (1, 2, 3)]
    Spy.log = []
    w = Warned(Spy, CFG)
    w.fit([s for s in sets if s.participant != "P4"])
    inner = Spy.log[1:]                     # the first fit is the final model on all training people
    assert len(inner) == 3 and all(len(people) == 2 for people in inner)
    assert all("P4" not in people for people in Spy.log)


def test_warned_model_runs_through_the_harness(tmp_path):
    fast = {**CFG, "model": {**CFG["model"]}}
    sets = load_sets(simulate(CFG, "normal", tmp_path / "n", people=4, seed=80))
    df = evaluate(lambda: Warned(lambda: FatigueModel(fast, ["1", "15", "17"], "ridge"), fast), sets, fast)
    assert len(df) == len(sets)
    assert df.w2_miss.dtype == bool and df.w2_false.dtype == bool
