"""Step 10.4-10.5: the one-command report and the nested knob sweep. Code checks only (simulated data)."""

import numpy as np
import pytest

from pipeline import load_config
from pipeline.evaluate import load_sets
from pipeline.model import config_key
from pipeline.report import build
from pipeline.simulate import simulate
from pipeline.sweep import Scorer, apply, settings, sweep

CFG = load_config()
FAST = {**CFG, "search": {**CFG["search"], "subsets": 50, "shuffles": 3, "bootstrap": 3}}


def test_the_grid_is_the_12_agreed_settings():
    grid = settings(CFG)
    assert len(grid) == 12 and len({tuple(sorted(g.items())) for g in grid}) == 12
    keys = {config_key(apply(CFG, g)) for g in grid}
    assert len(keys) == 12                                  # each setting gets its own feature cache
    assert apply(CFG, grid[0]) is not CFG and CFG["bins"]["halfwidth_deg"] == 15


class FakeScorer(Scorer):
    """Window 512 is always best; records which people each score was trained and tested on."""

    def __init__(self, sets):
        self.sets, self.calls = sets, []

    def error(self, k, setting, train_people, test_people):
        self.calls.append((set(train_people), set(test_people)))
        return 10.0 - setting["features.fixed_window"] / 100 + 0.01 * k


def test_sweep_chooses_on_training_people_only(tmp_path):
    from pipeline.evaluate import SetData
    from pathlib import Path
    sets = [SetData(name=f"{p}_set1", path=Path("x"), participant=p, set_no=1, rest_before_s=None,
                    set_start_s=0.0, stop_s=10.0, failure_s=None) for p in ("A", "B", "C", "D")]
    fake = FakeScorer(sets)
    folds, inner = sweep(CFG, sets, fake)
    assert set(folds.chosen) == {"halfwidth_deg 15, fixed_window 512, baseline_lifts 3"}
    for out in "ABCD":
        inner_calls = [c for c in fake.calls if out not in c[1] and out not in c[0]]
        assert inner_calls   # the held-out person's inner scoring never saw them
    held = [c for c in fake.calls if len(c[1]) == 1 and len(c[0]) == 3]
    assert all(not (tr & te) for tr, te in held)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return simulate(CFG, "normal", tmp_path_factory.mktemp("sim") / "n", people=3, seed=95)


def test_sweep_runs_end_to_end_on_a_small_grid(tiny):
    small = {**CFG, "sweep": {"features.fixed_window": [256, 384]}}
    folds, inner = sweep(small, load_sets(tiny))
    assert len(folds) == 3 and folds["held-out error s (swept)"].notna().all()
    assert len(inner) == 3 * 2


def test_report_refuses_simulated_data_unless_asked(tiny):
    with pytest.raises(SystemExit):
        build(tiny, FAST)


def test_report_runs_and_says_code_check(tiny):
    text = build(tiny, FAST, sim=True, fast=True)
    assert text.startswith("> **CODE CHECK")
    for heading in ("## 1. Recordings", "## 2. Labels", "## 3. Scores", "## 4. Features the search kept"):
        assert heading in text
    assert "core-v1 line" in text and "search + line" in text and "false alarms, weighted" in text
