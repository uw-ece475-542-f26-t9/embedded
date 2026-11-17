"""Step 9: turning predicted seconds-until-stop into the two haptic warnings, and tuning when they fire.

A warning fires the first time the prediction is at or below its threshold for `persistence` lifts in
a row, and then stays on for the set (latched: no flicker). "2 reps left" and "1 rep left" each have
their own threshold and persistence.

Tuning (Adi, 2026-11-17): on the training people only, using out-of-person predictions (each training
person's predictions come from a model fitted on the other training people, so the thresholds aren't
tuned on predictions that have seen the answer). Rule, per warning:
  1. weighted share of sets with a false alarm <= max_false_alarm_share (1 in 3),
  2. then the fewest weighted misses,
  3. then the fewest weighted false alarms, then the threshold closest to the untuned default.
Each set is weighted by its set number (set_weights): later sets count more.

A false alarm and a miss mean exactly what the step 3 harness scores; those definitions are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .evaluate import Model, SetData


@dataclass
class Policy:
    threshold_s: tuple[float, float]   # (2-rep warning, 1-rep warning)
    persistence: tuple[int, int]

    def fire_times(self, t: np.ndarray, pred: np.ndarray) -> tuple[float | None, float | None]:
        out = []
        for th, n in zip(self.threshold_s, self.persistence):
            below = np.asarray(pred) <= th + 1e-9
            run, when = 0, None
            for i, b in enumerate(below):
                run = run + 1 if b else 0
                if run >= n:
                    when = float(t[i])
                    break
            out.append(when)
        return out[0], out[1]


def outcome(s: SetData, when: float | None, false_s: float) -> tuple[bool, bool]:
    """(missed, false alarm) for one warning in one set, as the harness defines them."""
    if s.stop_s is None:
        return False, when is not None
    on_time = when is not None and when <= s.stop_s
    return (not on_time), (when is not None and s.stop_s - when > false_s)


def tune(preds: list[tuple[SetData, np.ndarray, np.ndarray]], cfg: dict) -> Policy:
    """Thresholds and persistence for both warnings, by the rule above."""
    w = cfg["warn"]
    weights = np.array([w["set_weights"][min(s.set_no, len(w["set_weights"])) - 1] for s, _, _ in preds])
    lo, hi, step = w["threshold_grid_s"]
    grid = np.arange(lo, hi + 1e-9, step)
    chosen_th, chosen_n = [], []
    for level, (default, false_s) in enumerate(zip(w["default_s"], cfg["eval"]["false_alarm_s"])):
        best = None
        for n in w["persistence"]:
            for th in grid:
                pol = Policy((th, th), (n, n))
                miss, fa = np.zeros(len(preds)), np.zeros(len(preds))
                for i, (s, t, p) in enumerate(preds):
                    miss[i], fa[i] = outcome(s, pol.fire_times(t, p)[0], false_s)
                fa_share = np.sum(weights * fa) / np.sum(weights)
                key = (fa_share > w["max_false_alarm_share"], np.sum(weights * miss), np.sum(weights * fa),
                       abs(th - default), n)
                if best is None or key < best[0]:
                    best = (key, th, n)
        chosen_th.append(float(best[1]))
        chosen_n.append(int(best[2]))
    return Policy(tuple(chosen_th), tuple(chosen_n))


class Warned:
    """A model plus a warning policy tuned on its training people. predict() gives the model's
    predictions; warnings() gives the policy's fire times; the harness scores those."""

    def __init__(self, make_model: Callable[[], Model], cfg: dict, tune_policy: bool = True):
        self.make_model, self.cfg, self.tune_policy = make_model, cfg, tune_policy
        self.name = f"{make_model().name} + {'tuned' if tune_policy else 'default'} warnings"

    def fit(self, train: list[SetData]) -> None:
        self.model = self.make_model()
        self.model.fit(train)
        if not self.tune_policy:
            d = self.cfg["warn"]["default_s"]
            self.policy = Policy((d[0], d[1]), (1, 1))
            return
        oof = []  # out-of-person predictions for every training set
        for person in sorted({s.participant for s in train}):
            inner = self.make_model()
            inner.fit([s for s in train if s.participant != person])
            for s in (s for s in train if s.participant == person):
                t, p = inner.predict(s)
                oof.append((s, t, p))
        self.policy = tune(oof, self.cfg)

    def predict(self, s: SetData):
        return self.model.predict(s)

    def warnings(self, s: SetData) -> tuple[float | None, float | None]:
        t, p = self.model.predict(s)
        return self.policy.fire_times(t, p)
