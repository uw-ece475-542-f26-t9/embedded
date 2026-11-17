"""Models scored by the step 3 harness: the baselines every real model must beat, and the oracle.

Baselines use no sensor data at all, only the clock and what the app knows about the session:
  time         seconds left = (typical set length) - (time since the first lift)
  time+set     the typical length for this set number
  time+set+rest  the typical length for this set number and the rest before it

The oracle cheats: it reads the simulator's hidden tiredness level, which no real model could
see, and works out when it will reach the stop level. It isn't a model. It tests the referee:
it should beat every baseline in the normal world and never warn in the effort world, and its
leftover error (rater error, reps slowing near the end) is the floor any real model faces.
"""

from __future__ import annotations

import json

import numpy as np

from .evaluate import SetData

HORIZON_S = 120.0  # baselines predict this far past the first lift
STEP_S = 0.5


def _countdown(s: SetData, length_s: float) -> tuple[np.ndarray, np.ndarray]:
    t = s.set_start_s + np.arange(0, HORIZON_S, STEP_S)
    if "cut_s" in s.meta:  # the real-time check: nothing past the cut
        t = t[t <= s.meta["cut_s"]]
    return t, np.maximum(length_s - (t - s.set_start_s), 0)


def _lengths(train: list[SetData]) -> list[tuple[SetData, float]]:
    return [(s, s.stop_s - s.set_start_s) for s in train if s.stop_s is not None]


class TimeOnly:
    name = "time"

    def fit(self, train):
        lengths = [d for _, d in _lengths(train)]
        self.length = float(np.mean(lengths)) if lengths else np.inf

    def predict(self, s):
        return _countdown(s, self.length)


class TimeSet:
    name = "time+set"

    def fit(self, train):
        pairs = _lengths(train)
        overall = float(np.mean([d for _, d in pairs])) if pairs else np.inf
        self.length = {k: float(np.mean([d for s, d in pairs if s.set_no == k] or [overall])) for k in (1, 2, 3)}

    def predict(self, s):
        return _countdown(s, self.length.get(s.set_no, np.inf))


class TimeSetRest:
    """Set length from a small ridge regression on set number and the rest before the set."""
    name = "time+set+rest"

    @staticmethod
    def _x(s: SetData) -> np.ndarray:
        rest = (s.rest_before_s or 0.0) / 180.0
        return np.array([s.set_no == 1, s.set_no == 2, s.set_no == 3, rest], float)

    def fit(self, train, lam: float = 1e-2):
        pairs = _lengths(train)
        if not pairs:
            self.coef = None
            return
        x = np.array([self._x(s) for s, _ in pairs])
        y = np.array([d for _, d in pairs])
        self.coef = np.linalg.solve(x.T @ x + lam * np.eye(x.shape[1]), x.T @ y)

    def predict(self, s):
        return _countdown(s, float(self._x(s) @ self.coef) if self.coef is not None else np.inf)


BASELINES = {"time": TimeOnly, "time+set": TimeSet, "time+set+rest": TimeSetRest}


class Oracle:
    """Reads the true tiredness F (simulated data only) and predicts when it reaches the stop
    level, from how fast it has risen since the first lift. Uses only truth up to each moment.

    own_level=True:  also knows this person's own stop level. The full cheat, the referee check:
                     it must beat every baseline.
    own_level=False: knows tiredness but only the average stop level. Shows what a model that
                     read tiredness perfectly from the sensors could do without knowing where
                     this person's form breaks."""

    def __init__(self, cfg: dict, own_level: bool = True):
        self.cfg, self.own_level = cfg, own_level
        self.name = "oracle (cheats)" if own_level else "oracle, tiredness only"

    def fit(self, train):
        pass

    def _stop_level(self, s: SetData) -> float:
        index = json.loads((s.path.parent / "index.json").read_text())
        if self.own_level:
            for p in index["people"]:
                if p["person"]["pid"] == s.participant:
                    return float(p["person"]["stop_level"])
        return float(index["config"]["stop_level"])

    def predict(self, s):
        tr = s.truth
        if not tr:
            raise ValueError(f"{s.name}: no simulator truth; the oracle only works on simulated data")
        stop_level = self._stop_level(s)
        t, F = tr["t"], tr["F"]
        keep = t >= s.set_start_s
        if "cut_s" in s.meta:
            keep &= t <= s.meta["cut_s"]
        t, F = t[keep], F[keep]
        F0 = F[0] if len(F) else 0.0
        elapsed = t - s.set_start_s
        with np.errstate(divide="ignore", invalid="ignore"):
            rate = np.where(elapsed >= 1.0, (F - F0) / elapsed, np.nan)
            left = np.where(rate > 1e-9, (stop_level - F) / rate, np.inf)
        left = np.where(F >= stop_level, 0.0, left)
        ok = ~np.isnan(rate)
        return t[ok], np.maximum(left[ok], 0)
