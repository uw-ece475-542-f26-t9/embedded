"""Step 8: models that turn each lift's filtered features into seconds until the stop point.

Trained on real recordings only. Simulated data is used only to check this code.

For every held-out person (the step 3 harness does the leave-one-person-out loop), a model:
  1. loads each training set's feature rows (step 6, cached per set and config),
  2. fits the Kalman q and r on the training sets only and filters every set (step 7),
  3. builds one row per lift: the chosen features -> target = video stop-point time - when the row
     is known (0 from the stop point on; set 3 runs past it),
  4. fits: a straight line (ridge), or a small neural net (weights <= half the training rows), or
     the physics-style comparison (distance to the breaking level / slope).
Time into the set, set number and rest are never inputs: they are the baselines.

Feature IDs are FEATURES.md numbers. Only those with a column below are computed so far.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import SetData
from .features import set_features
from .kalman import filter_rows, fit, specs_from

FEATURE_COLUMNS = {
    "1": "mdf_ratio_level", "2": "mnf_ratio_level", "3": "dimitrov_ratio_level", "4": "lowhigh_ratio_level",
    "6": "mdf_lower_ratio_level", "8": "loud_ratio_level", "9": "loud_max_level", "10": "fatigue_index_level",
    "11": "emg_cost_level", "13a": "mdf_ratio_slope", "13b": "loud_ratio_slope", "15": "speed_loss_level",
    "16": "duration_ratio_level", "17": "top_change", "18": "bottom_change", "19": "lateness",
    "20": "sticking_dip", "21": "wobble_change", "22": "tremor_ratio_level", "26": "mdf_session_level",
}
FILTER_OF = {"1": "mdf_ratio", "2": "mnf_ratio", "3": "dimitrov_ratio", "4": "lowhigh_ratio", "6": "mdf_lower_ratio",
             "8": "loud_ratio", "9": "loud_max", "10": "fatigue_index", "11": "emg_cost", "13a": "mdf_ratio",
             "13b": "loud_ratio", "15": "speed_loss", "16": "duration_ratio", "22": "tremor_ratio", "26": "mdf_session"}
CORE_V1 = ["1", "8", "13a", "15", "17", "22"]
CACHE_VERSION = 4  # bump when feature code changes in a way the config doesn't show


# ------------------------------------------------------------------ cached feature rows


def config_key(cfg: dict) -> str:
    parts = {k: cfg[k] for k in ("input", "clean", "angle", "bins", "features") if k in cfg}
    parts["version"] = CACHE_VERSION
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:12]


def cached_rows(s: SetData, cfg: dict) -> pd.DataFrame:
    """A set's step 6 rows, computed once per config and kept next to the recording."""
    cache = s.path.parent / ".feature_cache" / f"{s.name}_{config_key(cfg)}.pkl"
    sources = [s.path, s.path.with_name(f"{s.participant}_rest.bin"), s.path.with_name(f"{s.participant}_set1.bin")]
    newest = max(p.stat().st_mtime for p in sources if p.exists())
    if cache.exists() and cache.stat().st_mtime >= newest:
        return pd.read_pickle(cache)
    rows = set_features(s.path, cfg)
    cache.parent.mkdir(exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(rows, f)
    return rows


# ------------------------------------------------------------------ rows -> table


class Table:
    """Fits the Kalman filters on training sets and turns sets into (features, target) tables."""

    def __init__(self, cfg: dict, features: list[str]):
        unknown = [f for f in features if f not in FEATURE_COLUMNS]
        if unknown:
            raise ValueError(f"features not computed yet: {unknown}")
        self.cfg, self.features = cfg, features
        self.columns = [FEATURE_COLUMNS[f] for f in features]
        self.filters = sorted({FILTER_OF[f] for f in features if f in FILTER_OF})

    def fit(self, train: list[SetData]) -> "Table":
        raw = [cached_rows(s, self.cfg) for s in train]
        base = {sp.feature: sp for sp in specs_from(self.cfg)}
        fitted = {name: fit(raw, base[name], self.cfg["kalman"]["q_floor"] if base[name].model == "trend" else 0.0)
                  for name in self.filters}
        self.specs = [sp for sp in specs_from(self.cfg, fitted) if sp.feature in self.filters]
        X, _, _ = self.rows(train)
        self.fill = np.nanmean(X, axis=0) if len(X) else np.zeros(len(self.columns))
        self.fill = np.where(np.isnan(self.fill), 0.0, self.fill)
        return self

    def filtered(self, s: SetData) -> pd.DataFrame:
        rows = filter_rows(cached_rows(s, self.cfg), self.specs)
        rows = rows[~rows.is_baseline].reset_index(drop=True)
        # Lifts after the operator's "end" press (the recorder's tail: racking the weight) aren't part
        # of the set. Found on the board 2026-11-19: a lift in the 2 s tail.
        end = [t for t, label in s.meta.get("markers", []) if label == "end"]
        if end:
            rows = rows[rows.t_start <= end[0]].reset_index(drop=True)
        if "cut_s" in s.meta:  # the real-time check: what the board knew by then
            rows = rows[rows.t_known <= s.meta["cut_s"]].reset_index(drop=True)
        return rows

    def rows(self, sets: list[SetData]) -> tuple[np.ndarray, np.ndarray, list[pd.DataFrame]]:
        Xs, ys, kept = [], [], []
        for s in sets:
            r = self.filtered(s)
            for c in self.columns:
                if c not in r:
                    r[c] = np.nan
            Xs.append(r[self.columns].to_numpy(float))
            ys.append(np.maximum(s.stop_s - r.t_known.to_numpy(), 0.0) if s.stop_s is not None
                      else np.full(len(r), np.nan))
            kept.append(r)
        X = np.vstack(Xs) if Xs else np.zeros((0, len(self.columns)))
        y = np.concatenate(ys) if ys else np.zeros(0)
        return X, y, kept

    def complete(self, X: np.ndarray) -> np.ndarray:
        return np.where(np.isnan(X), self.fill, X)


# ------------------------------------------------------------------ regressors


class Ridge:
    def __init__(self, lam: float):
        self.lam = lam

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        Z = np.c_[(X - self.mu) / self.sd, np.ones(len(X))]
        pen = self.lam * np.eye(Z.shape[1])
        pen[-1, -1] = 0.0  # don't shrink the intercept
        self.coef = np.linalg.solve(Z.T @ Z + pen, Z.T @ y)
        return self

    def predict(self, X):
        return np.c_[(X - self.mu) / self.sd, np.ones(len(X))] @ self.coef


class SmallNet:
    """One hidden layer (tanh), trained with Adam and weight decay; averaged over several starts.
    Hidden size is capped so weights <= share x training rows (RepSense rule)."""

    def __init__(self, cfg: dict, seed: int = 0):
        self.c, self.seed = cfg["model"], seed

    def hidden_for(self, n_rows: int, n_in: int) -> int:
        cap = int(self.c["net_param_share"] * n_rows)
        h = min(self.c["net_max_hidden"], max(0, (cap - 1) // (n_in + 2)))
        return h

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-12] = 1.0
        self.ym, self.ys = y.mean(), y.std() or 1.0
        Z, t = (X - self.mu) / self.sd, (y - self.ym) / self.ys
        h = self.hidden_for(len(X), X.shape[1])
        if h < 1:
            raise ValueError(f"{len(X)} training rows: too few for any net under the size rule")
        self.n_params = h * (X.shape[1] + 2) + 1
        self.nets = [self._train(Z, t, h, np.random.default_rng([self.seed, k])) for k in range(self.c["net_ensemble"])]
        return self

    def _train(self, Z, t, h, rng):
        W1 = rng.normal(0, 1 / np.sqrt(Z.shape[1]), (Z.shape[1], h))
        b1, W2, b2 = np.zeros(h), rng.normal(0, 1 / np.sqrt(h), h), 0.0
        params = [W1, b1, W2, np.array(b2)]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        lr, l2, n = self.c["net_lr"], self.c["net_l2"], len(Z)
        for step in range(1, self.c["net_epochs"] + 1):
            W1, b1, W2, b2 = params
            H = np.tanh(Z @ W1 + b1)
            err = H @ W2 + b2 - t
            gW2 = H.T @ err / n + l2 * W2
            gb2 = np.array(err.mean())
            dH = np.outer(err, W2) * (1 - H**2)
            gW1 = Z.T @ dH / n + l2 * W1
            gb1 = dH.mean(0)
            for i, g in enumerate([gW1, gb1, gW2, gb2]):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g**2
                params[i] = params[i] - lr * (m[i] / (1 - 0.9**step)) / (np.sqrt(v[i] / (1 - 0.999**step)) + 1e-8)
        return params

    def predict(self, X):
        Z = (X - self.mu) / self.sd
        out = [np.tanh(Z @ W1 + b1) @ W2 + b2 for W1, b1, W2, b2 in self.nets]
        return np.mean(out, axis=0) * self.ys + self.ym


# ------------------------------------------------------------------ models the harness scores


class FatigueModel:
    """Features -> seconds until the stop point, with a straight line ("ridge") or small net ("net")."""

    def __init__(self, cfg: dict, features: list[str] = CORE_V1, kind: str = "ridge", name: str | None = None):
        self.cfg, self.features, self.kind = cfg, list(features), kind
        self.name = name or f"{kind} [{', '.join('#' + f for f in self.features)}]"

    def fit(self, train: list[SetData]) -> None:
        self.table = Table(self.cfg, self.features).fit(train)
        X, y, _ = self.table.rows(train)
        ok = ~np.isnan(y)
        X, y = self.table.complete(X[ok]), y[ok]
        self.reg = (Ridge(self.cfg["model"]["ridge_lambda"]) if self.kind == "ridge" else SmallNet(self.cfg)).fit(X, y)

    def predict(self, s: SetData) -> tuple[np.ndarray, np.ndarray]:
        X, _, kept = self.table.rows([s])
        if not len(X):
            return np.zeros(0), np.zeros(0)
        pred = np.clip(self.reg.predict(self.table.complete(X)), 0, self.cfg["model"]["max_seconds"])
        return kept[0].t_known.to_numpy(), pred


class PhysicsModel:
    """Comparison: learn the level this feature typically reaches at the stop point (training sets),
    then predict seconds left = (current level - that level) / how fast it's moving toward it."""

    def __init__(self, cfg: dict, feature: str | None = None):
        self.cfg = cfg
        self.feature = feature or cfg["model"]["physics_feature"]
        self.name = f"physics [#{self.feature} level / slope]"

    def fit(self, train: list[SetData]) -> None:
        self.table = Table(self.cfg, [self.feature]).fit(train)
        col = FEATURE_COLUMNS[self.feature]
        self.slope_col = col.replace("_level", "_slope")
        at_stop = []
        for s in train:
            r = self.table.filtered(s)
            if s.stop_s is None or col not in r:
                continue
            before = r[r.t_known <= s.stop_s][col].dropna()
            if len(before):
                at_stop.append(before.iloc[-1])
        self.stop_level = float(np.median(at_stop)) if at_stop else np.nan
        # Which way fatigue moves this feature (middle pitch falls toward its stop level, loudness rises).
        start = self.cfg["kalman"]["filters"][FILTER_OF[self.feature]].get("level0", 1.0)
        self.direction = np.sign(self.stop_level - (start if start is not None else 1.0)) or 1.0

    def predict(self, s: SetData) -> tuple[np.ndarray, np.ndarray]:
        r = self.table.filtered(s)
        col = FEATURE_COLUMNS[self.feature]
        if not len(r) or col not in r or self.slope_col not in r:
            return np.zeros(0), np.zeros(0)
        remaining = (self.stop_level - r[col].to_numpy()) * self.direction   # > 0: not there yet
        rate = r[self.slope_col].to_numpy() * self.direction                  # > 0: heading there
        with np.errstate(divide="ignore", invalid="ignore"):
            left = np.where(remaining <= 0, 0.0, np.where(rate > 0, remaining / rate, np.inf))
        left = np.where(np.isnan(left), np.inf, left)
        return r.t_known.to_numpy(), np.clip(left, 0, self.cfg["model"]["max_seconds"])
