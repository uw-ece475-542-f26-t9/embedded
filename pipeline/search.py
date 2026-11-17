"""Step 8b: random-subset feature ranking (FEATURES.md protocol, as run in RepSense). Real data only;
simulated data is used only to check this code (the planted test).

For a table of training rows (features X, target y, person of each row):
  1. Random subsets of `size` candidate features (all of them if there are fewer combinations).
  2. Each subset is scored by leave-one-person-out within these people: a straight line fitted on
     the others, mean absolute error (seconds) on the held-out one. Score = -error, higher is better.
     All subsets' lines are solved in one batch from shared sums, so 10,000 subsets are cheap.
  3. Credit, adjusted: each feature's weight in a regression of subset score on which features the
     subset had (raw "with - without" is reported next to it but doesn't decide).
  4. Luck: the same with the target shuffled within each person, `shuffles` times; a feature counts
     only if its credit beats the shuffled 95th percentile.
  5. Consistency: people resampled `bootstrap` times; kept if it beats luck in >= keep_share of them.

SearchedModel runs all of that inside fit(), on the training people only, so the step 3 harness
scores the whole procedure on people it never saw (fully nested). Inner folds share the outer
fold's Kalman q and r (fitted on the training people; the held-out person is never involved).
"""

from __future__ import annotations

import itertools
from math import comb

import numpy as np
import pandas as pd

from .evaluate import SetData
from .model import FEATURE_COLUMNS, Ridge, Table


def choose_subsets(n_features: int, size: int, count: int, rng: np.random.Generator) -> np.ndarray:
    size = min(size, n_features)
    if comb(n_features, size) <= count:
        return np.array(list(itertools.combinations(range(n_features), size)))
    seen = set()
    while len(seen) < count:
        seen.add(tuple(sorted(rng.choice(n_features, size, replace=False))))
    return np.array(sorted(seen))


def subset_scores(X: np.ndarray, y: np.ndarray, groups: np.ndarray, subsets: np.ndarray, lam: float) -> np.ndarray:
    """-(mean absolute error) of a ridge line on each subset, leave-one-group-out, batched."""
    S, k = subsets.shape
    F = X.shape[1]
    idx = np.c_[subsets, np.full(S, F)]  # the intercept is column F
    pen = lam * np.eye(k + 1)
    pen[-1, -1] = 0.0
    err = np.zeros(S)
    folds = np.unique(groups)
    if len(folds) < 2:   # nobody left to train on: no score
        return np.full(S, np.nan)
    for g in folds:
        tr, te = groups != g, groups == g
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd < 1e-12] = 1.0
        Ztr = np.c_[(X[tr] - mu) / sd, np.ones(tr.sum())]
        Zte = np.c_[(X[te] - mu) / sd, np.ones(te.sum())]
        G, b = Ztr.T @ Ztr, Ztr.T @ y[tr]
        Gs = G[idx[:, :, None], idx[:, None, :]] + pen
        coef = np.linalg.solve(Gs, b[idx][..., None])[..., 0]          # (S, k+1)
        pred = np.einsum("nsk,sk->sn", Zte[:, idx], coef)               # (S, n_te)
        err += np.mean(np.abs(np.clip(pred, 0, None) - y[te]), axis=1)
    return -err / len(folds)


def credits(subsets: np.ndarray, scores: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Adjusted credit (regression of score on membership) and raw credit (mean with - without)."""
    M = np.zeros((len(subsets), n_features))
    np.put_along_axis(M, subsets, 1.0, axis=1)
    A = np.c_[M, np.ones(len(M))]
    adjusted = np.linalg.lstsq(A, scores, rcond=None)[0][:n_features]
    raw = np.array([scores[M[:, j] == 1].mean() - scores[M[:, j] == 0].mean()
                    if 0 < M[:, j].sum() < len(M) else np.nan for j in range(n_features)])
    return adjusted, raw


def _shuffle_within(y, groups, rng):
    out = y.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] = rng.permutation(y[m])
    return out


def rank(X: np.ndarray, y: np.ndarray, groups: np.ndarray, names: list[str], cfg: dict,
         rng: np.random.Generator | None = None) -> pd.DataFrame:
    c, lam = cfg["search"], cfg["model"]["ridge_lambda"]
    rng = rng or np.random.default_rng(c["seed"])
    F = X.shape[1]
    if len(np.unique(groups)) < 2:   # can't leave anyone out: nothing can be ranked, nothing kept
        return pd.DataFrame({"feature": names, "credit": np.nan, "raw_credit": np.nan, "luck95": np.nan,
                             "bootstrap_share": 0.0, "kept": False})
    subsets = choose_subsets(F, c["size"], c["subsets"], rng)
    adjusted, raw = credits(subsets, subset_scores(X, y, groups, subsets, lam), F)
    luck = np.array([credits(subsets, subset_scores(X, _shuffle_within(y, groups, rng), groups, subsets, lam), F)[0]
                     for _ in range(c["shuffles"])])
    luck95 = np.percentile(luck, 95, axis=0)
    people = np.unique(groups)
    beats = np.zeros(F)
    for _ in range(c["bootstrap"]):
        pick = rng.choice(people, len(people), replace=True)
        if len(set(pick)) < 2:   # a resample of one person repeated can't be scored; it counts as a fail
            continue
        rows = [np.flatnonzero(groups == p) for p in pick]
        Xb = np.vstack([X[r] for r in rows])
        yb = np.concatenate([y[r] for r in rows])
        # A person drawn twice keeps one group label, so both copies are held out together: two labels
        # would let one copy train the model while its twin is scored (a leak that inflates credit).
        gb = np.concatenate([groups[r] for r in rows])
        beats += credits(subsets, subset_scores(Xb, yb, gb, subsets, lam), F)[0] > luck95
    share = beats / max(c["bootstrap"], 1)
    out = pd.DataFrame({"feature": names, "credit": adjusted, "raw_credit": raw, "luck95": luck95,
                        "bootstrap_share": share})
    out["kept"] = (out.credit > out.luck95) & (out.bootstrap_share >= c["keep_share"])
    return out.sort_values("credit", ascending=False).reset_index(drop=True)


class SearchedModel:
    """The whole search, run on the training people inside fit(); a straight line on what it keeps."""

    def __init__(self, cfg: dict, candidates: list[str] | None = None, name: str = "search + ridge"):
        self.cfg = cfg
        self.candidates = list(candidates or FEATURE_COLUMNS)
        self.name = name

    def fit(self, train: list[SetData]) -> None:
        self.table = Table(self.cfg, self.candidates).fit(train)
        X, y, kept = self.table.rows(train)
        groups = np.concatenate([np.full(len(k), s.participant) for k, s in zip(kept, train)])
        ok = ~np.isnan(y)
        X, y, groups = self.table.complete(X[ok]), y[ok], groups[ok]
        self.ranking = rank(X, y, groups, self.candidates, self.cfg)
        self.kept = [f for f in self.ranking[self.ranking.kept].feature]
        cols = [self.candidates.index(f) for f in self.kept]
        self.cols = cols
        self.reg = Ridge(self.cfg["model"]["ridge_lambda"]).fit(X[:, cols] if cols else np.zeros((len(X), 0)), y)

    def predict(self, s: SetData):
        X, _, kept = self.table.rows([s])
        if not len(X):
            return np.zeros(0), np.zeros(0)
        Xc = self.table.complete(X)[:, self.cols] if self.cols else np.zeros((len(X), 0))
        return kept[0].t_known.to_numpy(), np.clip(self.reg.predict(Xc), 0, self.cfg["model"]["max_seconds"])
