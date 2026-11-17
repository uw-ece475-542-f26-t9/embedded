"""Step 7: Kalman filters that turn noisy per-lift features into a level and a slope.

One small filter per feature. Same shape as RepSense's track_down (predict, then correct toward the
measurement), with two differences: there is no second sensor to predict from, so the filter keeps
its own slope state; and the pull toward each measurement is computed from the uncertainties
instead of fixed.

  model "trend":  state = (level, slope per second). Predict: level += slope * dt. The slope may
                  wander: process noise q per second (continuous white-noise acceleration).
                  Output: level (the filtered feature) and slope (#13a for middle pitch).
  model "level":  state = level only, a random walk with process noise q per second.

  r = measurement noise: the variance of one lift's value around the truth.

Each set starts fresh after its baseline lifts at (level0, slope 0) with starting uncertainties
from the config; a feature with no fixed starting value (#26) starts from its first measurement.
Rows with a missing value (too few bins, a jolt) are prediction only: uncertainty widens.

q and r are fitted by maximum likelihood on training sets (fit), and checked with the innovation
test: innovations divided by their predicted size should have spread ~1 and no lift-to-lift pattern.
Once fitted, the filter settles to fixed pulls (an alpha-beta filter), which is what the ESP32 can run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class FilterSpec:
    feature: str            # column in the feature rows
    model: str              # "trend" or "level"
    q: float                # process noise per second
    r: float                # measurement variance
    level0: float | None    # starting level (None: start from the first measurement)
    level_sd0: float        # starting uncertainty of the level
    slope_sd0: float        # starting uncertainty of the slope (per second)


def specs_from(cfg: dict, fitted: dict | None = None) -> list[FilterSpec]:
    out = []
    for name, c in cfg["kalman"]["filters"].items():
        q, r = (fitted or {}).get(name, (c["q"], c["r"]))
        out.append(FilterSpec(feature=name, model=c["model"], q=q, r=r, level0=c.get("level0"),
                              level_sd0=c["level_sd0"], slope_sd0=c.get("slope_sd0", 0.0)))
    return out


def run_filter(t: np.ndarray, z: np.ndarray, spec: FilterSpec, t0: float | None = None) -> pd.DataFrame:
    """Filter one set's values z at times t (seconds, increasing). t0: when the filter starts (the
    end of the baseline); default the first row. Uses only rows up to each one."""
    n = len(t)
    trend = spec.model == "trend"
    k = 2 if trend else 1
    level, slope = np.full(n, np.nan), np.full(n, np.nan)
    level_sd, slope_sd = np.full(n, np.nan), np.full(n, np.nan)
    innov, innov_sd = np.full(n, np.nan), np.full(n, np.nan)

    x, P, t_prev = None, None, t0
    if spec.level0 is not None:
        x = np.array([spec.level0, 0.0][:k])
        P = np.diag([spec.level_sd0**2, spec.slope_sd0**2][:k])
    for i in range(n):
        if x is None:  # no fixed start: begin at the first measurement
            if np.isnan(z[i]):
                continue
            x = np.array([z[i], 0.0][:k])
            P = np.diag([spec.r, spec.slope_sd0**2][:k])
            t_prev = t[i]
        else:
            dt = max(0.0, t[i] - (t_prev if t_prev is not None else t[i]))
            t_prev = t[i]
            if trend:
                F = np.array([[1.0, dt], [0.0, 1.0]])
                Q = spec.q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
            else:
                F, Q = np.eye(1), np.array([[spec.q * dt]])
            x = F @ x
            P = F @ P @ F.T + Q
            if not np.isnan(z[i]):
                S = P[0, 0] + spec.r
                v = z[i] - x[0]
                K = P[:, 0] / S
                x = x + K * v
                P = P - np.outer(K, P[0, :])
                innov[i], innov_sd[i] = v, np.sqrt(S)
        level[i], level_sd[i] = x[0], np.sqrt(P[0, 0])
        if trend:
            slope[i], slope_sd[i] = x[1], np.sqrt(P[1, 1])
    out = pd.DataFrame({"level": level, "level_sd": level_sd, "innov": innov, "innov_sd": innov_sd})
    if trend:
        out["slope"], out["slope_sd"] = slope, slope_sd
    return out


def _series(rows: pd.DataFrame, spec: FilterSpec):
    """The rows a filter runs over, and when it starts. The set-relative features equal 1.0 on
    average over the baseline lifts, so the filter starts at their mean time (not the last one: the
    muscle has already tired a little). #26 (no fixed start) also uses the baseline lifts' values."""
    has_base = "is_baseline" in rows and rows.is_baseline.any()
    t0 = float(rows[rows.is_baseline].t_known.mean()) if has_base and spec.level0 is not None else None
    use = rows if (spec.level0 is None or "is_baseline" not in rows) else rows[~rows.is_baseline]
    return use, t0


def filter_rows(rows: pd.DataFrame, specs: list[FilterSpec]) -> pd.DataFrame:
    """Add <feature>_level (and _slope, _sd, _innov) columns to one set's feature rows."""
    rows = rows.sort_values("t_known").reset_index(drop=True)
    out = rows.copy()
    for spec in specs:
        if spec.feature not in rows:
            continue
        use, t0 = _series(rows, spec)
        res = run_filter(use.t_known.to_numpy(), use[spec.feature].to_numpy(float), spec, t0)
        for col in res:
            out.loc[use.index, f"{spec.feature}_{col}"] = res[col].to_numpy()
    return out


# ------------------------------------------------------------------ fitting q and r


def _nll(series: list[tuple[np.ndarray, np.ndarray, float | None]], spec: FilterSpec) -> float:
    total = 0.0
    for t, z, t0 in series:
        res = run_filter(t, z, spec, t0)
        v, s = res.innov.to_numpy(), res.innov_sd.to_numpy()
        ok = ~np.isnan(v) & ~np.isnan(s)
        total += float(np.sum(np.log(s[ok] ** 2) + (v[ok] / s[ok]) ** 2))
    return total


def fit(sets: list[pd.DataFrame], spec: FilterSpec, q_floor: float = 0.0) -> tuple[float, float]:
    """q and r by maximum likelihood of the innovations over training sets. q_floor keeps the filter
    able to follow a changing slope: simulated fatigue is a straight line, so the fit alone drives q
    to zero, and real fatigue may speed up near the end."""
    series = []
    for rows in sets:
        if spec.feature not in rows:
            continue
        use, t0 = _series(rows.sort_values("t_known").reset_index(drop=True), spec)
        series.append((use.t_known.to_numpy(), use[spec.feature].to_numpy(float), t0))

    def cost(logp):
        q, r = np.exp(logp)
        return _nll(series, FilterSpec(**{**spec.__dict__, "q": q, "r": r}))
    start = np.log([max(spec.q, 1e-12), max(spec.r, 1e-12)])
    res = minimize(cost, start, method="Nelder-Mead", options={"xatol": 0.05, "fatol": 0.01, "maxiter": 200})
    q, r = np.exp(res.x)
    return float(max(q, q_floor)), float(r)


def innovation_check(filtered: list[pd.DataFrame], feature: str) -> dict:
    """Normalised innovations should be plain noise: spread ~1, no lift-to-lift pattern."""
    e_all, lag = [], []
    for rows in filtered:
        v, s = rows.get(f"{feature}_innov"), rows.get(f"{feature}_innov_sd")
        if v is None:
            continue
        e = (v / s).dropna().to_numpy()
        e_all.append(e)
        if len(e) > 2:
            lag.append(np.corrcoef(e[:-1], e[1:])[0, 1])
    e = np.concatenate(e_all) if e_all else np.zeros(0)
    return {"spread": float(np.std(e)) if len(e) else np.nan, "lag1": float(np.nanmean(lag)) if lag else np.nan,
            "n": int(len(e))}
