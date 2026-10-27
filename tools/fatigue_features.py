#!/usr/bin/env python3
"""EMG fatigue features: does our hardware see fatigue?

Reads sets recorded with tools/record.py, splits each into reps with the rep
counter's turning points, and computes four sEMG features per rep, each two
ways:

    MDF       frequency that splits spectral power in half      (expect: drops)
    MNF       power-weighted mean frequency                     (expect: drops)
    skewness  3rd standardized moment of the power values       (find which way)
    kurtosis  4th standardized moment of the power values       (find which way)

    stft  128 ms Hann frames, 64 ms hop; MDF/MNF averaged over frames,
          skewness/kurtosis over every power value in the time-frequency grid
    fft   one Hann-windowed FFT over the whole window

Every rep is analysed on a 500 ms window centred on the lift's EMG peak (the
highest 50 ms moving RMS). The lift is whichever half-rep has more EMG,
because the rep counter doesn't know which way is up. The signal is
band-passed 20-400 Hz and notched at 60 Hz first. (The whole recording is
filtered forward-backward and then cut, which avoids the edge transients of
filtering each 500 ms window on its own.) Each rep also logs its RMS and lift
duration, to spot confounds: lifting harder or faster also shifts the
spectrum.

Usage:
    python tools/fatigue_features.py recordings/
    python tools/fatigue_features.py recordings/S01_* --out analysis/S01

Writes results.md (the feature x method pass/fail table), results.csv,
reps.csv, sanity.csv and plots/ to --out (default: analysis/<timestamp>).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy import signal, stats

import capture_format as cf

FEATURES = ["mdf", "mnf", "skew", "kurt"]
FEATURE_NAMES = {"mdf": "MDF", "mnf": "MNF", "skew": "Skewness", "kurt": "Kurtosis"}
METHODS = ["stft", "fft"]
# Expected direction with fatigue; None = find out from the data.
EXPECTED = {"mdf": -1, "mnf": -1, "skew": None, "kurt": None}

BAND = (20.0, 400.0)
NOTCH_HZ = 60.0
WINDOW_S = 0.5
PEAK_RMS_S = 0.05
STFT_FRAME_S = 0.128
STFT_HOP_S = 0.064
BASELINE_REPS = 3       # normalize to the mean of the first 3 reps
MIN_REPS = 6            # fewer reps than this: no trend test

# Sanity-check thresholds (starting points, like the size threshold).
LOW_BAND_HZ = 250.0
LOW_BAND_MIN_SHARE = 0.80   # "most" of the 20-400 Hz power below 250 Hz
LINE_MAX_RATIO = 2.0        # PSD at 60 Hz vs its neighbours, after the notch
REST_MIN_RATIO = 5.0        # contraction RMS / resting RMS
SIZE_FACTOR = 2.0           # change must exceed 2x the scatter around the trend
P_THRESHOLD = 0.05


# ------------------------------------------------------------------ loading


@dataclass
class SetData:
    path: str
    meta: dict
    rec: cf.Recording
    fs: float = 2000.0
    filtered: np.ndarray = field(default_factory=lambda: np.zeros(0))
    reps: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def name(self) -> str:
        return os.path.splitext(os.path.basename(self.path))[0]

    @property
    def group(self) -> tuple:
        return (self.meta.get("subject", "?"), self.meta.get("session", 0))


def find_recordings(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.bin"), recursive=True))
        else:
            files += [f for f in sorted(glob.glob(p)) if f.endswith(".bin")]
    return files


def load_set(path: str) -> SetData:
    meta_path = os.path.splitext(path)[0] + ".json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    s = SetData(path=path, meta=meta, rec=cf.load(path))
    rec = s.rec
    # Frequencies come from the sample rate, so use the one measured against
    # the ESP32's clock if it's sane; the ADC's real rate can be off nominal.
    measured = rec.effective_emg_hz()
    s.fs = float(rec.emg_hz)
    if measured and abs(measured / rec.emg_hz - 1) < 0.1:
        s.fs = measured
    if measured and abs(measured / rec.emg_hz - 1) > 0.01:
        s.warnings.append(f"EMG rate {measured:.0f} Hz vs nominal {rec.emg_hz} Hz; using measured")
    return s


# ------------------------------------------------------------------ signal


def bandpass_notch(x: np.ndarray, fs: float) -> np.ndarray:
    """Zero-phase 20-400 Hz band-pass and 60 Hz notch. NaN gaps -> 0."""
    x = np.where(np.isnan(x), np.nanmean(x), x)
    x = x - np.mean(x)
    sos = signal.butter(4, BAND, btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, x)
    b, a = signal.iirnotch(NOTCH_HZ, Q=30, fs=fs)
    return signal.filtfilt(b, a, y)


def moving_rms(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) < n:
        return np.sqrt(np.mean(x**2)) * np.ones(1)
    c = np.convolve(x**2, np.ones(n) / n, mode="valid")
    return np.sqrt(c)


@dataclass
class Rep:
    number: int
    start: int
    mid: int
    end: int
    lift: tuple[int, int]
    window: tuple[int, int]
    rom_deg: float


def segment_reps(s: SetData) -> list[Rep]:
    """Rep k spans the turning point before its middle to the one after.

    The rep counter emits a turning point and, on every second one, a rep.
    So each rep's middle is the last turning point at or before the rep
    event; its start is the one before that, its end the one after (or, for
    the last rep, the middle plus the first half's length).
    """
    rec, fs, x = s.rec, s.fs, s.filtered
    turns = rec.turn_stamp - rec.first_index
    reps = []
    win = int(round(WINDOW_S * fs))
    peak_n = max(1, int(round(PEAK_RMS_S * fs)))
    for k, (stamp, rom) in enumerate(zip(rec.rep_stamp - rec.first_index, rec.rep_rom)):
        i = np.searchsorted(turns, stamp, side="right") - 1
        if i < 1:
            s.warnings.append(f"rep at sample {stamp} has no turning point before it; skipped")
            continue
        mid, start = int(turns[i]), int(turns[i - 1])
        first_half = mid - start
        end = int(turns[i + 1]) if i + 1 < len(turns) else mid + first_half
        if end - mid > 2 * first_half:
            end = mid + first_half  # arm rested before the next swing
        start, end = max(start, 0), min(end, len(x))
        if mid - start < peak_n or end - mid < peak_n:
            s.warnings.append(f"rep {k + 1} too short to analyse; skipped")
            continue
        halves = [(start, mid), (mid, end)]
        lift = max(halves, key=lambda h: np.sqrt(np.mean(x[h[0] : h[1]] ** 2)))
        r = moving_rms(x[lift[0] : lift[1]], peak_n)
        centre = lift[0] + int(np.argmax(r)) + peak_n // 2
        w0 = min(max(centre - win // 2, 0), len(x) - win)
        reps.append(Rep(k + 1, start, mid, end, lift, (w0, w0 + win), float(rom)))
    # Number reps within the set from 1.
    for n, r in enumerate(reps, 1):
        r.number = n
    return reps


# ------------------------------------------------------------------ features


def spectral_stats(freqs: np.ndarray, power: np.ndarray) -> tuple[float, float]:
    """MDF and MNF of one power spectrum (already limited to the band)."""
    total = power.sum()
    if total <= 0:
        return np.nan, np.nan
    cum = np.cumsum(power)
    j = int(np.searchsorted(cum, total / 2))
    # Interpolate within the bin where the cumulative power crosses half.
    prev = cum[j - 1] if j > 0 else 0.0
    frac = (total / 2 - prev) / power[j] if power[j] > 0 else 0.0
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 0.0
    mdf = freqs[j] - df / 2 + frac * df
    mnf = float(np.sum(freqs * power) / total)
    return float(mdf), mnf


def shape_stats(values: np.ndarray) -> tuple[float, float]:
    """Skewness and kurtosis (Pearson, normal = 3) of a set of power values."""
    v = values.ravel()
    return float(stats.skew(v)), float(stats.kurtosis(v, fisher=False))


def features_stft(w: np.ndarray, fs: float) -> dict:
    n = int(round(STFT_FRAME_S * fs))
    hop = int(round(STFT_HOP_S * fs))
    win = signal.get_window("hann", n)
    freqs = np.fft.rfftfreq(n, 1 / fs)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    grid = []
    for i in range(0, len(w) - n + 1, hop):
        seg = w[i : i + n]
        grid.append(np.abs(np.fft.rfft((seg - seg.mean()) * win)) ** 2)
    grid = np.array(grid)[:, band]
    per_frame = [spectral_stats(freqs[band], p) for p in grid]
    skew, kurt = shape_stats(grid)
    return {
        "mdf": float(np.mean([m for m, _ in per_frame])),
        "mnf": float(np.mean([m for _, m in per_frame])),
        "skew": skew,
        "kurt": kurt,
    }


def features_fft(w: np.ndarray, fs: float) -> dict:
    win = signal.get_window("hann", len(w))
    freqs = np.fft.rfftfreq(len(w), 1 / fs)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    p = np.abs(np.fft.rfft((w - w.mean()) * win)) ** 2
    mdf, mnf = spectral_stats(freqs[band], p[band])
    skew, kurt = shape_stats(p[band])
    return {"mdf": mdf, "mnf": mnf, "skew": skew, "kurt": kurt}


def rep_rows(s: SetData) -> list[dict]:
    rows = []
    for r in s.reps:
        w = s.filtered[r.window[0] : r.window[1]]
        row = {
            "set": s.name,
            "subject": s.meta.get("subject", ""),
            "session": s.meta.get("session", ""),
            "set_type": s.meta.get("set_type", ""),
            "rep": r.number,
            "rms_mv": float(np.sqrt(np.mean(w**2))),
            "lift_s": (r.lift[1] - r.lift[0]) / s.fs,
            "rom_deg": r.rom_deg,
            "lift_is_first_half": r.lift[0] == r.start,
        }
        for method, fn in (("stft", features_stft), ("fft", features_fft)):
            for f, v in fn(w, s.fs).items():
                row[f"{f}_{method}"] = v
        rows.append(row)
    return rows


# ------------------------------------------------------------------ sanity


def sanity(s: SetData, rest_rms: float | None) -> dict:
    rec, fs, x = s.rec, s.fs, s.filtered
    out = {
        "set": s.name,
        "crc_errors": rec.crc_errors,
        "missing_samples": rec.missing_samples,
        "adc_overflows": rec.adc_overflows,
        "frames_dropped": rec.frames_dropped,
        "fs_hz": round(fs, 1),
        "clipped": rec.clipped,
    }
    windows = [x[r.window[0] : r.window[1]] for r in s.reps]
    if windows:
        cat = np.concatenate(windows)
        f, p = signal.welch(cat, fs=fs, nperseg=min(512, len(cat)))
        band = (f >= BAND[0]) & (f <= BAND[1])
        out["share_below_250hz"] = float(p[band & (f < LOW_BAND_HZ)].sum() / p[band].sum())
        out["line_60hz_ratio"] = line_ratio(f, p, NOTCH_HZ)
        # Harmonics aren't notched; report them so we know if they matter.
        out["line_120hz_ratio"] = line_ratio(f, p, 120.0)
        out["line_180hz_ratio"] = line_ratio(f, p, 180.0)
        out["contraction_rms_mv"] = float(np.median([np.sqrt(np.mean(w**2)) for w in windows]))
    if rest_rms is None:
        rest_rms = quiet_rms(s)
        out["rest_source"] = "quietest still windows in this set"
    else:
        out["rest_source"] = "rest recording"
    out["rest_rms_mv"] = rest_rms
    if rest_rms and "contraction_rms_mv" in out:
        out["contraction_to_rest"] = out["contraction_rms_mv"] / rest_rms

    checks = {
        "below_250hz": out.get("share_below_250hz", 0) >= LOW_BAND_MIN_SHARE,
        "no_60hz": out.get("line_60hz_ratio", np.inf) < LINE_MAX_RATIO,
        "no_clipping": rec.clipped == 0,
        "rest_well_below": out.get("contraction_to_rest", 0) >= REST_MIN_RATIO,
        "stream_ok": rec.crc_errors == 0 and rec.missing_samples == 0 and rec.adc_overflows == 0,
    }
    out.update({f"ok_{k}": v for k, v in checks.items()})
    out["ok"] = all(checks.values())
    return out


def line_ratio(f: np.ndarray, p: np.ndarray, hz: float) -> float:
    at = p[np.abs(f - hz) <= 2]
    near = p[(np.abs(f - hz) > 5) & (np.abs(f - hz) <= 20)]
    if not len(at) or not len(near):
        return float("nan")
    return float(at.max() / np.median(near))


def quiet_rms(s: SetData) -> float | None:
    """Fallback resting level: quietest 500 ms windows while the arm is still."""
    x, rec = s.filtered, s.rec
    n = int(round(WINDOW_S * s.fs))
    if len(x) < 2 * n:
        return None
    r = moving_rms(x, n)[::n // 4]
    pos = np.arange(len(r)) * (n // 4) + n // 2
    if len(rec.imu_stamp):
        still_at = np.interp(pos, rec.imu_stamp - rec.first_index, rec.imu_still.astype(float)) > 0.5
        if still_at.any():
            r = r[still_at]
    return float(np.percentile(r, 10))


def rest_levels(sets: list[SetData]) -> dict:
    """Resting RMS per (subject, session) from any 'rest' recordings."""
    levels = {}
    for s in sets:
        if s.meta.get("set_type") == "rest" and len(s.filtered):
            levels.setdefault(s.group, []).append(float(np.median(moving_rms(s.filtered, int(WINDOW_S * s.fs)))))
    return {g: float(np.median(v)) for g, v in levels.items()}


# ------------------------------------------------------------------ evaluation


def set_trend(rows: list[dict], key: str) -> dict | None:
    """Trend of one feature over one set, normalized to the first reps."""
    y = np.array([r[key] for r in rows], dtype=float)
    x = np.array([r["rep"] for r in rows], dtype=float)
    ok = ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(y) < MIN_REPS:
        return None
    base = np.mean(y[:BASELINE_REPS])
    if base == 0:
        return None
    norm = y / base
    fit = stats.linregress(x, norm)
    resid = norm - (fit.intercept + fit.slope * x)
    scatter = float(np.std(resid, ddof=2))
    change = float(np.mean(norm[-BASELINE_REPS:]) - np.mean(norm[:BASELINE_REPS]))
    return {
        "slope": float(fit.slope),
        "p": float(fit.pvalue),
        "change": change,
        "scatter": scatter,
        "effect": change / scatter if scatter > 0 else np.nan,
        "norm": norm,
        "x": x,
    }


def evaluate(by_set: dict[str, list[dict]], sets: dict[str, SetData]) -> list[dict]:
    results = []
    fatigue = [n for n, s in sets.items() if s.meta.get("set_type") == "fatigue" and n in by_set]
    control = [n for n, s in sets.items() if s.meta.get("set_type") == "control" and n in by_set]
    for f in FEATURES:
        for m in METHODS:
            key = f"{f}_{m}"
            ft = {n: set_trend(by_set[n], key) for n in fatigue}
            ft = {n: t for n, t in ft.items() if t}
            ct = {n: set_trend(by_set[n], key) for n in control}
            ct = {n: t for n, t in ct.items() if t}

            direction = EXPECTED[f]
            found = ""
            if direction is None:
                slopes = [t["slope"] for t in ft.values()]
                direction = int(np.sign(np.median(slopes))) if slopes else 0
                found = {1: "rises", -1: "drops", 0: "?"}[direction]

            def trending(t):
                return bool(t["p"] < P_THRESHOLD and np.sign(t["slope"]) == direction)

            def big(t):
                return bool(np.sign(t["change"]) == direction and abs(t["change"]) > SIZE_FACTOR * t["scatter"])

            n = len(ft)
            trend_ok = n > 0 and sum(trending(t) for t in ft.values()) > n / 2
            size_ok = n > 0 and sum(big(t) for t in ft.values()) > n / 2
            # A control "shows such a trend" if it would pass as a fatiguing
            # set: significant and large, in the same direction.
            spec = None
            if ct:
                spec = not any(trending(t) and big(t) for t in ct.values())

            # Repeatability: every subject-session's fatiguing sets mostly
            # pass trend and size, with at least two people.
            groups: dict[tuple, list[bool]] = {}
            for name, t in ft.items():
                groups.setdefault(sets[name].group, []).append(trending(t) and big(t))
            people = {g[0] for g in groups}
            repeat = None
            if len(people) >= 2:
                repeat = all(sum(v) > len(v) / 2 for v in groups.values())

            criteria = [trend_ok, size_ok, spec, repeat]
            verdict = "PASS" if all(c is True for c in criteria) else ("FAIL" if any(c is False for c in criteria) else "INCOMPLETE")
            effects = [t["effect"] * direction for t in ft.values() if not np.isnan(t["effect"])]
            changes = [t["change"] for t in ft.values()]
            results.append({
                "feature": FEATURE_NAMES[f],
                "method": m,
                "direction": found or ("drops" if direction < 0 else "rises"),
                "fatigue_sets": n,
                "trend": trend_ok,
                "size": size_ok,
                "specificity": spec,
                "repeatability": repeat,
                "verdict": verdict,
                # Effect: (last 3 - first 3) / scatter, signed so + = expected way.
                "median_effect": float(np.median(effects)) if effects else np.nan,
                "median_change_pct": float(np.median(changes) * 100) if changes else np.nan,
                "sets_trending": f"{sum(trending(t) for t in ft.values())}/{n}",
                "controls_trending": f"{sum(trending(t) and big(t) for t in ct.values())}/{len(ct)}" if ct else "–",
            })
    return results


def confounds(by_set: dict[str, list[dict]], sets: dict[str, SetData]) -> list[str]:
    notes = []
    for name, rows in by_set.items():
        if sets[name].meta.get("set_type") != "fatigue":
            continue
        for key, label in (("lift_s", "lift duration"), ("rms_mv", "RMS")):
            t = set_trend(rows, key)
            if t and t["p"] < P_THRESHOLD:
                notes.append(f"{name}: {label} changed {t['change'] * 100:+.0f}% over the set (p={t['p']:.3f})")
    return notes


# ------------------------------------------------------------------ output


def fmt(v) -> str:
    if v is None:
        return "n/a"
    if v is True:
        return "✓"
    if v is False:
        return "✗"
    if isinstance(v, float):
        return "–" if np.isnan(v) else f"{v:.2f}"
    return str(v)


def write_markdown(path, results, sanity_rows, notes, sets):
    fatigue = sum(s.meta.get("set_type") == "fatigue" for s in sets.values())
    control = sum(s.meta.get("set_type") == "control" for s in sets.values())
    people = {s.meta.get("subject") for s in sets.values()}
    lines = [
        "# EMG fatigue features: results",
        "",
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. {len(sets)} recordings: "
        f"{fatigue} fatiguing sets, {control} control sets, {len(people)} people.",
        "",
        "## Feature x method",
        "",
        "Effect = (mean of last 3 reps - mean of first 3) / rep-to-rep scatter around the trend,",
        "signed so positive is the expected (or found) direction; median over fatiguing sets.",
        "",
        "| Feature | Method | Direction | Trend | Size | Specificity | Repeatability | Verdict | Effect | Change | Sets trending | Controls trending |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['feature']} | {r['method']} | {r['direction']} | {fmt(r['trend'])} | {fmt(r['size'])} | "
            f"{fmt(r['specificity'])} | {fmt(r['repeatability'])} | **{r['verdict']}** | "
            f"{fmt(r['median_effect'])} | {r['median_change_pct']:+.1f}% | {r['sets_trending']} | {r['controls_trending']} |"
        )
    lines += ["", "## Which method goes forward", ""]
    for f in FEATURES:
        pair = [r for r in results if r["feature"] == FEATURE_NAMES[f]]
        best = max(pair, key=lambda r: -np.inf if np.isnan(r["median_effect"]) else r["median_effect"])
        lines.append(f"- **{FEATURE_NAMES[f]}**: {best['method']} (effect {fmt(best['median_effect'])})")
    lines += ["", "## Sanity checks", "",
              "| Set | 20-400 Hz power < 250 Hz | 60 Hz ratio | 120 / 180 Hz | Clipped | Contraction / rest | Stream | OK |",
              "|---|---|---|---|---|---|---|---|"]
    for s in sanity_rows:
        share = s.get("share_below_250hz")
        lines.append(
            f"| {s['set']} | {'–' if share is None else f'{share:.0%}'} | {fmt(s.get('line_60hz_ratio'))} | "
            f"{fmt(s.get('line_120hz_ratio'))} / {fmt(s.get('line_180hz_ratio'))} | {s['clipped']} | "
            f"{fmt(s.get('contraction_to_rest'))} ({s['rest_source']}) | "
            f"{'ok' if s['ok_stream_ok'] else 'PROBLEM'} | {fmt(s['ok'])} |"
        )
    lines += ["", f"Thresholds: ≥{LOW_BAND_MIN_SHARE:.0%} of power below {LOW_BAND_HZ:.0f} Hz, 60 Hz under "
              f"{LINE_MAX_RATIO}x its neighbours, no clipping, contraction ≥{REST_MIN_RATIO}x rest."]
    lines += ["", "## Confounds", ""]
    lines += [f"- {n}" for n in notes] or ["- No significant change in lift duration or RMS within any fatiguing set."]
    warnings = [f"- {s.name}: {w}" for s in sets.values() for w in s.warnings]
    if warnings:
        lines += ["", "## Warnings", ""] + warnings
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot_set(path, s: SetData, rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    x = [r["rep"] for r in rows]
    for ax, f in zip(axes.flat[:4], FEATURES):
        for m, style in (("stft", "o-"), ("fft", "s--")):
            t = set_trend(rows, f"{f}_{m}")
            if t:
                ax.plot(t["x"], t["norm"], style, label=f"{m} (p={t['p']:.3f})", ms=4)
        ax.axhline(1, color="gray", lw=0.5)
        ax.set_title(FEATURE_NAMES[f])
        ax.legend(fontsize=8)
    axes[2, 0].plot(x, [r["rms_mv"] for r in rows], "o-", ms=4)
    axes[2, 0].set_title("RMS (mV) — confound check")
    axes[2, 1].plot(x, [r["lift_s"] for r in rows], "o-", ms=4)
    axes[2, 1].set_title("Lift duration (s) — confound check")
    for ax in axes[2]:
        ax.set_xlabel("rep")
    m = s.meta
    fail = f", failure at rep {m['failure_rep']}" if m.get("failure_rep") else ""
    fig.suptitle(f"{m.get('subject', '?')} session {m.get('session', '?')} — {m.get('set_type', '?')} set{fail}\n"
                 f"features normalized to the mean of reps 1-{BASELINE_REPS}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_overview(path, s: SetData):
    """The whole recording with each rep's halves and analysis window, plus its spectrum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1, 1.4]})
    t = np.arange(len(s.filtered)) / s.fs
    a1.plot(t, s.filtered, lw=0.3, color="k")
    label_y = np.percentile(np.abs(s.filtered), 99.9) if len(s.filtered) else 0
    for r in s.reps:
        a1.axvspan(r.lift[0] / s.fs, r.lift[1] / s.fs, color="tab:orange", alpha=0.15)
        a1.axvspan(r.window[0] / s.fs, r.window[1] / s.fs, color="tab:red", alpha=0.3)
        a1.text(r.mid / s.fs, label_y, str(r.number), fontsize=7, ha="center")
    a1.set_title("Filtered EMG — orange: lift half, red: 500 ms analysis window")
    a1.set_ylabel("mV")
    rec = s.rec
    if len(rec.imu_stamp):
        a2.plot((rec.imu_stamp - rec.first_index) / s.fs, rec.imu_angle, lw=0.8)
        a2.plot((rec.turn_stamp - rec.first_index) / s.fs, rec.turn_angle, "rv", ms=4)
        a2.set_ylabel("arm angle (deg)")
        a2.set_title("Arm angle and turning points")
    raw = rec.emg_mv[~np.isnan(rec.emg_mv)]
    if len(raw) > 1024:
        f, p_raw = signal.welch(raw - raw.mean(), fs=s.fs, nperseg=1024)
        f, p_f = signal.welch(s.filtered, fs=s.fs, nperseg=1024)
        a3.semilogy(f, p_raw, lw=0.8, label="raw")
        a3.semilogy(f, p_f, lw=0.8, label="filtered")
        for h in (60, 120, 180):
            a3.axvline(h, color="gray", lw=0.5, ls=":")
        a3.set_xlim(0, s.fs / 2)
        a3.set_xlabel("Hz")
        a3.set_title("Spectrum (whole recording)")
        a3.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_summary(path, by_set, sets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(FEATURES), len(METHODS), figsize=(11, 12), sharex=True)
    for i, f in enumerate(FEATURES):
        for j, m in enumerate(METHODS):
            ax = axes[i, j]
            for name, rows in by_set.items():
                t = set_trend(rows, f"{f}_{m}")
                if not t:
                    continue
                kind = sets[name].meta.get("set_type")
                ax.plot(t["x"], t["norm"], "-", lw=1.2 if kind == "fatigue" else 0.8,
                        color="tab:red" if kind == "fatigue" else "tab:gray", alpha=0.8)
            ax.axhline(1, color="k", lw=0.5)
            ax.set_title(f"{FEATURE_NAMES[f]} — {m}")
    for ax in axes[-1]:
        ax.set_xlabel("rep")
    fig.suptitle("All sets (red: to failure, gray: control), normalized to reps 1-3")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="recordings (.bin) or folders of them")
    ap.add_argument("--out", help="output folder (default: analysis/<timestamp>)")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    files = find_recordings(args.paths)
    if not files:
        print("No .bin recordings found.", file=sys.stderr)
        return 1
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis",
                                   dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)

    loaded = [load_set(f) for f in files]
    for s in loaded:
        s.filtered = bandpass_notch(s.rec.emg_mv, s.fs) if s.rec.n > 100 else np.zeros(0)
    rests = rest_levels(loaded)
    sets = {s.name: s for s in loaded if s.meta.get("set_type") != "rest"}

    by_set, all_rows, sanity_rows = {}, [], []
    for name, s in sets.items():
        if not len(s.filtered):
            s.warnings.append("no EMG data")
            continue
        s.reps = segment_reps(s)
        rows = rep_rows(s)
        by_set[name] = rows
        all_rows += rows
        sanity_rows.append(sanity(s, rests.get(s.group)))
        print(f"{name}: {len(rows)} reps, fs {s.fs:.1f} Hz" + (f" — {'; '.join(s.warnings)}" if s.warnings else ""))
        if not args.no_plots:
            plot_overview(os.path.join(out, "plots", f"{name}_overview.png"), s)
            if rows:
                plot_set(os.path.join(out, "plots", f"{name}_features.png"), s, rows)

    results = evaluate(by_set, sets)
    notes = confounds(by_set, sets)
    write_markdown(os.path.join(out, "results.md"), results, sanity_rows, notes, sets)
    write_csv(os.path.join(out, "results.csv"), results)
    write_csv(os.path.join(out, "reps.csv"), all_rows)
    write_csv(os.path.join(out, "sanity.csv"), sanity_rows)
    if not args.no_plots and by_set:
        plot_summary(os.path.join(out, "plots", "summary.png"), by_set, sets)

    print()
    print(f"{'feature':<10}{'method':<7}{'verdict':<12}{'effect':>8}  trend size spec repeat")
    for r in results:
        print(f"{r['feature']:<10}{r['method']:<7}{r['verdict']:<12}{fmt(r['median_effect']):>8}  "
              f"{fmt(r['trend']):^5} {fmt(r['size']):^4} {fmt(r['specificity']):^4} {fmt(r['repeatability']):^6}")
    bad = [s for s in sanity_rows if not s["ok"]]
    if bad:
        print(f"\nSanity checks failed for {len(bad)} set(s); see results.md before trusting the table.")
    print(f"\nResults in {os.path.abspath(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
