#!/usr/bin/env python3
"""Analysis for the MyoWare fatigue tests. One subcommand per test.

  python analyze.py signal --rest A_rest --squeeze A_squeeze
  python analyze.py alias  --rest A_rest --rest-alt A_rest_10500 --squeeze A_squeeze
  python analyze.py hold   --light A_light --heavy A_heavy1 A_heavy2 A_heavy3
  python analyze.py curls  --light B_light --sets B_set1 B_set2 [--cadence 2.0]

Names refer to recordings the logger saved in data/. Plots are saved to results/.

Terms: loudness = RMS (root mean square) of the signal; middle pitch = median
frequency, the frequency that splits the signal's power in half.
"""
import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal, stats

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = Path(os.environ.get("EMG_RESULTS_DIR", ROOT / "results"))  # tests point this elsewhere
TARGET_FS = 2000           # readings/s used for analysis, after the anti-aliasing filter
BAND = (20, 450)           # where surface EMG lives
HUM = (60, 120, 180)       # wall power and its multiples
FULL_SCALE = 4095          # 12-bit ADC


# ---------- loading and cleaning ----------

@dataclass
class Recording:
    name: str
    raw: np.ndarray
    env: np.ndarray
    fs: float
    markers: list  # [reading index, label]
    meta: dict


def load(name):
    path = Path(name).with_suffix("")
    if path.parent == Path("."):
        path = DATA_DIR / path
    meta = json.loads(path.with_suffix(".json").read_text())
    csv = path.with_suffix(".csv")
    if not csv.exists():
        csv = path.parent / f"{path.name}.csv.gz"  # recordings are stored gzipped in the repo
    data = np.loadtxt(csv, delimiter=",", skiprows=1)
    return Recording(path.name, data[:, 0], data[:, 1], meta["rate_hz"], meta["markers"], meta)


def bandpass_notch(x, fs):
    """Keep 20-450 Hz, then cut out wall-power hum."""
    x = signal.sosfiltfilt(signal.butter(4, BAND, btype="bandpass", fs=fs, output="sos"), x)
    for f0 in HUM:
        b, a = signal.iirnotch(f0, Q=30, fs=fs)
        x = signal.filtfilt(b, a, x)
    return x


def anti_alias(raw, fs):
    """Center RAW, then anti-aliasing filter + keep every q-th reading (~2000/s)."""
    x = raw - raw.mean()
    q = int(round(fs / TARGET_FS))
    if q > 1:
        x = signal.decimate(x, q, ftype="fir", zero_phase=True)
    return x, fs / q


def clean(raw, fs):
    """Anti-alias + downsample, then keep 20-450 Hz and remove hum."""
    x, fs = anti_alias(raw, fs)
    return bandpass_notch(x, fs), fs


def naive_downsample(raw, fs):
    """Same as clean() but WITHOUT the anti-aliasing filter: what plain 2000/s sampling would see."""
    q = int(round(fs / TARGET_FS))
    return bandpass_notch((raw - raw.mean())[::q], fs / q), fs / q


def mdf(x, fs):
    """Middle pitch (median frequency): where the running total of power crosses half."""
    f, p = signal.welch(x, fs=fs, nperseg=min(256, len(x)))
    keep = (f >= BAND[0]) & (f <= BAND[1])
    total = np.cumsum(p[keep])
    return float(np.interp(total[-1] / 2, total, f[keep]))


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def slices(x, fs, seconds):
    """Back-to-back slices with no overlap, so the trend statistics stay honest."""
    n = int(seconds * fs)
    return [(i / fs, x[i:i + n]) for i in range(0, len(x) - n + 1, n)]


def marked_span(rec, fs, n):
    """Index range between the first and last marker, for a signal sampled at fs."""
    if len(rec.markers) >= 2:
        return int(rec.markers[0][0] / rec.fs * fs), int(rec.markers[-1][0] / rec.fs * fs)
    return int(fs), n - int(fs)  # no markers: drop the first and last second


def band_power(x, fs, lo, hi):
    f, p = signal.welch(x, fs=fs, nperseg=min(int(fs), len(x)))
    keep = (f >= lo) & (f <= hi)
    return float(p[keep].sum() * (f[1] - f[0]))


# ---------- reporting ----------

@dataclass
class Check:
    label: str
    ok: bool | None  # None = information only
    detail: str


def report(title, checks, plot_path):
    print(f"\n{title}")
    for c in checks:
        tag = "INFO" if c.ok is None else ("PASS" if c.ok else "FAIL")
        print(f"  {tag}  {c.label:<44} {c.detail}")
    graded = [c for c in checks if c.ok is not None]
    passed = sum(c.ok for c in graded)
    print(f"Result: {'PASS' if passed == len(graded) else 'FAIL'} ({passed}/{len(graded)})   plot: {plot_path}\n")


def save(fig, name):
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def no_loss_check(rec):
    m = rec.meta
    return Check(f"No readings lost ({rec.name})", m["ok"],
                 f"{m['board_overflows']} overflows, {m['resyncs']} resyncs, {rec.fs:,.0f}/s")


def control_check(what, name, change, fit, extra):
    """A light (control) trial passes unless it shows a believable trend bigger than 5%."""
    return Check(f"{what}: no real trend ({name})", abs(change) < 5 or fit.pvalue >= 0.05,
                 f"{change:+.1f}%, chance random {fit.pvalue:.1e}, {extra}")


def squeeze_slices(x, fs):
    """Quarter-second slices louder than half the loudest: the squeezes."""
    parts = slices(x, fs, 0.25)
    levels = np.array([rms(s) for _, s in parts])
    return [p for p, lv in zip(parts, levels) if lv > 0.5 * np.percentile(levels, 95)]


# ---------- Test 1: signal check ----------

def cmd_signal(args):
    rest, sq = load(args.rest), load(args.squeeze)
    rest_x, rest_fs = clean(rest.raw, rest.fs)
    sq_x, sq_fs = clean(sq.raw, sq.fs)
    squeezes = squeeze_slices(sq_x, sq_fs)
    sq_level = np.median([rms(s) for _, s in squeezes])
    rest_level = np.median([rms(s) for _, s in slices(rest_x, rest_fs, 0.25)])
    flat = np.mean((sq.raw <= 5) | (sq.raw >= FULL_SCALE - 5))
    rest_bp, bp_fs = anti_alias(rest.raw, rest.fs)  # hum measured before the notch filter removes it
    hum_share = band_power(rest_bp, bp_fs, 55, 65) / band_power(np.concatenate([s for _, s in squeezes]), sq_fs, *BAND)
    pitch = np.median([mdf(s, sq_fs) for _, s in squeezes])

    checks = [
        no_loss_check(rest),
        no_loss_check(sq),
        Check("Squeeze loudness >= 10x rest", sq_level / rest_level >= 10, f"{sq_level / rest_level:.1f}x"),
        Check("No flat tops (< 0.1% of readings at the rails)", flat < 0.001, f"{flat:.3%}"),
        Check("60 Hz hum < 1% of squeeze power", hum_share < 0.01, f"{hum_share:.2%}"),
        Check("Squeeze middle pitch 50-120 Hz", 50 <= pitch <= 120, f"{pitch:.0f} Hz"),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    ax1.plot(np.arange(len(sq.raw)) / sq.fs, sq.raw, lw=0.3)
    ax1.set(title=f"{sq.name}: RAW readings", xlabel="seconds", ylabel="ADC reading", ylim=(0, FULL_SCALE))
    for label, x, fs in (("rest (before hum filter)", rest_bp, bp_fs),
                         ("squeezes", np.concatenate([s for _, s in squeezes]), sq_fs)):
        f, p = signal.welch(x, fs=fs, nperseg=min(int(fs), len(x)))
        ax2.semilogy(f, p, label=label)
    ax2.set(title="Frequency breakdown", xlabel="Hz", ylabel="power", xlim=(0, 500))
    ax2.legend()
    report("Test 1 - Signal check", checks, save(fig, f"signal_{sq.name}"))


# ---------- Test 2: aliasing check ----------

def find_spikes(raw, fs):
    """Frequencies (20-450 Hz) that stick out 10 dB above their neighbours."""
    f, p = signal.welch(raw - raw.mean(), fs=fs, nperseg=int(fs))
    keep = (f >= BAND[0]) & (f <= BAND[1])
    f, db = f[keep], 10 * np.log10(p[keep])
    peaks, _ = signal.find_peaks(db - signal.medfilt(db, 31), height=10)
    return f[peaks]


def cmd_alias(args):
    a, b, sq = load(args.rest), load(args.rest_alt), load(args.squeeze)
    spikes_a, spikes_b = find_spikes(a.raw, a.fs), find_spikes(b.raw, b.fs)
    moved = sorted({float(round(f)) for f in spikes_a if not np.any(np.abs(spikes_b - f) <= 3)} |
                   {float(round(f)) for f in spikes_b if not np.any(np.abs(spikes_a - f) <= 3)})

    centered = sq.raw - sq.raw.mean()
    above_1k = band_power(centered, sq.fs, 1000, sq.fs / 2) / band_power(centered, sq.fs, *BAND)
    proper, fs = clean(sq.raw, sq.fs)
    naive, _ = naive_downsample(sq.raw, sq.fs)
    windows = squeeze_slices(proper, fs)
    n = len(windows[0][1])
    mdf_proper = np.median([mdf(s, fs) for _, s in windows])
    mdf_naive = np.median([mdf(naive[int(t * fs):int(t * fs) + n], fs) for t, _ in windows])
    diff = abs(mdf_naive - mdf_proper) / mdf_proper

    checks = [
        Check(f"No spikes move between {a.fs:,.0f}/s and {b.fs:,.0f}/s", not moved,
              "none" if not moved else "moved: " + ", ".join(f"{f:.0f} Hz" for f in moved)),
        Check("Squeeze power above 1000 Hz (vs 20-450 Hz)", None, f"{above_1k:.2%}"),
        Check("Middle pitch: plain 2000/s vs filtered", None,
              f"{mdf_naive:.1f} vs {mdf_proper:.1f} Hz ({diff:.1%}) -> "
              + ("final device can sample at plain 2000/s" if diff < 0.02 else "final device needs fast sampling + filter")),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    for rec in (a, b):
        f, p = signal.welch(rec.raw - rec.raw.mean(), fs=rec.fs, nperseg=int(rec.fs))
        ax1.semilogy(f, p, label=f"rest at {rec.fs:,.0f}/s", lw=0.8)
    for f in moved:
        ax1.axvline(f, color="red", ls=":")
    ax1.set(title="Resting signal: spikes that move between rates are folded-down interference",
            xlabel="Hz", ylabel="power", xlim=BAND)
    ax1.legend()
    f, p = signal.welch(centered, fs=sq.fs, nperseg=int(sq.fs))
    ax2.semilogy(f, p, lw=0.8)
    ax2.axvline(1000, color="red", ls=":")
    ax2.set(title=f"{sq.name}: full frequency breakdown", xlabel="Hz", ylabel="power")
    report("Test 2 - Aliasing check", checks, save(fig, f"alias_{sq.name}"))


# ---------- Test 3: still hold ----------

def hold_series(rec):
    """Middle pitch and loudness for each half-second slice between the markers."""
    x, fs = clean(rec.raw, rec.fs)
    a, b = marked_span(rec, fs, len(x))
    parts = slices(x[a:b], fs, 0.5)
    t = np.array([t for t, _ in parts])
    pitch, level = np.array([mdf(s, fs) for _, s in parts]), np.array([rms(s) for _, s in parts])
    # Timed holds run a fixed length; if the arm gave out early, loudness collapses. Stop there.
    collapsed = np.flatnonzero(level < 0.4 * level[:10].mean())
    if len(collapsed) and collapsed[0] >= 10:
        t, pitch, level = t[:collapsed[0]], pitch[:collapsed[0]], level[:collapsed[0]]
    return t, pitch, level


def trend(x, y):
    """Straight-line fit through y. Returns y as % of the line's starting value, the fit, the line's
    total change from start to end (%), and its starting value. Using the whole line rather than
    the first few points keeps one jumpy slice or rep from deciding the result."""
    fit = stats.linregress(x, y)
    start = fit.intercept + fit.slope * x[0]
    return 100 * y / start, fit, 100 * fit.slope * (x[-1] - x[0]) / start, start


def max_squeeze_level(name):
    """Loudness of the strongest squeeze moments, used as '100% effort'."""
    rec = load(name)
    x, fs = clean(rec.raw, rec.fs)
    return np.percentile([rms(s) for _, s in squeeze_slices(x, fs)], 90)


def cmd_hold(args):
    checks, starts = [], []
    mvc = max_squeeze_level(args.squeeze) if args.squeeze else None
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, heavy in [(n, True) for n in args.heavy] + [(n, False) for n in args.light or []]:
        rec = load(name)
        t, pitch, level = hold_series(rec)
        pitch_pct, fit, change, start = trend(t, pitch)
        level_pct, level_fit, level_change, _ = trend(t, level)
        style = "-" if heavy else "--"
        ax1.plot(t, pitch_pct, style, label=rec.name)
        ax2.plot(t, level_pct, style, label=rec.name)
        checks.append(no_loss_check(rec))
        if mvc:
            checks.append(Check(f"Effort vs max squeeze ({rec.name})", None,
                                f"{level[:10].mean() / mvc:.0%} at start (heavy target 50-60%)"))
        if heavy:
            starts.append((rec.name, start))
            checks.append(Check(f"Middle pitch drops >= 10% ({rec.name})", change <= -10 and fit.pvalue < 0.05,
                                f"{change:+.1f}%, chance random {fit.pvalue:.1e}, held {t[-1]:.0f} s"))
            checks.append(Check(f"Loudness rises ({rec.name})", level_change > 0 and level_fit.pvalue < 0.05,
                                f"{level_change:+.1f}%, chance random {level_fit.pvalue:.1e}"))
        else:
            checks.append(control_check("Light hold", rec.name, change, fit, f"held {t[-1]:.0f} s"))
    if len(starts) >= 2:
        (n1, b1), (n2, b2) = starts[0], starts[1]
        checks.append(Check(f"Recovery: {n2} starts within 5% of {n1}", abs(b2 / b1 - 1) < 0.05,
                            f"{b2:.1f} vs {b1:.1f} Hz ({b2 / b1 - 1:+.1%})"))

    ax1.set(title="Middle pitch (% of trend-line start)", ylabel="%")
    ax2.set(title="Loudness (% of trend-line start)", xlabel="seconds since start marker", ylabel="%")
    ax1.legend()
    report("Test 3 - Still hold", checks, save(fig, f"hold_{Path(args.heavy[0]).stem}"))


# ---------- Test 4: curls ----------

def find_reps(rec, cadence=None):
    """Start times (s) of the 400 ms window analysed in each rep.

    Default: each time the smoothed ENV line rises through 30% of the way from the resting
    level (first 2 s, arm relaxed) to the peak level. With cadence (fallback if ENV doesn't
    dip between reps): the loudest 400 ms of each `cadence`-second period after the first marker.
    """
    a = rec.markers[0][0] / rec.fs if rec.markers else 0
    b = rec.markers[-1][0] / rec.fs if len(rec.markers) >= 2 else len(rec.env) / rec.fs
    if cadence:
        loud = ndimage.uniform_filter1d(rec.env, int(0.4 * rec.fs))
        starts = []
        for p0 in np.arange(a, b - cadence / 2, cadence):
            i0, i1 = int(p0 * rec.fs), min(len(loud), int((p0 + cadence) * rec.fs))
            starts.append((i0 + np.argmax(loud[i0:i1])) / rec.fs - 0.2)
        return np.array(starts)
    env = ndimage.uniform_filter1d(rec.env, max(1, int(0.05 * rec.fs)))
    rest, peak = np.median(env[:int(2 * rec.fs)]), np.percentile(env, 95)
    above = env > rest + 0.3 * (peak - rest)
    starts = []
    for i in np.flatnonzero(above[1:] & ~above[:-1]) + 1:
        if not starts or i - starts[-1] >= 1.2 * rec.fs:  # ignore bounces around the line
            starts.append(i)
    starts = np.array(starts) / rec.fs
    return starts[(starts >= a) & (starts <= b)]


def rep_series(rec, cadence):
    """Middle pitch and loudness of a 400 ms window per rep (mostly the lifting half)."""
    x, fs = clean(rec.raw, rec.fs)
    reps = find_reps(rec, cadence)
    n = int(0.4 * fs)
    parts = [x[int(s * fs):int(s * fs) + n] for s in reps]
    parts = [p for p in parts if len(p) == n]
    return reps, np.array([mdf(p, fs) for p in parts]), np.array([rms(p) for p in parts])


def cmd_curls(args):
    checks, counts = [], []
    recs = [(n, True) for n in args.sets] + [(n, False) for n in args.light or []]
    fig, axes = plt.subplots(len(recs) + 1, 1, figsize=(10, 3 + 2.2 * len(recs)))
    for ax_env, (name, real) in zip(axes[1:], recs):
        rec = load(name)
        reps, pitch, level = rep_series(rec, args.cadence)
        counts.append(f"{rec.name}={len(pitch)}")
        checks.append(no_loss_check(rec))
        if len(pitch) < 6:
            checks.append(Check(f"Enough reps found ({rec.name})", False, f"{len(pitch)} reps; try --cadence 2.0"))
            continue
        rep_no = np.arange(1, len(pitch) + 1)
        pct, fit, change, _ = trend(rep_no, pitch)
        axes[0].plot(rep_no, pct, "o-" if real else "o--", label=rec.name)
        if real:
            checks.append(Check(f"Middle pitch drops >= 10% ({rec.name})", change <= -10 and fit.pvalue < 0.05,
                                f"{change:+.1f}% over {len(pitch)} reps, chance random {fit.pvalue:.1e}"))
            checks.append(Check(f"Loudness change ({rec.name})", None, f"{trend(rep_no, level)[2]:+.1f}%"))
        else:
            checks.append(control_check("Light curls", rec.name, change, fit, f"{len(pitch)} reps"))
        ax_env.plot(np.arange(len(rec.env)) / rec.fs, rec.env, lw=0.5)
        for s in reps:
            ax_env.axvline(s, color="red", lw=1)
        ax_env.set(title=f"{rec.name}: ENV with detected rep starts (red)", ylabel="ENV")
    checks.append(Check("Rep counts -> compare with the video", None, ", ".join(counts)))

    axes[0].set(title="Middle pitch per rep (% of trend-line start)", xlabel="rep", ylabel="%")
    axes[0].legend()
    axes[-1].set_xlabel("seconds")
    report("Test 4 - Curls", checks, save(fig, f"curls_{Path(args.sets[0]).stem}"))


# ---------- Squeeze comparison ----------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, always in this order
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def find_squeezes(x, fs):
    """(start, end) indices of each squeeze: runs of 0.1 s slices louder than 5x the resting level."""
    n = int(0.1 * fs)
    levels = np.array([rms(x[i:i + n]) for i in range(0, len(x) - n + 1, n)])
    loud = levels > 5 * np.median(levels)  # the recording is mostly rest, so the median is the rest level
    loud[:10] = loud[-10:] = False  # first/last second hold filter start-up blips, not squeezes
    edges = np.flatnonzero(np.diff(np.r_[0, loud.astype(int), 0]))
    runs = []
    for a, b in zip(edges[::2], edges[1::2]):
        if runs and a - runs[-1][1] <= 5:  # merge dips shorter than 0.5 s
            runs[-1][1] = b
        else:
            runs.append([a, b])
    return [(a * n, b * n) for a, b in runs if b - a >= 5]  # at least 0.5 s long


def rest_gaps(squeezes, n, fs):
    """The quiet stretches around the squeezes, keeping 1 s away from each squeeze and the recording ends."""
    pad = int(fs)
    bounds = [0] + [i for sq in squeezes for i in sq] + [n]
    gaps = [(a + pad, b - pad) for a, b in zip(bounds[::2], bounds[1::2])]
    return [(a, b) for a, b in gaps if b - a >= 2 * fs]  # at least 2 s long


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)


def cmd_squeezes(args):
    rec = load(args.squeeze)
    x, fs = anti_alias(rec.raw, rec.fs)
    # Display without the hum notches so the curves have no artificial dips (hum was checked in Test 1).
    x = signal.sosfiltfilt(signal.butter(4, BAND, btype="bandpass", fs=fs, output="sos"), x)
    squeezes = find_squeezes(x, fs)
    kind = "Rest" if args.rest else "Squeeze"
    found = (rest_gaps(squeezes, len(x), fs) if args.rest else squeezes)[:len(SERIES)]
    ref = rms(x[squeezes[0][0]:squeezes[0][1]])  # squeeze 1's loudness, the yardstick in the table

    rows = []
    for a, b in found:
        f, p = signal.welch(x[a:b], fs=fs, nperseg=512)
        keep = (f >= BAND[0]) & (f <= BAND[1])
        f, p = f[keep], p[keep]
        total = np.cumsum(p)
        rows.append(dict(t0=a / fs, t1=b / fs, f=f, p=p, level=rms(x[a:b]),
                         mdf=float(np.interp(total[-1] / 2, total, f)),
                         mean=float((f * p).sum() / p.sum()), peak=float(f[np.argmax(p)])))

    print(f"\n{kind} periods in {rec.name} (loudness also shown as % of squeeze 1)")
    print(f"  {'#':<3}{'time':>12}{'loudness':>19}{'middle pitch':>15}{'mean freq':>12}{'peak':>9}")
    for i, r in enumerate(rows, 1):
        print(f"  {i:<3}{r['t0']:>5.1f}-{r['t1']:<5.1f}s{r['level']:>8.1f} ({r['level'] / ref:>5.1%})"
              f"{r['mdf']:>12.0f} Hz{r['mean']:>9.0f} Hz{r['peak']:>6.0f} Hz")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [1, 2, 2]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_axes(ax)

    t = np.arange(len(x)) / fs
    axes[0].plot(t, x, color=INK_2, lw=0.4)
    for i, r in enumerate(rows):
        axes[0].axvspan(r["t0"], r["t1"], color=SERIES[i], alpha=0.18, lw=0)
        axes[0].text((r["t0"] + r["t1"]) / 2, 0.92, f"{kind} {i + 1}", transform=axes[0].get_xaxis_transform(),
                     ha="center", va="top", color=INK, fontsize=9)
    axes[0].set_title(f"{rec.name}: filtered signal, {kind.lower()} periods shaded", color=INK, loc="left",
                      fontsize=11)
    axes[0].set_xlabel("seconds", color=INK_2)

    for i, r in enumerate(rows):
        label = f"{kind} {i + 1}"
        axes[1].plot(r["f"], r["p"], color=SERIES[i], lw=1.6, label=label)
        if not args.rest:  # rest curves share their peaks (hum), so labels there would pile up
            k = int(np.argmax(r["p"]))
            axes[1].annotate(label, (r["f"][k], r["p"][k]), xytext=(8, 0), textcoords="offset points",
                             color=INK, fontsize=9, va="center")
        share = 100 * r["p"] / r["p"].sum()
        axes[2].plot(r["f"], share, color=SERIES[i], lw=1.6, label=f"{label}: middle pitch {r['mdf']:.0f} Hz")
        axes[2].axvline(r["mdf"], color=SERIES[i], lw=1.2, ls=(0, (4, 3)))
    axes[1].set_title("Power at each frequency (height = how strongly the muscle fired)",
                      color=INK, loc="left", fontsize=11)
    axes[1].set_ylabel("power (ADC counts\u00b2/Hz)", color=INK_2)
    axes[2].set_title(f"Shape only: each {kind.lower()} period scaled to its own total (dashed = middle pitch)",
                      color=INK, loc="left", fontsize=11)
    axes[2].set_ylabel("share of power (%)", color=INK_2)
    axes[2].set_xlabel("frequency (Hz)", color=INK_2)
    for ax in axes[1:]:
        # squeeze power sits below 300 Hz; resting noise spreads across the whole band
        ax.set_xlim(BAND if args.rest else (BAND[0], 300))
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    print(f"Plot: {save(fig, f'{kind.lower()}_{rec.name}')}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("signal", help="Test 1: is the signal clean?")
    p.add_argument("--rest", required=True)
    p.add_argument("--squeeze", required=True)
    p.set_defaults(func=cmd_signal)
    p = sub.add_parser("alias", help="Test 2: is interference folding into the muscle range?")
    p.add_argument("--rest", required=True, help="relaxed recording at the normal rate")
    p.add_argument("--rest-alt", required=True, help="relaxed recording at a different rate (e.g. 10500)")
    p.add_argument("--squeeze", required=True)
    p.set_defaults(func=cmd_alias)
    p = sub.add_parser("hold", help="Test 3: still hold")
    p.add_argument("--heavy", nargs="+", required=True, help="heavy holds, in the order done")
    p.add_argument("--light", nargs="+", help="light hold(s)")
    p.add_argument("--squeeze", help="Test 1 squeeze recording; reports each hold's effort as % of max")
    p.set_defaults(func=cmd_hold)
    p = sub.add_parser("curls", help="Test 4: curls")
    p.add_argument("--sets", nargs="+", required=True, help="real sets to failure")
    p.add_argument("--light", nargs="+", help="light-curl set(s)")
    p.add_argument("--cadence", type=float, help="seconds per rep; use if ENV-based rep finding fails")
    p.set_defaults(func=cmd_curls)
    p = sub.add_parser("squeezes", help="compare the frequency content of each squeeze in a Test 1 recording")
    p.add_argument("--squeeze", required=True)
    p.add_argument("--rest", action="store_true", help="compare the rest periods between the squeezes instead")
    p.set_defaults(func=cmd_squeezes)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
