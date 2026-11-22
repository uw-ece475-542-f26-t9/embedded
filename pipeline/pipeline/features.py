"""Step 6: one row of features per lift, every value relative to the set's own first lifts.

A lift is a continuous rise of the forearm angle (at least min_lift_deg; short stalls joined). Its
row is stamped with when the board would know it: the top is final once lowering starts (or
top_wait_s after the rise ends), and every EMG pass in it has been cleaned. Nothing is numbered for
the model; lift order is only used to pick the baseline (the set's first baseline_lifts lifts).

Core features (FEATURES.md):
  #1  mdf_ratio      middle pitch of the lift's pooled spectrum / the same bins' baseline. Each
                     pass is analysed over a fixed-length window (pass_window = "fixed"), so the
                     short-window bias stays the same however fast the lift is
                     (pool = "equal": each bin's spectrum divided by its own baseline power first,
                     so rising loudness can't change which bins dominate the mix)
  #8  loud_ratio     mean over the lift's bins of sqrt(pass power / that bin's baseline power)
  #15 speed_loss     1 - peak turning speed / the baseline lifts' median
  #17 top_change     top angle - the baseline lifts' median top (up: drift or swing; down: partial)
  #22 tremor_ratio   gyro wobble in tremor_band_hz during the lift / the baseline lifts' median
  #26 mdf_session    as #1 but against set 1's baseline (sets 2-3 start below 1.0 by the carried-over
                     tiredness); from a set's first lift
  #13a (middle pitch slope) comes from the Kalman filter, step 7.

Candidates (step 6b), same pooled spectra and baselines:
  #2  mnf_ratio       power-weighted mean frequency / baseline
  #3  dimitrov_ratio  spectral moment ratio M(-1)/M(5) / baseline (rises with fatigue)
  #4  lowhigh_ratio   power 20-50 Hz / power 100-450 Hz, / baseline (rises)
  #6  mdf_lower_ratio #1 on the lowering just before this lift, against the baseline lifts' lowerings
  #9  loud_max        lift loudness / the person's best baseline max pull (not set-relative)
  #10 fatigue_index   (log #8 - log #1)/sqrt 2: up for fatigue (pitch down, loudness up); effort
                      (both up) barely moves it. effort_index (log #8 + log #1)/sqrt 2 is kept beside it
  #11 emg_cost        #8 / (1 - #15): loudness per unit of lift speed
  #16 duration_ratio  rise time / the baseline lifts' median
  #18 bottom_change   lowest angle before the lift - the baseline lifts' median (up: not lowered fully)
  #19 lateness        time since the previous lift started - the baseline lifts' median interval
  #20 sticking_dip    deepest speed dip inside the lift, as a share of its peak speed (0 = smooth)
  #21 wobble_change   rotation around the forearm that isn't the curl, deg (RepSense labels.lift_motion
                      "wobble"), - the baseline lifts' median. RepSense's hinge residual was never
                      implemented there and isn't here either
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

from .angle import AngleTrack, forearm_angle
from .bins import Pass, find_passes, pass_spectrum
from .clean import SpectrumCleaner, band_power, clean_emg, middle_pitch, rest_check
from .load import load_session
from .session import Session


@dataclass
class Lift:
    i0: int            # IMU index where the rise starts
    i1: int            # one past where it ends
    t_start: float
    t_end: float
    t_top_known: float  # when the top angle is final
    top_deg: float
    peak_dps: float
    tremor_dps: float
    bottom_deg: float = np.nan   # lowest angle between the previous lift and this one
    dip: float = 0.0             # deepest mid-lift speed dip / peak speed (sticking point)
    wobble_deg: float = np.nan   # rotation around the forearm that isn't the curl (RepSense)


def lift_motion_wobble(t: np.ndarray, gyro: np.ndarray, axis: np.ndarray, start: int, end: int) -> float:
    """RepSense labels.lift_motion, wobble part: roll about the forearm axis is removed; the rest
    of the rotation's main direction is the curl; what's left over, added up, is the wobble (deg)."""
    span = slice(start, end + 1)
    dt = np.diff(t[span], prepend=t[start])
    roll = gyro[span] @ axis
    around = gyro[span] - np.outer(roll, axis)
    curl_axis = np.linalg.svd(around, full_matrices=False)[2][0]
    yaw = around @ np.cross(axis, curl_axis)
    return float(np.ptp(np.cumsum(yaw * dt)))


def speed_dip(v: np.ndarray) -> float:
    """Deepest dip between two speed peaks inside a lift, as a share of the lift's peak speed."""
    if len(v) < 5 or np.nanmax(v) <= 0:
        return 0.0
    best = 0.0
    for m in range(1, len(v) - 1):
        if v[m] < v[m - 1] and v[m] <= v[m + 1]:
            depth = min(np.max(v[:m]), np.max(v[m:])) - v[m]
            best = max(best, depth)
    return float(best / np.max(v))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def find_lifts(track: AngleTrack, gyro: np.ndarray, cfg: dict) -> list[Lift]:
    f = cfg["features"]
    t, ang, d = track.t, track.angle_deg, track.direction
    dt = float(np.median(np.diff(t)))
    gap = int(round(f["lift_merge_gap_s"] / dt))
    runs = []
    for i, j in _runs(d == 1):
        if runs and i - runs[-1][1] <= gap and not np.any(d[runs[-1][1]:i] == -1):
            runs[-1] = (runs[-1][0], j)
        else:
            runs.append((i, j))

    # Tremor band, one-way, on each gyro axis.
    lo, hi = f["tremor_band_hz"]
    fs = 1 / dt
    sos = signal.butter(2, [lo, min(hi, 0.45 * fs)], btype="bandpass", fs=fs, output="sos")
    shake = np.linalg.norm(signal.sosfilt(sos, gyro, axis=0), axis=1)

    lifts = []
    wait = int(round(f["top_wait_s"] / dt))
    back = int(round(1.5 / dt))
    for i, j in runs:
        a = ang[i:j]
        if np.all(np.isnan(a)) or np.nanmax(a) - np.nanmin(ang[max(0, i - 5):i + 1]) < f["min_lift_deg"]:
            continue
        # The top is final when lowering starts (or after top_wait_s); hold time counts toward the top.
        k = j
        while k < min(len(t), j + wait) and d[k] != -1:
            k += 1
        known = min(k, len(t) - 1)
        since = max(lifts[-1].i1 if lifts else 0, i - back)   # the bottom: since the previous lift, at most 1.5 s
        bottom = ang[since:i + 1]
        wobble = (lift_motion_wobble(t, gyro, track.forearm_axis, i, j - 1)
                  if track.forearm_axis is not None and j - i >= 3 else np.nan)
        lifts.append(Lift(i0=int(i), i1=int(j), t_start=float(t[i]), t_end=float(t[j - 1]),
                          t_top_known=float(t[known]), top_deg=float(np.nanmax(ang[i:known + 1])),
                          peak_dps=float(np.max(track.rate_dps[i:j])),
                          tremor_dps=float(np.sqrt(np.mean(shake[i:j] ** 2))),
                          bottom_deg=float(np.nanmin(bottom)) if np.any(~np.isnan(bottom)) else np.nan,
                          dip=speed_dip(track.rate_dps[i:j]), wobble_deg=wobble))
    return lifts


def pass_window(p: Pass, cl, cfg: dict) -> tuple[int, int, np.ndarray]:
    """Samples and taper used for a pass's spectrum. "fixed": the same length whatever the speed,
    centred on the pass, so the short-window bias can't change as tired lifts slow down."""
    if cfg["features"]["pass_window"] == "pass":
        return p.start, p.stop, p.weights
    n = int(cfg["features"]["fixed_window"])
    c = int(round(p.t_mid_s * cl.fs))
    a, b = c - n // 2, c - n // 2 + n
    if a < 0 or b > len(cl.emg):
        return p.start, p.stop, np.zeros(p.stop - p.start)
    return a, b, np.hanning(n) * ~cl.skip[a:b]


def _bin_spectra(passes: list[Pass], cl, cleaner: SpectrumCleaner | None, nfft: int, cfg: dict):
    out = {}
    for p in passes:
        a, b, w = pass_window(p, cl, cfg)
        if not w.any():
            continue
        f, s = pass_spectrum(cl.emg[a:b], w, cl.fs, nfft)
        if cleaner is not None:
            s = cleaner(f, s)
        out.setdefault(p.bin_deg, []).append(s)
    return f if passes else None, out


@dataclass
class Reference:
    """A set's baseline per angle bin: its mean cleaned spectrum and power. Set 1's reference is
    reused by sets 2-3 for the session-relative feature (#26)."""
    freqs: np.ndarray | None
    spectra: dict
    power: dict


def mean_frequency(f, p, band) -> float:
    k = (f >= band[0]) & (f <= band[1])
    return float(np.sum(f[k] * p[k]) / np.sum(p[k])) if np.sum(p[k]) > 0 else np.nan


def dimitrov(f, p, band) -> float:
    """Spectral moment ratio M(-1)/M(5) (Dimitrov et al. 2006)."""
    k = (f >= band[0]) & (f <= band[1])
    m5 = np.sum(f[k] ** 5 * p[k])
    return float(np.sum(f[k] ** -1.0 * p[k]) / m5) if m5 > 0 else np.nan


def low_high(f, p, band) -> float:
    lo, hi = (f >= band[0]) & (f <= 50), (f >= 100) & (f <= band[1])
    return float(np.sum(p[lo]) / np.sum(p[hi])) if np.sum(p[hi]) > 0 else np.nan


SPECTRAL = {"mdf_ratio": middle_pitch, "mnf_ratio": mean_frequency, "dimitrov_ratio": dimitrov,
            "lowhigh_ratio": low_high}


def pooled_pair(spec: dict, ref: Reference, cfg: dict):
    """A lift's pooled spectrum and the reference's, over the bins both have (None if too few)."""
    bins = [b for b in spec if b in ref.spectra and ref.power[b] > 0]
    if len(bins) < cfg["features"]["min_bins"]:
        return None, None
    scale = {b: 1 / ref.power[b] if cfg["features"]["pool"] == "equal" else 1.0 for b in bins}
    return sum(spec[b] * scale[b] for b in bins), sum(ref.spectra[b] * scale[b] for b in bins)


def pooled_ratio(spec: dict, ref: Reference, cfg: dict, band, stat=middle_pitch) -> float:
    """A spectral statistic of a lift's bins / the reference's same bins, pooled as cfg says (#1, #26)."""
    pooled, base = pooled_pair(spec, ref, cfg)
    if pooled is None:
        return np.nan
    return stat(ref.freqs, pooled, band) / stat(ref.freqs, base, band)


def _reference(passes, cl, cleaner, cfg) -> Reference:
    fc, band = cfg["features"], tuple(cfg["clean"]["band_hz"])
    freqs, spec = _bin_spectra(passes, cl, cleaner, cfg["clean"]["nfft"], cfg)
    spectra = {b: np.mean(v, axis=0) for b, v in spec.items() if len(v) >= fc["min_baseline_passes"]}
    return Reference(freqs, spectra, {b: band_power(freqs, v, band) for b, v in spectra.items()})


def lift_features(s: Session, cfg: dict, cleaner: SpectrumCleaner | None = None,
                  session_ref: Reference | None = None, pull_rms_mv: float | None = None) -> pd.DataFrame:
    """Feature rows for one set recording. session_ref: set 1's reference, for #26 (for set 1
    itself, leave it None and its own baseline is used). pull_rms_mv: the person's best baseline
    max-pull loudness, for #9."""
    fc, band = cfg["features"], tuple(cfg["clean"]["band_hz"])
    cl = clean_emg(s, cfg)
    track = forearm_angle(s, cfg)
    lifts = find_lifts(track, s.gyro, cfg)
    every = find_passes(cl, track, cfg)
    passes = [p for p in every if p.direction == 1]
    lowering = [p for p in every if p.direction == -1]
    nfft = cfg["clean"]["nfft"]

    def lift_passes(lf: Lift):
        return [p for p in passes if lf.t_start - 0.1 <= p.t_mid_s <= lf.t_end + 0.1]

    def lowering_after(k: int):
        """The lowering between lift k and lift k+1 (or up to 3 s after lift k)."""
        end = lifts[k + 1].t_start if k + 1 < len(lifts) else lifts[k].t_end + 3.0
        return [p for p in lowering if lifts[k].t_end <= p.t_mid_s <= end]

    nb = fc["baseline_lifts"]
    base = lifts[:nb]
    rows = []
    own_ref = _reference([p for lf in base for p in lift_passes(lf)], cl, cleaner, cfg)
    freqs, baseline, base_power = own_ref.freqs, own_ref.spectra, own_ref.power
    sess_ref = session_ref if session_ref is not None else own_ref
    low_ref = _reference([p for k in range(min(nb, len(lifts))) for p in lowering_after(k)], cl, cleaner, cfg)
    med = lambda xs: float(np.nanmedian(xs)) if len(xs) and np.any(~np.isnan(xs)) else np.nan  # noqa: E731
    b_peak = med([lf.peak_dps for lf in base])
    b_top = med([lf.top_deg for lf in base])
    b_tremor = med([lf.tremor_dps for lf in base])
    b_dur = med([lf.t_end - lf.t_start for lf in base])
    b_bottom = med([lf.bottom_deg for lf in base])
    b_wobble = med([lf.wobble_deg for lf in base])
    b_gap = med([base[i].t_start - base[i - 1].t_start for i in range(1, len(base))])
    have_base = len(base) == nb
    half = fc["fixed_window"] / 2 / cl.fs if fc["pass_window"] == "fixed" else 0.0

    for k, lf in enumerate(lifts):
        mine = lift_passes(lf)
        before = lowering_after(k - 1) if k >= 1 else []
        known = max([lf.t_top_known + cl.latency_s] +
                    [max(p.t_known_s, p.t_mid_s + half + cl.latency_s) for p in mine + before])
        row = {"t_start": lf.t_start, "t_end": lf.t_end, "t_known": known, "is_baseline": k < nb,
               "top_deg": lf.top_deg, "peak_dps": lf.peak_dps, "tremor_dps": lf.tremor_dps,
               "n_passes": len(mine), "weight": float(sum(p.weight_sum for p in mine))}
        ok = have_base and k >= nb
        row["speed_loss"] = 1 - lf.peak_dps / b_peak if ok and b_peak > 0 else np.nan
        row["top_change"] = lf.top_deg - b_top if ok else np.nan
        row["tremor_ratio"] = lf.tremor_dps / b_tremor if ok and b_tremor > 0 else np.nan
        row["duration_ratio"] = (lf.t_end - lf.t_start) / b_dur if ok and b_dur > 0 else np.nan        # 16
        row["bottom_change"] = lf.bottom_deg - b_bottom if ok else np.nan                                # 18
        row["lateness"] = (lf.t_start - lifts[k - 1].t_start - b_gap) if ok and k >= 1 else np.nan      # 19
        row["sticking_dip"] = lf.dip if ok else np.nan                                                    # 20
        row["wobble_change"] = lf.wobble_deg - b_wobble if ok else np.nan                                # 21

        f, spec = _bin_spectra(mine, cl, cleaner, nfft, cfg)
        lift_spec = {b: np.mean(v, axis=0) for b, v in spec.items()}
        bins = [b for b in lift_spec if b in baseline and base_power[b] > 0]
        row["n_bins"] = len(bins)
        row["emg_ok"] = bool(ok and len(bins) >= fc["min_bins"])
        for col in ["mdf_hz", "rms_mv", "loud_ratio", "mdf_session", "mdf_lower_ratio", "loud_max",
                    "fatigue_index", "effort_index", "emg_cost", *SPECTRAL]:
            row[col] = np.nan
        if lift_spec and sess_ref.freqs is not None and (session_ref is not None or ok):
            # #26: against set 1's baseline (sets 2-3 from their first lift; set 1 after its baseline)
            row["mdf_session"] = pooled_ratio(lift_spec, sess_ref, cfg, band)
        if bins:
            row["mdf_hz"] = middle_pitch(freqs, sum(lift_spec[b] for b in bins), band)  # absolute, for reference
            row["rms_mv"] = float(np.sqrt(np.mean([band_power(freqs, lift_spec[b], band) for b in bins])))
            if pull_rms_mv:
                row["loud_max"] = row["rms_mv"] / pull_rms_mv                                            # 9
            if row["emg_ok"]:
                for col, stat in SPECTRAL.items():                                                       # 1, 2, 3, 4
                    row[col] = pooled_ratio(lift_spec, own_ref, cfg, band, stat)
                row["loud_ratio"] = float(np.mean([np.sqrt(band_power(freqs, lift_spec[b], band) / base_power[b])
                                                   for b in bins]))
                lp, lm = np.log(row["loud_ratio"]), np.log(row["mdf_ratio"])
                row["fatigue_index"], row["effort_index"] = (lp - lm) / np.sqrt(2), (lp + lm) / np.sqrt(2)  # 10
                if row["speed_loss"] < 1:
                    row["emg_cost"] = row["loud_ratio"] / (1 - row["speed_loss"])                        # 11
        if ok and before and low_ref.freqs is not None:                                                  # 6
            _, low_spec = _bin_spectra(before, cl, cleaner, nfft, cfg)
            row["mdf_lower_ratio"] = pooled_ratio({b: np.mean(v, axis=0) for b, v in low_spec.items()},
                                                  low_ref, cfg, band)
        rows.append(row)
    return pd.DataFrame(rows)


def pull_window(pull: Session, cfg: dict) -> tuple[int, int]:
    """EMG sample range of the pull itself: "Pull!" to "Stop" plus pull_after_stop_s (people stop a
    moment late). Leftover activity from the set before, or a late start after "Stop", is outside it
    (pilot 2026-11-20: pull2 scored above the baseline pulls from activity after "Stop"). The whole
    recording if it has no pull markers."""
    marks = {label: t for t, label in pull.markers}
    if "pull_start" not in marks or "pull_end" not in marks:
        return 0, len(pull.emg)
    a = int(marks["pull_start"] * pull.emg_hz)
    b = int((marks["pull_end"] + cfg["lab"]["pull_after_stop_s"]) * pull.emg_hz)
    return max(a, 0), min(b, len(pull.emg))


def max_pull_paths(folder: Path, participant: str) -> list[Path]:
    """The pulls that set 100%: all 6 before set 1 (3 cold, 3 after their own warm-up)."""
    return sorted(list(folder.glob(f"{participant}_pull_base*.bin")) + list(folder.glob(f"{participant}_pull_warm*.bin")))


def pull_lead_noise(pull: Session, cfg: dict) -> float:
    """RMS of cleaned EMG in the relaxed lead-in before "Pull!" (skips the first 0.3 s)."""
    cl = clean_emg(pull, cfg)
    marks = {label: t for t, label in pull.markers}
    end = int(marks.get("pull_start", cfg["lab"]["pull_lead_s"]) * cl.fs)
    x = np.where(cl.skip, np.nan, cl.emg)[int(0.3 * cl.fs):end]
    return float(np.sqrt(np.nanmean(x ** 2))) if np.isfinite(x).any() else float("nan")


def pull_loudness(pull: Session, cfg: dict, window_s: float = 1.0) -> float:
    """Loudness of a max pull: RMS of cleaned EMG over its loudest window_s inside the pull window."""
    cl = clean_emg(pull, cfg)
    a, b = pull_window(pull, cfg)
    x = np.where(cl.skip, np.nan, cl.emg)[a:b]
    n = max(1, int(window_s * cl.fs))
    sq = pd.Series(x**2).rolling(n, min_periods=int(0.8 * n)).mean()
    return float(np.sqrt(sq.max())) if sq.notna().any() else float("nan")


def reference(s: Session, cfg: dict, cleaner: SpectrumCleaner | None = None) -> Reference:
    """A set's own baseline reference (built from its first baseline_lifts lifts)."""
    cl, track = clean_emg(s, cfg), forearm_angle(s, cfg)
    base = find_lifts(track, s.gyro, cfg)[: cfg["features"]["baseline_lifts"]]
    passes = [p for p in find_passes(cl, track, cfg) if p.direction == 1
              and any(lf.t_start - 0.1 <= p.t_mid_s <= lf.t_end + 0.1 for lf in base)]
    return _reference(passes, cl, cleaner, cfg)


def set_features(path: str | Path, cfg: dict) -> pd.DataFrame:
    """Rows for one set file, with the person's rest-recording cleaner and, for sets 2-3, set 1's
    reference for #26. Everything used was recorded before this set."""
    path = Path(path)
    participant, set_name = path.stem.rsplit("_", 1)
    cleaner = cleaner_for(path, participant, cfg)
    ref = None
    if set_name != "set1":
        first = path.with_name(f"{participant}_set1.bin")
        ref = reference(load_session(first, cfg), cfg, cleaner) if first.exists() else None
    pulls = max_pull_paths(path.parent, participant)
    pull_rms = max((pull_loudness(load_session(p, cfg), cfg) for p in pulls), default=None)
    return lift_features(load_session(path, cfg), cfg, cleaner, ref, pull_rms)


def cleaner_for(set_path: Path, participant: str, cfg: dict) -> SpectrumCleaner | None:
    """The spectrum cleaner from this person's rest recording, if there is one."""
    rest = Path(set_path).parent / f"{participant}_rest.bin"
    return rest_check(load_session(rest, cfg), cfg).cleaner(cfg) if rest.exists() else None


def truncated(s: Session, cut_s: float) -> Session:
    """The session as the board had it at cut_s (for the real-time check)."""
    n = int(cut_s * s.emg_hz)
    keep = s.imu_t <= cut_s
    return Session(name=s.name, source=s.source, emg=s.emg[:n], emg_hz=s.emg_hz, imu_t=s.imu_t[keep],
                   accel=s.accel[keep], gyro=s.gyro[keep], markers=s.markers, meta=s.meta, health=s.health)
