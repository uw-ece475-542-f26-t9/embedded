"""Step 4: cleaning. Gets EMG ready for features, in real time (past samples only).

Time domain (clean_emg):
  1. Subtract a running estimate of the mid-rail level (RAW sits at ~1,440 mV at rest).
  2. Band-pass 20-450 Hz with a linear-phase FIR. A real-time filter can only use past samples,
     so it delays its output. A linear-phase FIR delays every frequency by the same amount,
     (taps - 1) / 2 samples, so nothing is smeared and the delay can simply be subtracted: the
     output is shifted back onto the original clock. In real time that is a fixed latency
     (100 ms with 401 taps at 2 kHz). The cutoffs sit just outside 20-450 Hz so the band is flat.
  3. Mark samples to skip: missing data, clipping, filter warm-up, and motion jolts (low-frequency
     swings from electrodes sliding or leads tugging, which add false loudness and drag middle
     pitch down).

Spectrum (applied to every spectrum the features compute):
  4. Drop and fill lines: bins within hum_halfwidth_hz of mains hum (60/120/180 Hz) and of any
     line found in the rest recording (e.g. USB packet noise) are replaced by a straight line
     between their neighbours. A notch filter would also remove the real EMG at those
     frequencies, which sits right in the middle of the band.
  5. Subtract the noise floor: the rest recording's noise spectrum is subtracted, so quiet people
     don't get a middle pitch pulled up by flat background noise.

Rest check (rest_check): finds hum and narrow lines on the rest recording before any set is
recorded, so the session can be fixed (move the laptop, unplug the charger) and redone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from .session import Session


# ------------------------------------------------------------------ time domain


def bandpass_taps(fs: float, cfg: dict) -> np.ndarray:
    c = cfg["clean"]
    return signal.firwin(c["fir_taps"], c["fir_band_hz"], pass_zero=False, fs=fs,
                         window=("kaiser", c["fir_kaiser_beta"]))


@dataclass
class Cleaned:
    emg: np.ndarray      # band-passed EMG in mV, on the session's clock (0 where skipped)
    fs: float
    skip: np.ndarray     # True where the sample must not be used
    delay_s: float       # filter delay, already subtracted: sample t is on the original clock
    latency_s: float     # the board knows cleaned sample t (and whether to skip it) at t + latency_s
    reasons: dict = field(default_factory=dict)  # samples skipped for each reason


def running_baseline(x: np.ndarray, fs: float, tau_s: float) -> np.ndarray:
    """One-pole running mean (past samples only), started at the first valid sample."""
    a = 1 / (tau_s * fs)
    start = x[np.flatnonzero(~np.isnan(x))[0]] if np.any(~np.isnan(x)) else 0.0
    filled = np.where(np.isnan(x), start, x)
    zi = signal.lfilter_zi([a], [1, a - 1]) * start
    return signal.lfilter([a], [1, a - 1], filled, zi=zi)[0]


def _dilate(mask: np.ndarray, n: int) -> np.ndarray:
    if n <= 0 or not mask.any():
        return mask
    return np.convolve(mask.astype(float), np.ones(2 * n + 1), mode="same") > 0


def clean_emg(s: Session, cfg: dict) -> Cleaned:
    c, fs = cfg["clean"], s.emg_hz
    x = s.emg.astype(float)
    missing = np.isnan(x)
    clipped = ~missing & ((x <= 1) | (x >= 3099))
    centred = np.where(missing, 0.0, x - running_baseline(x, fs, c["baseline_tau_s"]))

    taps = bandpass_taps(fs, cfg)
    d = (len(taps) - 1) // 2
    y = signal.lfilter(taps, 1.0, centred)          # real time: output lags by d samples
    emg = np.r_[y[d:], np.zeros(d)]                 # back onto the original clock

    # Jolts: a slow swing (below jolt_hz) bigger than jolt_mv and as big as the EMG itself. Real
    # EMG has some low-frequency content (about 20% of its in-band loudness in the real squeezes),
    # so a fixed limit would throw away hard contractions. Loudness is over 100 ms centred on each
    # sample: 50 ms of look-ahead, inside the filter's 100 ms latency.
    sos = signal.butter(2, c["jolt_hz"], fs=fs, output="sos")
    low = signal.sosfilt(sos, centred)
    w = np.ones(max(1, int(0.1 * fs))) / max(1, int(0.1 * fs))
    low_rms = np.sqrt(np.convolve(low**2, w, mode="same"))
    emg_rms = np.sqrt(np.convolve(emg**2, w, mode="same"))
    jolt = (low_rms > c["jolt_mv"]) & (low_rms > c["jolt_ratio"] * emg_rms)

    pad = int(c["jolt_pad_s"] * fs)
    warmup = np.zeros(len(x), bool)
    warmup[: len(taps)] = True
    warmup[len(x) - d:] = True                       # the last d samples need future input
    gap = _dilate(missing | clipped, d)              # a gap spoils every output the filter spans
    skip = gap | warmup | _dilate(jolt, pad)
    emg[skip] = 0.0
    reasons = {"gap or clip": int(gap.sum()), "warm-up": int(warmup.sum()),
               "jolt": int(_dilate(jolt, pad).sum())}
    latency = (d + len(w) // 2 + pad) / fs  # filter delay + half the loudness window + jolt padding
    return Cleaned(emg=emg, fs=fs, skip=skip, delay_s=d / fs, latency_s=latency, reasons=reasons)


# ------------------------------------------------------------------ spectrum


def spectrum(x: np.ndarray, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Power spectral density (mV^2/Hz) of one stretch, Hann window, zero-padded to nfft.
    Normalised by the window's energy, so stretches of different lengths are comparable."""
    x = np.asarray(x, float)
    if len(x) > nfft:
        raise ValueError(f"stretch of {len(x)} samples is longer than nfft={nfft}")
    w = np.hanning(len(x))
    X = np.fft.rfft((x - x.mean()) * w, nfft)
    p = np.abs(X) ** 2 / (fs * np.sum(w**2))
    p[1:-1] *= 2
    return np.fft.rfftfreq(nfft, 1 / fs), p


def line_bins(f: np.ndarray, lines_hz, halfwidth_hz: float) -> np.ndarray:
    mask = np.zeros(len(f), bool)
    for h in lines_hz:
        mask |= np.abs(f - h) <= halfwidth_hz
    return mask


def drop_lines(f: np.ndarray, p: np.ndarray, lines_hz, halfwidth_hz: float) -> np.ndarray:
    """Replace the bins around each line with a straight line between their neighbours."""
    bad = line_bins(f, lines_hz, halfwidth_hz)
    if not bad.any() or bad.all():
        return p
    out = p.copy()
    out[bad] = np.interp(f[bad], f[~bad], p[~bad])
    return out


def subtract_noise(p: np.ndarray, noise: np.ndarray | None, share: float) -> np.ndarray:
    if noise is None or share == 0:
        return p
    return np.maximum(p - share * noise, 0.0)


def middle_pitch(f: np.ndarray, p: np.ndarray, band=(20.0, 450.0)) -> float:
    """Frequency splitting the in-band power in half, interpolated between bins."""
    k = (f >= band[0]) & (f <= band[1])
    total = np.cumsum(p[k])
    if total[-1] <= 0:
        return float("nan")
    return float(np.interp(total[-1] / 2, total, f[k]))


def band_power(f: np.ndarray, p: np.ndarray, band=(20.0, 450.0)) -> float:
    """Power in the band (mV^2); its square root is the loudness (RMS) of the band."""
    k = (f >= band[0]) & (f <= band[1])
    return float(np.sum(p[k]) * (f[1] - f[0]))


@dataclass
class SpectrumCleaner:
    """Everything the features need to clean one spectrum: which lines to drop, and the noise floor."""
    lines_hz: list
    halfwidth_hz: float
    noise: np.ndarray | None
    share: float
    band: tuple

    def __call__(self, f: np.ndarray, p: np.ndarray) -> np.ndarray:
        p = drop_lines(f, p, self.lines_hz, self.halfwidth_hz)
        return subtract_noise(p, self.noise, self.share)


# ------------------------------------------------------------------ rest check


@dataclass
class RestCheck:
    hum_mv: float                  # 60 Hz hum, RMS
    noise_mv: float                # background in the band, RMS (lines dropped)
    lines: list                    # (Hz, mV, kind) narrow peaks in the band
    noise_spectrum: np.ndarray     # on the shared nfft bins, lines dropped: subtracted from every spectrum
    problems: list = field(default_factory=list)

    def cleaner(self, cfg: dict) -> SpectrumCleaner:
        c = cfg["clean"]
        lines = sorted({*c["hum_hz"], *(hz for hz, _, _ in self.lines)})
        return SpectrumCleaner(lines, c["hum_halfwidth_hz"], self.noise_spectrum, c["noise_subtract"],
                               tuple(c["band_hz"]))


KNOWN_BASES = {50.0: "IMU frame rate", 62.5: "EMG frame rate", 78.125: "old 128-reading packets"}


def _is_multiple(hz: float, base: float, tol: float) -> bool:
    k = round(hz / base)
    return k >= 1 and abs(hz - k * base) <= tol


def classify_lines(freqs: list[float], tol: float) -> list[str]:
    """Name each line: mains hum, a harmonic family (3+ lines at multiples of one base, e.g.
    USB packet noise), or other."""
    kinds = []
    others = [f for f in freqs if not _is_multiple(f, 60.0, tol)]
    bases = sorted({*KNOWN_BASES, *others})
    families = {b: [f for f in others if _is_multiple(f, b, tol)] for b in bases}
    families = {b: m for b, m in families.items() if len(m) >= 3}
    for f in freqs:
        if _is_multiple(f, 60.0, tol):
            kinds.append("mains")
            continue
        fam = next((b for b, m in families.items() if f in m), None)
        if fam is None:
            kinds.append("other")
        else:
            kinds.append(f"harmonic of {fam:g} Hz" + (f" ({KNOWN_BASES[fam]})" if fam in KNOWN_BASES else ""))
    return kinds


def rest_check(rest: Session, cfg: dict) -> RestCheck:
    """Hum, background noise and narrow lines on a rest recording (arm relaxed, no EMG)."""
    c = cfg["clean"]
    cl = clean_emg(rest, cfg)
    x = cl.emg[~cl.skip]
    fs, band = cl.fs, tuple(c["band_hz"])

    # Fine spectrum (0.5 Hz bins) to find lines.
    f, p = signal.welch(x, fs, nperseg=min(len(x), int(4 * fs)))
    df = f[1] - f[0]
    smooth = signal.medfilt(p, 2 * int(10 / df) + 1)        # neighbourhood level, +-10 Hz
    in_band = (f >= band[0]) & (f <= band[1])
    peaks, _ = signal.find_peaks(np.where(in_band, p, 0), height=c["line_ratio"] * smooth)
    found = []
    for k in peaks:
        mv = float(np.sqrt(max(p[k] - smooth[k], 0) * df * 1.5))  # Hann spreads a line over ~1.5 bins
        if mv >= c["line_min_mv"]:
            found.append((round(float(f[k]), 1), round(mv, 2)))
    kinds = classify_lines([hz for hz, _ in found], c["mains_tolerance_hz"])
    lines = [(hz, mv, kind) for (hz, mv), kind in zip(found, kinds)]
    near60 = np.abs(f - 60) <= 1.5
    hum = float(np.sqrt(max(np.sum(p[near60] - smooth[near60]), 0) * df))

    # Noise spectrum on the shared bins, with every line dropped, for subtraction.
    nfft = c["nfft"]
    chunks = [x[i:i + nfft] for i in range(0, len(x) - nfft + 1, nfft // 2)]
    fb, _ = spectrum(chunks[0], fs, nfft)
    noise = np.mean([spectrum(ch, fs, nfft)[1] for ch in chunks], axis=0)
    all_lines = sorted({*c["hum_hz"], *(hz for hz, _, _ in lines)})
    noise = drop_lines(fb, noise, all_lines, c["hum_halfwidth_hz"])
    noise_mv = float(np.sqrt(band_power(fb, noise, band)))

    problems = []
    if hum > c["max_hum_mv"]:
        problems.append(f"60 Hz hum {hum:.1f} mV (limit {c['max_hum_mv']} mV): unplug the charger, "
                        "move away from power strips, redo the rest recording")
    bad = [ln for ln in lines if ln[2] != "mains" and (ln[1] >= c["line_problem_mv"] or ln[2].startswith("harmonic"))]
    if bad:
        problems.append("interference peaks at " + ", ".join(f"{hz:g} Hz ({kind}, {mv} mV)" for hz, mv, kind in bad)
                        + ": find the source and redo the rest recording (they are dropped from spectra either way)")
    return RestCheck(hum_mv=round(hum, 2), noise_mv=round(noise_mv, 2), lines=lines, noise_spectrum=noise,
                     problems=problems)


def alias_check(rest: Session, rest_alt: Session, cfg: dict) -> dict:
    """The spike check (protocol step 3): rest recorded at two ADC rates. Real signals and interference
    in the band (hum, USB packets) sit at the same frequency in both; anything above the band that
    folded down (aliasing; the board has no anti-aliasing capacitor) lands somewhere else at the other
    rate. A narrow peak present in one recording but not the other moved: that's aliasing."""
    tol = cfg["lab"]["alias_tolerance_hz"]
    a, b = rest_check(rest, cfg), rest_check(rest_alt, cfg)
    fa = [hz for hz, _, _ in a.lines]
    fb = [hz for hz, _, _ in b.lines]
    moved = sorted({*(f for f in fa if not any(abs(f - g) <= tol for g in fb)),
                    *(g for g in fb if not any(abs(g - f) <= tol for f in fa))})
    stayed = sorted(f for f in fa if any(abs(f - g) <= tol for g in fb))
    problems = []
    if moved:
        problems.append("peaks that move between the two rates (aliasing: interference above the band folding "
                        f"in): {', '.join(f'{f:g} Hz' for f in moved)}. Move or unplug nearby devices, redo both "
                        "rest recordings")
    return {"moved_hz": moved, "stayed_hz": stayed, "problems": problems}
