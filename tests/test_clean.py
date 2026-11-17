"""Step 4: cleaning keeps the band, removes what isn't muscle, runs in real time, and catches interference."""

import numpy as np
import pytest
from scipy import signal

from pipeline import load_config, load_session
from pipeline.clean import (bandpass_taps, clean_emg, drop_lines, middle_pitch, rest_check, spectrum,
                            subtract_noise)
from pipeline.config import ROOT
from pipeline.session import Session
from pipeline.simulate import simulate, synth_emg

CFG = load_config()
FS = 2000.0


def fake(emg):
    return Session(name="x", source="legacy", emg=np.asarray(emg, float), emg_hz=FS)


def test_filter_is_linear_phase_and_flat_across_the_band():
    h = bandpass_taps(FS, CFG)
    np.testing.assert_allclose(h, h[::-1])  # symmetric taps = the same delay at every frequency
    _, H = signal.freqz(h, worN=[3, 8, 20, 25, 50, 100, 300, 450], fs=FS)
    db = 20 * np.log10(np.abs(H))
    assert db[0] < -50 and db[1] < -20                 # motion jolts and drift
    assert np.all(np.abs(db[2:]) < 1.0)                # 20-450 Hz kept (201 taps lost 6 dB at 20 Hz)


def test_output_is_on_the_original_clock():
    x = np.full(20000, 1440.0)
    t = np.arange(len(x)) / FS
    x += 50 * np.exp(-0.5 * ((t - 5.0) / 0.05) ** 2) * np.sin(2 * np.pi * 100 * t)  # burst at 5 s
    y = clean_emg(fake(x), CFG).emg
    env = np.abs(signal.hilbert(y))
    assert t[np.argmax(env)] == pytest.approx(5.0, abs=0.003)


def test_cleaning_is_real_time():
    rng = np.random.default_rng(0)
    x = 1440 + 20 * rng.standard_normal(30000)
    full = clean_emg(fake(x), CFG)
    k = 15000
    cut = clean_emg(fake(x[: k + int(full.latency_s * FS) + 1]), CFG)
    np.testing.assert_allclose(cut.emg[:k], full.emg[:k], atol=1e-9)
    np.testing.assert_array_equal(cut.skip[:k], full.skip[:k])


@pytest.mark.parametrize("true", [45.0, 51.0, 72.0])
def test_hum_and_noise_cleaning_recovers_quiet_emg(true):
    rng = np.random.default_rng(int(true))
    t = np.arange(120000) / FS
    emg = 6 * synth_emg(t, np.full(len(t), true), np.ones(len(t)), rng)   # a quiet person
    hum = 1.4 * np.sqrt(2) * np.sin(2 * np.pi * 60 * t)
    white = lambda n: rng.standard_normal(n) * 1.9 / np.sqrt(430 / 1000)  # noqa: E731
    rest = hum[:20000] + white(20000)
    f, _ = spectrum(rest[:512], FS, 512)
    noise = drop_lines(f, np.mean([spectrum(rest[i:i + 512], FS, 512)[1] for i in range(0, 19400, 256)], 0),
                       [60, 120, 180], 6)
    y = emg + hum + white(len(t))
    plain, cleaned, ref = [], [], []
    for i in range(0, len(t) - 500, 500):
        f, P = spectrum(y[i:i + 500], FS, 512)
        plain.append(middle_pitch(f, P))
        cleaned.append(middle_pitch(f, subtract_noise(drop_lines(f, P, [60, 120, 180], 6), noise, 1.0)))
        ref.append(middle_pitch(*spectrum(emg[i:i + 500], FS, 512)))
    ref = np.median(ref)
    assert abs(np.median(cleaned) / ref - 1) < 0.04
    assert abs(np.median(cleaned) - ref) < abs(np.median(plain) - ref)   # cleaning helps


def test_middle_pitch_estimate_is_accurate():
    t = np.arange(120000) / FS
    for true in (45.0, 51.0, 72.0):
        x = synth_emg(t, np.full(len(t), true), np.ones(len(t)), np.random.default_rng(1))
        est = np.median([middle_pitch(*spectrum(x[i:i + 500], FS, 512)) for i in range(0, len(t) - 500, 500)])
        assert est == pytest.approx(true, rel=0.05)


@pytest.fixture(scope="module")
def sims(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    loud = dict(CFG)
    loud["sim"] = {**CFG["sim"], "packet_noise_mv": 1.0, "motion_artifact_mv": 120.0}
    return {"normal": simulate(CFG, "normal", root / "n", people=1, seed=2),
            "noisy": simulate(loud, "normal", root / "p", people=1, seed=2)}


def test_rest_check_passes_a_clean_session_and_flags_packet_noise(sims):
    ok = rest_check(load_session(sims["normal"] / "P01_rest.bin", CFG), CFG)
    assert ok.problems == [] and ok.hum_mv == pytest.approx(1.0, rel=0.3)
    bad = rest_check(load_session(sims["noisy"] / "P01_rest.bin", CFG), CFG)
    kinds = {kind for _, _, kind in bad.lines}
    assert any("50 Hz" in k for k in kinds) and any("62.5 Hz" in k for k in kinds)
    assert bad.problems


def test_jolts_are_found_where_they_happened(sims):
    import json
    for key in ("normal", "noisy"):
        d = sims[key]
        cl = clean_emg(load_session(d / "P01_set1.bin", CFG), CFG)
        info = json.loads((d / "P01_set1_truth.json").read_text())
        # the simulator puts a jolt at each lift start and each start of lowering
        events = [r["start_s"] for r in info["reps"]] + [r["start_s"] + r["lift_s"] + 0.15 for r in info["reps"]]
        t = np.arange(len(cl.emg)) / cl.fs
        near = np.min(np.abs(t[:, None] - np.array(events)[None, :]), axis=1) < 0.2
        flagged = cl.skip & ~np.isin(np.arange(len(t)), np.r_[:601, len(t) - 200:len(t)])
        assert flagged[near].sum() >= 0.9 * flagged.sum(), key   # what's flagged is at a jolt
    jolty = clean_emg(load_session(sims["noisy"] / "P01_set1.bin", CFG), CFG).reasons["jolt"]
    quiet = clean_emg(load_session(sims["normal"] / "P01_set1.bin", CFG), CFG).reasons["jolt"]
    assert jolty > quiet


REAL = (ROOT / CFG["input"]["legacy_data_dir"]).resolve()
real = pytest.mark.skipif(not (REAL / "B_rest.csv.gz").exists(), reason="sensor-testing data not here")


@real
def test_real_rest_check_catches_the_old_packet_noise():
    assert rest_check(load_session("A_rest", CFG), CFG).problems == []
    assert rest_check(load_session("B_rest2", CFG), CFG).problems == []
    old = rest_check(load_session("B_rest", CFG), CFG)
    assert old.problems and sum("78.125" in kind for _, _, kind in old.lines) >= 3


@real
def test_real_squeezes_are_not_mistaken_for_jolts():
    for name in ("A_squeeze", "B_squeeze2"):
        cl = clean_emg(load_session(name, CFG), CFG)
        assert cl.reasons["jolt"] < 0.01 * len(cl.emg)


@real
def test_legacy_recordings_have_no_edge_ramps():
    s = load_session("A_rest", CFG)
    assert abs(s.emg[0] - np.median(s.emg)) < 20 and abs(s.emg[-1] - np.median(s.emg)) < 20
