"""Step 2: the simulator makes what it claims to, and the test worlds differ the way they should."""

import json

import numpy as np
import pytest
from scipy import signal

from pipeline import load_config, load_session
from pipeline.simulate import simulate, synth_emg

CFG = load_config()


def mdf(x, fs=2000.0, nperseg=512):
    """Middle pitch over 20-450 Hz, interpolated within the bin."""
    f, p = signal.welch(x, fs, nperseg=min(nperseg, len(x)))
    k = (f >= 20) & (f <= 450)
    f, c = f[k], np.cumsum(p[k])
    i = np.searchsorted(c, c[-1] / 2)
    prev = c[i - 1] if i else 0.0
    return float(f[i] - (f[1] - f[0]) * (c[i] - c[-1] / 2) / (c[i] - prev))


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    return {w: simulate(CFG, w, root / w, people=4, seed=1) for w in CFG["sim"]["worlds"]}


def reps(d, name):
    return json.loads((d / f"{name}_truth.json").read_text())


@pytest.mark.parametrize("target", [35.0, 55.0, 75.0])
def test_emg_has_the_middle_pitch_asked_for(target):
    t = np.arange(40000) / 2000
    x = synth_emg(t, np.full(len(t), target), np.ones(len(t)), np.random.default_rng(0))
    assert mdf(x) == pytest.approx(target, rel=0.03)
    assert x.std() == pytest.approx(1, rel=0.05)


def test_slice_scatter_is_like_the_real_holds():
    """Real holds scattered about 10% between 0.5 s slices; the EMG model alone should give a
    similar amount, so the pipeline faces realistic noise."""
    t = np.arange(120000) / 2000
    x = synth_emg(t, np.full(len(t), 60.0), np.ones(len(t)), np.random.default_rng(2))
    slices = [mdf(x[i:i + 1000], nperseg=256) for i in range(0, len(x) - 1000, 1000)]
    scatter = np.std(slices) / np.mean(slices)
    assert 0.03 < scatter < 0.15, scatter


def test_every_recording_loads_clean(worlds):
    d = worlds["normal"]
    names = [p.stem for p in sorted(d.glob("*.bin"))]
    assert len(names) == 4 * 10  # rest, 3 base pulls, 3 sets, 3 pulls
    for name in names:
        s = load_session(d / f"{name}.bin", CFG)
        assert s.problems(CFG) == [], name
        assert s.imu_hz == pytest.approx(50, rel=0.05)


def test_sets_end_at_the_stop_point_and_set_3_at_failure(worlds):
    d = worlds["normal"]
    for pid in ("P01", "P02", "P03", "P04"):
        for k in (1, 2):
            info = reps(d, f"{pid}_set{k}")
            assert info["stop_rep"] == len(info["reps"]) and info["failure_rep"] is None
        info = reps(d, f"{pid}_set3")
        assert info["failure_rep"] == len(info["reps"]) and info["stop_rep"] < info["failure_rep"]
        side = json.loads((d / f"{pid}_set3.json").read_text())
        assert abs(side["labels"]["stop_point_s"] - info["stop_point_s"]) in (0, CFG["sim"]["cadence_s"])


def test_short_rest_leaves_the_next_set_shorter(worlds):
    d, ratio = worlds["normal"], {60: [], 180: []}
    for pid in ("P01", "P02", "P03", "P04"):
        rest = json.loads((d / f"{pid}_set2.json").read_text())["rest_before_s"]
        ratio[rest].append(len(reps(d, f"{pid}_set2")["reps"]) / len(reps(d, f"{pid}_set1")["reps"]))
    assert np.mean(ratio[60]) < np.mean(ratio[180])


def lift_features(d, name):
    s = load_session(d / f"{name}.bin", CFG)
    sos = signal.butter(4, [20, 450], "bandpass", fs=s.emg_hz, output="sos")
    x = signal.sosfiltfilt(sos, s.emg - np.median(s.emg))
    out = []
    for r in reps(d, name)["reps"]:
        if r["kind"] != "good":
            continue
        a, b = int(r["start_s"] * s.emg_hz), int((r["start_s"] + r["lift_s"]) * s.emg_hz)
        out.append((mdf(x[a:b], s.emg_hz, 256), np.sqrt(np.mean(x[a:b] ** 2))))
    return np.array(out)


def test_fatigue_shows_in_emg_only_when_it_should(worlds):
    """Normal: middle pitch falls and loudness rises over set 1. EMG-blind: neither moves."""
    falls = {}
    for w in ("normal", "emg_blind"):
        f = np.array([np.polyfit(np.arange(len(m)), m[:, 0] / m[:3, 0].mean(), 1)[0] * len(m)
                      for m in (lift_features(worlds[w], f"P0{i}_set1") for i in (1, 2, 3, 4))])
        falls[w] = f.mean()
    assert falls["normal"] < -0.07
    assert abs(falls["emg_blind"]) < 0.05


def test_shortcut_world_is_fully_predictable(worlds):
    d = worlds["shortcut"]
    for k in (1, 2, 3):
        stops = {reps(d, f"P0{i}_set{k}")["stop_rep"] for i in (1, 2, 3, 4)}
        assert len(stops) == 1  # every person, every set: the same rep


def test_effort_world_has_no_fatigue_and_no_stop_point(worlds):
    d = worlds["effort"]
    for k in (1, 2, 3):
        info = reps(d, f"P01_set{k}")
        assert info["stop_rep"] is None and info["failure_rep"] is None
        assert any(r["effort"] > 0 for r in info["reps"])
        assert np.all(np.load(d / f"P01_set{k}_truth.npz")["F"] == 0)


def test_imu_reads_gravity_and_the_curl(worlds):
    s = load_session(worlds["normal"] / "P01_set1.bin", CFG)
    still = s.imu_t < 2.5
    assert np.linalg.norm(s.accel[still], axis=1).mean() == pytest.approx(1, abs=0.03)
    turning = np.linalg.norm(s.gyro, axis=1)
    assert np.percentile(turning[s.imu_t > 7], 95) > 150  # a 125 deg lift in about a second


def test_same_seed_same_bytes(tmp_path):
    a = simulate(CFG, "normal", tmp_path / "a", people=1, seed=3)
    b = simulate(CFG, "normal", tmp_path / "b", people=1, seed=3)
    assert (a / "P01_set1.bin").read_bytes() == (b / "P01_set1.bin").read_bytes()
