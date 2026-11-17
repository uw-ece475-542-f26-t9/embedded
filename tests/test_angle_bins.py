"""Step 5: the forearm angle (RepSense port, one-way) and the soft angle bins."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from pipeline import load_config, load_session
from pipeline.angle import forearm_angle, track_down
from pipeline.bins import bin_weights, find_passes, pass_spectrum
from pipeline.clean import clean_emg, middle_pitch
from pipeline.simulate import simulate, synth_emg

CFG = load_config()
REPSENSE = Path.home() / "Documents" / "repsense"


@pytest.fixture(scope="module")
def sims(tmp_path_factory):
    root = tmp_path_factory.mktemp("sim")
    tilted = dict(CFG)
    tilted["sim"] = {**CFG["sim"], "mount_tilt_deg": 35.0}
    return {"normal": simulate(CFG, "normal", root / "n", people=2, seed=6),
            "tilted": simulate(tilted, "normal", root / "t", people=2, seed=7)}


def load(d, name):
    s = load_session(d / f"{name}.bin", CFG)
    truth = dict(np.load(d / f"{name}_truth.npz"))
    truth.update(json.loads((d / f"{name}_truth.json").read_text()))
    return s, truth


@pytest.mark.parametrize("world", ["normal", "tilted"])
def test_angle_follows_the_truth_whatever_the_mount(sims, world):
    for pid in ("P01", "P02"):
        s, truth = load(sims[world], f"{pid}_set1")
        tr = forearm_angle(s, CFG)
        est = np.interp(truth["t"], tr.t, tr.angle_deg)
        ref = truth["angle_deg"] - truth["angle_deg"][truth["t"] < 2.5].mean()  # 0 = where the set started
        v = np.gradient(truth["angle_deg"], truth["t"])
        moving = (np.abs(v) > 40) & ~np.isnan(est)
        assert np.sqrt(np.mean((est - ref)[moving] ** 2)) < 5
        d = np.interp(truth["t"], tr.t, tr.direction)
        assert np.mean(np.sign(d[moving]) == np.sign(v[moving])) > 0.95


def test_hanging_is_fixed_at_the_first_lift_not_by_the_sync_flicks(sims):
    s, truth = load(sims["normal"], "P01_set1")
    tr = forearm_angle(s, CFG)
    assert truth["sync_s"] + 1.0 < tr.hang_known_s < truth["set_start_s"] + 1.0


def test_angle_is_real_time(sims):
    s, _ = load(sims["normal"], "P01_set1")
    full = forearm_angle(s, CFG)
    cut_s = full.hang_known_s + 8
    keep = s.imu_t <= cut_s
    s.imu_t, s.accel, s.gyro = s.imu_t[keep], s.accel[keep], s.gyro[keep]
    part = forearm_angle(s, CFG)
    n = len(part.t)
    np.testing.assert_allclose(part.angle_deg, full.angle_deg[:n], equal_nan=True)


def lifts(truth):
    """Each rep's lift window and kind, from the truth."""
    return [(r["start_s"], r["start_s"] + r["lift_s"], r["kind"]) for r in truth["reps"]]


def test_each_lift_fills_the_bins_it_reaches(sims):
    s, truth = load(sims["normal"], "P01_set3")
    passes = find_passes(clean_emg(s, CFG), forearm_angle(s, CFG), CFG)
    for a, b, kind in lifts(truth):
        bins = sorted(p.bin_deg for p in passes if p.direction == 1 and a - 0.05 <= p.t_mid_s <= b + 0.2)
        assert len(bins) == len(set(bins)), f"a bin filled twice in one {kind} lift: {bins}"
        if kind == "good":
            assert bins[0] == 15 and bins[-1] >= 120 and len(bins) >= 8
        if kind == "partial":
            assert bins[-1] <= 75  # a 55% rep tops out around 70 deg


def test_lowering_has_its_own_bins(sims):
    s, _ = load(sims["normal"], "P01_set1")
    passes = find_passes(clean_emg(s, CFG), forearm_angle(s, CFG), CFG)
    up = [p for p in passes if p.direction == 1]
    down = [p for p in passes if p.direction == -1]
    assert len(down) > 0.7 * len(up)


def test_passes_are_real_time(sims):
    s, _ = load(sims["normal"], "P01_set1")
    cl, tr = clean_emg(s, CFG), forearm_angle(s, CFG)
    full = find_passes(cl, tr, CFG)
    cut = full[len(full) // 2].t_known_s
    early = find_passes(cl, tr, CFG, cut_s=cut)
    assert [(p.bin_deg, p.direction, p.start) for p in early] == \
           [(p.bin_deg, p.direction, p.start) for p in full if p.t_known_s <= cut]


def test_soft_weights_overlap_and_sum_to_one_inside():
    a = np.linspace(15, 135, 241)
    total = sum(bin_weights(a, c, 15) for c in CFG["bins"]["centres_deg"])
    np.testing.assert_allclose(total, 1.0, atol=1e-9)


def test_pass_spectrum_gives_the_right_middle_pitch_even_for_long_passes():
    fs, rng = 2000.0, np.random.default_rng(3)
    t = np.arange(300000) / fs
    x = synth_emg(t, np.full(len(t), 60.0), np.ones(len(t)), rng)
    for n in (260, 400, 1500):  # typical passes, and one longer than nfft (chunked)
        est = [middle_pitch(*pass_spectrum(x[i:i + n], np.bartlett(n), fs, 512)) for i in range(0, len(x) - n, n)]
        assert np.median(est) == pytest.approx(60, rel=0.03), n
    assert len(pass_spectrum(x[:1500], np.bartlett(1500), fs, 512)[0]) == 257


@pytest.mark.skipif(not (REPSENSE / "src" / "labels.py").exists(), reason="RepSense not on this machine")
def test_track_down_is_an_exact_port_of_repsense():
    sys.path.insert(0, str(REPSENSE))
    try:
        from src.labels import track_down as repsense_track_down
    finally:
        sys.path.pop(0)
    rng = np.random.default_rng(0)
    t = np.cumsum(np.full(500, 0.02))
    gyro = rng.normal(0, 80, (500, 3))
    accel = rng.normal(0, 0.3, (500, 3)) + [0, 0, -1]
    np.testing.assert_array_equal(track_down(t, gyro, accel, 1.0), repsense_track_down(t, gyro, accel, 1.0))
