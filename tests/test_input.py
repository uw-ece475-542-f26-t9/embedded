"""Step 1: the input layer reads both formats and catches damaged recordings."""

import gzip
import json

import numpy as np
import pytest

from pipeline import load_config, load_session
from pipeline.capture import (EMG, IMU, STATUS, FrameReader, crc16, emg_payload, frame,
                              imu_payload, parse, status_payload)
from pipeline.config import ROOT

CFG = load_config()


def capture_stream(seconds=5.0, true_hz=2004.0, drop_blocks=(), overflow_at=None, clipped_per_block=0,
                   imu_skip=(), start_index=0, start_ms=0):
    """Bytes the firmware would send: 32-sample EMG blocks, IMU at 50 Hz, STATUS every second.
    The board's EMG clock really runs at true_hz (the ADC isn't exactly 8 kHz)."""
    n = int(seconds * true_hz)
    t = np.arange(n) / true_hz
    emg = 1500 + 400 * np.sin(2 * np.pi * 80 * t)  # mV, 80 Hz tone on the mid-rail level
    out, overflows, next_imu_ms, next_status_ms = bytearray(), 0, start_ms, start_ms + 1000
    for b, first in enumerate(range(0, n - 31, 32)):
        now_ms = start_ms + first / true_hz * 1000
        if b not in drop_blocks:
            out += frame(EMG, emg_payload(start_index + first, emg[first:first + 32], clipped_per_block))
        while next_imu_ms <= now_ms:
            k = int(round((next_imu_ms - start_ms) / 20))
            if k not in imu_skip:
                stamp = start_index + int((next_imu_ms - start_ms) / 1000 * true_hz)
                out += frame(IMU, imu_payload(stamp, (0, 0, 1), (0, 0, 0)))
            next_imu_ms += 20
        if now_ms >= next_status_ms:
            if overflow_at is not None and now_ms >= overflow_at * 1000:
                overflows = 3
            out += frame(STATUS, status_payload(start_index + first, int(now_ms), overflows=overflows))
            next_status_ms += 1000
    return bytes(out), emg


def test_crc_matches_the_standard_check_value():
    assert crc16(b"123456789") == 0x29B1  # CRC-16/CCITT-FALSE


def test_clean_stream_round_trips():
    data, truth = capture_stream()
    s = parse(data, CFG)
    assert s.problems(CFG) == []
    assert s.health.missing == 0 and s.health.crc_errors == 0
    assert s.emg_hz == pytest.approx(2004, abs=1)  # measured, not assumed 2000
    np.testing.assert_allclose(s.emg, truth[:len(s.emg)], atol=0.051)  # sent in 0.1 mV steps
    assert s.imu_hz == pytest.approx(50, rel=0.05)
    assert s.imu_t[0] == pytest.approx(0, abs=0.02)  # IMU on the EMG clock


def test_chunked_and_noisy_stream_gives_the_same_frames():
    data, _ = capture_stream(seconds=2)
    whole = FrameReader().feed(data)
    rng = np.random.default_rng(0)
    first_frame = 7 + 6 + 2 * 32  # junk goes between frames, not inside one
    noisy = b"\x00\xa5junk" + data[:first_frame] + b"\xa5\x5a\xff\xff" + data[first_frame:]
    reader, cuts, frames = FrameReader(), np.sort(rng.integers(0, len(noisy), 40)), []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(noisy)]):
        frames += reader.feed(noisy[a:b])
    frames += reader.finish()
    assert frames == whole
    assert reader.skipped_bytes > 0


def test_corrupted_frame_becomes_a_short_gap():
    data, _ = capture_stream()
    damaged = bytearray(data)
    damaged[77 + 28 + 40] ^= 0xFF  # inside the 2nd EMG frame (after EMG 0 and IMU 0)
    s = parse(bytes(damaged), CFG)
    assert s.health.crc_errors >= 1
    assert 0 < s.health.missing <= 64
    assert s.problems(CFG) == []  # one lost block is tolerable
    assert any("checksum" in n for n in s.notes())


def test_long_gap_is_rejected():
    s = parse(capture_stream(drop_blocks=range(50, 60))[0], CFG)  # 10 blocks = 160 ms
    assert any("EMG gap" in p for p in s.problems(CFG))


def test_adc_overflow_is_rejected():
    s = parse(capture_stream(overflow_at=3)[0], CFG)
    assert s.health.adc_overflows == 3
    assert any("overflowed" in p for p in s.problems(CFG))


def test_clipping_is_rejected():
    s = parse(capture_stream(clipped_per_block=2)[0], CFG)  # 2 of 128 raw readings per block
    assert any("clipped" in p for p in s.problems(CFG))


def test_imu_gap_is_rejected():
    s = parse(capture_stream(imu_skip=range(100, 120))[0], CFG)  # 0.4 s without IMU
    assert any("IMU gap" in p for p in s.problems(CFG))


def test_reboot_keeps_the_longest_stretch():
    before, _ = capture_stream(seconds=2, start_index=50_000, start_ms=40_000)
    after, _ = capture_stream(seconds=4)  # index and millis restart from 0
    s = parse(before + after, CFG)
    assert s.health.reboots == 1
    assert s.duration_s == pytest.approx(4, abs=0.05)
    assert any("rebooted" in n for n in s.notes())


def test_capture_file_with_sidecar(tmp_path):
    data, _ = capture_stream(seconds=2)
    (tmp_path / "P01_set1.bin").write_bytes(data)
    (tmp_path / "P01_set1.json").write_text(json.dumps(
        {"participant": "P01", "set": 1, "markers": [[0.5, "start"], [1.5, "stop_point"]]}))
    s = load_session(tmp_path / "P01_set1.bin", CFG)
    assert s.name == "P01_set1" and s.meta["participant"] == "P01"
    assert s.markers == [(0.5, "start"), (1.5, "stop_point")]


def tone_amplitude(x, fs, f):
    x = x - x.mean()
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) / (np.hanning(len(x)).sum() / 2)
    return spec[int(round(f * len(x) / fs))]


def test_legacy_recording_keeps_the_band_and_blocks_aliasing(tmp_path):
    fs, t = 10000, np.arange(50000) / 10000
    # 100 Hz is in the EMG band. 1,500 Hz would fold to 500 Hz at 2,000/s if not filtered out.
    raw = 2000 + 300 * np.sin(2 * np.pi * 100 * t) + 300 * np.sin(2 * np.pi * 1500 * t)
    with gzip.open(tmp_path / "X.csv.gz", "wt") as f:
        f.write("raw,env\n" + "\n".join(f"{int(round(v))},0" for v in raw))
    (tmp_path / "X.json").write_text(json.dumps({"rate_hz": fs, "markers": [[20000, "start"]]}))
    s = load_session(tmp_path / "X.csv.gz", CFG)
    assert s.emg_hz == 2000 and s.markers == [(2.0, "start")]
    mv_per_count = CFG["input"]["legacy_full_scale_mv"] / 4095
    assert tone_amplitude(s.emg, 2000, 100) == pytest.approx(300 * mv_per_count, rel=0.02)
    assert tone_amplitude(s.emg, 2000, 500) < 300 * mv_per_count * 0.01  # at least 40 dB down
    assert s.problems(CFG) == []


REAL = (ROOT / CFG["input"]["legacy_data_dir"]).resolve()


@pytest.mark.skipif(not (REAL / "A_squeeze.csv.gz").exists(), reason="sensor-testing data not here")
def test_real_sensor_testing_recordings_load():
    s = load_session("A_squeeze", CFG)
    assert s.duration_s == pytest.approx(75, abs=0.1) and s.emg_hz == 2000
    assert 1400 < np.median(s.emg) < 1650  # rest level ~1,900 counts
    assert s.problems(CFG) == []
    bad = load_session("B_light", CFG)  # the loose RAW jumper: rail to rail
    assert any("clipped" in p for p in bad.problems(CFG))
