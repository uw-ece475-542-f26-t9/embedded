"""Step 10 items 1-3: the protocol recorder, the on-the-spot checks and the label import, driven by
simulated byte streams (no board here). Code checks only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import load_config
from pipeline.capture import EMG, EMG_HEAD, FrameReader, frame
from pipeline.evaluate import load_sets
from pipeline.lab import STEPS, Recorder, Session, Sound
from pipeline.labels import agreement, apply, find_flicks, template
from pipeline.simulate import simulate

CFG = load_config()


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakePort:
    """Serves a recorded stream at its real pace: each frame arrives when its last EMG sample is due."""

    def __init__(self, data: bytes, clock: Clock, hz: float = 2000.0, step: float = 0.01):
        frames, first = FrameReader().feed(data), None
        self.queue = []
        for ftype, p in frames:
            due = 0.0
            if ftype == EMG:
                f, n, _ = EMG_HEAD.unpack_from(p)
                first = f if first is None else first
                due = (f + n - first) / hz
            elif self.queue:
                due = self.queue[-1][0]
            self.queue.append((due, frame(ftype, p)))
        self.clock, self.step = clock, step

    def write(self, data):
        self.written = getattr(self, "written", []) + [data]

    def read(self, n):
        self.clock.now += self.step
        out = bytearray()
        while self.queue and self.queue[0][0] <= self.clock.now:
            out += self.queue.pop(0)[1]
        return bytes(out)


class FakeEnter:
    """Enter presses at given recording times."""

    def __init__(self, clock, times):
        self.clock, self.times, self.start = clock, times, None

    @property
    def presses(self):
        if self.start is None:
            self.start = self.clock.now
        return sum(self.clock.now - self.start >= t for t in self.times)


class Quiet(Sound):
    def __init__(self):
        super().__init__(CFG, enabled=False)
        self.said = []

    def say(self, text):
        self.said.append(text)


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    return simulate(CFG, "normal", tmp_path_factory.mktemp("sim") / "n", people=1, seed=90)


def session(sim, tmp_path, step, enter_times=()):
    clock = Clock()
    rec = Recorder(FakePort((sim / f"P01_{step}.bin").read_bytes(), clock), CFG, Quiet(), clock=clock)
    rec.enter = FakeEnter(clock, enter_times)
    return Session("P01", CFG, rec, rec.sound, data_dir=tmp_path, weight_kg=10.0, wait=lambda p: "")


def test_rest_recording_is_saved_named_and_checked(sim, tmp_path):
    ses = session(sim, tmp_path, "rest")
    result = ses.run("rest")
    assert (tmp_path / "P01_rest.bin").exists() and result["ok"]
    assert result["rest"]["hum_mv"] == pytest.approx(1.0, rel=0.4)
    meta = json.loads((tmp_path / "P01_rest.json").read_text())
    assert meta["recording"] == "rest" and meta["participant"] == "P01"


def test_set_recording_marks_the_end_on_the_recordings_clock(sim, tmp_path):
    truth = json.loads((sim / "P01_set1_truth.json").read_text())
    end_at = truth["stop_point_s"] + 2.0
    ses = session(sim, tmp_path, "set1", enter_times=[end_at])
    result = ses.run("set1")
    meta = json.loads((tmp_path / "P01_set1.json").read_text())
    marks = dict((label, t) for t, label in meta["markers"])
    assert marks["end"] == pytest.approx(end_at, abs=0.1)
    assert marks["sync_cue"] == pytest.approx(CFG["lab"]["still_s"], abs=0.1)
    assert meta["set"] == 1 and meta["weight_kg"] == 10.0 and meta["rest_before_s"] is None
    assert result["lifts_seen"] >= len(truth["reps"]) - 2 and result["ok"]


def test_set_3_needs_two_presses_stop_point_then_failure(sim, tmp_path):
    truth = json.loads((sim / "P01_set3_truth.json").read_text())
    ses = session(sim, tmp_path, "set3", enter_times=[truth["stop_point_s"], truth["failure_s"] + 1])
    ses.run("set3")
    labels = [lab for _, lab in json.loads((tmp_path / "P01_set3.json").read_text())["markers"]]
    assert "stop_called" in labels and "end" in labels


def test_pull_marks_and_loudness(sim, tmp_path):
    result = session(sim, tmp_path, "pull_base1").run("pull_base1")
    marks = dict((lab, t) for t, lab in json.loads((tmp_path / "P01_pull_base1.json").read_text())["markers"])
    assert marks["pull_start"] == pytest.approx(2.0, abs=0.1) and marks["pull_end"] == pytest.approx(5.0, abs=0.1)
    assert result["pull_rms_mv"] > 5


def test_steps_follow_the_protocol_and_rest_order(tmp_path):
    ses = Session("P02", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path)
    assert ses.next_steps() == STEPS
    (tmp_path / "P02_rest.bin").write_bytes(b"")
    assert ses.next_steps()[0] == "rest_alt"
    assert ses.rest_before(1) is None and ses.rest_before(2) == 180 and ses.rest_before(3) == 60


def test_sync_flicks_are_found_in_the_imu(sim):
    for k in (1, 2, 3):
        truth = json.loads((sim / f"P01_set{k}_truth.json").read_text())
        assert find_flicks(sim / f"P01_set{k}.bin", CFG) == pytest.approx(truth["sync_s"], abs=0.15)


def test_labels_from_video_times_land_on_the_recordings_clock(sim, tmp_path):
    for p in sim.glob("P01_*"):
        (tmp_path / p.name).write_bytes(p.read_bytes())
    sheet = pd.read_csv(template(tmp_path))
    video_offset = 12.7    # the camera started 12.7 s before... anything; only differences matter
    rows = []
    for k in (1, 2, 3):
        truth = json.loads((sim / f"P01_set{k}_truth.json").read_text())
        to_video = lambda t: t - truth["sync_s"] + video_offset  # noqa: E731
        for rater, err in (("rater1", 0.0), ("rater2", 0.4 if k < 3 else 4.5)):   # set 3: >1 rep apart
            rows.append({"recording": f"P01_set{k}", "rater": rater, "video_sync_s": video_offset,
                         "set_start_video_s": to_video(truth["set_start_s"]),
                         "stop_point_video_s": to_video(truth["stop_point_s"]) + err,
                         "failure_video_s": to_video(truth["failure_s"]) if truth["failure_s"] else None})
    pd.DataFrame(rows, columns=sheet.columns).to_csv(tmp_path / "labels.csv", index=False)
    report = apply(tmp_path, CFG)
    for k in (1, 2):
        truth = json.loads((sim / f"P01_set{k}_truth.json").read_text())
        lab = json.loads((tmp_path / f"P01_set{k}.json").read_text())["labels"]
        assert lab["stop_point_s"] == pytest.approx(truth["stop_point_s"] + 0.2, abs=0.2)  # mean of raters
        assert lab["set_start_s"] == pytest.approx(truth["set_start_s"], abs=0.2)
    s3 = report.set_index("recording").loc["P01_set3"]
    assert "disagree" in s3.status                                  # left for discussion
    assert "within one rep 67%" in agreement(report)
    loaded = load_sets(tmp_path)                                    # labelled sets load for scoring
    assert {s.name for s in loaded} >= {"P01_set1", "P01_set2"}


def test_unlabelled_sets_are_skipped_not_crashed_on(sim, tmp_path):
    for p in sim.glob("P01_set*"):
        (tmp_path / p.name).write_bytes(p.read_bytes())
    side = tmp_path / "P01_set2.json"
    meta = json.loads(side.read_text())
    meta["labels"] = {"set_start_s": None, "stop_point_s": None, "failure_s": None}
    side.write_text(json.dumps(meta))
    assert {s.name for s in load_sets(tmp_path)} == {"P01_set1", "P01_set3"}


def test_rest_alt_switches_the_rate_and_checks_the_board_took_it(sim, tmp_path):
    (tmp_path / "P01_rest.bin").write_bytes((sim / "P01_rest.bin").read_bytes())
    ses = session(sim, tmp_path, "rest")       # the simulated stream stays at 2,000/s: the board "ignored" it
    result = ses.run("rest_alt")
    assert ses.rec.port.written == [b"RATE 8400\n", b"RATE 8000\n"]
    assert any("RATE command" in p for p in result["problems"])


def test_alias_check_flags_peaks_that_move_between_rates():
    from pipeline.clean import alias_check
    from pipeline.session import Session
    rng = np.random.default_rng(5)
    t = np.arange(40000) / 2000.0

    def rest(alias_hz):
        x = 1440 + rng.normal(0, 3, len(t)) + 1.5 * np.sin(2 * np.pi * 60 * t) + 1.0 * np.sin(2 * np.pi * alias_hz * t)
        return Session(name="r", source="legacy", emg=x, emg_hz=2000.0)

    out = alias_check(rest(150.0), rest(250.0), CFG)    # a folded-in peak lands elsewhere at the other rate
    assert any(abs(f - 150) < 2 for f in out["moved_hz"]) and any(abs(f - 250) < 2 for f in out["moved_hz"])
    assert any(abs(f - 60) < 2 for f in out["stayed_hz"]) and out["problems"]
    same = alias_check(rest(150.0), rest(150.0), CFG)  # a real peak stays put: no aliasing problem
    assert not same["problems"]
