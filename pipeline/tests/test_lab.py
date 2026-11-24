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
from pipeline import lab
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
    assert ses.next_steps() == lab.MAIN_STEPS
    (tmp_path / "P02_rest.bin").write_bytes(b"")
    assert ses.next_steps()[0] == "rest_alt"
    assert ses.rest_before(1) is None and ses.rest_before(2) == 180 and ses.rest_before(3) == 60


def test_set_3_says_what_each_press_marked(sim, tmp_path):
    truth = json.loads((sim / "P01_set3_truth.json").read_text())
    ses = session(sim, tmp_path, "set3", enter_times=[truth["stop_point_s"], truth["failure_s"] + 1])
    ses.run("set3")
    said = " | ".join(ses.sound.said)
    assert "Stop point marked" in said and "Keep going to failure" in said and "Failure marked" in said
    assert said.index("Stop point marked") < said.index("Failure marked")


def test_ctrl_c_mid_recording_keeps_what_was_recorded(sim, tmp_path):
    ses = session(sim, tmp_path, "set3", enter_times=[10.0])

    def ctrl_c():
        raise KeyboardInterrupt
    real = ses.rec.record
    ses.rec.record = lambda path, cues, *a, **k: real(path, cues + [lab.Cue(20.0, ctrl_c)], *a, **k)
    result = ses.run("set3")
    meta = json.loads((tmp_path / "P01_set3.json").read_text())
    assert result["interrupted"] and meta["interrupted"]
    assert meta["duration_s"] == pytest.approx(20.0, abs=0.2)
    assert "stop_called" in [label for _, label in meta["markers"]]   # the stop point mark survives
    assert (tmp_path / "P01_set3.bin").stat().st_size > 0


def test_two_scripts_cold_pulls_then_the_rest(tmp_path):
    from pipeline.lab import COLD_STEPS, MAIN_STEPS
    assert COLD_STEPS == ["pull_base1", "pull_base2", "pull_base3"] and not set(COLD_STEPS) & set(MAIN_STEPS)
    assert MAIN_STEPS[:8] == ["rest", "rest_alt", "own_warmup", "pull_warm1", "pull_warm2", "pull_warm3", "pace", "set1"]
    ses = Session("P02", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path)
    assert ses.next_steps() == MAIN_STEPS and ses.next_steps(COLD_STEPS) == COLD_STEPS


def test_no_file_steps_are_due_only_until_the_next_recording(tmp_path):
    ses = Session("P02", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path)
    (tmp_path / "P02_pull_warm1.bin").write_bytes(b"")
    assert "own_warmup" not in ses.next_steps() and "pace" in ses.next_steps()
    (tmp_path / "P02_set1.bin").write_bytes(b"")
    assert "pace" not in ses.next_steps()


def test_rest_counters_time_from_the_step_before(tmp_path):
    ses = Session("P02", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path)
    assert ses.rest_due("pull_base2") is None            # nothing recorded this run yet: no counter
    ses.end_host.update(pull_base1=100.0, own_warmup=150.0, pull_warm2=170.0, pace=200.0, set1=300.0, set2=400.0)
    assert ses.rest_due("pull_base2") == (CFG["lab"]["pull_rest_s"], 100.0, "rest between max pulls")
    w = CFG["lab"]["warmup_to_pull_s"]          # 0 = straight from their warm-up into the warm pulls
    assert ses.rest_due("pull_warm1") is None if not w else ses.rest_due("pull_warm1")[:2] == (w, 150.0)
    assert ses.rest_due("pull_warm3")[:2] == (CFG["lab"]["pull_rest_s"], 170.0)
    assert ses.rest_due("set1")[:2] == (CFG["lab"]["pace_rest_s"], 200.0)
    assert ses.rest_due("set2")[:2] == (180, 300.0) and ses.rest_due("set3")[:2] == (60, 400.0)  # P02: 180 then 60
    assert ses.rest_due("pull1") is None and ses.rest_due("rest") is None


class TypedLines:
    def __init__(self, lines=()):
        self.lines, self.presses = list(lines), len(lines)


class RecWithKeys:
    def __init__(self, enter):
        self.enter = enter


def test_countdown_runs_to_zero_and_speaks(tmp_path, capsys):
    clock = [0.0]
    ses = Session("P02", CFG, recorder=RecWithKeys(TypedLines()), sound=Quiet(), data_dir=tmp_path)
    ses.sound.say_quiet = ses.sound.said.append
    ses.countdown(65, 0.0, "rest", sleep=lambda dt: clock.__setitem__(0, clock[0] + dt), now=lambda: clock[0])
    assert clock[0] == pytest.approx(65, abs=0.1)
    assert ses.sound.said == ["60 seconds", "30 seconds", "10 seconds", "5 seconds", "Rest over."]
    assert "rest: done" in capsys.readouterr().out


def test_countdown_can_be_skipped(tmp_path):
    clock, enter = [0.0], TypedLines()
    ses = Session("P02", CFG, recorder=RecWithKeys(enter), sound=Quiet(), data_dir=tmp_path)
    ses.sound.say_quiet = lambda t: None

    def sleep(dt):
        clock[0] += dt
        if clock[0] >= 3:
            enter.lines.append("skip")
    ses.countdown(120, 0.0, "rest", sleep=sleep, now=lambda: clock[0])
    assert clock[0] < 5


def test_pace_practice_plays_the_metronome_until_enter(tmp_path):
    import time as _t
    ses = Session("P02", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path, weight_kg=10.0,
                  wait=lambda p: _t.sleep(2.7))
    beats = []
    ses.sound.beat = beats.append
    ses.pace()
    n = len(beats)
    _t.sleep(1.2)
    assert beats[:2] == [True, False] and len(beats) == n     # up, down ... and silent once Enter came
    assert "pace" in ses.end_host


def test_sets_start_only_on_go(tmp_path):
    typed = iter(["", "", "yes", "go"])
    ses = Session("P02", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path, wait=lambda p: next(typed))
    ses.wait_for_go("set1")
    assert next(typed, "used up") == "used up"     # stray Enters and other words were ignored until go


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


# ---- pilot 2026-11-20 fixes: pull window, dead-sensor pulls, gentle flicks

from pipeline.capture import EMG as _EMG, IMU as _IMU, STATUS as _STATUS, emg_payload, imu_payload, status_payload
from pipeline.capture import frame as _frame


def _write_capture(path, emg_mv, gyro_dps, markers, hz=2000, imu_hz=50):
    """A capture .bin (+ .json markers) from EMG in mV and gyro rows at imu_hz; arm hanging (accel = 1 g down)."""
    out = bytearray(_frame(_STATUS, status_payload(0, 0)))
    for i in range(0, len(emg_mv), 32):
        out += _frame(_EMG, emg_payload(i, emg_mv[i:i + 32]))
    step = hz // imu_hz
    for k, g in enumerate(gyro_dps):
        out += _frame(_IMU, imu_payload(k * step, [0.0, 0.0, -1.0], g))
    path.write_bytes(bytes(out))
    path.with_suffix(".json").write_text(json.dumps({"markers": markers}))
    return path


BAND = np.sqrt((450 - 20) / 1000)   # white noise at 2 kHz keeps this share of its RMS in the 20-450 Hz band


def _noise(seconds, mv, rng, hz=2000):
    return 1300 + rng.normal(0, mv, int(seconds * hz))


def test_pull_loudness_ignores_activity_outside_the_pull(tmp_path):
    from pipeline.features import pull_loudness
    from pipeline.load import load_session
    rng = np.random.default_rng(0)
    x = _noise(9, 3.0, rng)
    x[int(2.0 * 2000):int(5.0 * 2000)] += rng.normal(0, 30, 3 * 2000)      # the pull: 30 mV
    x[int(6.5 * 2000):int(8.5 * 2000)] += rng.normal(0, 80, 2 * 2000)      # late, 1.5 s after "Stop": not counted
    p = _write_capture(tmp_path / "P01_pull1.bin", x, np.zeros((450, 3)), [[2.0, "pull_start"], [5.0, "pull_end"]])
    assert pull_loudness(load_session(p, CFG), CFG) == pytest.approx(30 * BAND, rel=0.15)


def test_pull_counts_one_second_past_stop(tmp_path):
    from pipeline.features import pull_loudness
    from pipeline.load import load_session
    rng = np.random.default_rng(1)
    x = _noise(9, 3.0, rng)
    x[int(4.6 * 2000):int(5.9 * 2000)] += rng.normal(0, 50, int(1.3 * 2000))   # a late pull, mostly after "Stop"
    p = _write_capture(tmp_path / "P01_pull2.bin", x, np.zeros((450, 3)), [[2.0, "pull_start"], [5.0, "pull_end"]])
    assert pull_loudness(load_session(p, CFG), CFG) == pytest.approx(50 * BAND, rel=0.15)


def test_a_pull_with_no_muscle_signal_is_redone(tmp_path):
    rng = np.random.default_rng(2)
    _write_capture(tmp_path / "P01_rest.bin", _noise(10, 3.0, rng), np.zeros((500, 3)), [])
    _write_capture(tmp_path / "P01_pull_base1.bin", _noise(7, 3.0, rng), np.zeros((350, 3)),
                   [[2.0, "pull_start"], [5.0, "pull_end"]])
    ses = Session("P01", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path)
    r = ses.check("pull_base1")
    assert not r["ok"] and "isn't reading the muscle" in r["problems"][0]


def test_gentle_flicks_after_the_cue_are_found(tmp_path):
    from pipeline.labels import find_flicks
    rng = np.random.default_rng(3)
    t = np.arange(0, 8, 1 / 50)
    g = rng.normal(0, 3, (len(t), 3))                       # hanging still: a few deg/s
    for t0, peak in [(4.0, 75), (4.3, 60), (4.6, 70)]:      # flicks with a dumbbell: 60-75 deg/s
        g[:, 0] += peak * np.exp(-0.5 * ((t - t0) / 0.04) ** 2)
    p = _write_capture(tmp_path / "P01_set1.bin", _noise(8, 3.0, rng), g, [[3.0, "sync_cue"]])
    assert find_flicks(p, CFG) == pytest.approx(4.0, abs=0.05)
    p2 = _write_capture(tmp_path / "P01_set2.bin", _noise(8, 3.0, rng), g, [])   # no cue: strict 90 deg/s
    assert find_flicks(p2, CFG) is None


def test_cold_pull_with_no_muscle_signal_is_caught_without_a_rest_recording(tmp_path):
    rng = np.random.default_rng(4)        # cold pulls come first: the floor is the pull's own lead-in
    _write_capture(tmp_path / "P01_pull_base1.bin", _noise(7, 3.0, rng), np.zeros((350, 3)),
                   [[2.0, "pull_start"], [5.0, "pull_end"]])
    x = _noise(7, 3.0, rng)
    x[4000:10000] += rng.normal(0, 40, 6000)
    _write_capture(tmp_path / "P01_pull_base2.bin", x, np.zeros((350, 3)), [[2.0, "pull_start"], [5.0, "pull_end"]])
    ses = Session("P01", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path)
    assert not ses.check("pull_base1")["ok"] and ses.check("pull_base2")["ok"]


def test_100_percent_is_the_best_of_cold_and_warm_pulls(tmp_path):
    from pipeline.features import max_pull_paths
    for name in ("P01_pull_base1", "P01_pull_base3", "P01_pull_warm2", "P01_pull1", "P02_pull_warm1"):
        (tmp_path / f"{name}.bin").write_bytes(b"")
    assert [p.stem for p in max_pull_paths(tmp_path, "P01")] == ["P01_pull_base1", "P01_pull_base3", "P01_pull_warm2"]


def test_scale_reading_is_asked_and_saved(tmp_path):
    (tmp_path / "P01_pull1.json").write_text(json.dumps({"step": "pull1"}))
    typed = iter(["", "abc", "500", "18,5"])
    ses = Session("P01", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path, wait=lambda p: next(typed))
    assert ses.ask_force("pull1") == 18.5            # blank, words and impossible numbers are asked again
    assert json.loads((tmp_path / "P01_pull1.json").read_text())["force_kg"] == 18.5
    ses.wait = lambda p: "none"
    assert ses.ask_force("pull1") is None and json.loads((tmp_path / "P01_pull1.json").read_text())["force_kg"] is None


def test_report_pull_table_uses_the_best_of_6_and_shows_the_drop(tmp_path):
    from pipeline.report import pull_table
    rng = np.random.default_rng(5)
    for step, kg, mv in [("pull_base1", 20, 30), ("pull_base2", 22, 32), ("pull_base3", 21, 31),
                         ("pull_warm1", 24, 36), ("pull_warm2", 25, 40), ("pull_warm3", 23, 38), ("pull1", 20, 30)]:
        x = _noise(7, 3.0, rng)
        x[4000:10000] += rng.normal(0, mv / BAND, 6000)
        p = _write_capture(tmp_path / f"P01_{step}.bin", x, np.zeros((350, 3)), [[2.0, "pull_start"], [5.0, "pull_end"]])
        meta = json.loads(p.with_suffix(".json").read_text())
        meta["force_kg"] = kg
        p.with_suffix(".json").write_text(json.dumps(meta))
    t = pull_table(tmp_path, CFG).set_index("measure")
    assert t.loc["force", "100%"] == "25.0 kg" and t.loc["force", "warm / cold"] == "114%"   # 25 / 22
    assert t.loc["force", "pull1"] == "80%"                                                 # 20 / 25
    assert t.loc["EMG", "pull1"] in ("73%", "74%", "75%", "76%", "77%", "78%")              # about 30 / 40


def test_pace_asks_for_the_weight_when_it_wasnt_given(tmp_path):
    import time as _t
    typed = iter(["", "heavy", "7,5", "enter"])

    def wait(prompt):
        if "metronome" in prompt:
            _t.sleep(0.1)
        return next(typed)
    ses = Session("P02", CFG, recorder=None, sound=Quiet(), data_dir=tmp_path, wait=wait)
    ses.sound.beat = lambda up: None
    ses.pace()
    assert ses.weight_kg == 7.5


def test_sets_without_flicks_line_up_on_the_first_lift_and_raters_have_own_sheets(sim, tmp_path, monkeypatch):
    import pipeline.labels as labels
    for p in sim.glob("P01_set1*"):
        (tmp_path / p.name).write_bytes(p.read_bytes())
    truth = json.loads((sim / "P01_set1_truth.json").read_text())
    monkeypatch.setattr(labels, "find_flicks", lambda *a, **k: None)      # flicks missed (7 of 12 real sets)
    lift0 = labels.first_lift_s(tmp_path / "P01_set1.bin", CFG)
    assert lift0 == pytest.approx(truth["set_start_s"], abs=0.2)
    for rater, err in (("ana", 0.0), ("ben", 0.4)):
        out = template(tmp_path, (rater,))
        assert out.name == f"labels_{rater}.csv"
        sheet = pd.read_csv(out)
        to_video = lambda t: t - truth["set_start_s"] + 30.0  # noqa: E731
        sheet.loc[0, ["set_start_video_s", "stop_point_video_s"]] = [to_video(truth["set_start_s"]),
                                                                    to_video(truth["stop_point_s"]) + err]
        sheet.to_csv(out, index=False)
    report = labels.apply(tmp_path, CFG)
    assert report.iloc[0].sync == "first lift"
    lab = json.loads((tmp_path / "P01_set1.json").read_text())["labels"]
    assert lab["stop_point_s"] == pytest.approx(truth["stop_point_s"] + 0.2, abs=0.3)
