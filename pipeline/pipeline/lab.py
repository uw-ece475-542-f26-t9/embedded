"""Step 10: the lab kit's recorder. Records one protocol step at a time from the team firmware in
capture mode (CAPTURE_MODE 1), names it the protocol's way, runs the cues, and checks it on the spot.

Two scripts (DATA_COLLECTION_PROTOCOL.md §2):

    python -m pipeline.cold P01                        # 1. the 3 cold max pulls, as soon as they arrive
    python -m pipeline.lab P01 --weight-kg 10          # 2. everything else, in order
    python -m pipeline.lab P01 --only rest             # one step

    cold:  pull_base1..3 (60 s apart)
    main:  rest -> rest_alt (aliasing check) -> own_warmup (their own; not recorded) -> pull_warm1..3 (60 s
           apart) -> pace (metronome practice until Enter; not recorded) -> [2 min] -> set1 -> pull1 -> [rest]
           -> set2 -> pull2 -> [rest] -> set3 -> pull3 -> pull_recovery
100% = the best of all 6 max pulls (cold and warm).
Every rest shows a live counter (spoken at 60/30/10/5 s; type skip + Enter to end it). Sets start only on
'go' + Enter, so a stray Enter can't start one.
Each recording is <data_dir>/<P01>_<step>.bin (the raw stream, as received) plus .json (who, what,
cue markers on the recording's clock, empty labels for the video raters).

A set: arm still -> "flick three times" -> settle -> metronome (lift on one beat, lower on the next)
-> the operator presses Enter when the spotter calls the stop point (set 3: Enter at the stop point,
then Enter again at failure) -> a short tail -> done. The pull after it starts automatically, within
the protocol's 10 s. Rests between sets are timed from the end of a set, from the rest-order table.

Right after each recording it is loaded and checked: rest recordings get the hum and interference
check (fix the setup and redo before any set), sets get stream health, the IMU, and how many lifts were
seen, pulls get their loudness. The port is opened plainly (no DTR/RTS changes: on the ESP32-S3 those
reboot the board, learned in sensor testing).

Everything that touches hardware or time (port, clock, sounds, speech, the Enter key) is passed in,
so the tests drive the recorder with simulated byte streams.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .capture import EMG, STATUS, STATUS_FMT, EMG_HEAD, FrameReader
from .clean import alias_check, rest_check
from .config import ROOT, load_config
from .load import load_session

COLD_STEPS = ["pull_base1", "pull_base2", "pull_base3"]
MAIN_STEPS = ["rest", "rest_alt", "own_warmup", "pull_warm1", "pull_warm2", "pull_warm3", "pace", "set1", "pull1", "set2", "pull2", "set3", "pull3",
         "pull_recovery"]
STEPS = COLD_STEPS + MAIN_STEPS
NO_FILE = {"own_warmup": "pull_warm1", "pace": "set1"}   # steps that record nothing: due until this one is recorded


# ------------------------------------------------------------------ outside world


class Sound:
    """Metronome clicks and spoken cues (macOS afplay / say), fire-and-forget."""

    def __init__(self, cfg: dict, enabled: bool = True):
        self.c, self.enabled = cfg["lab"], enabled

    def beat(self, up: bool) -> None:
        if self.enabled:
            subprocess.Popen(["afplay", "-v", str(self.c.get("beat_volume", 1.0)),
                              self.c["beat_sound"] if up else self.c["off_sound"]],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def say_quiet(self, text: str) -> None:
        if self.enabled:
            subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def say(self, text: str) -> None:
        print(f"  >> {text}")
        if self.enabled:
            subprocess.Popen(["say", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class EnterKey:
    """The only reader of the keyboard: counts Enter presses in the background and keeps what was
    typed, so prompts and the mid-recording Enter never fight over a key press."""

    def __init__(self):
        self.presses, self.lines = 0, []
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        for line in sys.stdin:
            self.lines.append(line.strip())
            self.presses += 1

    def wait(self, prompt: str) -> str:
        print(prompt, end="", flush=True)
        seen = self.presses
        while self.presses == seen:
            time.sleep(0.02)
        return self.lines[-1]


def probe(port, seconds: float = 1.5) -> tuple[int, bytes]:
    """Listen briefly: (capture frames heard, a sample of anything else). Text and no frames means the
    normal firmware is on the board, not the capture build."""
    reader, other, frames, t0 = FrameReader(), bytearray(), 0, time.monotonic()
    while time.monotonic() - t0 < seconds:
        data = port.read(4096)
        if data:
            frames += len(reader.feed(data))
            other += data[:200] if not frames else b""
    return frames, bytes(other[:200])


def open_port(cfg: dict, port: str | None = None):
    import serial
    import serial.tools.list_ports

    if port is None:
        likely = [p for p in serial.tools.list_ports.comports() if p.vid in cfg["lab"]["usb_vids"]]
        if len(likely) != 1:
            names = ", ".join(p.device for p in serial.tools.list_ports.comports()) or "none"
            sys.exit(f"Couldn't pick the board's port (found: {names}); pass --port.")
        port = likely[0].device
    return serial.Serial(port, cfg["lab"]["baud"], timeout=0.05)  # plain open: no DTR/RTS changes


# ------------------------------------------------------------------ one recording


@dataclass
class Cue:
    at_s: float
    action: Callable[[], None]


@dataclass
class Recording:
    path: Path
    markers: list = field(default_factory=list)   # (s on the recording's clock, label)
    host_start: float = 0.0
    duration_s: float = 0.0
    interrupted: bool = False   # Ctrl-C during the recording: what came in so far is still saved


class Recorder:
    def __init__(self, port, cfg: dict, sound: Sound, clock: Callable[[], float] = time.monotonic,
                 wall: Callable[[], float] = time.time):
        self.port, self.cfg, self.sound, self.clock, self.wall = port, cfg, sound, clock, wall

    def record(self, path: Path, cues: list[Cue], until: Callable[[float, int], bool],
               marks: Callable[[float, int], list[str]] | None = None) -> Recording:
        """Record until until(seconds, enter_presses) is true, running cues on time. marks() turns new
        Enter presses into marker labels."""
        reader = FrameReader()
        arrivals = []                       # (host seconds, first EMG index of a frame, index after it)
        enter = getattr(self, "enter", None)
        base_presses = enter.presses if enter else 0
        rate = float(self.cfg["input"]["emg_hz"])
        out = bytearray()
        pending = sorted(cues, key=lambda c: c.at_s)
        host_marks = []
        self.port.reset_input_buffer() if hasattr(self.port, "reset_input_buffer") else None
        # Drop whatever piled up while waiting for Enter: it arrives as a burst of full reads. Found on
        # the board 2026-11-19: 3.5 s of old data at the start of a recording otherwise.
        t_drain = self.clock()
        while self.clock() - t_drain < 2.0:
            if len(self.port.read(4096)) < 4096 and self.clock() - t_drain >= 0.1:
                break
        start, host_start = self.clock(), self.wall()
        interrupted = False
        try:
            self._loop(reader, arrivals, enter, base_presses, out, pending, host_marks, start, until, marks)
        except KeyboardInterrupt:
            interrupted = True
            print("\n  Ctrl-C: stopped. Saving what was recorded.")
        rate = self._rate or rate
        path.write_bytes(bytes(out))
        # Host times -> the recording's clock (EMG sample 0 of this file), via the EMG frames' arrivals.
        rec = Recording(path=path, host_start=host_start, duration_s=self.clock() - start, interrupted=interrupted)
        if arrivals:
            first_index = min(f for _, f, _ in arrivals)
            h = np.array([a for a, _, _ in arrivals])
            idx = np.array([e for _, _, e in arrivals], float)
            for t, label in host_marks + [(c, lab) for c, lab in getattr(self, "_cue_marks", [])]:
                rec.markers.append((round(float(np.interp(t, h, idx) - first_index) / rate, 3), label))
        self._cue_marks = []
        return rec

    def _loop(self, reader, arrivals, enter, base_presses, out, pending, host_marks, start, until, marks):
        self._rate, presses_seen = None, 0
        while True:
            data = self.port.read(4096)
            now = self.clock() - start
            if data:
                out += data
                for ftype, p in reader.feed(data):
                    if ftype == EMG and len(p) >= EMG_HEAD.size:
                        first, n, _ = EMG_HEAD.unpack_from(p)
                        arrivals.append((now, first, first + n))
                    elif ftype == STATUS and len(p) >= STATUS_FMT.size:
                        self._rate = float(STATUS_FMT.unpack_from(p)[1]) or self._rate
            while pending and pending[0].at_s <= now:
                pending.pop(0).action()
            presses = (enter.presses - base_presses) if enter else 0
            if marks and presses > presses_seen:
                for label in marks(now, presses)[presses_seen:presses]:
                    host_marks.append((now, label))
                presses_seen = presses
            if until(now, presses):
                return

    def note(self, label: str, at: float) -> None:
        """A cue that also leaves a marker (e.g. when the sync flicks were asked for)."""
        self._cue_marks = getattr(self, "_cue_marks", []) + [(at, label)]


# ------------------------------------------------------------------ the protocol steps


def kind_of(step: str) -> str:
    if step in NO_FILE:
        return step
    return "rest" if step.startswith("rest") else "set" if step.startswith("set") else "pull"


class Session:
    def __init__(self, participant: str, cfg: dict, recorder: Recorder, sound: Sound,
                 data_dir: Path | None = None, weight_kg: float | None = None,
                 wait: Callable[[str], None] = lambda prompt: input(prompt)):
        self.p, self.cfg, self.rec, self.sound = participant, cfg, recorder, sound
        self.c = cfg["lab"]
        self.dir = Path(data_dir or ROOT / self.c["data_dir"])
        self.dir.mkdir(parents=True, exist_ok=True)
        self.weight_kg, self.wait = weight_kg, wait
        self.set_end_host: dict[int, float] = {}
        self.end_host: dict[str, float] = {}   # host time each step finished (rest counters time from it)
        self.checks: dict[str, dict] = {}

    def path(self, step: str) -> Path:
        return self.dir / f"{self.p}_{step}.bin"

    def next_steps(self, steps: list[str] | None = None) -> list[str]:
        """Steps not yet recorded. The warm-up and pace practice leave no file: each is due only while
        the recording after it isn't there."""
        return [s for s in (steps or MAIN_STEPS) if not self.path(NO_FILE.get(s, s)).exists()]

    def rest_before(self, set_no: int) -> int | None:
        order = self.c["rest_orders"].get(self.p)
        return order[set_no - 2] if order and set_no > 1 else None

    # one step
    def run(self, step: str) -> dict:
        kind = kind_of(step)
        if kind == "rest":
            rec = self._rest(step)
        elif kind == "pull":
            rec = self._pull(step)
        else:
            rec = self._set(step)
        meta = {"participant": self.p, "recording": kind, "step": step, "host_start": rec.host_start,
                "duration_s": round(rec.duration_s, 2), "markers": rec.markers,
                "labels": {"set_start_s": None, "stop_point_s": None, "failure_s": None}}
        if kind == "set":
            n = int(step[3:])
            meta.update(set=n, rest_before_s=self.rest_before(n), weight_kg=self.weight_kg,
                        rest_order=self.c["rest_orders"].get(self.p))
        if kind == "pull":
            meta["after_set"] = int(step[4:]) if step[4:].isdigit() else 0
        if rec.interrupted:
            meta["interrupted"] = True
        rec.path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        result = self.check(step)
        result["interrupted"] = rec.interrupted
        self.checks[step] = result
        return result

    def _rest(self, step: str) -> Recording:
        alt = step == "rest_alt"
        if alt:  # the aliasing check: the same rest at another ADC rate (firmware capture-mode command)
            self._command(f"RATE {self.c['alt_adc_hz']}")
        self.sound.say("Rest recording" + (" again" if alt else "") + ". Forearm supported, palm up, hand limp. Keep still.")
        dur = self.c["rest_record_s"]
        try:
            return self.rec.record(self.path(step), [Cue(dur - 0.05, lambda: self.sound.say("Done"))],
                                   until=lambda t, _: t >= dur)
        finally:
            if alt:
                self._command(f"RATE {self.c['adc_hz']}")

    def _command(self, text: str) -> None:
        self.rec.port.write((text + "\n").encode())

    def _pull(self, step: str) -> Recording:
        lead, pull = self.c["pull_lead_s"], self.c["pull_s"]
        cues = [Cue(0.0, lambda: self.sound.say("Elbow at a right angle. Ready.")),
                Cue(lead, lambda: self.sound.say("Pull!")),
                Cue(lead + pull, lambda: self.sound.say("Stop. Relax."))]
        self.rec.note("pull_start", lead)
        self.rec.note("pull_end", lead + pull)
        return self.rec.record(self.path(step), cues, until=lambda t, _: t >= 2 * lead + pull)

    def _set(self, step: str) -> Recording:
        n = int(step[3:])
        c = self.c
        go = c["still_s"] + c["sync_s"] + c["settle_s"]
        cues = [Cue(0.0, lambda: self.sound.say(f"Set {n}. Hold the dumbbell, arm hanging, still.")),
                Cue(c["still_s"], lambda: self.sound.say("Flick your wrist three times.")),
                Cue(c["still_s"] + c["sync_s"], lambda: self.sound.say("Settle.")),
                Cue(go - 0.6, lambda: self.sound.say("Go."))]
        half = c["cadence_s"] / 2
        cues += [Cue(go + k * half, (lambda up: lambda: self.sound.beat(up))(k % 2 == 0)) for k in range(400)]
        self.rec.note("sync_cue", c["still_s"])
        presses_to_end = 2 if n == 3 else 1   # set 3: stop point, then failure
        labels = ["stop_called", "end"] if n == 3 else ["end"]
        ended = {}

        def until(t, presses):
            if n == 3 and presses >= 1 and "stop" not in ended:
                ended["stop"] = t
                self.sound.say(f"Stop point marked at {t:.0f} seconds. Keep going to failure, then Enter.")
            if presses >= presses_to_end and "t" not in ended:
                ended["t"] = t
                self.sound.say(f"End marked at {t:.0f} seconds. Rack the weight." if n < 3 else
                               f"Failure marked at {t:.0f} seconds. Done. Rack the weight.")
            return "t" in ended and t >= ended["t"] + c["tail_s"]

        self.sound.say("Spotter: call the stop point. Operator: press Enter when it's called"
                       + (", then again at failure." if n == 3 else "."))
        rec = self.rec.record(self.path(step), cues, until, marks=lambda t, p: labels)
        self.set_end_host[n] = time.time()
        return rec

    # checks right after recording
    def check(self, step: str) -> dict:
        s = load_session(self.path(step), self.cfg)
        problems = s.problems(self.cfg)
        out = {"step": step, "problems": problems, "notes": s.notes(), "summary": s.summary()}
        if len(s.emg) == 0:
            out["problems"] = ["no data received from the board: is the capture-mode firmware on it, and is "
                               "it the right port?"]
            out["ok"] = False
            return out
        kind = kind_of(step)
        if kind == "rest":
            rc = rest_check(s, self.cfg)
            out["problems"] = problems + rc.problems
            out["rest"] = {"hum_mv": rc.hum_mv, "noise_mv": rc.noise_mv, "lines": rc.lines}
            if step == "rest_alt":
                expect = self.c["alt_adc_hz"] / 4
                if abs(s.emg_hz - expect) > 0.02 * expect:
                    out["problems"].append(f"recorded at {s.emg_hz:.0f} samples/s, expected {expect:.0f}: did the "
                                           "board take the RATE command? (needs the firmware with it)")
                elif self.path("rest").exists():
                    ac = alias_check(load_session(self.path("rest"), self.cfg), s, self.cfg)
                    out["problems"] += ac["problems"]
                    out["alias"] = ac
        elif kind == "set" and len(s.imu_t):
            from .angle import forearm_angle
            from .features import find_lifts
            tr = forearm_angle(s, self.cfg)
            out["lifts_seen"] = len(find_lifts(tr, s.gyro, self.cfg))
            from .labels import find_flicks
            out["flicks_s"] = find_flicks(self.path(step), self.cfg)
            if np.isnan(tr.hang_known_s):
                out["problems"].append("no hanging posture found: was the arm still and hanging at the start?")
        elif kind == "pull":
            from .features import pull_loudness
            out["pull_rms_mv"] = loud = round(pull_loudness(s, self.cfg), 1)
            # The floor: the rest recording's noise, or for the cold pulls (recorded before it) the
            # pull's own relaxed lead-in before "Pull!".
            from .features import pull_lead_noise
            floor = rest_check(load_session(self.path("rest"), self.cfg), self.cfg).noise_mv \
                if self.path("rest").exists() else round(pull_lead_noise(s, self.cfg), 2)
            if floor and loud < self.c["pull_min_ratio"] * floor:
                out["problems"].append(f"pull loudness {loud} mV is under {self.c['pull_min_ratio']:g}x the "
                                       f"resting noise ({floor} mV): the sensor isn't reading the muscle (pads not "
                                       "making contact yet?). Check with the meter, then redo")
        out["ok"] = not out["problems"]
        return out

    def report(self, result: dict) -> None:
        print(("OK      " if result["ok"] else "REDO    ") + result["summary"])
        for p in result["problems"]:
            print(f"    problem: {p}")
        if "rest" in result:
            r = result["rest"]
            print(f"    rest: hum {r['hum_mv']} mV, noise {r['noise_mv']} mV, peaks {[hz for hz, _, _ in r['lines']]}")
        if "alias" in result:
            a = result["alias"]
            print(f"    aliasing check: peaks that stayed {a['stayed_hz']}, peaks that moved {a['moved_hz'] or 'none'}")
        if "lifts_seen" in result:
            print(f"    lifts seen: {result['lifts_seen']}")
        if "flicks_s" in result:
            f = result["flicks_s"]
            print(f"    sync flicks at {f:.1f} s" if f is not None else
                  "    note: sync flicks not found (flick sharper next set; raters line this one up by the first lift)")
        if "pull_rms_mv" in result:
            print(f"    pull loudness: {result['pull_rms_mv']} mV")

    def rest_due(self, step: str) -> tuple[float, float, str] | None:
        """(seconds, host time the rest started, what it's for) before a step, or None. Rests count from
        the end of the step before: 60 s between max pulls, after their own warm-up, 2 min after pace
        practice, and the participant's rest order between sets."""
        if step in ("pull_base2", "pull_base3", "pull_warm2", "pull_warm3"):
            prev, secs, what = step[:-1] + str(int(step[-1]) - 1), self.c["pull_rest_s"], "rest between max pulls"
        elif step == "pull_warm1":
            prev, secs, what = "own_warmup", self.c["warmup_to_pull_s"], "rest after the warm-up"
        elif step == "set1":
            prev, secs, what = "pace", self.c["pace_rest_s"], "rest before set 1"
        elif step in ("set2", "set3"):
            prev, secs, what = f"set{int(step[3]) - 1}", self.rest_before(int(step[3])), f"rest before {step}"
        else:
            return None
        if not secs or prev not in self.end_host:
            return None
        return float(secs), self.end_host[prev], what

    def countdown(self, secs: float, since: float, what: str, sleep=time.sleep, now=time.time) -> None:
        """A rest counter. It prints a new line only at a few moments (start, 60/30/10 s left, done), never
        redrawing a line every second: a redraw wipes out whatever is being typed, so 'skip' couldn't be
        typed (pilot 2026-11-20). Spoken at 60/30/10/5 s. Typing 'skip' + Enter ends it early."""
        enter = getattr(self.rec, "enter", None)
        typed = len(enter.lines) if enter else 0
        spoken, shown = set(), set()
        left = secs - (now() - since)
        if left > 0:
            print(f"  {what}: {self._mmss(left)} (type skip + Enter to start early)", flush=True)
            shown.add(int(left + 0.999))
        while True:
            left = secs - (now() - since)
            if left <= 0:
                break
            if enter and any(x.lower() == "skip" for x in enter.lines[typed:]):
                print(f"  {what}: skipped with {left:.0f} s left.")
                return
            whole = int(left + 0.999)
            if whole in (60, 30, 10) and whole not in shown:
                shown.add(whole)
                print(f"  {what}: {self._mmss(whole)} left", flush=True)
            if whole in (60, 30, 10, 5) and whole not in spoken:
                spoken.add(whole)
                self.sound.say_quiet(f"{whole} seconds")
            sleep(min(0.25, max(left - (whole - 1), 0.05)))
        print(f"  {what}: done.")
        self.sound.say_quiet("Rest over.")

    @staticmethod
    def _mmss(secs: float) -> str:
        whole = int(secs + 0.999)
        return f"{whole // 60}:{whole % 60:02d}"

    def own_warmup(self) -> None:
        """Their own warm-up, between the cold and the warm pulls. Not recorded and not timed."""
        self.sound.say("Own warm-up: warm up the right arm the way you normally would.")
        self.wait("  Press Enter when the warm-up is done. ")
        self.end_host["own_warmup"] = time.time()

    def pace(self) -> None:
        """Pace practice before set 1 (not recorded): curls to the same metronome as the sets, only to
        learn the speed. The beat keeps going until Enter. Asks for the dumbbell weight first if it wasn't
        given (load selection comes just before this, after the warm pulls)."""
        while self.weight_kg is None:
            text = self.wait("  Dumbbell weight for the sets (kg): ").strip().replace(",", ".")
            try:
                kg = float(text)
                if 0 < kg < 100:
                    self.weight_kg = kg
                    continue
            except ValueError:
                pass
            print("  (type the dumbbell's weight, e.g. 7.5)")
        self.sound.say("Pace practice: light curls to the beat. Lift on the high beat, lower on the low one.")
        stop = threading.Event()

        def beats():
            k, half = 0, self.c["cadence_s"] / 2
            while not stop.wait(half if k else 1.5):   # first beat 1.5 s after the cue
                self.sound.beat(k % 2 == 0)
                k += 1
        threading.Thread(target=beats, daemon=True).start()
        try:
            self.wait("  Press Enter to stop the metronome (the 2 min rest before set 1 starts then). ")
        finally:
            stop.set()
        self.end_host["pace"] = time.time()

    def ask_force(self, step: str) -> float | None:
        """The luggage scale's peak for a pull, typed by the operator, saved in the pull's .json as
        force_kg (kilogram-force; only ratios are used). 'none' if there's no reading."""
        while True:
            text = self.wait(f"  Scale peak for {step} (kg, or 'none'): ").strip().lower().replace(",", ".")
            if text == "none":
                kg = None
                break
            try:
                kg = float(text)
            except ValueError:
                print("  (type the number on the scale, e.g. 18.5, or 'none')")
                continue
            if 0 < kg < 200:
                break
            print("  (that doesn't look like a scale reading: 0-200 kg)")
        meta_path = self.path(step).with_suffix(".json")
        meta = json.loads(meta_path.read_text())
        meta["force_kg"] = kg
        meta_path.write_text(json.dumps(meta, indent=1))
        return kg

    def wait_for_go(self, step: str) -> None:
        """Sets start only on 'go': a stray Enter (left over from a pull or a rest) must not start one."""
        while self.wait(f"\n[{self.p}] next: {step}. Type go + Enter to start (Ctrl-C to stop). ").strip().lower() != "go":
            print("  (not started: type go)")

    def walk(self, only: str | None = None, steps: list[str] | None = None) -> None:
        steps = [only] if only else self.next_steps(steps)
        for step in steps:
            kind = kind_of(step)
            due = self.rest_due(step)
            if due:
                self.countdown(*due)
            if kind in NO_FILE:
                getattr(self, kind)()
                continue
            if kind == "set":
                self.wait_for_go(step)
            elif not (kind == "pull" and step[4:].isdigit()):   # the pull after a set starts on its own
                self.wait(f"\n[{self.p}] next: {step}. Press Enter to start (Ctrl-C to stop). ")
            else:
                self.sound.say("Pull in five seconds.")
                time.sleep(self.c["pull_after_set_s"])
            result = self.run(step)
            self.report(result)
            if result["interrupted"]:
                print(f"  Saved {self.path(step).name} (cut short). Session stopped; run the same command to "
                      "carry on from the next step, or delete that file to redo it.")
                return
            while not result["ok"]:
                if self.wait(f"  {step} has problems. Enter = redo it, or type 'keep' to keep it: ").strip() == "keep":
                    break
                result = self.run(step)
                self.report(result)
            self.end_host[step] = time.time()   # the rest counts from the end of the recording,
            if kind == "pull":                  # not from when the scale reading was typed
                self.ask_force(step)


def main(steps: list[str] | None = None, what: str = "the session") -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("participant", help="P01..P08")
    ap.add_argument("--only", choices=steps or STEPS, help="record just this step")
    ap.add_argument("--weight-kg", type=float)
    ap.add_argument("--port")
    ap.add_argument("--quiet", action="store_true", help="no sounds")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sound = Sound(cfg, enabled=not args.quiet)
    port = open_port(cfg, args.port)
    frames, other = probe(port)
    if not frames:
        hint = ("it's sending text, so the normal firmware is on it: flash the capture build (CAPTURE_MODE=1)"
                if other.strip() else "it's sending nothing: check the cable, the port, and that it's powered")
        sys.exit(f"No capture frames from the board on {port.port}: {hint}.")
    rec = Recorder(port, cfg, sound)
    rec.enter = EnterKey()
    try:
        Session(args.participant, cfg, rec, sound, weight_kg=args.weight_kg,
                wait=rec.enter.wait).walk(args.only, steps or MAIN_STEPS)
        print(f"\n  {what} done.")
    except KeyboardInterrupt:   # outside a recording (a prompt or a rest counter): nothing to save
        print("\n  Stopped. Run the same command to carry on from the next step.")


if __name__ == "__main__":
    main()
