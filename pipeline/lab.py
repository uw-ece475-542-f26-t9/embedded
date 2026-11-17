"""Step 10: the lab kit's recorder. Records one protocol step at a time from the team firmware in
capture mode (CAPTURE_MODE 1), names it the protocol's way, runs the cues, and checks it on the spot.

    python -m pipeline.lab P01 --weight-kg 10          # walks through the whole session in order
    python -m pipeline.lab P01 --only rest             # one step

Steps, in protocol order (DATA_COLLECTION_PROTOCOL.md §2):
    rest -> pull_base1..3 -> set1 -> pull1 -> [rest] -> set2 -> pull2 -> [rest] -> set3 -> pull3 -> pull_recovery
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
from .clean import rest_check
from .config import ROOT, load_config
from .load import load_session

STEPS = ["rest", "pull_base1", "pull_base2", "pull_base3", "set1", "pull1", "set2", "pull2", "set3", "pull3",
         "pull_recovery"]


# ------------------------------------------------------------------ outside world


class Sound:
    """Metronome clicks and spoken cues (macOS afplay / say), fire-and-forget."""

    def __init__(self, cfg: dict, enabled: bool = True):
        self.c, self.enabled = cfg["lab"], enabled

    def beat(self, up: bool) -> None:
        if self.enabled:
            subprocess.Popen(["afplay", self.c["beat_sound"] if up else self.c["off_sound"]],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        start, host_start = self.clock(), self.wall()
        pending = sorted(cues, key=lambda c: c.at_s)
        host_marks, presses_seen = [], 0
        self.port.reset_input_buffer() if hasattr(self.port, "reset_input_buffer") else None
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
                        rate = float(STATUS_FMT.unpack_from(p)[1]) or rate
            while pending and pending[0].at_s <= now:
                pending.pop(0).action()
            presses = (enter.presses - base_presses) if enter else 0
            if marks and presses > presses_seen:
                for label in marks(now, presses)[presses_seen:presses]:
                    host_marks.append((now, label))
                presses_seen = presses
            if until(now, presses):
                break
        path.write_bytes(bytes(out))
        # Host times -> the recording's clock (EMG sample 0 of this file), via the EMG frames' arrivals.
        rec = Recording(path=path, host_start=host_start, duration_s=self.clock() - start)
        if arrivals:
            first_index = min(f for _, f, _ in arrivals)
            h = np.array([a for a, _, _ in arrivals])
            idx = np.array([e for _, _, e in arrivals], float)
            for t, label in host_marks + [(c, lab) for c, lab in getattr(self, "_cue_marks", [])]:
                rec.markers.append((round(float(np.interp(t, h, idx) - first_index) / rate, 3), label))
        self._cue_marks = []
        return rec

    def note(self, label: str, at: float) -> None:
        """A cue that also leaves a marker (e.g. when the sync flicks were asked for)."""
        self._cue_marks = getattr(self, "_cue_marks", []) + [(at, label)]


# ------------------------------------------------------------------ the protocol steps


def kind_of(step: str) -> str:
    return "rest" if step == "rest" else "set" if step.startswith("set") else "pull"


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
        self.checks: dict[str, dict] = {}

    def path(self, step: str) -> Path:
        return self.dir / f"{self.p}_{step}.bin"

    def next_steps(self) -> list[str]:
        return [s for s in STEPS if not self.path(s).exists()]

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
        rec.path.with_suffix(".json").write_text(json.dumps(meta, indent=1))
        result = self.check(step)
        self.checks[step] = result
        return result

    def _rest(self, step: str) -> Recording:
        self.sound.say("Rest recording. Forearm supported, palm up, hand limp. Keep still.")
        dur = self.c["rest_record_s"]
        return self.rec.record(self.path(step), [Cue(dur - 0.05, lambda: self.sound.say("Done"))],
                               until=lambda t, _: t >= dur)

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
            if presses >= presses_to_end and "t" not in ended:
                ended["t"] = t
                self.sound.say("Stop. Rack the weight." if n < 3 else "Done. Rack the weight.")
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
        kind = kind_of(step)
        if kind == "rest":
            rc = rest_check(s, self.cfg)
            out["problems"] = problems + rc.problems
            out["rest"] = {"hum_mv": rc.hum_mv, "noise_mv": rc.noise_mv, "lines": rc.lines}
        elif kind == "set" and len(s.imu_t):
            from .angle import forearm_angle
            from .features import find_lifts
            tr = forearm_angle(s, self.cfg)
            out["lifts_seen"] = len(find_lifts(tr, s.gyro, self.cfg))
            if np.isnan(tr.hang_known_s):
                out["problems"].append("no hanging posture found: was the arm still and hanging at the start?")
        elif kind == "pull":
            from .features import pull_loudness
            out["pull_rms_mv"] = round(pull_loudness(s, self.cfg), 1)
        out["ok"] = not out["problems"]
        return out

    def report(self, result: dict) -> None:
        print(("OK      " if result["ok"] else "REDO    ") + result["summary"])
        for p in result["problems"]:
            print(f"    problem: {p}")
        if "rest" in result:
            r = result["rest"]
            print(f"    rest: hum {r['hum_mv']} mV, noise {r['noise_mv']} mV, peaks {[hz for hz, _, _ in r['lines']]}")
        if "lifts_seen" in result:
            print(f"    lifts seen: {result['lifts_seen']}")
        if "pull_rms_mv" in result:
            print(f"    pull loudness: {result['pull_rms_mv']} mV")

    def countdown_rest(self, set_no: int, sleep=time.sleep) -> None:
        rest = self.rest_before(set_no)
        if not rest or set_no - 1 not in self.set_end_host:
            return
        left = rest - (time.time() - self.set_end_host[set_no - 1])
        while left > 0:
            if int(left) in (60, 30, 10):
                self.sound.say(f"{int(left)} seconds")
            sleep(min(1.0, left))
            left = rest - (time.time() - self.set_end_host[set_no - 1])

    def walk(self, only: str | None = None) -> None:
        steps = [only] if only else self.next_steps()
        for step in steps:
            kind = kind_of(step)
            if kind == "set":
                self.countdown_rest(int(step[3:]))
            if not (kind == "pull" and step[4:].isdigit()):   # the pull after a set starts on its own
                self.wait(f"\n[{self.p}] next: {step}. Press Enter to start (Ctrl-C to stop). ")
            else:
                self.sound.say("Pull in five seconds.")
                time.sleep(self.c["pull_after_set_s"])
            result = self.run(step)
            self.report(result)
            while not result["ok"]:
                if self.wait(f"  {step} has problems. Enter = redo it, or type 'keep' to keep it: ").strip() == "keep":
                    break
                result = self.run(step)
                self.report(result)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("participant", help="P01..P08")
    ap.add_argument("--only", choices=STEPS, help="record just this step")
    ap.add_argument("--weight-kg", type=float)
    ap.add_argument("--port")
    ap.add_argument("--quiet", action="store_true", help="no sounds")
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sound = Sound(cfg, enabled=not args.quiet)
    port = open_port(cfg, args.port)
    rec = Recorder(port, cfg, sound)
    rec.enter = EnterKey()
    Session(args.participant, cfg, rec, sound, weight_kg=args.weight_kg, wait=rec.enter.wait).walk(args.only)


if __name__ == "__main__":
    main()
