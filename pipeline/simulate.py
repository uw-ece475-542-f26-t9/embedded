"""Step 2: fake sessions shaped like the protocol, with the right answers known.

Each fake person runs the protocol: a rest recording, 3 max pulls, then sets 1-3 at the
metronome pace with 60/180 s rests in their order and a pull after each set. Files come out in
the team firmware's capture format (so they go through step 1's loader), plus a truth file.

How fatigue works here:
  - One hidden tiredness level F per person: 0 fresh, stop_level when form breaks, 1 at failure.
    Each rep adds to it; it decays during rest (recovery_tau_s), so a 60 s rest leaves the next
    set starting more tired than a 180 s rest.
  - With x = F / stop_level: middle pitch falls (mdf_drop at x = 1, and keeps falling), loudness
    rises (loud_rise at x = 1, then levels off), lifts slow down (speed_loss * x^2).
  - EMG and speed follow the actual tiredness (x uses the average stop_level). Form breaks at
    each person's own stop level (stop_level_spread), so a person who breaks late shows a
    bigger middle-pitch drop by their stop point.
  - The stop point is the first rep with F >= the person's stop level. It is a bad rep (partial, swing or
    upper-arm drift). Sets 1-2 end there. Set 3 carries on until F >= 1, the failed rep.
  - Effort surges (a hard or fast rep) push middle pitch and loudness up together: not fatigue.

The EMG is random noise shaped to a chosen middle pitch (an EMG-like spectrum, time-warped so the
middle pitch can change smoothly), times a loudness envelope that follows the lift, plus
background noise, mains hum, optional USB packet noise and a motion jolt at each turn.

This proves the pipeline works. It says nothing about real muscle: the simulator contains our
own assumptions. The test worlds in config.toml ([sim.worlds]) are there to catch a pipeline
that only works because of them.

    python -m pipeline.simulate --world normal --out sim/normal
    python -m pipeline.simulate --all-worlds --out sim
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation

from .capture import EMG, IMU, STATUS, emg_payload, frame, imu_payload, status_payload
from .config import load_config

NOMINAL_HZ = 2000
IMU_HZ = 50
FULL_SCALE_MV = 3100
G = 9.81

# ------------------------------------------------------------------ EMG-like noise

REF_MDF = 60.0     # middle pitch of the base noise
BASE_HZ = 8000     # base noise is made 4x faster than the output so warping it stays accurate
BAND = (20.0, 450.0)


def emg_psd(f: np.ndarray, fh: float) -> np.ndarray:
    """A standard surface-EMG spectrum shape (rises from 0, peaks, falls off)."""
    fl = 0.5 * fh
    return fh**4 * f**2 / ((f**2 + fl**2) * (f**2 + fh**2) ** 2)


def mdf_of(f: np.ndarray, p: np.ndarray, band=BAND) -> float:
    """Middle pitch: the frequency splitting the in-band power in half."""
    keep = (f >= band[0]) & (f <= band[1])
    c = np.cumsum(p[keep])
    return float(f[keep][np.searchsorted(c, c[-1] / 2)])


def _ref_fh() -> float:
    f = np.linspace(0, 1000, 20001)
    lo, hi = 10.0, 500.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if mdf_of(f, emg_psd(f, mid)) < REF_MDF else (lo, mid)
    return (lo + hi) / 2


REF_FH = _ref_fh()


def _warp_table() -> tuple[np.ndarray, np.ndarray]:
    """Warp factor -> in-band middle pitch. Not proportional: as the spectrum slides, power
    crosses the 20 Hz band edge (asking for 36 Hz by proportion would give 40.7 Hz). So the
    warp for a wanted middle pitch is looked up in this table instead."""
    f = np.linspace(0, 4000, 40001)
    scales = np.linspace(0.15, 3.0, 300)
    return scales, np.array([mdf_of(f, emg_psd(f / s, REF_FH)) for s in scales])


WARP_SCALES, WARP_MDF = _warp_table()


def warp_for(mdf_hz: np.ndarray) -> np.ndarray:
    return np.interp(mdf_hz, WARP_MDF, WARP_SCALES)


def shaped_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-RMS noise at BASE_HZ whose spectrum has middle pitch REF_MDF."""
    x = np.fft.rfft(rng.standard_normal(n))
    x *= np.sqrt(emg_psd(np.fft.rfftfreq(n, 1 / BASE_HZ), REF_FH))
    y = np.fft.irfft(x, n)
    return y / y.std()


def synth_emg(t: np.ndarray, mdf_hz: np.ndarray, rms_mv: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """EMG whose middle pitch follows mdf_hz and loudness follows rms_mv, sample by sample.
    Playing the base noise faster scales every frequency in it, so a time warp sets the pitch."""
    dt = np.diff(t, prepend=t[0] - (t[1] - t[0]))
    tau = np.cumsum(warp_for(mdf_hz) * dt) * BASE_HZ
    base = shaped_noise(int(tau[-1]) + 16, rng)
    return rms_mv * np.interp(tau, np.arange(len(base)), base)


# ------------------------------------------------------------------ people and plans


def sim_config(cfg: dict, world: str) -> dict:
    worlds = cfg["sim"].get("worlds", {})
    if world not in worlds:
        raise ValueError(f"unknown world {world!r}; choose from {sorted(worlds)}")
    c = {k: v for k, v in cfg["sim"].items() if k != "worlds"}
    c.update(worlds[world])
    c["world"] = world
    return c


@dataclass
class Person:
    pid: str
    reps_fresh: float      # reps to the stop point from fresh
    stop_level: float      # tiredness at which this person's form breaks
    mdf0: float            # fresh middle pitch, Hz
    mdf_drop: float
    loud_rise: float
    scale: float           # skin/electrode loudness scale
    clock_hz: float        # the board's real EMG rate
    strength_kg: float
    tremor_hz: float       # this person's physiological tremor frequency
    rest_order: tuple
    mount: np.ndarray      # IMU mount rotation (3x3)

    def truth(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "mount"}
        d["mount_tilt_deg"] = float(np.degrees(Rotation.from_matrix(self.mount).magnitude()))
        return d


def make_person(i: int, c: dict, rng: np.random.Generator) -> Person:
    spread = c["fatigue_spread"]
    tilt = rng.normal(size=3)
    tilt *= np.radians(abs(rng.normal(0, c["mount_tilt_deg"]))) / np.linalg.norm(tilt)
    order = c["rest_orders"][i % len(c["rest_orders"])]
    return Person(
        pid=f"P{i + 1:02d}",
        reps_fresh=c["reps_to_stop_fresh"] * float(np.exp(rng.normal(0, c["reps_to_stop_spread"]))),
        stop_level=min(0.95, c["stop_level"] * float(np.exp(rng.normal(0, c["stop_level_spread"])))),
        mdf0=float(rng.uniform(*c["mdf_fresh_hz"])),
        mdf_drop=c["mdf_drop"] * float(np.exp(rng.normal(0, spread))),
        loud_rise=c["loud_rise"] * float(np.exp(rng.normal(0, spread))),
        scale=float(np.exp(rng.normal(0, c["loudness_scale_spread"]))),
        clock_hz=NOMINAL_HZ * (1 + float(rng.normal(0, c["board_clock_spread"]))),
        strength_kg=float(rng.uniform(15, 30)),
        tremor_hz=float(rng.uniform(*c["tremor_hz"])),
        rest_order=tuple(order),
        mount=Rotation.from_rotvec(tilt).as_matrix(),
    )


@dataclass
class Rep:
    n: int
    F: float            # tiredness at the start of the rep
    kind: str           # good, partial, swing, drift, failed
    effort: float
    lift_s: float
    lower_s: float
    hold_s: float
    pause_s: float
    rom_deg: float
    swing_deg: float
    drift_deg: float
    jitter: float       # middle pitch wobble for this rep
    start_s: float = 0.0  # lift start, filled in when rendered


def plan_set(p: Person, set_no: int, F0: float, c: dict, rng: np.random.Generator) -> tuple[list[Rep], float]:
    """Reps of one set, and the tiredness added per rep. Tiredness builds at the same pace for a
    given endurance; where form breaks is the person's own stop_level."""
    dF = c["stop_level"] / p.reps_fresh if p.reps_fresh > 0 else 0.0
    reps, F, stopped = [], F0, False
    for n in range(1, 61):
        x = F / c["stop_level"]
        if dF == 0:
            kind = "good"
        elif F >= 1.0:
            kind = "failed"
        elif F >= p.stop_level:
            kind = str(rng.choice(["partial", "swing", "drift"]))
        else:
            kind = "good"
        effort = float(rng.uniform(0.5, 1) * c["effort_size"]) if rng.random() < c["effort_prob"] else 0.0
        speed = max(0.4, 1 - c["speed_loss"] * min(x, 1.25) ** 2) * (1 + 0.5 * effort)
        lift = c["lift_s"] / speed * (1.3 if kind != "good" else 1) * (2 if kind == "failed" else 1)
        rom = c["rom_deg"] * (1 - 0.08 * float(np.clip((x - 0.8) / 0.2, 0, 1)))
        rom *= {"partial": c["partial_rom"], "failed": 0.4}.get(kind, 1.0)
        swing = c["pre_stop_swing_deg"] * float(np.clip((x - 0.6) / 0.4, 0, 1)) ** 2
        swing = c["bad_swing_deg"] if kind == "swing" else swing
        hold, lower = 0.15, 0.8
        reps.append(Rep(n=n, F=F, kind=kind, effort=effort, lift_s=lift, lower_s=lower, hold_s=hold,
                        pause_s=max(0.1, c["cadence_s"] - lift - hold - lower), rom_deg=rom,
                        swing_deg=swing, drift_deg=c["bad_drift_deg"] if kind == "drift" else 0.0,
                        jitter=float(rng.normal(0, c["mdf_rep_jitter"]))))
        F += dF
        if kind == "failed" or (dF == 0 and n >= c["effort_world_reps"]):
            break
        if kind != "good" and not stopped:
            stopped = True
            if set_no < 3:
                break
    return reps, dF


# ------------------------------------------------------------------ rendering


def min_jerk(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0, 1)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


@dataclass
class Track:
    """Everything about one recording on the EMG sample grid (real time)."""
    t: np.ndarray
    elbow: np.ndarray          # forearm angle from hanging from the elbow alone, deg
    body: np.ndarray           # extra angle from body swing / upper-arm drift, deg
    twist_dps: np.ndarray      # forearm twist rate (the sync flicks)
    activity: np.ndarray       # loudness as a share of max-pull loudness, before fatigue
    lifting: np.ndarray        # True while lifting or holding at the top
    F: np.ndarray
    effort: np.ndarray
    jitter: np.ndarray
    events: list = field(default_factory=list)  # (t, kind) where a motion jolt happens


def _empty_track(duration: float, clock_hz: float, c: dict, angle: float) -> Track:
    t = np.arange(int(duration * clock_hz)) / clock_hz
    z = np.zeros(len(t))
    return Track(t=t, elbow=z + angle, body=z.copy(), twist_dps=z.copy(), activity=z.copy(),
                 lifting=np.zeros(len(t), bool), F=z.copy(), effort=z.copy(), jitter=z.copy())


def render_set(reps: list[Rep], dF: float, p: Person, c: dict) -> tuple[Track, dict]:
    pre, flicks, settle, post = 3.0, 1.5, 2.0, 3.0
    starts, t0 = [], pre + flicks + settle
    for r in reps:
        r.start_s = t0
        starts.append(t0)
        t0 += r.lift_s + r.hold_s + r.lower_s + r.pause_s
    tr = _empty_track(t0 + post, p.clock_hz, c, c["bottom_deg"])
    t, b = tr.t, c["bottom_deg"]
    tr.activity[:] = 0.1  # holding the dumbbell with the arm hanging
    for k in range(3):    # sync flicks: three quick forearm twists
        m = np.abs(t - (pre + 0.3 + 0.4 * k)) < 0.075
        tr.twist_dps[m] = 300 * np.sin(2 * np.pi * (t[m] - (pre + 0.3 + 0.4 * k) + 0.075) / 0.15)
    tr.F[:] = reps[0].F
    for r in reps:
        a, top, down, end = (r.start_s, r.start_s + r.lift_s, r.start_s + r.lift_s + r.hold_s,
                             r.start_s + r.lift_s + r.hold_s + r.lower_s)
        lift, hold, lower = (t >= a) & (t < top), (t >= top) & (t < down), (t >= down) & (t < end)
        s_up, s_dn = (t[lift] - a) / r.lift_s, (t[lower] - down) / r.lower_s
        tr.elbow[lift] = b + r.rom_deg * min_jerk(s_up)
        tr.elbow[hold] = b + r.rom_deg
        tr.elbow[lower] = b + r.rom_deg * (1 - min_jerk(s_dn))
        tr.body[lift] = r.swing_deg * np.sin(np.pi * s_up) + r.drift_deg * min_jerk(s_up)
        tr.body[hold] = r.drift_deg
        tr.body[lower] = r.drift_deg * (1 - min_jerk(s_dn))
        torque = lambda m: 0.6 + 0.4 * np.sin(np.radians(tr.elbow[m]))  # noqa: E731
        tr.activity[lift] = c["curl_effort"] * torque(lift)
        tr.activity[hold] = c["curl_effort"] * 0.5 * np.sin(np.radians(tr.elbow[hold]))
        tr.activity[lower] = c["curl_effort"] * c["lowering_loudness"] * torque(lower)
        tr.lifting |= lift | hold
        rep = (t >= a) & (t < end + r.pause_s)
        tr.F[rep] = r.F + dF * (t[rep] - a) / (end + r.pause_s - a)
        tr.F[t >= end + r.pause_s] = r.F + dF
        tr.effort[rep] = r.effort
        tr.jitter[rep] = r.jitter
        tr.events += [(a, "swing" if r.kind == "swing" else "turn"), (down, "turn")]
    stop = next((r for r in reps if r.kind != "good"), None)
    failed = next((r for r in reps if r.kind == "failed"), None)
    info = {
        "reps": [{"n": r.n, "start_s": round(r.start_s, 4), "F": round(r.F, 4), "kind": r.kind,
                  "effort": round(r.effort, 3), "lift_s": round(r.lift_s, 3), "rom_deg": round(r.rom_deg, 1)}
                 for r in reps],
        "sync_s": pre + 0.3,
        "set_start_s": reps[0].start_s,
        "stop_rep": stop.n if stop else None,
        "stop_point_s": stop.start_s if stop else None,
        "failure_rep": failed.n if failed else None,
        "failure_s": failed.start_s if failed else None,
        "end_s": t0,
    }
    return tr, info


def render_pull(F: float, p: Person, c: dict) -> tuple[Track, dict]:
    tr = _empty_track(7.0, p.clock_hz, c, 90.0)  # elbow at a right angle, strap under the foot
    m = (tr.t >= 2) & (tr.t < 5)
    ramp = np.clip((tr.t[m] - 2) / 0.3, 0, 1) * np.clip((5 - tr.t[m]) / 0.2, 0, 1)
    tr.activity[:] = 0.05
    # An all-out pull can't recruit more than everything, so fatigue doesn't make it louder:
    # divide out the loudness rise signals() applies, and let it sag a little instead.
    tr.activity[m] = np.maximum(0.05, ramp / (1 + p.loud_rise * min(F / c["stop_level"], 1)) * (1 - 0.1 * F))
    tr.lifting[m] = True
    tr.F[:] = F
    tr.events = [(2.0, "turn")]
    return tr, {"pull_kg": round(p.strength_kg * (1 - 0.35 * F / c["stop_level"]), 2), "pull_start_s": 2.0,
                "pull_end_s": 5.0}


def render_rest(p: Person, c: dict) -> tuple[Track, dict]:
    tr = _empty_track(10.0, p.clock_hz, c, 90.0)  # forearm supported, relaxed
    return tr, {}


def signals(tr: Track, p: Person, c: dict, rng: np.random.Generator) -> dict:
    """Turn a track into what the board would measure: EMG in mV, IMU readings, and the truth."""
    t, n = tr.t, len(tr.t)
    x = tr.F / c["stop_level"]
    angle = tr.elbow + tr.body
    mdf = p.mdf0 * (1 - p.mdf_drop * x) * (1 + c["angle_mdf_effect"] * (angle - 75) / 65)
    mdf *= (1 + 0.25 * tr.effort) * (1 + tr.jitter)
    loud = (1 + p.loud_rise * np.minimum(x, 1)) * (1 + tr.effort)
    rms = c["max_pull_rms_mv"] * p.scale * tr.activity * loud
    rms = np.convolve(rms, np.ones(60) / 60, mode="same")  # muscles don't switch instantly (30 ms)
    emg = synth_emg(t, np.maximum(mdf, 15), rms, rng)

    emg += rng.standard_normal(n) * c["rest_noise_mv"] / np.sqrt((BAND[1] - BAND[0]) / (p.clock_hz / 2))
    for k, a in enumerate(c["hum_mv"]):
        emg += a * np.sqrt(2) * np.sin(2 * np.pi * 60 * (k + 1) * t + rng.uniform(0, 2 * np.pi))
    if c["packet_noise_mv"]:
        for base in (50.0, 62.5):
            for k in range(1, 9):
                emg += c["packet_noise_mv"] / k * np.sin(2 * np.pi * base * k * t + rng.uniform(0, 2 * np.pi))
    for te, kind in tr.events:
        amp = c["motion_artifact_mv"] * rng.normal() * (2 if kind == "swing" else 1)
        emg += amp * np.exp(-0.5 * ((t - te) / 0.025) ** 2)
    emg_mv = c["midrail_mv"] + emg
    clipped = (emg_mv <= 0) | (emg_mv >= FULL_SCALE_MV)
    emg_mv = np.clip(emg_mv, 0, FULL_SCALE_MV)

    # IMU at 50 Hz on the board's millis() clock; the forearm turns about the elbow (sensor y).
    imu_t = np.arange(0, t[-1], 1 / IMU_HZ) + rng.uniform(0, 1 / IMU_HZ)
    th = np.radians(np.interp(imu_t, t, angle))
    w = np.gradient(np.radians(angle), t)
    th_d, th_dd = np.interp(imu_t, t, w), np.interp(imu_t, t, np.gradient(w, t))
    r = c["forearm_m"]
    accel = np.c_[-np.cos(th) - r * th_d**2 / G, np.zeros_like(th), np.sin(th) + r * th_dd / G]
    # Sign: the accelerometer reading u turns as du/dt = -w x u for a rigid body, so with
    # u = (-cos th, 0, sin th) the forearm turns about sensor y at -th'. (Getting this wrong makes
    # the gyro and accelerometer disagree, which no real IMU does.)
    gyro = np.c_[np.interp(imu_t, t, tr.twist_dps), -np.degrees(th_d), np.zeros_like(th)]
    accel = accel @ p.mount.T + rng.normal(0, c["accel_noise_g"], accel.shape)
    gyro = gyro @ p.mount.T + rng.normal(0, c["gyro_noise_dps"], gyro.shape) + rng.normal(0, 0.2, 3)
    # Physiological tremor: a narrow wobble near the person's tremor frequency, growing with
    # tiredness, on every gyro axis (vector RMS = tremor_dps when fresh). Size is an assumption.
    if c["tremor_dps"]:
        band = [max(0.5, p.tremor_hz - 1.5), min(IMU_HZ / 2 - 0.5, p.tremor_hz + 1.5)]
        sos = signal.butter(2, band, btype="bandpass", fs=IMU_HZ, output="sos")
        wobble = signal.sosfilt(sos, rng.standard_normal((len(imu_t) + 200, 3)), axis=0)[200:]
        wobble /= np.sqrt(np.mean(np.sum(wobble**2, axis=1)))
        grow = 1 + c["tremor_rise"] * np.minimum(np.interp(imu_t, t, x), 1.3)
        gyro += c["tremor_dps"] * grow[:, None] * wobble

    keep = np.arange(0, n, max(1, int(round(p.clock_hz / IMU_HZ))))
    truth = {"t": t[keep], "F": tr.F[keep], "x": x[keep], "mdf_hz": mdf[keep], "loudness": loud[keep],
             "angle_deg": angle[keep], "elbow_deg": tr.elbow[keep], "lifting": tr.lifting[keep]}
    return {"emg_mv": emg_mv, "clipped": clipped, "imu_t": imu_t, "accel": accel, "gyro": gyro, "truth": truth}


def write_capture(path: Path, sig: dict, first_index: int, ms_at_first: float, clock_hz: float) -> None:
    """The byte stream the firmware would send: 32-sample EMG blocks, IMU frames by stamp, STATUS each second."""
    emg, clipped = sig["emg_mv"], sig["clipped"]
    stamps = first_index + np.floor(sig["imu_t"] * clock_hz).astype(np.int64)
    out, j, next_status = bytearray(), 0, (np.floor(ms_at_first / 1000) + 1) * 1000
    for b in range(0, len(emg) - 31, 32):
        idx = first_index + b
        out += frame(EMG, emg_payload(idx, emg[b:b + 32], clipped=4 * int(clipped[b:b + 32].sum())))
        while j < len(stamps) and stamps[j] < idx + 32:
            out += frame(IMU, imu_payload(int(stamps[j]), sig["accel"][j], sig["gyro"][j]))
            j += 1
        now_ms = ms_at_first + (b + 32) / clock_hz * 1000
        if now_ms >= next_status:
            out += frame(STATUS, status_payload(idx + 32, int(now_ms)))
            next_status += 1000
    path.write_bytes(bytes(out))


# ------------------------------------------------------------------ one person, one dataset


def simulate_person(i: int, c: dict, out: Path, rng: np.random.Generator) -> dict:
    p = make_person(i, c, rng)
    tau = c["recovery_tau_s"]
    recover = lambda F, s: F * float(np.exp(-s / tau))  # noqa: E731
    T, F, written = 5.0, 0.0, []  # board booted 5 s before the first recording

    def save(name: str, tr: Track, info: dict, meta: dict) -> None:
        sig = signals(tr, p, c, rng)
        write_capture(out / f"{name}.bin", sig, int(round(T * p.clock_hz)), (T + 5.0) * 1000, p.clock_hz)
        markers = [[info[k], k.removesuffix("_s")] for k in ("sync_s", "pull_start_s", "pull_end_s", "end_s")
                   if info.get(k) is not None]
        labels = {k: info.get(k) for k in ("set_start_s", "stop_point_s", "failure_s")}
        if labels["stop_point_s"] is not None and rng.random() < c["label_error_prob"]:
            labels["stop_point_s"] += float(rng.choice([-1, 1])) * c["cadence_s"]  # rater off by a rep
        side = {"participant": p.pid, "world": c["world"], **meta, "markers": markers, "labels": labels}
        if "pull_kg" in info:
            side["pull_kg"] = info["pull_kg"]
        (out / f"{name}.json").write_text(json.dumps(side, indent=1))
        np.savez_compressed(out / f"{name}_truth.npz", **sig["truth"])
        (out / f"{name}_truth.json").write_text(json.dumps(info, indent=1))
        written.append(name)

    tr, info = render_rest(p, c)
    save(f"{p.pid}_rest", tr, info, {"recording": "rest"})
    T += 10 + 60
    for k in (1, 2, 3):
        tr, info = render_pull(0.0, p, c)
        save(f"{p.pid}_pull_base{k}", tr, info, {"recording": "pull", "after_set": 0})
        T += 7 + 60
    T += 120  # load selection and warm-up
    for set_no in (1, 2, 3):
        rest_before = None if set_no == 1 else p.rest_order[set_no - 2]
        reps, dF = plan_set(p, set_no, F, c, rng)
        tr, info = render_set(reps, dF, p, c)
        save(f"{p.pid}_set{set_no}", tr, info, {"recording": "set", "set": set_no, "rest_before_s": rest_before,
                                                 "rest_order": list(p.rest_order)})
        F = float(tr.F[-1])
        T += info["end_s"] + 8.0  # the pull starts 8 s after the last rep
        F = recover(F, 8.0)
        tr, pinfo = render_pull(F, p, c)
        save(f"{p.pid}_pull{set_no}", tr, pinfo, {"recording": "pull", "after_set": set_no})
        if set_no < 3:
            # Rest is timed from the end of the set to the next set's first lift, which comes
            # 6.5 s into its recording (still, sync flicks, settle).
            rest = p.rest_order[set_no - 1]
            T += rest - 8.0 - 6.5
            F = recover(F, rest - 8.0)
    return {"person": p.truth(), "recordings": written}


def simulate(cfg: dict, world: str, out: str | Path, people: int | None = None, seed: int | None = None) -> Path:
    c = sim_config(cfg, world)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seed = c["seed"] if seed is None else seed
    index = {"world": world, "seed": seed, "config": {k: v for k, v in c.items()}, "people": []}
    for i in range(people or c["people"]):
        rng = np.random.default_rng([seed, i])
        index["people"].append(simulate_person(i, c, out, rng))
    (out / "index.json").write_text(json.dumps(index, indent=1, default=float))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world", default="normal")
    ap.add_argument("--all-worlds", action="store_true", help="every world, each in its own folder under --out")
    ap.add_argument("--out", default="sim")
    ap.add_argument("--people", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--config")
    args = ap.parse_args()
    cfg = load_config(args.config)
    worlds = sorted(cfg["sim"]["worlds"]) if args.all_worlds else [args.world]
    for w in worlds:
        dest = Path(args.out) / w if args.all_worlds else Path(args.out)
        simulate(cfg, w, dest, args.people, args.seed)
        print(f"{w}: {dest}")


if __name__ == "__main__":
    main()
