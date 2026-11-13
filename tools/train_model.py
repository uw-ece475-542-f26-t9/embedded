#!/usr/bin/env python3
"""Trains the on-device fatigue model and writes it into the firmware.

The model estimates reps in reserve (RIR: how many more reps until failure)
after each rep, from the features the ESP32 measures itself
(fatigue_sensor/fatigue_features.h):

    rep_index       rep number in the set
    velocity_loss   1 - peak lift speed / the set's first 3 reps
    duration_ratio  moving time of the lift / the first 3 reps
    rom_ratio       range of motion / the first 3 reps
    emg_ratio       EMG RMS / the first 3 reps
    mdf_ratio       EMG median frequency / the first 3 reps

"The first 3 reps" is their median, and the model estimates from rep 4 on.

It's a small neural network (6 -> 16 -> 8 -> 1, ReLU) trained here with
numpy, then quantized to int8 weights with one scale per layer, which is
under 300 bytes of weights. The output is fatigue_sensor/fatigue_model_data.h;
re-flash to use it. The header also holds test vectors, and the firmware
checks at boot that it gets the same answers as this script.

Training data, either:

  Real: capture-mode recordings (tools/record.py). The firmware sends its
  features for every rep in FEAT frames, so the model trains on exactly
  what it will see. Fatigue sets need the failure rep in their .json;
  RIR = failure rep - rep number.

      python tools/train_model.py recordings/

  Synthetic: made-up sets where speed drops, reps slow, EMG rises and its
  median frequency falls toward failure, with noise and person-to-person
  differences. Only for testing the pipeline before real data exists. A
  model trained on it says nothing about real fatigue.

      python tools/train_model.py --synthetic

Accuracy is checked by leaving one person out at a time, and compared with
a straight line on rep number and velocity loss (the velocity-only
baseline from the PRD).
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import struct

import numpy as np

import capture_format as cf

HERE = os.path.dirname(os.path.abspath(__file__))
HEADER = os.path.join(HERE, "..", "fatigue_sensor", "fatigue_model_data.h")

FEATURES = ["rep_index", "velocity_loss", "duration_ratio", "rom_ratio", "emg_ratio", "mdf_ratio"]
BASELINE_REPS = 3  # FatigueFeatures::BASELINE_REPS
HIDDEN = (16, 8)

# Training. Full-batch Adam on mean squared error with weight decay; small
# enough to take a few seconds.
EPOCHS = 1500
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.1  # strong: the data sets are small, and it overfits otherwise


# ------------------------------------------------------------------ data


def synthetic(rng: np.random.Generator, subjects: int = 12, sets: int = 4):
    """Rows of (subject, set, features..., rir), computed the way the
    firmware computes them, from made-up raw measurements."""
    rows = []
    for s in range(subjects):
        # Each person fatigues a little differently.
        v0 = rng.uniform(150, 300)  # peak speed, deg/s
        moving0 = rng.uniform(0.9, 1.6)
        rom0 = rng.uniform(110, 140)
        emg0 = rng.uniform(50, 300)
        mdf0 = rng.uniform(70, 110)
        vl_fail = rng.uniform(0.35, 0.6)  # velocity loss at failure
        curve = rng.uniform(1.0, 2.0)     # >1: speed drops faster near failure
        rom_drop = rng.uniform(0, 0.15)
        emg_rise = rng.uniform(0.1, 0.5)
        mdf_drop = rng.uniform(0.05, 0.2)
        for k in range(sets):
            n = int(rng.integers(8, 17))  # reps to failure
            no_emg = rng.random() < 0.2   # electrodes off or not recorded
            raw = []
            for r in range(1, n + 1):
                x = (r - 1) / (n - 1)
                vl = vl_fail * x**curve
                raw.append((
                    v0 * (1 - vl) * (1 + 0.04 * rng.standard_normal()),
                    moving0 / (1 - 0.8 * vl) * (1 + 0.05 * rng.standard_normal()),
                    rom0 * (1 - rom_drop * x**2) * (1 + 0.03 * rng.standard_normal()),
                    emg0 * (1 + emg_rise * x) * (1 + 0.10 * rng.standard_normal()),
                    mdf0 * (1 - mdf_drop * x) * (1 + 0.04 * rng.standard_normal()),
                ))
            raw = np.array(raw)
            base = np.median(raw[:BASELINE_REPS], axis=0)
            for r in range(BASELINE_REPS + 1, n + 1):
                peak, moving, rom, emg, mdf = raw[r - 1] / base
                if no_emg:
                    emg = mdf = 1.0
                rows.append((f"syn{s:02d}", k, r, 1 - peak, moving, rom, emg, mdf, n - r))
    return rows


def parse_feat(p: bytes) -> dict:
    stamp, rep, flags = struct.unpack_from("<IHB", p)
    vals = struct.unpack_from("<11f", p, 7)
    return {"stamp": stamp, "rep_in_set": rep, "emg_ok": bool(flags & 1),
            "has_baseline": bool(flags & 2), "alert": bool(flags & 4),
            "peak_dps": vals[0], "moving_s": vals[1], "rom_deg": vals[2],
            "emg_rms_mv": vals[3], "mdf_hz": vals[4], "velocity_loss": vals[5],
            "duration_ratio": vals[6], "rom_ratio": vals[7], "emg_ratio": vals[8],
            "mdf_ratio": vals[9], "rir_pred": vals[10]}


def recordings(folder: str):
    """Rows from capture recordings of fatigue sets that have a failure rep."""
    rows, used = [], 0
    for path in sorted(glob.glob(os.path.join(folder, "*.bin"))):
        meta_path = path[:-4] + ".json"
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        failure = meta.get("failure_rep")
        if meta.get("set_type") != "fatigue" or not failure:
            continue
        with open(path, "rb") as f:
            frames = cf.FrameReader().feed(f.read())
        feats = [parse_feat(p) for t, p in frames if t == cf.FEAT and len(p) >= 51]
        n0 = len(rows)
        for ft in feats:
            r = ft["rep_in_set"]
            if not ft["has_baseline"] or r > failure:
                continue
            session = f"{meta.get('session')}:{os.path.basename(path)}"
            rows.append((meta["subject"], session, r, ft["velocity_loss"], ft["duration_ratio"],
                         ft["rom_ratio"], ft["emg_ratio"], ft["mdf_ratio"], failure - r))
        if len(rows) > n0:
            used += 1
        else:
            print(f"  {os.path.basename(path)}: no FEAT frames with a baseline; skipped "
                  "(recorded before the model firmware?)")
    print(f"Loaded {len(rows)} reps from {used} fatigue sets in {folder}")
    return rows


# ----------------------------------------------------------------- model


class Mlp:
    def __init__(self, sizes, rng):
        self.w = [rng.standard_normal((o, i)) * np.sqrt(2 / i) for i, o in zip(sizes, sizes[1:])]
        self.b = [np.zeros(o) for o in sizes[1:]]

    def forward(self, x):
        acts = [x]
        for k, (w, b) in enumerate(zip(self.w, self.b)):
            x = x @ w.T + b
            if k < len(self.w) - 1:
                x = np.maximum(x, 0)
            acts.append(x)
        return acts

    def predict(self, x):
        return self.forward(x)[-1][:, 0]

    def fit(self, x, y, rng):
        params = self.w + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2 = 0.9, 0.999
        for t in range(1, EPOCHS + 1):
            acts = self.forward(x)
            grad = 2 * (acts[-1][:, 0] - y)[:, None] / len(y)
            gw, gb = [], []
            for k in range(len(self.w) - 1, -1, -1):
                gw.insert(0, grad.T @ acts[k] + WEIGHT_DECAY * self.w[k])
                gb.insert(0, grad.sum(axis=0))
                if k:
                    grad = (grad @ self.w[k]) * (acts[k] > 0)
            for p, g, mi, vi in zip(params, gw + gb, m, v):
                mi += (1 - b1) * (g - mi)
                vi += (1 - b2) * (g * g - vi)
                p -= LEARNING_RATE * (mi / (1 - b1**t)) / (np.sqrt(vi / (1 - b2**t)) + 1e-8)
        return self


def quantize(model: Mlp):
    """int8 weights with one symmetric scale per layer; float biases."""
    layers = []
    for w, b in zip(model.w, model.b):
        scale = float(np.abs(w).max() / 127) or 1.0
        q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
        layers.append((q, np.float32(scale), b.astype(np.float32)))
    return layers


def quantized_predict(layers, mean, std, x):
    """What the firmware computes, in float32 like the ESP32."""
    h = ((x - mean) / std).astype(np.float32)
    for k, (q, scale, b) in enumerate(layers):
        h = (h @ q.T.astype(np.float32)) * scale + b
        if k < len(layers) - 1:
            h = np.maximum(h, 0)
    return np.maximum(h[:, 0], 0)


def ridge(x, y, lam=1e-2):
    a = np.c_[x, np.ones(len(x))]
    coef = np.linalg.solve(a.T @ a + lam * np.eye(a.shape[1]), a.T @ y)
    return lambda z: np.c_[z, np.ones(len(z))] @ coef


def train(x, y, rng):
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-6] = 1.0  # a constant input (e.g. no EMG at all) is ignored
    model = Mlp((x.shape[1], *HIDDEN, 1), rng).fit((x - mean) / std, y, rng)
    return model, mean.astype(np.float32), std.astype(np.float32)


# ----------------------------------------------------------- evaluation


def scores(err):
    err = np.abs(err)
    return f"MAE {err.mean():4.2f} reps   within 2: {np.mean(err <= 2):4.0%}   within 5: {np.mean(err <= 5):4.0%}"


def leave_one_person_out(subjects, x, y, rng):
    people = sorted(set(subjects))
    if len(people) < 2:
        print("Only one person: skipping the leave-one-person-out check.")
        return
    err = {"velocity-only line": [], "all-features line": [], "network": [], "network, int8": []}
    for p in people:
        test = subjects == p
        tr = ~test
        vel = [FEATURES.index("rep_index"), FEATURES.index("velocity_loss")]
        err["velocity-only line"] += list(np.maximum(ridge(x[tr][:, vel], y[tr])(x[test][:, vel]), 0) - y[test])
        err["all-features line"] += list(np.maximum(ridge(x[tr], y[tr])(x[test]), 0) - y[test])
        model, mean, std = train(x[tr], y[tr], rng)
        err["network"] += list(np.maximum(model.predict((x[test] - mean) / std), 0) - y[test])
        err["network, int8"] += list(quantized_predict(quantize(model), mean, std, x[test]) - y[test])
    print(f"\nLeave one person out ({len(people)} people, {len(y)} reps):")
    for name, e in err.items():
        print(f"  {name:20s} {scores(np.array(e))}")


# ---------------------------------------------------------------- export


def c_array(a, fmt):
    if a.ndim == 1:
        return "{" + ", ".join(fmt(v) for v in a) + "}"
    return "{\n    " + ",\n    ".join(c_array(r, fmt) for r in a) + "}"


def f32(v):
    s = f"{float(v):.9g}"
    return s + ("f" if any(c in s for c in ".e") else ".0f")


def write_header(path, layers, mean, std, test_x, test_y, source, n_reps, n_people):
    (q1, s1, b1), (q2, s2, b2), (q3, s3, b3) = layers
    n_weights = sum(q.size for q, _, _ in layers)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"// Generated by tools/train_model.py on {stamp}. Don't edit; retrain instead.",
        f"// Trained on {source}: {n_reps} reps from {n_people} people.",
        f"// {n_weights} int8 weights.",
        "",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        "namespace fatigue_model {",
        "",
        f'constexpr const char *SOURCE = "{source}";',
        f"constexpr int N_IN = {len(FEATURES)}, N_H1 = {HIDDEN[0]}, N_H2 = {HIDDEN[1]};",
        "",
        "// Inputs: " + ", ".join(FEATURES) + ".",
        f"constexpr float IN_MEAN[N_IN] = {c_array(mean, f32)};",
        f"constexpr float IN_STD[N_IN] = {c_array(std, f32)};",
        "",
        f"constexpr int8_t W1[N_H1][N_IN] = {c_array(q1, str)};",
        f"constexpr float W1_SCALE = {f32(s1)};",
        f"constexpr float B1[N_H1] = {c_array(b1, f32)};",
        "",
        f"constexpr int8_t W2[N_H2][N_H1] = {c_array(q2, str)};",
        f"constexpr float W2_SCALE = {f32(s2)};",
        f"constexpr float B2[N_H2] = {c_array(b2, f32)};",
        "",
        f"constexpr int8_t W3[N_H2] = {c_array(q3[0], str)};",
        f"constexpr float W3_SCALE = {f32(s3)};",
        f"constexpr float B3 = {f32(b3[0])};",
        "",
        "// Inputs and the RIR this script's int8 model gives for them, for the",
        "// boot-time self-test.",
        f"constexpr int N_TEST = {len(test_y)};",
        f"constexpr float TEST_IN[N_TEST][N_IN] = {c_array(test_x.astype(np.float32), f32)};",
        f"constexpr float TEST_OUT[N_TEST] = {c_array(test_y, f32)};",
        "",
        "}  // namespace fatigue_model",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", help="capture recordings (tools/record.py)")
    ap.add_argument("--synthetic", action="store_true", help="train on made-up data, to test the pipeline")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=HEADER)
    args = ap.parse_args()
    if bool(args.folder) == args.synthetic:
        ap.error("give a recordings folder or --synthetic")

    rng = np.random.default_rng(args.seed)
    if args.synthetic:
        rows, source = synthetic(rng), "synthetic data (pipeline test only)"
        print(f"Made {len(rows)} synthetic reps. These test the pipeline, not real fatigue.")
    else:
        rows, source = recordings(args.folder), f"recordings in {os.path.basename(os.path.abspath(args.folder))}"
    if len(rows) < 20:
        raise SystemExit("Not enough reps to train on (need 20+ after the baseline reps).")

    subjects = np.array([r[0] for r in rows])
    x = np.array([r[2:8] for r in rows], dtype=float)
    y = np.array([r[8] for r in rows], dtype=float)
    leave_one_person_out(subjects, x, y, rng)

    model, mean, std = train(x, y, rng)
    layers = quantize(model)
    fit = np.maximum(model.predict((x - mean) / std), 0)
    qfit = quantized_predict(layers, mean, std, x)
    print(f"\nFinal model on all data: float {scores(fit - y)}")
    print(f"                         int8  {scores(qfit - y)}")
    print(f"  int8 vs float: largest difference {np.abs(qfit - fit).max():.3f} reps")

    pick = rng.choice(len(x), size=min(8, len(x)), replace=False)
    test_x = x[pick].astype(np.float32)
    write_header(args.out, layers, mean, std, test_x, quantized_predict(layers, mean, std, test_x),
                 source, len(y), len(set(subjects)))
    print(f"\nWrote {os.path.relpath(args.out)}. Flash fatigue_sensor to use it.")


if __name__ == "__main__":
    main()
