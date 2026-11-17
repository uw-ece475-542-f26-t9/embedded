# Bicep fatigue pipeline

Signal processing and ML for the bicep fatigue wearable. The plan and design rules are in
[`../PIPELINE_PLAN.md`](../PIPELINE_PLAN.md). Every knob is in [`config.toml`](config.toml).

## Status
| Step | What | State |
|---|---|---|
| 1 | Input layer: `load_session()` for team capture `.bin` files and sensor-testing `.csv.gz` recordings, with health checks | Done |
| 2 | Simulator: protocol-shaped fake sessions with known truth, in 4 test worlds (normal, emg_blind, shortcut, effort) | Done |
| 3 | Scoring harness: leave-one-person-out, 3 clock baselines, warning misses / false alarms / lead / error, cheating oracle to check the referee | Done |
| 4 | Cleaning: real-time linear-phase band-pass, jolt skipping, hum/interference lines dropped from spectra, noise floor subtracted, rest check | Done |
| 5 | Forearm angle (RepSense port, one-way) and soft angle bins: 9 lifting + 9 lowering, one pass per sweep | Done |
| 6 | Per-lift features, relative to the set's first 3 lifts: core #1, #8, #15, #17, #22 (#13a is step 7), and #26 against set 1 | Done |
| 6b | The remaining 12 candidate features (#2–4, #6, #9–11, #16, #18–21): all 20 candidates now computed | Done |
| 7 | Kalman filters: level + slope per feature, q and r fitted by maximum likelihood on training people, innovation check | Done |
| 8 | Models (ridge, small net under the size rule, physics-style comparison) and the nested random-subset search; feature cache. Trained on real data only | Built (code-tested) |
| 10 (1-3) | Lab kit: protocol recorder with metronome and cues (`python -m pipeline.lab P01`), on-the-spot checks, video label import with sync-flick alignment and rater agreement (`python -m pipeline.labels`) | Built (code-tested; needs the board) |
| 10.4-10.5 | One-command report (`python -m pipeline.report recordings/`) and nested 12-setting knob sweep (`python -m pipeline.sweep recordings/`); both refuse simulated data unless `--sim` (code check) | Built (code-tested) |
| 9 | Warnings: threshold + persistence + latch, tuned on out-of-person predictions: weighted false alarms <= 1 in 3 sets (later sets weigh more), then fewest misses | Built (code-tested) |

## Use
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```
```bash
.venv/bin/python -m pytest -q
```
Check recordings before using them (exit code 1 if any should be rejected):
```bash
.venv/bin/python -m pipeline.check A_squeeze B_light
```
Make fake data (every world, 8 people each):
```bash
.venv/bin/python -m pipeline.simulate --all-worlds --out sim
```
Each recording gets `NAME.bin` + `NAME.json` (what the lab would have: markers, video labels with
rater error) and `NAME_truth.npz` / `NAME_truth.json` (what really happened: tiredness, true middle
pitch, every rep).

Score the baselines (and the oracle, on simulated data):
```bash
.venv/bin/python -m pipeline.evaluate sim/normal --oracle
```

In code:
```python
from pipeline import load_session
s = load_session("A_squeeze")          # or "recordings/P01_set1.bin"
s.emg, s.emg_hz, s.imu_t, s.accel, s.gyro, s.markers, s.meta
s.problems(cfg), s.notes()
```

## Layout
| Path | What |
|---|---|
| `pipeline/session.py` | `Session` and `Health`: the object every stage works on, and the reject rules |
| `pipeline/capture.py` | Team firmware capture stream reader, written from the `capture.h` spec; frame writers for tests |
| `pipeline/legacy.py` | Sensor-testing recordings: ADC counts to mV, 10,000/s to 2,000/s |
| `pipeline/load.py` | `load_session()`: picks the reader |
| `pipeline/check.py` | Command-line health check |
| `pipeline/simulate.py` | Step 2 simulator; its knobs and test worlds are `[sim]` in `config.toml` |
| `pipeline/clean.py` | Step 4: `clean_emg`, `spectrum`, `middle_pitch`, `rest_check` and its spectrum cleaner |
| `pipeline/angle.py` | Step 5a: `forearm_angle`, ported from RepSense `track_down` / `hang_axis` / `segment_angle` |
| `pipeline/bins.py` | Step 5b: `find_passes`, `pass_spectrum` |
| `pipeline/features.py` | Step 6: `find_lifts`, `lift_features`, `cleaner_for` |
| `pipeline/kalman.py` | Step 7: `run_filter`, `filter_rows`, `fit`, `innovation_check` |
| `pipeline/model.py` | Step 8: `FatigueModel`, `PhysicsModel`, feature cache, `CORE_V1` |
| `pipeline/search.py` | Step 8b: random-subset ranking (`rank`), `SearchedModel` (fully nested) |
| `pipeline/warn.py` | Step 9: `Policy`, `tune`, `Warned` |
| `pipeline/lab.py` | Step 10: protocol recorder, cues, on-the-spot checks |
| `pipeline/labels.py` | Step 10: video label sheet, sync-flick alignment, rater agreement |
| `pipeline/report.py` | Step 10.4: recordings to markdown report |
| `pipeline/sweep.py` | Step 10.5: nested knob sweep |
| `pipeline/evaluate.py` | Step 3 harness: `SetData`, `load_sets`, `evaluate`, `report`, `causal_check` |
| `pipeline/models.py` | Baselines and the oracle |
| `tests/` | Tests |
