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
| `pipeline/evaluate.py` | Step 3 harness: `SetData`, `load_sets`, `evaluate`, `report`, `causal_check` |
| `pipeline/models.py` | Baselines and the oracle |
| `tests/` | Tests |
