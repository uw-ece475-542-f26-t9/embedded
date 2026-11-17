# Signal processing and ML pipeline plan

*Owner: Adi · started 2026-11-15 · updated 2026-11-17 · code in `pipeline/`*

## Design rules
- **Real time.** Every stage runs sample by sample on past data only: one-way filters, no look-ahead. The Python is the reference the firmware copies.
- **No rep counts.** The model never sees rep numbers. EMG is analysed in **soft angle bins** (every 15°, 9 lifting + 9 lowering): each bin is compared only with itself in other reps, then pooled (**binning, not counting**).
- **Labels are moments in time** from the video: the stop point time. The target is **seconds until the stop point**.
- **Cold start.** The headline model works on a new person with no calibration beyond the max pulls. Features are changes from the start of the set.
- **Knobs live in `config.toml`.** They are tuned on training people only (nested sweep); the pilot checks the setup and tunes nothing.
- **Real data only for training and selection.** Simulated data tests the code; nothing fitted on it carries forward.

## Steps
| # | Step | Done when | Status |
|---|---|---|---|
| 1 | **Input layer.** `load_session()` for team capture files and the sensor-testing recordings, with health checks | Both formats load; deliberately damaged files are caught | Done |
| 2 | **Simulator.** Protocol-shaped sessions (8 people, 3 sets, 60/180 s rest, stop point, set 3 to failure). Noise is based on real holds | Labelled sessions with tunable effect size, noise and spread between people | Done (+ tremor, per-person stop level) |
| 3 | **Scoring harness.** Train on 7, test on 1. Baselines: time in set; time + set number; time + set number + rest. Metrics: warning lead time, misses, false alarms, error in seconds | Baselines get scores on simulated data (code check) | Done: + time+set+rest baseline, two oracles |
| 4 | **Cleaning.** Linear-phase band-pass (401 taps), hum and interference dropped from spectra (no notches), noise floor subtracted, jolts skipped, one-way | Middle pitch on the real recordings matches `analyze.py` | Done |
| 5 | **Forearm angle and soft angle bins.** RepSense's gravity tracker (98% gyro / 2% accelerometer), run one-way; EMG tagged with angle and direction, one pass per sweep through each bin | Angle within 5° of truth; each lift fills the bins it reaches | Done: RepSense angle port + soft bins |
| 6 | **Features per gated window.** Middle pitch, loudness (% of max pull), lift speed, range of motion, each relative to the set's start | Unit tests on signals with known properties | Done: all 20 candidates (6b) |
| 7 | **Kalman filter.** Tracks each feature's level and slope. Measurement noise is measured; process noise is a knob | Follows the simulated fatigue trend; innovation check passes | Done |
| 8 | **Model.** A straight-line model, then the small neural net | Runs through the harness; code checks pass on simulated data (performance is judged on real data only) | Built; trains on real data only |
| 9 | **Warning.** Fires when predicted seconds left stay at or below a threshold (default 4 s / 2 s = 2 / 1 reps); thresholds tuned on training people | Scored with step 3's metrics | Built: ≤ 1 in 3 false alarms (later sets weighted), then fewest misses |
| 10 | **Lab kit.** `config.toml` holds every knob. One command takes a folder of recordings and produces a report. A sweep tunes knobs on training people only | Runs end to end on simulated data | Built: lab kit, labels, report, sweep |
| 11 | **Lab day.** Pilot session → check signal quality, the checks, cues and label sheet → fix what's broken. **No tuning** (one person can't be left out) | Setup works end to end | Waiting for the board |
| 12 | **Port to device.** Firmware matches the Python on the same input | Host test agrees | After the 8-person collection |

**Secondary result:** personalising sets 2–3 using set 1's trend (past data only).

**Protocol doc:** updated 2026-11-17 (labels in seconds, warnings at 4 s / 2 s, lab kit, label sheet, the 1-in-3 rule, pilot tunes nothing).
