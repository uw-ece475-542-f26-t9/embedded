# Signal processing and ML pipeline plan

*Owner: Adi · started 2026-11-15 · code in `pipeline/`*

## Design rules
- **Real time.** Every stage runs sample by sample on past data only: one-way filters, no look-ahead. The Python is the reference the firmware copies.
- **No rep counts.** The model never sees rep numbers. EMG is analysed only while the forearm passes a fixed angle window (**gating, not counting**).
- **Labels are moments in time** from the video: the stop point time. The target is **seconds until the stop point**.
- **Cold start.** The headline model works on a new person with no calibration beyond the max pulls. Features are changes from the start of the set.
- **Knobs live in `config.toml`.** They are tuned on training people only, and frozen before the 8-person collection.

## Steps
| # | Step | Done when |
|---|---|---|
| 1 | **Input layer.** `load_session()` for team capture files and the sensor-testing recordings, with health checks | Both formats load; deliberately damaged files are caught |
| 2 | **Simulator.** Protocol-shaped sessions (8 people, 3 sets, 60/180 s rest, stop point, set 3 to failure). Noise is based on real holds | Labelled sessions with tunable effect size, noise and spread between people |
| 3 | **Scoring harness.** Train on 7, test on 1. Baselines: time in set; time + set number. Metrics: warning lead time, misses, false alarms, error in seconds | Baselines get scores on simulated data |
| 4 | **Cleaning.** Band-pass and hum notches, one-way | Middle pitch on the real recordings matches `analyze.py` |
| 5 | **Angle gate.** A real-time forearm angle from gravity and the gyro (RepSense's approach, made one-way). EMG is kept only inside the angle window on the way up | Gate opens once per lift on simulated and real IMU |
| 6 | **Features per gated window.** Middle pitch, loudness (% of max pull), lift speed, range of motion, each relative to the set's start | Unit tests on signals with known properties |
| 7 | **Kalman filter.** Tracks each feature's level and slope. Measurement noise is measured; process noise is a knob | Follows the simulated fatigue trend; innovation check passes |
| 8 | **Model.** A straight-line model, then the small neural net | Beats both baselines on simulated data |
| 9 | **Warning.** Predicted time until the threshold is crossed is under 4 s / 2 s (about 2 / 1 reps at the metronome pace) | Scored with step 3's metrics |
| 10 | **Lab kit.** `config.toml` holds every knob. One command takes a folder of recordings and produces a report. A sweep tunes knobs on training people only | Runs end to end on simulated data |
| 11 | **Lab day.** Pilot session → check signal quality → tune → **freeze knobs** (record the date) | Knobs locked |
| 12 | **Port to device.** Firmware matches the Python on the same input | Host test agrees |

**Secondary result:** personalising sets 2–3 using set 1's trend (past data only).

**Protocol doc to update:** labels in seconds until the stop point; warnings at 4 s and 2 s.
