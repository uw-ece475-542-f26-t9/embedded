# Signal processing and ML pipeline: team catch-up

*Adi · 2026-11-17 · branch `signal-pipeline` of `embedded`, folder `pipeline/`*

Adi took over all signal processing and ML for the project. This is everything that was built and every decision that changed along the way, so nobody is working from the 11-03 version of the plan.

## TL;DR
- **A full pipeline from raw recordings to haptic warnings is built and tested** (112 automated tests): input and health checks, cleaning, forearm angle, EMG angle bins, 20 candidate features, Kalman filters, models, a nested feature search, warning tuning, a lab kit and one-command reports. It's in `pipeline/` on branch `signal-pipeline`. `main` is untouched.
- **Labels are now time, not rep counts.** The model predicts **seconds until the stop point** (the first bad rep, marked on video). Warnings fire at **4 s and 2 s**, which is 2 reps and 1 rep at the metronome pace. Nothing in the model counts reps.
- **The metronome stays** (1 s up, 1 s down) for data collection *and* the demo.
- **Nothing has been trained yet.** Training and feature selection happen only on real recordings. Simulated data was used only to test the code.
- **Next:** test the firmware changes on the board, run the pilot, then the 8-person collection. We need two teammates to rate the videos.

## 1. Where things are and how to run them
| What | Where |
|---|---|
| Pipeline code, tests, every tunable setting | `pipeline/` (settings in `pipeline/config.toml`) |
| Feature list and selection rules | `pipeline/FEATURES.md` |
| Plan, step by step | `pipeline/docs/PIPELINE_PLAN.md` |
| Data collection protocol (updated 2026-11-17) | `pipeline/docs/DATA_COLLECTION_PROTOCOL.md` |
| Sensor testing (the MyoWare proof of concept) | branch `emg-sensor-testing`, folder `emg-sensor-testing/` |
| Firmware changes | `fatigue_sensor/` on `signal-pipeline` (see §6) |
| Pre-registered baseline feature set | tag `pipeline-core-v1` (2026-11-16) |

```bash
cd pipeline && python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                                  # 112 tests, ~3.5 min
.venv/bin/python -m pipeline.lab P01 --weight-kg 10            # run a whole session with the board
.venv/bin/python -m pipeline.labels template recordings/       # label sheet for the video raters
.venv/bin/python -m pipeline.labels apply recordings/          # write the agreed labels into the recordings
.venv/bin/python -m pipeline.report recordings/                # everything scored, one markdown report
.venv/bin/python -m pipeline.sweep recordings/                 # the 12-setting knob sweep, nested by person
```
Capture-mode firmware for the XIAO:
```bash
arduino-cli compile --fqbn esp32:esp32:XIAO_ESP32S3 --build-property "compiler.cpp.extra_flags=-DCAPTURE_MODE=1" fatigue_sensor
```
(`build.extra_flags` looks like the obvious choice but replaces the board's own flags and breaks the Adafruit libraries.)

## 2. What changed in the data collection protocol (vs the 11-03 version)
| Topic | Before (11-03) | Now | Why |
|---|---|---|---|
| **Label** | Reps until the stop point | **Seconds until the stop point.** The stop point is a moment: the start of the first bad rep | No rep counting anywhere, so the model can't learn "rep 12 = stop". Seconds map to reps through the metronome |
| **Warnings** | 2 reps and 1 rep before | **4 s and 2 s before** (the same at 2 s/rep) | Follows from the label |
| **Metronome** | Yes | **Kept, for collection and the demo** | Keeps EMG comparable (speed changes the EMG), makes "missed beat" an objective stop sign, makes seconds equal reps. Costs: realism, and time-into-set becomes a stronger shortcut (the varied 60/180 s rests counter that) |
| **Running a session** | Manual, with `log_emg.py` | **The lab kit runs it:** spoken cues (hold still, three flicks, settle, go), metronome clicks, rest timers, automatic pull 5 s after each set, and every recording checked as soon as it ends | Fewer operator mistakes, and bad recordings are caught on the spot |
| **Operator keys** | – | **Enter when the spotter calls the stop point.** Set 3: Enter at the stop point, then again at failure | Gives a live marker (the official label still comes from video) |
| **Spike (aliasing) check** | 10,000 vs 10,500 readings/s, dev-kit firmware | **8,000 vs 8,400/s on the team firmware**, through a new `RATE` command; the lab kit does it automatically (`P01_rest`, `P01_rest_alt`) | The anti-aliasing capacitor isn't fitted, so this runs every session. Peaks that move between rates are aliasing |
| **Video labels** | Raters mark the stop-point rep | **Raters type times as the video shows them** (first sync flick, first lift, first bad rep, failure) into `labels.csv`. The tool finds the flicks in the IMU and lines everything up | Raters don't need to sync anything; labels are times |
| **Rater agreement** | Same rep / within 1 rep | Same rule, defined in time: **same rep** if under 1 s apart, **within one rep** if under 3 s. More than a rep apart: discuss, add an `agreed` row; until then the set isn't scored | Makes the rule mechanical |
| **Baselines to beat** | Rep number; rep + set number | **Time into set; time + set number; time + set number + rest before the set** | No rep numbers exist any more. The rest baseline is the toughest fair one |
| **How warnings are tuned** | Fixed bands | **Fires when predicted seconds stay at or below a threshold; thresholds tuned on the training people only: false alarms in at most 1 in 3 sets, weighted towards later sets (×1 / ×1.5 / ×2), then the fewest misses** (Adi's call) | A miss breaks the claim "warns before form breaks"; later sets are the most fatigued |
| **Pilot session** | Tune knobs, then freeze | **Checks the setup and procedure only; tunes nothing** | One person can't be left out, so tuning on them can't be honest. Knobs are tuned nested on the 8 instead |
| **Calibration** | – | Proposed and **dropped**: a dead hold to exhaustion before the workout | It pre-fatigues the muscle, a still hold doesn't match curls, and electrodes move between sessions. Set 1 already shows each person's own range (see #25/#26) |

Unchanged: 8 people, 1 session each, 3 sets (sets 1–2 end at the stop point, set 3 to failure), the 4 stop-point signs, max pulls, varied rest order, safety rules, leave-one-person-out testing.

## 3. What changed vs the model on `main`
The on-device model on `main` stays as it is. The new pipeline differs in these ways:

| | `main` (on-device model) | `signal-pipeline` |
|---|---|---|
| Target | Reps in reserve: operator types the "failure rep" after the set | **Seconds until the stop point**, from two video raters |
| Rep number as an input | Yes | **Never** (it's the shortcut the baselines test) |
| Training data | Synthetic (pipeline test) | **Real recordings only**; synthetic data only tests code |
| Middle-pitch window | The whole lift | **Soft angle bins**: the same arm angle compared with the same arm angle |
| Baseline | Speed only | Three clock baselines, plus two "oracles" that check the scoring itself |

## 4. The pipeline, step by step
Design rules for every step: **real time** (past data only, checked by tests), **relative** (each feature compared with the start of its own set, set 1, or the max pull), **no rep counting**, **nothing tuned on the person being tested**.

1. **Input layer.** Reads the team firmware's capture stream (parser written from the `capture.h` spec) and the sensor-testing recordings. It measures the real sample rate, puts the IMU on the EMG clock, marks gaps, survives reboots, and rejects bad recordings (gaps, clipping, ADC buffer overflow, IMU gaps).
2. **Simulator (code tests only).** Fake protocol-shaped sessions with known truth, in 4 test worlds: normal, EMG-blind (EMG carries no fatigue), shortcut (stop point predictable from time), and effort (effort swings, no fatigue). It models the left shift *and* the loudness rise, tremor, form breakdown and person-to-person differences, including where each person's form breaks.
3. **Scoring harness.** Leave one person out; misses, false alarms, lead time and error in seconds; three clock baselines. Two "oracles" that cheat by reading the simulator's hidden tiredness check that the scoring itself is right.
4. **Cleaning.** A 401-tap linear-phase band-pass, flat from 20 to 450 Hz. The planned 201 taps cut 20 Hz in half. It adds a 100 ms delay, subtracted so EMG stays aligned with the IMU. Hum and interference peaks are **dropped from spectra and filled from neighbours**, not notched: a 60 Hz notch sits in the middle of the EMG band. The rest recording's noise floor is subtracted (quiet people otherwise read 3–8 Hz high). Motion jolts are skipped when a slow swing is both over 10 mV *and* as big as the EMG itself. A rest check flags hum and interference families.
5. **Forearm angle and soft angle bins.** RepSense's gravity tracker, ported exactly and made real time (τ = 1 s, the 98% gyro / 2% accelerometer blend). "Hanging" is the stillest 0.3 s before the first lift, which is mount-proof and unaffected by the sync flicks. Bins sit every 15° from 15° to 135°, each 30° wide with weights that fade at the edges, so a small angle error nudges a weight instead of flipping bins. Lifting and lowering get separate bins (18 in total). RepSense's *counter* isn't used: it only counts in hindsight (a 12 s wait).
6. **Features.** One row per lift, relative to the set's first 3 lifts, with bins pooled at equal weight. Each pass is analysed over a fixed 384-sample window, so lift speed can't bias middle pitch. **All 20 candidates are computed** (`FEATURES.md`), including the new **#26 session-relative middle pitch**, which compares sets 2–3 with set 1's start so carried-over tiredness shows.
7. **Kalman filters.** One per noisy feature, with a level and a slope. The two noise settings are fitted by maximum likelihood on the training people, and an innovation check confirms the filter's errors are plain noise. The top-angle feature isn't filtered: one bad rep has to show up immediately.
8. **Models and feature search.** A straight line (ridge), a small neural net capped at half as many weights as training rows (the RepSense rule), and a physics-style comparison (distance to the breaking level ÷ slope). The **random-subset search** (10,000 random 6-feature models, as in RepSense) runs fully nested, with adjusted credit, a shuffled-label luck check and a bootstrap. A full run takes about 1 minute.
9. **Warnings.** A threshold plus persistence (1 or 2 lifts in a row), latched once fired, tuned as in §2.
10. **Lab kit, report, sweep.** See §1. The sweep tries 12 settings (bin width 15/20°, window 256/384/512 samples, baseline 3/4 lifts) and picks the best using only training people, fold by fold.

## 5. Features
Full table in `pipeline/FEATURES.md`: 20 candidates across EMG frequency, EMG loudness, trends, IMU movement and form, and across sets.
- **Baseline feature set (core-v1, locked before any data, tag `pipeline-core-v1`):** #1 middle pitch, #8 loudness, #13a middle-pitch slope, #15 speed loss, #17 top-angle change, #22 tremor. It's the reference; **the best features are chosen empirically** by the nested search.
- **Never model inputs:** time into the set, set number and rest. They are the baselines.

## 6. Firmware changes (on `signal-pipeline`, compile-tested for the XIAO, **not yet tested on the board**)
- **`RATE <hz>` command in capture mode** (6,000–12,000 Hz). It restarts the ADC at the new rate and keeps the sample index continuous; STATUS frames report the live rate. The lab kit uses it for the spike check.
- **`CAPTURE_MODE` and `EMG_PIN` can be set at build time,** so nobody has to edit the source to switch. The default pin stays GPIO 1 (XIAO A0); the sensor-testing dev kit used GPIO 4.

## 7. What we learned so far
These come from **code checks on simulated data** and from earlier real data (sensor testing, RepSense wrist recordings). None of them are results about real fatigue yet.
- **Where each person's form breaks matters most.** A cheating model that knows the exact tiredness but not the person's own breaking point barely beats the clock (about 2.7–4.2 s error vs 2.8–4.4 s). Knowing both drops error to about 1.9 s with no misses. So per-person signals (top-angle drift, #26, set-1 personalisation) are where the EMG and IMU can add most.
- **The time + set + rest baseline is strong,** partly because the protocol's per-person weight choice makes set lengths predictable.
- **The middle-pitch slope (#13a, in core-v1) is mostly noise within one set** (signal/noise 0.55). Loudness slope (#13b): 3.6, but that leans on assumed effect sizes. Logged, not acted on: the search decides.
- **#26 shows carried-over tiredness:** set 2 starts at 0.94 after a 60 s rest vs 1.00 after 180 s.
- **#3 (Dimitrov index) and #4 (low/high ratio) move about 6–7× more than middle pitch, and the opposite way for effort surges.** Promising against the effort confound.
- **One lift's middle pitch scatters about 8.4%** against a true drift of about 1% per lift, which is why everything is filtered over lifts.
- **The sensor-testing analysis (`analyze.py`) read middle pitch up to 9% low.** Person B is probably about 54 Hz, not 51. The conclusions don't change.
- **Real wrist data (RepSense recordings):** sticking point 0.38 in exhaustion sets vs 0.02; bottom angle +49° in partial sets vs +4°; lateness 1.0 s in exhaustion vs 0.1 s. Axis wobble (#21) isn't specific.
- **Tremor (#22) also rises on jerky bad reps,** so it partly measures form breakdown.

## 8. Bugs caught along the way (all fixed and covered by tests)
- **The simulator's gyro turned the wrong way** relative to its own accelerometer. That's physically impossible, and it made the angle 45° wrong.
- **The 10 kHz→2 kHz converter padded file edges with zeros,** a fake 1,400 mV swing at each end of every old recording.
- **The feature search's bootstrap leaked:** a person drawn twice was treated as two people, so one copy trained the model while its twin was scored. A "fingerprint" feature passed 60% of resamples; it now passes 15%.
- **Simulated middle pitch didn't scale in proportion** with the time-warp (the 20 Hz band edge), which squeezed the fatigue drop.
- **Floating-point rounding** made warnings look 0.5 s late. Also: the jolt rule threw away hard squeezes; "hanging" detection was stricter than real hand tremor allows.
- **Statistics checked properly before conclusions:** one apparent confound (slowing faking a middle-pitch drop) turned out to be noise from too few simulated people; with 8 seeds it was +1.1% ± 1.8%.

## 9. What's needed next
| Item | Who | Notes |
|---|---|---|
| Flash capture-mode firmware; test `RATE`, the IMU stream and the EMG pin on the XIAO | Anyone with the board | The lab kit's rest_alt step checks the rate switch |
| Pilot session with a teammate | Team | Checks the setup, cues, checks and label sheet; nothing tuned, data not in the 8 |
| **Two video raters** | Two teammates | They mustn't see each other's marks or any model output |
| Instructor check on human-subjects approval | Team | Still open |
| 8-person collection → `pipeline.report` and `pipeline.sweep` | Team / Adi | Then the feature search on real data |
| Merge `signal-pipeline` into `main` | Team | Via a pull request, when ready |
| README rewrite | Team | Not done yet, on purpose. `pipeline/README.md` has one stale link (`../PIPELINE_PLAN.md`, now at `docs/`) |
| ESP32 port of the final features and model | Adi | After the collection, once knobs and features are final |
| App: show the new warnings | App owners | The app reads only the first 8 bytes of the rep packet today |
| Bluetooth streaming instead of USB | If needed | Only if the pilot shows hum or packet-noise problems |
