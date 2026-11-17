# Data collection protocol

*Bicep fatigue capstone · design finalized 2026-11-03 · updated 2026-11-17 for the pipeline and lab kit*

## 1. Goal and claim
The wearable predicts, lift by lift, **how long remains before form breaks down**. It warns at **2 reps** and **1 rep** before that point: **4 s and 2 s** at the metronome pace (2 s per rep).

**Claim:** "warns you before your form breaks down."
We do **not** claim it "prevents injury". Injury risk can't be measured directly or ethically. Form breakdown is the closest observable stand-in, and it is the point where coaches tell lifters to stop.

## 2. Session procedure (about 50 minutes)
**Before you start, four things to know:**
- **The wearable:**
  - an EMG sensor (electromyography: the muscle's electrical signal), a MyoWare 2.0 stuck on the biceps;
  - an IMU (inertial measurement unit: motion sensor) worn on the wrist like a watch.

  Both are wired to one ESP32-S3 board (the XIAO), which streams to a laptop. The board must run the team firmware built in **capture mode** with the **RATE** command (§11).
- **The lab kit runs the session:** `python -m pipeline.lab P01 --weight-kg <kg>`. It walks through the steps below in order, names every recording (§3), speaks the cues, plays the metronome, times the rests, and **checks each recording as soon as it ends**. If it says redo, redo. Full equipment list: §7.
- **Three roles:**
  - **operator:** runs the laptop;
  - **spotter:** watches form, calls the stop point, and ends the set on pain;
  - **camera:** films the side view.
- **Stop point:** the first rep showing torso swing, the upper arm drifting forward or away from the side, a partial rep, or a missed metronome beat. Sets 1–2 end there. Full definition in §4; safety rules in §9.
- **Max pull:** an all-out 3 s pull on a luggage scale, with a strap under the foot and the elbow at a right angle. It measures how much strength is left. It is done once at baseline and again after every set.

**Setup (15 min)**
1. **Electrode placement:** measure from the elbow crease to the shoulder tip. Centre both electrodes ⅓ of that distance above the crease, along the muscle, MID toward the shoulder and END toward the elbow. Put the reference electrode on the elbow bone. **Record the distance in cm.**
2. **Skin prep:** shave if needed, rub with alcohol, let dry 1 min, press each electrode for 10 s. Tape all jumpers and cables.
3. **Rest recordings and spike check:**
   - **Arm fully relaxed:** forearm supported, palm up, hand limp, shoulder dropped. Wait 5 s after settling before starting each recording.
   - **The lab kit records 10 s at the normal rate** (`P01_rest`, also the session's noise baseline), **then switches the board to 8,400 readings/s and records 10 s more** (`P01_rest_alt`), then switches back. It checks each for hum and interference peaks, and compares the two.
   - **Why:** the board has no anti-aliasing filter, so electrical interference from nearby devices can fold into the muscle signal. Real signals appear at the same frequency in both recordings; interference shifts between them.
   - **It must PASS before continuing.** If it fails, move or unplug nearby devices and redo.

**Baseline (5 min)**

4. **3 max pulls** (`P01_pull_base1` to `P01_pull_base3`), each 3 s, with 60 s between. Upper arm against the side, elbow at a right angle. EMG is recorded during each pull. **The best pull = 100% strength.** It is also the max-effort EMG reference: the analysis uses it to express muscle activity as a % of max.
5. **Load selection:** the participant picks a dumbbell they could curl about 12–15 times at the metronome pace (60 beats per minute: lift on one beat, lower on the next). Record the weight.
6. **Warm-up:** 8 light curls at the metronome pace, then 2 min rest.

**Sets (about 20 min)**

7. **Set 1** (`P01_set1`). The lab kit speaks each cue:
   - **"Hold still"** (3 s, arm hanging with the dumbbell).
   - **Sync mark: "flick your wrist three times"** in view of the camera. The flicks show up in the wrist IMU as sharp twists, which lines the phone video up with the sensor data.
   - **"Settle", "go", then the metronome:** curl on it, upper arm pinned to the side.
   - **Stop when the spotter calls the stop point.** The operator presses **Enter** at that moment.
8. **Pull within 10 s** of the set ending (`P01_pull1`). The lab kit starts it automatically 5 s after the set. Strength recovers within minutes, so a late pull understates the fatigue.
9. **Rest:** 60 s or 180 s, per the order table below. Rest is timed from the end of the set to the start of the next set.
10. **Set 2** (`P01_set2`): same as set 1, followed by a pull within 10 s (`P01_pull2`).
11. **Rest:** the other of 60 s / 180 s.
12. **Set 3** (`P01_set3`):
    - **Goes to true failure:** 2 missed beats in a row, or the participant can't complete a rep.
    - **The spotter still calls the stop point when it happens** (operator: **Enter**), but the set continues past it. Operator: **Enter again at failure.**
    - **Then a pull within 10 s** (`P01_pull3`).
13. **Optional:** a recovery pull 5 min later (`P01_pull_recovery`).

**Rest order:** 4 people get short then long, 4 get long then short. Varying rest makes later sets start at different levels of tiredness. That way the model can't predict the stop point just by counting reps or sets (see §10).

| Participant | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 |
|---|---|---|---|---|---|---|---|---|
| Rest 1 / Rest 2 | 60/180 | 180/60 | 180/60 | 60/180 | 60/180 | 180/60 | 180/60 | 60/180 |

## 3. What gets recorded
**Data files:** the lab kit saves each recording in `recordings/` as `NAME.bin` (the raw stream, as received) plus `NAME.json` (who, what, cue and Enter markers, and the video labels once they're applied). One set per participant (P01 shown):
- `P01_rest`, `P01_rest_alt` (setup step 3)
- `P01_pull_base1` to `P01_pull_base3` (baseline step 4)
- `P01_set1`, `P01_pull1`, `P01_set2`, `P01_pull2`, `P01_set3`, `P01_pull3` (steps 7–12)
- `P01_pull_recovery` (optional step 13)

**Session sheet** (one per participant):
- participant ID, arm, electrode distance (cm), dumbbell weight, rest order;
- baseline pull values;
- reps completed per set;
- each pull value;
- set 3 failure rep;
- notes, including anything unusual (a loose cable, a pause, a pain stop).

**Video:** one clip per set, starting before the sync flicks.

**Label sheet:** `recordings/labels.csv`, filled in by the raters (§5).

## 4. Labels
**Primary label: seconds until the stop point.**
- The **stop point** is a moment in time: the start of the first rep showing any of these:
  1. torso swing (leaning back or using momentum);
  2. upper arm drifting forward or away from the side;
  3. a partial rep (no full curl at the top, or no full extension at the bottom);
  4. a missed metronome beat.
- **The spotter's live call only ends the set.** The official stop point for each set comes from the video (§5).
- The model's target at each lift is **the time from that lift to the stop point**. Example: a lift that ends 6 s before the stop point → **6 s** (3 reps at the metronome pace). Rep numbers are never used: nothing in the model counts reps.
- In set 3 (taken to failure), every lift from the stop point onward is labelled **0** ("past stop").

**Secondary label: strength drop.**
- Each max pull is reported as a % of the baseline pull.
- It shows that the stop point coincides with real strength loss.

**Set 3 margin.** Set 3 runs to true failure, so we also record the number of reps between the stop point and failure.

## 5. Labelling
1. **Two teammates independently mark each set's video** against the 4 signs in §4, without seeing each other's marks or any model output. Each types **times as the video shows them** into `recordings/labels.csv` (make it with `python -m pipeline.labels template recordings/`):
   - the first of the three sync flicks;
   - the start of the first lift;
   - the start of the first bad rep (the stop point);
   - set 3 only: the rep the participant couldn't complete (failure).
2. **`python -m pipeline.labels apply recordings/`** finds the flicks in the IMU, moves every video time onto the recording's clock, and writes the labels into each set's `.json`. It reports agreement: **same rep** if the two stop points are under 1 s apart, **within one rep** if under 3 s. Two raters within one rep → the label is their mean. **More than one rep apart → settle it by discussion and add a row with rater `agreed`**; until then the set stays unlabelled and isn't scored.
3. **Elbow-drift model check:** our existing model on the wrist IMU already flags bad reps. It merges elbow drift and torso swing, which is acceptable because the label is "first bad rep". Report its agreement with the video stop point. **Video remains the reference.**
4. **Alignment:** the sync flicks line up video with data. No rep numbers are carried over: labels are times.

## 6. Analysis and evaluation
The pipeline is in `pipeline/` (plan: `docs/PIPELINE_PLAN.md`; features: `FEATURES.md`). One command scores everything: `python -m pipeline.report recordings/`.
- **Inputs, per lift:** features from `FEATURES.md`, each relative to the start of its own set (or set 1, or the max pull). EMG is analysed in soft angle bins (the same arm angle compared with the same arm angle), cleaned of hum, interference and the noise floor, and smoothed by Kalman filters. IMU features: speed loss, range of motion (top angle), tremor, timing, sticking points.
  - **Baseline feature set (core-v1, locked 2026-11-16, tag `pipeline-core-v1`):** #1 middle pitch, #8 loudness, #13a middle-pitch slope, #15 speed loss, #17 top angle, #22 tremor.
  - **Best features chosen empirically** by the nested random-subset search over all 20 candidates (`FEATURES.md` protocol).
- **Output:** seconds until the stop point, after every lift.
- **Testing:** train on 7 people, test on the 8th, and repeat for all 8 (leave-one-person-out). **Nothing is tuned on the person being tested**: not the model, the Kalman filters, the feature choice, the warning thresholds or the knob sweep (`python -m pipeline.sweep recordings/`, 12 settings).
- **Baselines the model must beat** (they never see a sensor):
  1. time into the set;
  2. time + set number;
  3. time + set number + rest before the set.
- **Warnings:** each fires when the prediction stays at or below its threshold (1 or 2 lifts in a row), then stays on. Thresholds are tuned on the training people's out-of-person predictions: **false alarms in at most 1 in 3 sets, weighted towards later sets** (set 1 ×1, set 2 ×1.5, set 3 ×2), **then the fewest misses** (Adi, 2026-11-17).
- **Headline metrics** (definitions fixed in advance):
  - **Lead time:** how many seconds before the stop point each warning fires.
  - **Misses:** the stop point is reached with no warning.
  - **False alarms:** the 2-rep warning fires with more than 8 s (4+ reps) really left, or the 1-rep warning with more than 6 s (3+ reps); any warning in a set with no stop point.
  - **Error:** mean absolute error, in seconds.
- **Secondary results:**
  - strength drop at each stop point;
  - set 3 margin (stop point → failure);
  - rater agreement;
  - drift model vs video agreement.
- **Exploratory:** anything tried after looking at test results is labelled "exploratory" and is never the headline.

**Expected size:** about 290 reps in total, about 70 of them inside the 2-rep warning zone.

## 7. Equipment
| Item | Spec |
|---|---|
| EMG | MyoWare 2.0, RAW → GPIO 4, ENV → GPIO 2, powered from 3.3 V; fresh gel electrodes every session |
| IMU | Worn on the wrist like a watch, wired to the **same ESP32-S3**, so it shares the EMG clock |
| Recorder | ESP32-S3 at 10,000 readings/s; laptop on battery (charger unplugged) |
| Force | Luggage scale with peak-hold, plus a strap anchored under the foot |
| Load | Dumbbells; metronome at 60 beats per minute |
| Video | Phone on a tripod, side view, hips to head, 60 fps if possible |
| People | Operator (laptop), spotter (calls the stop point, ends the set on pain), camera |

## 8. Participants
- **8 people, 1 session each,** dominant arm.
- **Inclusion:** healthy adults, no current or recent arm, elbow or shoulder injury, no pain when curling.
- **Participation is voluntary;** anyone can stop at any time.
- **Before the first session, check with the capstone instructor** whether human-subjects approval or a consent form is required.

## 9. Safety rules
- **Pain** (sharp or joint pain, as opposed to the normal burn) **ends the set immediately**, including in set 3.
- The spotter watches for the stop point in sets 1–2 and ends the set there.
- In set 3, the spotter stays ready to take the weight.
- No participant does more than one session.

## 10. Decisions and reasons
| Decision | Reason |
|---|---|
| Label = reps until form breakdown | "About to get hurt" can't be measured; form breakdown is the observable danger zone |
| Sets 1–2 end at the stop point | The safety rule and the label are the same thing; nobody pushes past bad form |
| Set 3 to failure | Measures the margin between breakdown and failure, showing the warning comes first |
| Strength pull after each set | Measures fatigue in the strict sense, confirms the label, and gives the max-effort EMG reference |
| Weight chosen per person (about 12–15 reps) | Same relative difficulty for everyone |
| Metronome, 1 s up / 1 s down | Steady speed keeps EMG comparable; a missed beat clearly marks failure |
| Rest of 60 s / 180 s, order balanced | Stops the model from predicting the stop point by counting reps or sets |
| 8 people, 1 session, 3 sets | Electrode placement stays fixed within a person; the test is on unseen people |
| Warning definitions fixed before any data; thresholds tuned on training people only | Choosing them after seeing test results would inflate the reported numbers |
| Video with 2 raters is the reference | The agreement rate makes the label credible; the drift model is checked against it |
| Spike check every session | The anti-aliasing capacitor isn't fitted; this catches interference instead. Rest at two ADC rates (8,000 and 8,400/s, the firmware's RATE command): peaks that move are aliasing |
| Metronome kept (2026-11-17) | Steady speed keeps EMG comparable and makes seconds equal reps; the demo uses it too. Cost: realism and a stronger clock shortcut (the varied rest counters it) |
| Labels in seconds, never rep numbers | Nothing counts reps, so the model can't learn "rep 12 = stop"; warnings at 4 s / 2 s = 2 / 1 reps at the pace |
| Warnings: false alarms ≤ 1 in 3 sets (later sets weighted), then fewest misses | A miss breaks the claim; later sets are where the person is most tired |
| The pilot checks the setup, it doesn't tune | One person can't be left out, so it can't tune honestly; knobs keep their defaults and the sweep runs nested on the 8 |

## 11. Before the first session
1. **Firmware:** the team firmware already streams EMG and the wrist IMU on one clock in capture mode. Build it with `--build-property "compiler.cpp.extra_flags=-DCAPTURE_MODE=1"` for the XIAO (`esp32:esp32:XIAO_ESP32S3`). The **RATE command** for the spike check is written and compiles; it needs testing on the board.
2. **Lab kit:** done (`pipeline.lab`, `pipeline.labels`, `pipeline.report`, `pipeline.sweep`).
3. **Analysis:** done (`pipeline/`, steps 1–10).
4. **A pilot session with a teammate,** run end to end. It **checks the setup and the procedure only**: signal quality, the checks, the cues, the label sheet. Nothing is tuned on it, and its data is not included in the 8.
5. **The instructor check** on human-subjects approval.
