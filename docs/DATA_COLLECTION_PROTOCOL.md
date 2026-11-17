# Data collection protocol

*Bicep fatigue capstone · design finalized 2026-11-03*

## 1. Goal and claim
The wearable predicts, rep by rep, **how many reps remain before form breaks down**. It warns at **2 reps** and **1 rep** before that point.

**Claim:** "warns you before your form breaks down."
We do **not** claim it "prevents injury". Injury risk can't be measured directly or ethically. Form breakdown is the closest observable stand-in, and it is the point where coaches tell lifters to stop.

## 2. Session procedure (about 50 minutes)
**Before you start, four things to know:**
- **The wearable:**
  - an EMG sensor (electromyography: the muscle's electrical signal), a MyoWare 2.0 stuck on the biceps;
  - an IMU (inertial measurement unit: motion sensor) worn on the wrist like a watch.

  Both are wired to one ESP32-S3 board, which streams to a laptop. The laptop recorder (`log_emg.py`) saves each recording under the names listed in §3. Full equipment list: §7.
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
   - **Record 10 s at 10,000 readings/s** (`P01_rest`, which is also the session's resting baseline), **then 10 s at 10,500 readings/s** (`P01_rest_10500`). Then run `analyze.py alias` on the two.
   - **Why:** the board has no anti-aliasing filter, so electrical interference from nearby devices can fold into the muscle signal. Real signals appear at the same frequency in both recordings; interference shifts between them.
   - **It must PASS before continuing.** If it fails, move or unplug nearby devices and redo.

**Baseline (5 min)**

4. **3 max pulls** (`P01_pull_base1` to `P01_pull_base3`), each 3 s, with 60 s between. Upper arm against the side, elbow at a right angle. EMG is recorded during each pull. **The best pull = 100% strength.** It is also the max-effort EMG reference: the analysis uses it to express muscle activity as a % of max.
5. **Load selection:** the participant picks a dumbbell they could curl about 12–15 times at the metronome pace (60 beats per minute: lift on one beat, lower on the next). Record the weight.
6. **Warm-up:** 8 light curls at the metronome pace, then 2 min rest.

**Sets (about 20 min)**

7. **Set 1** (`P01_set1`):
   - **Sync mark first:** 3 sharp wrist flicks in view of the camera. They show up as spikes in the wrist IMU, which lines the phone video up with the sensor data.
   - **Then curl on the metronome,** upper arm pinned to the side.
   - **Stop when the spotter calls the stop point.**
8. **Pull within 10 s** of the set ending (`P01_pull1`). Strength recovers within minutes, so a late pull understates the fatigue.
9. **Rest:** 60 s or 180 s, per the order table below. Rest is timed from the end of the set to the start of the next set.
10. **Set 2** (`P01_set2`): same as set 1, followed by a pull within 10 s (`P01_pull2`).
11. **Rest:** the other of 60 s / 180 s.
12. **Set 3** (`P01_set3`):
    - **Goes to true failure:** 2 missed beats in a row, or the participant can't complete a rep.
    - **The spotter still notes the stop-point rep when it happens,** but the set continues past it.
    - **Then a pull within 10 s** (`P01_pull3`).
13. **Optional:** a recovery pull 5 min later (`P01_pull_recovery`).

**Rest order:** 4 people get short then long, 4 get long then short. Varying rest makes later sets start at different levels of tiredness. That way the model can't predict the stop point just by counting reps or sets (see §10).

| Participant | P01 | P02 | P03 | P04 | P05 | P06 | P07 | P08 |
|---|---|---|---|---|---|---|---|---|
| Rest 1 / Rest 2 | 60/180 | 180/60 | 180/60 | 60/180 | 60/180 | 180/60 | 180/60 | 60/180 |

## 3. What gets recorded
**Data files:** the laptop recorder saves each recording under these names in `data/`, one set per participant (P01 shown):
- `P01_rest`, `P01_rest_10500` (setup step 3)
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

## 4. Labels
**Primary label: reps until stop point.**
- The **stop point** is the first rep showing any of these:
  1. torso swing (leaning back or using momentum);
  2. upper arm drifting forward or away from the side;
  3. a partial rep (no full curl at the top, or no full extension at the bottom);
  4. a missed metronome beat.
- **The spotter's live call only ends the set.** The official stop point for each set comes from the video (§5).
- Each earlier rep is labelled by its distance from the stop point. Example: rep 9 of a set whose stop point is rep 12 → **3**.
- In set 3 (taken to failure), every rep from the stop point onward is labelled **0** ("past stop").

**Secondary label: strength drop.**
- Each max pull is reported as a % of the baseline pull.
- It shows that the stop point coincides with real strength loss.

**Set 3 margin.** Set 3 runs to true failure, so we also record the number of reps between the stop point and failure.

## 5. Labelling
1. **Two teammates independently mark the stop point** in each set's video against the 4 signs in §4, without seeing each other's marks or any model output.
2. **Disagreements are settled by discussion.** Report agreement: % of sets with the same rep, and % within 1 rep.
3. **Elbow-drift model check:** our existing model on the wrist IMU already flags bad reps. It merges elbow drift and torso swing, which is acceptable because the label is "first bad rep". Report its agreement with the video stop point. **Video remains the reference.**
4. **Rep alignment:** the sync flicks line up video with data; rep numbers in the data must match the video.

## 6. Analysis and evaluation
- **Inputs, per rep:**
  - EMG middle pitch (median frequency) and loudness (RMS, root mean square). As a muscle tires, middle pitch falls and loudness rises; sensor testing confirmed that middle pitch falls under load. Both are taken at the same arm angle in every rep, using the wrist IMU.
  - IMU features: rep timing, range of motion, drift.
- **Output:** exact reps until stop point.
- **Testing:** train on 7 people, test on the 8th, and repeat for all 8 (leave-one-person-out). **Nothing is tuned on the person being tested.**
- **Baselines the model must beat:**
  1. rep number only;
  2. rep number + set number.
- **Headline metrics** (bands fixed in advance: 3+ = fine, 2 = warn, ≤1 = stop):
  - **Lead time:** how many reps before the stop point each warning fires.
  - **Misses:** the stop point is reached with no warning.
  - **False alarms:** the 2-rep warning fires with 4+ reps really left, or the 1-rep warning with 3+ left.
  - **Exact-rep error:** mean absolute error, in reps.
- **Secondary results:**
  - strength drop at each stop point;
  - set 3 margin (stop point → failure);
  - rater agreement;
  - drift model vs video agreement.
- **Exploratory:** other band splits may be tried on the test results only if labelled "exploratory". They are never the headline.

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
| Bands fixed before any data | Choosing bands after seeing test results would inflate the reported numbers |
| Video with 2 raters is the reference | The agreement rate makes the label credible; the drift model is checked against it |
| Spike check every session | The anti-aliasing capacitor isn't fitted; this catches interference instead |

## 11. Before the first session
1. **Firmware:** add the wrist-IMU stream to the recorder, on the same ESP32 clock as the EMG.
2. **Logger:** a set mode with spoken metronome and sync cues and automatic start/end markers.
3. **Analysis:** rep segmentation by forearm angle; label import from the session sheet.
4. **A pilot session with a teammate,** run end to end. The pilot data is not included in the 8.
5. **The instructor check** on human-subjects approval.
