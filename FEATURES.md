# Features

The living list of every feature the fatigue model could use. **Numbers never change:** a dropped
feature keeps its number and is marked dropped, so results stay traceable. Anyone adds rows; every
change goes in the log at the bottom with the date and the reason.

## Rules every feature must follow
1. **Real time.** Computed from past data only: it must pass `causal_check` (no value changes when
   later data is removed).
2. **Relative to the set's own start** (or to the person's max pull), never absolute. Electrode
   placement, skin and strength differ between people; changes within a set don't.
3. **No rep counting.** Features come from angle-bin passes and lifts found by the angle, never from
   "rep number N".
4. **Wrist IMU plus biceps EMG only.** Nothing the product won't have (no video, no upper-arm sensor).
5. **Mount-proof.** The same movement with the strap turned on the wrist gives the same value.

## Status key
- **core** = in the pre-registered confirmatory model (fixed before real data; changing it after
  seeing results makes the main result meaningless)
- **candidate** = in the random-subset search
- **exploratory** = reported, never the headline
- **shortcut** = recreates the clock baseline; kept out of the model, used only as a baseline

## Core model (locked 2026-11-16, before any real data)
**#1 middle pitch change, #8 loudness change, #13a middle pitch slope, #15 speed loss, #17 top-angle
change, #22 tremor.** The headline result is this model, fitted and scored by leave-one-person-out.
Changing it after seeing real results makes the main result meaningless; any change before then goes
in the log with the reason. Core features are also in the random-subset search, like any candidate.

## A. EMG frequency (pooled pass spectra, each bin vs its own fresh baseline)
| # | Feature | Computation | Tired | Proposed by | Status |
|---|---|---|---|---|---|
| 1 | Middle pitch change | Median frequency of the lift's pooled spectrum ÷ the same bins' baseline | ↓ | Adi | **core** |
| 2 | Mean frequency change | Power-weighted mean frequency, same pooling | ↓ | Adi | candidate |
| 3 | Dimitrov index change | Spectral moment ratio M(−1)/M(5), same pooling; reported as more sensitive than middle pitch in dynamic contractions | ↑ | Adi | candidate |
| 4 | Low/high band ratio change | Power 20–50 Hz ÷ power 100–450 Hz | ↑ | Adi | candidate |
| 5 | Spectral shape change | Skewness, kurtosis, entropy of the spectrum | ? | team (earlier) | exploratory |
| 6 | Lowering middle pitch change | #1 using lowering bins only | ↓ | Adi | candidate |
| 7 | Per-bin middle pitch | #1 per angle bin (which angles tire first) | ↓ | Adi | exploratory |

## B. EMG loudness
| # | Feature | Computation | Tired | Proposed by | Status |
|---|---|---|---|---|---|
| 8 | Loudness change | RMS of the lift's passes ÷ the same bins' baseline | ↑ then plateau | Adi | **core** |
| 9 | Loudness as % of max pull | Lift RMS ÷ RMS of the best baseline pull | ↑ | Adi | candidate |
| 10 | Pitch–loudness quadrant | Position in the (Δlog pitch, Δlog loudness) plane: fatigue = pitch↓ loudness↑, effort = both ↑ | – | Adi (sensor testing) | candidate |
| 11 | EMG cost of movement | Loudness ÷ lift speed | ↑ | Adi | candidate |
| 12 | Lifting/lowering loudness ratio | Lifting RMS ÷ lowering RMS | ? | Adi | exploratory |

## C. Trends (Kalman filter, step 7)
| # | Feature | Computation | Tired | Proposed by | Status |
|---|---|---|---|---|---|
| 13a | Middle pitch slope | Kalman slope state of #1 | more negative | Adi | **core** |
| 13b | Loudness slope | Kalman slope state of #8 | more positive | Adi | candidate |
| 14 | Accumulated EMG | Integrated EMG since the set started | ↑ | Adi | shortcut (tracks time into the set) |

## D. IMU movement and form (per lift, from the forearm angle)
| # | Feature | Computation | Tired | Proposed by | Status |
|---|---|---|---|---|---|
| 15 | Speed loss | 1 − peak lift speed ÷ the set's first lifts | ↑ | team / Adi | **core** |
| 16 | Lift duration change | Time to sweep the lifting bins ÷ the first lifts | ↑ | team | candidate |
| 17 | Top-angle change | Lift's top angle − the first lifts' (RepSense drift flag made relative): up = drift/swing, down = partial | ± | Adi (RepSense) | **core** |
| 18 | Bottom-angle change | Lowest angle before the lift − the first lifts' | ↑ | Adi | candidate |
| 19 | Metronome lateness | Lift start vs the expected beat (2 s cycle) | ↑ | Adi | candidate |
| 20 | Sticking point | Depth of a mid-lift speed dip | appears | Adi | candidate |
| 21 | Hinge residual / axis wobble | Accel vs pure-elbow-hinge prediction; how much the rotation axis wanders (RepSense features #20–21) | ↑ | Adi (RepSense) | candidate |
| 22 | Tremor | IMU power 8–12 Hz while holding at the top or bottom | ↑ | Adi | **core** |
| 23 | Electromechanical delay | EMG onset to movement onset | ↑ | Adi | exploratory (20 ms IMU resolution is borderline) |

## E. Context
| # | Feature | Computation | Proposed by | Status |
|---|---|---|---|---|
| 24a | Time into the set | Seconds since the first lift | team | shortcut (baseline 1) |
| 24b | Set number | 1–3 | team | shortcut (baseline 2) |
| 24c | Rest before the set | Seconds | Adi | shortcut (baseline 3) |
| 25 | Set-1 personalisation | This person's #1 and #8 change at their set-1 stop point, used in sets 2–3 | Adi | secondary result |

## Selection protocol: random-subset ranking (Adi's method, as run in RepSense)
Fixed here before any real data, mirroring RepSense FEATURES.md (2026-11-04 entry).
1. **Nested by person.** For each held-out person, the whole ranking runs on the other 7 only, scored
   by leave-one-person-out *within* those 7. The held-out person is touched once, at the end.
2. **Subsets.** 10,000 models, each on a random 6 of the candidate features.
3. **Credit, adjusted.** Each feature's credit is its weight in a regression of subset score on
   which features the subset contained. The raw "mean score with it − without it" is reported next to
   it but doesn't decide. (In RepSense's planted test, raw credit let noise features borrow credit
   from a strong feature they shared subsets with; adjusted credit didn't.)
4. **Luck check.** The same run on shuffled labels, 20 times: a feature counts only if its credit
   beats the shuffled 95th percentile.
5. **Consistency.** Bootstrap the people 20 times; a feature is kept if it passes in ≥ 70% of runs
   and in most held-out folds.
6. **Planted test first.** Before the real run, check the method on simulated data where the
   deciding features are known (one planted feature, the rest noise): it must keep the planted one
   and drop the noise.
7. **Size cap.** Model parameters ≤ 50% of training examples (RepSense rule).
8. **Reporting.** The pre-registered core model is the headline. The search result is exploratory
   and needs new data to be claimed.

Cost note: 10,000 subsets × 8 held-out people × 7 inner folds = 560,000 fits. Fine for straight-line
models (seconds each, parallel); the small neural net only for the final chosen set.

## Log
| Date | Change | Why |
|---|---|---|
| 2026-11-16 | List created: 25 features (13 split into 13a/13b, 24 into 24a–c). Proposed core: #1, #8, #13a, #15, #17, #22 (not yet agreed) | Starting point for discussion; selection protocol fixed before any data |
| 2026-11-16 | **Core locked:** #1, #8, #13a, #15, #17, #22 (Adi) | Pre-registered so the headline is a real test: with 8 people, features picked after seeing results would be partly picked for luck. The nested search stays as the exploratory result |
