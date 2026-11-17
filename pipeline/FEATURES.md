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

## Core model (locked 2026-11-16, before any real data; serves as the baseline feature set)
**#1 middle pitch change, #8 loudness change, #13a middle pitch slope, #15 speed loss, #17 top-angle
change, #22 tremor.** The headline result is this model, fitted and scored by leave-one-person-out.
Changing it after seeing real results makes the main result meaningless; any change before then goes
in the log with the reason. Core features are also in the random-subset search, like any candidate.

**Role (Adi, 2026-11-17):** core-v1 is kept unchanged as the baseline feature set. The best features
are chosen empirically by the nested random-subset search; core-v1 is the reference they're compared
against. The step 7 simulation findings below are noted, not acted on.

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

## F. Across sets
| # | Feature | Computation | Tired | Proposed by | Status |
|---|---|---|---|---|---|
| 26 | Session-relative middle pitch | #1 computed against **set 1's** baseline bins instead of this set's own start, so sets 2–3 start below 1.0 by the tiredness carried over from the last rest; available from a set's first lift. Risk: electrodes and sweat drift over ~10 min | ↓ | Adi | candidate |

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

Implementation (2026-11-17, `pipeline/search.py`): subset score = −(mean absolute error in seconds) of a
straight line, leave-one-person-out within the training people. Inner folds share the outer fold's
Kalman q and r (fitted on the training people; the held-out person is never involved). All subsets'
lines are solved in one batch: a full nested run (10,000 subsets, 20 shuffles, 20 bootstraps, 8 people)
takes about a minute. The planted test passes (one deciding feature among ten: only it is kept).
Training and selection run on real recordings only; simulated data is used only to test the code.

Cost note: 10,000 subsets × 8 held-out people × 7 inner folds = 560,000 fits. Fine for straight-line
models (seconds each, parallel); the small neural net only for the final chosen set.

## Log
| Date | Change | Why |
|---|---|---|
| 2026-11-16 | List created: 25 features (13 split into 13a/13b, 24 into 24a–c). Proposed core: #1, #8, #13a, #15, #17, #22 (not yet agreed) | Starting point for discussion; selection protocol fixed before any data |
| 2026-11-16 | **Core locked:** #1, #8, #13a, #15, #17, #22 (Adi) | Pre-registered so the headline is a real test: with 8 people, features picked after seeing results would be partly picked for luck. The nested search stays as the exploratory result |
| 2026-11-16 | Note on #22 (tremor), no status change | Simulated check: with tremor growth switched off, jerky bad reps (swing, partial, failed) still raise the 7–14 Hz band ~25%, so #22 partly measures form-breakdown jerkiness, not only tremor. Slowing alone doesn't fake it (good lifts: corr 0.20 with tiredness) |
| 2026-11-17 | Added #26 (session-relative middle pitch), candidate | Each set is otherwise compared with its own start, which hides the tiredness carried over from a short rest: the information the oracles showed matters most |
| 2026-11-17 | Clarification, no status change: the model reads #1, #8 and #26 as their Kalman-filtered level (step 7), and #22 and #15 as filtered levels; #13a/#13b are the filters' slope states; #17 is unfiltered | Same features, less noise (one lift scatters ~8.4%). #17 is an event (one bad rep), so smoothing would delay it |

| 2026-11-17 | **Simulation findings, step 7 (no status change; core-v1 kept as the baseline set, Adi).** Per set, the middle-pitch slope #13a is mostly noise: true slopes differ between sets by sd 0.46%/s, a straight line through one set's values misses by sd 0.85%/s (signal/noise 0.55; Kalman slope vs truth corr ≈ 0). Loudness slope #13b: 1.97%/s vs 0.55%/s (signal/noise 3.6; corr 0.50), but that rests on the simulator's 40% loudness rise vs 15% pitch drop, and loudness also rises with effort. #26 shows carried-over tiredness: set 2's first lifts read 0.943 ± 0.022 after a 60 s rest vs 1.005 ± 0.019 after 180 s (EMG-blind world ~0.99). Per-lift middle-pitch noise is ~10% and bell-shaped (excess kurtosis 0.1). Options noted for later: keep #13a, swap to #13b, or swap to #26 | Found before any real data. To be settled empirically by the search, with core-v1 as the reference |
| 2026-11-17 | **Step 6b: all 20 candidate features now computed** (#2, #3, #4, #6, #9, #10, #11, #16, #18, #19, #20, #21 added; statuses unchanged). #21 is axis wobble only, ported from RepSense `labels.lift_motion`; RepSense's hinge residual was never implemented there and isn't here. #6 uses the lowering just *before* each lift, so it adds no delay. Simulated checks (12 people): every EMG spectral feature moves with fatigue in the normal world and not in the EMG-blind one; #3 Dimitrov and #4 move ~7× more than middle pitch and *opposite* to effort surges. Caveats: #10 fatigue index is only partly effort-proof (simulated effort raises loudness 4× more than pitch); #11 EMG cost is half speed loss, so not EMG-only. The simulator has no bottom shortfall, sticking points or sideways wobble, so #18, #20 and #21 were checked on real RepSense wrist recordings instead: partial sets bottom +49° vs normal +4°; exhaustion sets sticking dip 0.38 vs 0.02 and lateness 1.0 s vs 0.1 s; #21 drift 2.2° vs normal 0.8°, but alternating, pause and slow sets also 2.4–2.9° (not specific) | Gives the search the full candidate pool: 20 features, 38,760 possible 6-feature subsets |
