# Recordings

Every recording from both test sessions is here, including the broken ones, which are kept as evidence for the debugging log. Each recording has two files:

- **`NAME.csv.gz`**: gzipped CSV with columns `raw,env`. These are 12-bit ADC readings (0–4095) of the MyoWare RAW and ENV outputs, one row per reading.
- **`NAME.json`**: metadata.
  - `rate_hz` is the measured readings per second.
  - `markers` lists `[reading index, label]` pairs, for example the automatic `start`/`end` of timed holds.
  - `board_overflows`, `resyncs`, `board_reboots` and `ok` record lost-data checks.

`analyze.py` reads them by name, for example `python analyze.py signal --rest A_rest --squeeze A_squeeze`.

**Person A:** session 2026-10-27, electrodes at about mid-biceps, RAW on GPIO 1, firmware sending 128 readings per packet.
**Person B:** session 2026-10-29, electrodes at the SENIAM ⅓ spot, RAW on GPIO 4.

## Person A
| Recording | What | Status |
|---|---|---|
| `smoke` | First working 5 s recording after board bring-up | Setup check |
| `check` | Quick check: relax 5 s, squeeze 5 s | Setup check |
| `A_rest` | Relaxed arm, 10 s at 10,000/s | **Valid**: Tests 1–2 |
| `A_rest_10500` | Relaxed arm, 10 s at 10,500/s | **Valid**: Test 2 |
| `A_squeeze` | 3 max squeezes over 75 s | **Valid**: Tests 1–2, max-effort reference |
| `A_light` | 60 s light hold, 22% effort | **Valid**: Test 3 control |
| `A_heavy1`–`A_heavy3` | 60 s "heavy" holds, 26–30% effort (too light) | **Valid**: Test 3 |

## Person B
| Recording | What | Status |
|---|---|---|
| `B_check` | Quick check | **Excluded**: firmware read GPIO 1 while RAW was wired to GPIO 4 (see `results/diagnostics/check_B_check.png`) |
| `B_check2` | Quick check after the pin fix | Diagnostic: the rest half shows muscle activity because the arm wasn't relaxed |
| `B_rest`, `B_rest_10500`, `B_squeeze` | Tests 1–2 on firmware with 128 readings per packet | **Superseded**: packet noise put spikes at 78/156/234… Hz, so Test 2 fails. Kept as the failure example. `B_squeeze` also has an accidental small squeeze at the start |
| `B_rest2`, `B_rest2_10500`, `B_squeeze2` | Tests 1–2 on firmware with 10 readings per packet | **Valid**: Tests 1–2, max-effort reference |
| `B_try1`–`B_try3` | 10 s weight-finding holds at 84%, 43% and 74% effort | **Valid**: B's loaded holds |
| `B_light` | 60 s light hold | **Excluded**: the RAW jumper came loose, so RAW swings rail to rail (see `results/diagnostics/check_B_light.png`) |
| `B_light2` | 60 s light hold, 18% effort, after re-seating the jumper | **Valid**: Test 3 control |
