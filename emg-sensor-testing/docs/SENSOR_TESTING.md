# MyoWare 2.0 sensor testing: summary

*Bicep fatigue capstone · sessions 2026-10-27 (Person A, Adi Bhagwani) and 2026-10-29 (Person B, Milo Ridenour)*

**Bottom line:** we have enough to go on. The MyoWare 2.0 + ESP32-S3 chain records clean biceps EMG (electromyography: the muscle's electrical signal). It also shows the drop in median frequency that research links to fatigue during loaded holds. Our own pre-set pass rule for the fatigue test was not fully met, so that result is **supportive, not proven**. **Decision: go.** Move on to dynamic curls, the main open risk.

## Setup
- **Hardware:** MyoWare 2.0 (powered from 3.3 V) → ESP32-S3-N16R8. RAW output → GPIO 4 (GPIO 1 in Person A's session), ENV (envelope) output → GPIO 2. The laptop ran on battery. The 1 kΩ / 100 nF anti-aliasing filter is **not fitted**.
- **Firmware:** the ADC (analog-to-digital converter) samples in hardware at 10,000 readings/s per channel with no lost readings, and streams over USB in packets of 10 readings.
- **Analysis (`analyze.py`):** anti-alias filter and downsample to 2,000/s → keep 20–450 Hz → remove 60/120/180 Hz hum.
  - **Loudness** = RMS (root mean square) of the signal.
  - **Middle pitch** = MDF (median frequency), computed for each 0.5 s slice.
  - **Trends** are measured with a straight-line fit. "Chance random" is that fit's p-value.
  - **Effort** = hold loudness ÷ loudness of the strongest squeeze.
- **Electrode placement:** Person A at about the midpoint of the biceps. Person B at the SENIAM standard spot, ⅓ of the way from the elbow crease to the shoulder tip (now the standard for all sessions).

## What we did and what came out

| Test | Person A | Person B |
|---|---|---|
| **1. Signal check** (rest + 3 max squeezes) | PASS 6/6: squeeze 31.5× rest, 0% clipping, hum 0.01%, MDF 72 Hz | PASS 6/6: 28.4× rest, 0% clipping, hum 0.08%, MDF 51 Hz |
| **2. Aliasing check** (rest at 10,000 vs 10,500/s) | PASS: no moving spikes; plain 2,000/s vs filtered MDF 72.2 vs 72.3 Hz | First FAIL (board's own packet noise, see below) → fixed → PASS: 51.2 vs 51.0 Hz |
| **Squeeze/rest comparison** | Squeeze loudness fell 100→89→62% while MDF held at 71–73 Hz: less effort, not fatigue. Rest ≈ 2% of squeeze | Rest 3–4% of squeeze; muscle fully relaxed between squeezes |
| **3. Static holds** | 3 heavy holds (60 s) + 1 light | 1 light hold (60 s) + 3 heavy 10 s weight-finding trials |

**Static holds** (✓ = believable trend, chance random < 5%):

| Hold | Effort | Length | MDF change | Loudness change |
|---|---|---|---|---|
| A light | 22% | 60 s | −2.6% | −24% ✓ (eased off) |
| A heavy 1 / 2 / 3 | 30 / 26 / 26% | 60 s | **−7.8 ✓ / −9.2 ✓ / −14.6% ✓** | +5 / +3 / **+20% ✓** |
| B light | 18% | 60 s | −5.9% (borderline, 5.2%) | −21% ✓ (eased off) |
| B trial 1 / 2 / 3 | 84 / 43 / 74% | 10 s | **−28.1 ✓ / −22.7 ✓** / +15.5% | +21 / −4 / −3% |

## Proven
1. **Reliable data chain:** 10,000 readings/s, 0 lost readings in every recording after the fixes below.
2. **Clean signal in both people:** squeezes 28–31× above rest, no clipping, hum under 0.1%.
3. **No aliasing at 10,000/s.** Plain 2,000/s sampling gives the same MDF (within 0.6%), so the final device can use it.
4. **The sensor tells less effort apart from fatigue.** Less effort lowers loudness while MDF holds steady; fatigue lowers MDF.
5. **MDF falls under load:** a believable drop in **5 of 6 loaded holds** across 2 people. Light holds showed no believable drop, and higher effort gave bigger, faster drops.

## Inconclusive
1. **Pre-set fatigue rule** (MDF −10% or more *and* loudness rising): met by only **1 of 6** loaded holds (A heavy 3).
2. **No 60 s hold at the target 50–60% effort.** A's "heavy" holds were 26–30%; B's loaded holds were 10 s weight-finding trials.
3. **Loudness rising,** the second fatigue sign, was believable in only 1 of 6 loaded holds.
4. **One loaded trial went the other way:** B trial 3, +15.5%, not believable.
5. **Small sample:** 2 people, 1 session each. Person A was on the old electrode spot, A's recovery check was borderline (+5.1% vs a 5% limit), and A's max-squeeze reference may have been below true max.
6. **Dynamic curls,** the real use case, are **untested.**

## Problems found and fixed
| Problem | Fix |
|---|---|
| No data reached the laptop | Firmware built with USB CDC on boot (output over the board's USB port) |
| Opening the port rebooted the board | Logger opens the port normally and discards stale data |
| RAW read from the wrong pin | Firmware reads GPIO 4 |
| Board's USB packets leaked spikes at 78, 156, 234… Hz | 10-reading packets move the leak to 1 kHz steps, outside the muscle band |
| RAW jumper came loose (B's first light hold) | Re-seated and taped the jumper; that recording was discarded |
| No anti-aliasing capacitor | Rest at 10,000 vs 10,500/s (Test 2 spike check) run at the start of every session |

## Next
1. **Test 4, curls,** on a fresh arm at the SENIAM spot, after the spike check.
2. **Optional confirmation:** one 60 s hold at 50–60% effort would upgrade the static result from "supportive" to "passed".
3. **Repeat with 2–3 teammates,** then run the confound tests (changing load, elbow angle, electrode shift).
