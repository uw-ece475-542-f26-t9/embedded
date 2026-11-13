# Team 9 Embedded Code

Authors: Milo Ridenour, Bernardo Lin, Adi Bhagwani, James Lee
ECE 475/542 Team 9

Embedded code for the esp-32.

| Folder | What it is |
|---|---|
| `fatigue_sensor/` | The wearable's firmware: MyoWare 2.0 EMG, MPU-6050 IMU with rep counter, DRV2605L haptic, the on-device fatigue model, BLE to the app, and a USB capture mode for EMG analysis |
| `tools/` | Python: record capture sessions, analyse EMG fatigue features, train the on-device model, simulate recordings, host tests |
| `imu_test/`, `myoware_test/` | Early bring-up sketches |

## Firmware

Open `fatigue_sensor/fatigue_sensor.ino` in the Arduino IDE with the
**esp32 by Espressif** core (3.x) and the **Adafruit MPU6050** and
**Adafruit DRV2605 Library** libraries. Select **XIAO_ESP32S3** and flash.

- **EMG** is sampled by the ADC in continuous (DMA) mode at 8 kHz and
  decimated to 2 kHz with a 31-tap FIR low-pass (`emg_adc.h`). Sampling in
  the background means IMU reads and BLE calls can't delay samples. The old
  polled `analogReadMilliVolts` stalled ~2 ms on each IMU read, which put a
  50 Hz artifact in the spectrum.
- **Reps** are counted from the IMU (`rep_counter.h`). The counter also
  records when each turning point happened, so an analysis can split
  recordings into reps.
- **Fatigue model** runs after every rep; see below.
- **BLE** IMU and EMG packets are unchanged. The rep packet grew from 8 to
  12 bytes with the model's output (below); apps that read only the first 8
  still work. See `mobile-app/lib/wearable/protocol.dart`.
- **Serial monitor** (115200): a status line every second, a block per rep,
  and commands: `i` model and sensor info, `h` test the haptic alert, `n`
  start a new set, `f` I'm at failure (see below).

## On-device fatigue model

After each rep the ESP32 estimates **reps in reserve** (RIR: reps left
before failure), and buzzes the haptic once it's at `ALERT_RIR` (2) or
fewer.

1. **Features** (`fatigue_features.h`). From the rep's lift: peak turning
   speed, moving time, range of motion, EMG RMS (20–400 Hz band-pass, 60 Hz
   notch) and EMG median frequency (spectrum averaged over 256 ms frames).
   Each is divided by its median over the set's first 3 reps, so the
   model sees change within the set, not absolute levels, and one odd rep
   at the start doesn't skew the rest. Predictions start at rep 4. A new
   set starts after a 15 s rest. The lift is the half of the rep with more
   EMG, picked once the first rep is complete, so it works whichever way
   up the board is mounted.
2. **Model** (`fatigue_model.h`, weights in `fatigue_model_data.h`). A small
   neural network (6 → 16 → 8 → 1) with int8 weights: 232 bytes, about
   20 µs per prediction. At boot it checks it gives the same answers as
   Python on test vectors stored with the weights (`i` shows the result).
3. **Output.** Serial log, BLE rep packet bytes 8–11 (u16 RIR × 10, `0xFFFF`
   before rep 3; u8 % of the set done, `0xFF` before rep 3; u8 flags, bit 0
   = alert sent this set), and a FEAT frame in capture mode.
4. **Checking it by hand.** At the end of a set taken to failure, type `f`
   in the serial monitor. It lists the model's estimate at each rep next to
   the true reps in reserve, with the average error, and starts a new set.

**The model in the repo is trained on synthetic data.** It proves the
pipeline works end to end on the board; its numbers mean nothing about real
fatigue yet. To train on real data:

1. Flash with `CAPTURE_MODE 1` and record fatigue sets to failure with
   `tools/record.py` (enter the failure rep when asked). The firmware puts
   its own per-rep features in the recording, so the model trains on
   exactly what the device measures.
2. Train. This prints leave-one-person-out accuracy against a velocity-only
   baseline and overwrites `fatigue_sensor/fatigue_model_data.h`:

   ```bash
   .venv/bin/python tools/train_model.py recordings/
   ```

3. Set `CAPTURE_MODE 0` and flash.

`--synthetic` instead of a folder regenerates the synthetic model.

To check the feature and model code on a computer (builds it with clang,
runs simulated sets through it, and compares with numpy):

```bash
.venv/bin/python tools/host_test/run.py
```

## EMG fatigue feature test (MVP)

Question: can our hardware see fatigue in four sEMG features (median
frequency, mean frequency, skewness and kurtosis of the power spectrum)
clearly enough to build on? The ESP32 records; a laptop computes the
features.

### 1. Flash capture mode

In `fatigue_sensor.ino`, set `#define CAPTURE_MODE 1` and flash. The board
then streams every EMG sample (2 kHz), every IMU sample (50 Hz), and the
turning points and reps over USB as framed binary (~6.5 KB/s; format in
`capture.h`). Text logging is off in this mode; BLE still works.

- On a dev kit's **UART** USB port, the stream runs at 921600 baud
  (`CAPTURE_BAUD`).
- On the S3's **native USB** port, enable *Tools → USB CDC On Boot*; baud
  doesn't matter.

Set it back to `0` for normal use.

### 2. Python setup (once)

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r tools/requirements.txt
```

### 3. Record

**Safety: keep the laptop on battery, charger unplugged, whenever electrodes
are on and USB is connected.** This also cuts 60 Hz pickup.

Setup: use the same electrode spot every time (mark it), and clean the skin
with alcohol. Use a metronome at 2 s up, 2 s down. Record one file per set:

```bash
.venv/bin/python tools/record.py --subject S01 --session 1 --set rest
```

```bash
.venv/bin/python tools/record.py --subject S01 --session 1 --set control --weight-kg 4
```

```bash
.venv/bin/python tools/record.py --subject S01 --session 1 --set fatigue --weight-kg 10
```

- `rest`: about 20 s with the arm relaxed. This is the noise baseline for
  the sanity check.
- `control`: a light weight, 12 reps. It shouldn't cause fatigue.
- `fatigue`: a weight that fails at around 10–15 reps, taken to failure.
  Do two per session, with 3–5 min rest between sets.

A live line shows EMG level, reps, and any clipping or stream problems.
Press **Ctrl-C** when the set ends; it then asks for the failure rep and RPE.
Files go to `recordings/` (git-ignored): a `.bin` with the raw stream and a
`.json` with the labels. Aim for 2–3 people with 2 sessions each.

### 4. Analyse

```bash
.venv/bin/python tools/fatigue_features.py recordings/
```

For every rep it takes the lift (the half-rep with more EMG) and a 500 ms
window centred on its EMG peak, band-passes 20–400 Hz and notches 60 Hz,
then computes MDF, MNF, skewness and kurtosis two ways:

- **STFT:** 128 ms Hann frames with a 64 ms hop.
- **Single FFT:** one Hann-windowed FFT over the whole window.

It also logs RMS and lift duration, to rule out confounds. The output goes
to `analysis/<timestamp>/`:

- **`results.md`:** the feature × method table, marked pass or fail with
  effect sizes, plus which method goes forward, the sanity checks, the
  confounds and any warnings.
- **`plots/`:** per set, an overview (EMG with rep windows, arm angle with
  turning points, spectrum) and the features against rep number. Also
  `summary.png` with all sets overlaid.
- **`reps.csv`, `results.csv`, `sanity.csv`:** the numbers behind the
  report.

**Sanity checks, per set.** All must pass before the table means anything:

- ≥80% of the 20–400 Hz power sits below 250 Hz.
- No 60 Hz spike is left after the notch (under 2× its neighbours).
- No ADC clipping.
- Contraction RMS is ≥5× resting RMS.
- The stream is clean: no CRC errors, gaps or ADC overflows.

**Pass criteria, per feature and method.** Each set is normalized to the
mean of its first 3 reps.

- **Trend:** in most fatiguing sets, the slope against rep number is
  significant (p < 0.05) and in the expected direction. MDF and MNF should
  drop. For skewness and kurtosis, the direction is found from the data.
- **Size:** the change from the first 3 reps to the last 3 is more than 2×
  the scatter around the trend line.
- **Specificity:** no control set shows such a trend, meaning significant
  and large in that direction.
- **Repeatability:** in every person-session, most fatiguing sets pass trend
  and size. This needs at least 2 people.

The thresholds are constants at the top of `fatigue_features.py`.

### Testing the pipeline without hardware

```bash
.venv/bin/python tools/simulate_recording.py --out /tmp/sim
```

```bash
.venv/bin/python tools/fatigue_features.py /tmp/sim --out /tmp/sim-out
```

The simulator writes the exact capture format. The fatigued sets' EMG
spectrum compresses about 25% over the set, and the control sets' doesn't.
MDF and MNF should pass; skewness and kurtosis, which it doesn't model,
should fail. This proves the pipeline only, not anything about muscle.

### Limits to know about

- **Stamp accuracy.** IMU, turning-point and rep stamps are the EMG sample
  count when the IMU was read. The ADC delivers in 8 ms blocks, so they can
  lag by up to ~10 ms. Turning points where the arm pauses can be off by
  ~0.2 s. Neither matters for a 500 ms window centred on the EMG peak.
- **Sample rate.** The ADC's real rate can differ slightly from 8 kHz. The
  analysis measures it against the ESP32's clock (STATUS frames) and uses
  the measured rate for frequencies.
- **Mains harmonics.** Only 60 Hz is notched, as planned. The report shows
  the 120 and 180 Hz levels, so you can tell whether they need notching too.
