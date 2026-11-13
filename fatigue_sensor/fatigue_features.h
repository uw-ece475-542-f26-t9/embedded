// Measures each rep for the fatigue model.
//
// It keeps the last ~8 s of band-passed EMG and the last ~10 s of IMU
// turning speed. Each time the rep counter confirms a turning point, the
// swing that just ended is complete; if it was a lift (the swing up with
// the weight), it's measured:
//
//   peak speed   fastest the forearm turned during the lift (deg/s)
//   moving time  time spent turning faster than MOVING_DPS during it (s)
//   range        how far the forearm turned (deg)
//   EMG RMS      over the lift, after a 20-400 Hz band-pass and 60 Hz notch
//   EMG MDF      median frequency from 20-400 Hz of the lift's spectrum,
//                averaged over 256 ms Hann frames that overlap by half
//                (Welch). One FFT of one window is a noisy estimate; on
//                simulated EMG averaging cut the rep-to-rep scatter from
//                ±13% to ±6%. Louder parts of the lift count for more.
//
// Each is then divided by the median over the set's first BASELINE_REPS
// lifts, so the model sees how this rep compares with the start of the set
// rather than absolute values that change with the person, the weight and
// where the electrodes sit. A median, so one odd rep at the start (a jerk,
// a half rep, picking the weight up) doesn't skew every rep after it.
// Predictions start at the rep after the baseline.
//
// Which swing is the lift: the rep counter's angle has no fixed up, since
// its sign depends on how the IMU is mounted. So once the set's first two
// swings are both complete, the one with more EMG is taken as the lift
// (lifting the weight takes more muscle than lowering it), and every lift
// of that set is a swing in the same direction. Without EMG, the first
// swing is taken as the lift, as a set normally starts with the arm
// straight. Either way the first rep is measured too.
//
// A new set starts after a swing that took longer than SET_GAP_S: a rest.
// A turning point where the arm rested is stamped when the next swing
// began, so the rest is inside the swing before it.
//
// Timestamps are EMG sample indexes (2 kHz), the same stamps the rep
// counter's turning points carry.
//
// No Arduino code in here, so it can be tested on a computer.

#pragma once

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "rep_counter.h"

struct RepFeatures {
  uint16_t repInSet = 0;  // 1 for the first lift of the set

  // This rep's lift: when it started and ended, and what it measured.
  uint32_t liftFrom = 0, liftTo = 0;
  float peakDps = 0;
  float movingS = 0;
  float romDeg = 0;
  float emgRmsMv = 0;
  float mdfHz = 0;
  bool emgOk = false;  // enough EMG was recorded to measure it

  // Relative to the start of the set. Only meaningful once hasBaseline,
  // from the lift after the baseline lifts; the EMG ratios are 1 when
  // there's no EMG, so the model leans on motion.
  bool hasBaseline = false;
  float velocityLoss = 0;   // 1 - peak speed / baseline
  float durationRatio = 1;  // moving time / baseline
  float romRatio = 1;
  float emgRatio = 1;       // EMG RMS / baseline
  float mdfRatio = 1;       // EMG MDF / baseline
};

class FatigueFeatures {
 public:
  static constexpr uint32_t EMG_HZ = 2000;
  static constexpr uint32_t IMU_HZ = 50;
  static constexpr float MOVING_DPS = 20;
  static constexpr int BASELINE_REPS = 3;
  static constexpr float SET_GAP_S = 15;

  // Spectrum frames: 512 samples (256 ms), 3.9 Hz bins, every 256 samples.
  static constexpr int FFT_N = 512;
  static constexpr int FFT_HOP = FFT_N / 2;
  static constexpr float MDF_LO_HZ = 20;
  static constexpr float MDF_HI_HZ = 400;

  static constexpr int EMG_RING = 16384;  // 8.2 s
  static constexpr int IMU_RING = 512;    // 10 s

  // Band-pass 20-400 Hz (2nd-order Butterworth, two sections) then a 60 Hz
  // notch with Q 30, at 2 kHz. Rows are b0 b1 b2 a0 a1 a2, from
  // scipy.signal.butter(2, [20, 400], "bandpass", fs=2000, output="sos")
  // and iirnotch(60, 30, fs=2000).
  static constexpr int SECTIONS = 3;
  static constexpr float SOS[SECTIONS][6] = {
      {0.1906166f, 0.3812332f, 0.1906166f, 1, -0.423531165f, 0.225709868f},
      {1, -2, 1, 1, -1.91155124f, 0.9156533f},
      {0.996868236f, -1.95842192f, 0.996868236f, 1, -1.95842192f, 0.993736472f}};

  FatigueFeatures() {
    for (int i = 0; i < FFT_N; i++) hann[i] = 0.5f - 0.5f * cosf(2 * (float)M_PI * i / FFT_N);
  }

  // One EMG sample in mV, with its sample index. Indexes must count up by
  // one per sample.
  void addEmg(uint32_t index, float mv) {
    float x = mv;
    for (int s = 0; s < SECTIONS; s++) {
      const float *c = SOS[s];
      float y = c[0] * x + z[s][0];
      z[s][0] = c[1] * x - c[4] * y + z[s][1];
      z[s][1] = c[2] * x - c[5] * y;
      x = y;
    }
    // Stored in 0.1 mV; the MyoWare can't swing more than ±3.3 V.
    float q = fminf(fmaxf(x * 10, -32767), 32767);
    emg[index % EMG_RING] = (int16_t)lroundf(q);
    emgNewest = index;
    if (emgCount < EMG_RING) emgCount++;
  }

  // Turning speed about the elbow (deg/s, gyro bias removed) at one IMU
  // sample.
  void addImu(uint32_t stamp, float dps) {
    imu[imuHead] = {stamp, dps};
    imuHead = (imuHead + 1) % IMU_RING;
    if (imuCount < IMU_RING) imuCount++;
  }

  // Every turning point the rep counter confirms. Returns true if it ended
  // a lift, which `out` then describes.
  bool addTurn(TurningPoint t, RepFeatures *out) {
    if (!havePrev) {
      prev = t;
      havePrev = true;
      return false;
    }
    TurningPoint from = prev;
    prev = t;
    if (t.stamp - from.stamp > SET_GAP_S * EMG_HZ) {
      newSet();  // that swing had a rest in it
      return false;
    }
    int dir = t.angleDeg > from.angleDeg ? 1 : -1;
    if (liftDir != 0) {
      if (dir != liftDir) return false;
      *out = measure(from, t);
      lift(out);
      return true;
    }

    // Start of a set: wait for two swings, then pick the lift.
    RepFeatures m = measure(from, t);
    if (!firstSwingDone || dir == firstDir) {
      firstSwing = m;
      firstDir = dir;
      firstSwingDone = true;
      return false;
    }
    bool emg = firstSwing.emgOk && m.emgOk;
    liftDir = emg && m.emgRmsMv > firstSwing.emgRmsMv ? dir : firstDir;
    *out = liftDir == dir ? m : firstSwing;
    lift(out);
    return true;
  }

  uint16_t repsInSet() const { return repInSet; }
  // +1 or -1 once the set's lift direction is known, 0 before.
  int liftDirection() const { return liftDir; }

  // Starts a new set now (otherwise one starts after a rest of SET_GAP_S).
  void newSet() {
    repInSet = 0;
    liftDir = 0;
    firstSwingDone = false;
  }

 private:
  struct ImuSample {
    uint32_t stamp;
    float dps;
  };

  RepFeatures measure(const TurningPoint &from, const TurningPoint &to) {
    RepFeatures f;
    f.liftFrom = from.stamp;
    f.liftTo = to.stamp;
    f.romDeg = fabsf(to.angleDeg - from.angleDeg);
    measureMotion(from.stamp, to.stamp, &f);
    measureEmg(from.stamp, to.stamp, &f);
    return f;
  }

  // Counts a lift in the set and compares it with the baseline.
  void lift(RepFeatures *f) {
    f->repInSet = ++repInSet;
    if (repInSet <= BASELINE_REPS) {
      base[repInSet - 1] = *f;
      if (repInSet == BASELINE_REPS) setBaseline();
      return;
    }
    f->hasBaseline = basePeak > 0;
    if (!f->hasBaseline) return;
    f->velocityLoss = 1 - f->peakDps / basePeak;
    f->durationRatio = baseMoving > 0 ? f->movingS / baseMoving : 1;
    f->romRatio = baseRom > 0 ? f->romDeg / baseRom : 1;
    if (f->emgOk && baseEmg > 0 && baseMdf > 0) {
      f->emgRatio = f->emgRmsMv / baseEmg;
      f->mdfRatio = f->mdfHz / baseMdf;
    }
  }

  // Median of up to 3 values (the mean of 2).
  static float median(float *v, int n) {
    if (n == 0) return 0;
    if (n == 1) return v[0];
    if (n == 2) return (v[0] + v[1]) / 2;
    return fmaxf(fminf(v[0], v[1]), fminf(fmaxf(v[0], v[1]), v[2]));
  }

  void setBaseline() {
    float peak[BASELINE_REPS], moving[BASELINE_REPS], rom[BASELINE_REPS];
    float rms[BASELINE_REPS], mdf[BASELINE_REPS];
    int nEmg = 0;
    for (int i = 0; i < BASELINE_REPS; i++) {
      peak[i] = base[i].peakDps;
      moving[i] = base[i].movingS;
      rom[i] = base[i].romDeg;
      if (base[i].emgOk) {
        rms[nEmg] = base[i].emgRmsMv;
        mdf[nEmg++] = base[i].mdfHz;
      }
    }
    basePeak = median(peak, BASELINE_REPS);
    baseMoving = median(moving, BASELINE_REPS);
    baseRom = median(rom, BASELINE_REPS);
    baseEmg = median(rms, nEmg);
    baseMdf = median(mdf, nEmg);
  }

  void measureMotion(uint32_t from, uint32_t to, RepFeatures *f) {
    int moving = 0;
    for (int k = 0; k < imuCount; k++) {
      const ImuSample &s = imu[(imuHead + IMU_RING - 1 - k) % IMU_RING];
      if (s.stamp < from || s.stamp > to) continue;
      f->peakDps = fmaxf(f->peakDps, s.dps);
      if (s.dps > MOVING_DPS) moving++;
    }
    f->movingS = (float)moving / IMU_HZ;
  }

  // Oldest EMG index still in the ring.
  uint32_t emgOldest() const { return emgNewest + 1 - emgCount; }

  float emgAt(uint32_t index) const { return emg[index % EMG_RING] * 0.1f; }

  // RMS of the stored EMG over [from, to), or NaN if it isn't all there.
  float emgRms(uint32_t from, uint32_t to) const {
    if (emgCount == 0 || to <= from || from < emgOldest() || to > emgNewest + 1) {
      return NAN;
    }
    double sum = 0;
    for (uint32_t i = from; i < to; i++) {
      float v = emgAt(i);
      sum += v * v;
    }
    return sqrtf(sum / (to - from));
  }

  void measureEmg(uint32_t from, uint32_t to, RepFeatures *f) {
    // A lift longer than the ring keeps its end.
    if (emgCount == 0) return;
    if (from < emgOldest()) from = emgOldest();
    if (to > emgNewest + 1) to = emgNewest + 1;
    if (to < from + FFT_N) return;
    f->emgRmsMv = emgRms(from, to);
    f->mdfHz = medianFrequency(from, to);
    f->emgOk = f->emgRmsMv > 0 && f->mdfHz > 0;
  }

  // Median frequency of the power spectrum averaged over frames in
  // [from, to), or 0 if there's no power in the band.
  float medianFrequency(uint32_t from, uint32_t to) {
    const float binHz = (float)EMG_HZ / FFT_N;
    const int lo = (int)ceilf(MDF_LO_HZ / binHz), hi = (int)floorf(MDF_HI_HZ / binHz);
    for (int k = 0; k <= hi; k++) power[k] = 0;
    for (uint32_t start = from; start + FFT_N <= to; start += FFT_HOP) {
      for (int i = 0; i < FFT_N; i++) {
        re[i] = emgAt(start + i) * hann[i];
        im[i] = 0;
      }
      fft(re, im, FFT_N);
      for (int k = lo; k <= hi; k++) power[k] += re[k] * re[k] + im[k] * im[k];
    }
    double total = 0;
    for (int k = lo; k <= hi; k++) total += power[k];
    if (total <= 0) return 0;
    // Where the running total passes half, interpolated within that bin.
    double cum = 0;
    for (int k = lo; k <= hi; k++) {
      if (cum + power[k] >= total / 2) {
        double frac = power[k] > 0 ? (total / 2 - cum) / power[k] : 0;
        return (float)((k - 0.5 + frac) * binHz);
      }
      cum += power[k];
    }
    return hi * binHz;
  }

  // In-place radix-2 FFT; n must be a power of two.
  static void fft(float *re, float *im, int n) {
    for (int i = 1, j = 0; i < n; i++) {
      int bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        float t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (int len = 2; len <= n; len <<= 1) {
      float ang = -2 * (float)M_PI / len;
      for (int k = 0; k < len / 2; k++) {
        float wr = cosf(ang * k), wi = sinf(ang * k);
        for (int i = k; i < n; i += len) {
          int j = i + len / 2;
          float tr = re[j] * wr - im[j] * wi, ti = re[j] * wi + im[j] * wr;
          re[j] = re[i] - tr;
          im[j] = im[i] - ti;
          re[i] += tr;
          im[i] += ti;
        }
      }
    }
  }

  // EMG
  float z[SECTIONS][2] = {};
  int16_t emg[EMG_RING] = {};
  uint32_t emgNewest = 0;
  int emgCount = 0;
  float re[FFT_N], im[FFT_N], power[FFT_N / 2 + 1];
  float hann[FFT_N];

  // IMU
  ImuSample imu[IMU_RING] = {};
  int imuHead = 0;
  int imuCount = 0;

  // The last turning point
  TurningPoint prev = {};
  bool havePrev = false;

  // This set
  uint16_t repInSet = 0;
  int liftDir = 0;
  RepFeatures firstSwing;  // waiting for the second swing to pick the lift
  int firstDir = 0;
  bool firstSwingDone = false;
  RepFeatures base[BASELINE_REPS];
  float basePeak = 0, baseMoving = 0, baseRom = 0, baseEmg = 0, baseMdf = 0;
};
