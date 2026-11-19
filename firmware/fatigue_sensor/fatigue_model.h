// The on-device fatigue model: estimates reps in reserve (RIR, how many
// more reps until failure) from one rep's features.
//
// A small neural network, 6 inputs -> 16 ReLU -> 8 ReLU -> 1, with int8
// weights and one float scale per layer; the maths is in float, which the
// ESP32-S3 does in hardware. tools/train_model.py trains it and writes the
// weights to fatigue_model_data.h. A prediction takes a few microseconds.
//
// No Arduino code in here, so it can be tested on a computer.

#pragma once

#include <math.h>

#include "fatigue_features.h"
#include "fatigue_model_data.h"

namespace fatigue_model {

// The inputs in the order the model was trained on.
inline void inputsFrom(const RepFeatures &f, float x[N_IN]) {
  x[0] = f.repInSet;
  x[1] = f.velocityLoss;
  x[2] = f.durationRatio;
  x[3] = f.romRatio;
  x[4] = f.emgRatio;
  x[5] = f.mdfRatio;
}

// Estimated reps in reserve, never below zero.
inline float predict(const float in[N_IN]) {
  float x[N_IN], h1[N_H1], h2[N_H2];
  for (int i = 0; i < N_IN; i++) x[i] = (in[i] - IN_MEAN[i]) / IN_STD[i];
  for (int j = 0; j < N_H1; j++) {
    float s = 0;
    for (int i = 0; i < N_IN; i++) s += W1[j][i] * x[i];
    h1[j] = fmaxf(s * W1_SCALE + B1[j], 0);
  }
  for (int j = 0; j < N_H2; j++) {
    float s = 0;
    for (int i = 0; i < N_H1; i++) s += W2[j][i] * h1[i];
    h2[j] = fmaxf(s * W2_SCALE + B2[j], 0);
  }
  float s = 0;
  for (int i = 0; i < N_H2; i++) s += W3[i] * h2[i];
  return fmaxf(s * W3_SCALE + B3, 0);
}

inline float predict(const RepFeatures &f) {
  float x[N_IN];
  inputsFrom(f, x);
  return predict(x);
}

// Largest difference, in reps, between this code and the Python model on
// the test vectors in fatigue_model_data.h. Should be well under 0.01.
inline float selfTestError() {
  float worst = 0;
  for (int t = 0; t < N_TEST; t++) worst = fmaxf(worst, fabsf(predict(TEST_IN[t]) - TEST_OUT[t]));
  return worst;
}

}  // namespace fatigue_model
