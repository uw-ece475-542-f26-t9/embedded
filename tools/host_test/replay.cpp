// Runs the firmware's rep counter, fatigue features and model on a
// computer, for tools/host_test/run.py.
//
// stdin, one sample per line:
//   E <emg index> <mV>
//   I <stamp> <ax> <ay> <az> <gx> <gy> <gz> <dt>   (g, deg/s, s)
// stdout: the model self-test, then one line per lift with the features.

#include <stdio.h>

#include "../../fatigue_sensor/fatigue_model.h"

int main() {
  RepCounter counter;
  FatigueFeatures features;
  printf("selftest %.6f\n", fatigue_model::selfTestError());

  char kind;
  while (scanf(" %c", &kind) == 1) {
    if (kind == 'E') {
      unsigned idx;
      float mv;
      if (scanf("%u %f", &idx, &mv) != 2) return 1;
      features.addEmg(idx, mv);
    } else if (kind == 'I') {
      unsigned stamp;
      Vec3 a, w;
      float dt;
      if (scanf("%u %f %f %f %f %f %f %f", &stamp, &a.x, &a.y, &a.z, &w.x, &w.y, &w.z, &dt) != 8) {
        return 1;
      }
      uint32_t turns = counter.turnCount();
      counter.update(a, w, dt, stamp);
      features.addImu(stamp, norm(w - counter.gyroBias()));
      RepFeatures f;
      if (counter.turnCount() != turns && features.addTurn(counter.lastTurn(), &f)) {
        float rir = f.hasBaseline ? fatigue_model::predict(f) : -1;
        printf("rep %u %u %u %d %d %.4f %.4f %.4f %.4f %.4f %.5f %.5f %.5f %.5f %.5f %.4f\n",
               f.repInSet, f.liftFrom, f.liftTo, f.emgOk, f.hasBaseline, f.peakDps,
               f.movingS, f.romDeg, f.emgRmsMv, f.mdfHz, f.velocityLoss, f.durationRatio,
               f.romRatio, f.emgRatio, f.mdfRatio, rir);
      }
    }
  }
  return 0;
}
