// Counts bicep curls from an IMU strapped to the forearm, however it's
// mounted.
//
// The forearm turns about the elbow, so during a curl the gyro spins mostly
// about one axis that is fixed relative to the sensor. We learn that axis as
// the main direction of rotation, integrate the turning about it into an arm
// angle, and count a swing each time the angle moves SWING_DEG away from the
// last turning point. A swing up plus a swing back down is one rep, counted
// part way down.
//
// The angle is relative (zero is wherever the arm was at boot) and would
// drift with gyro bias. So the bias is learned whenever the arm is still, and
// the angle is frozen while it's still. A swing only compares the angle with
// where it was a second or two earlier, so what drift is left doesn't matter.
//
// Each turning point is also recorded with the caller's timestamp for the
// moment the arm was there (the capture build passes the EMG sample index),
// so an analysis can split the recording into reps and half-reps. If the
// arm rests at a turning point, its time is the end of the rest, when the
// next swing began.
//
// No Arduino code in here, so it can be tested on a computer.

#pragma once

#include <math.h>
#include <stdint.h>

struct Vec3 {
  float x, y, z;
};

// Where the arm turned around.
struct TurningPoint {
  uint32_t stamp;  // caller's timestamp when the arm was there
  float angleDeg;  // arm angle there
  int8_t top;      // +1 if the angle peaked here, -1 if it bottomed out
};

inline Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(float k, Vec3 v) { return {k * v.x, k * v.y, k * v.z}; }
inline float dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline float norm(Vec3 v) { return sqrtf(dot(v, v)); }

class RepCounter {
 public:
  // Half a rep: how far the arm has to swing one way, in degrees. A full curl
  // is ~120-140 deg; this lets short reps count but ignores small movements.
  static constexpr float SWING_DEG = 45;

  // Still = accelerometer steady (staying within STILL_G of where it was
  // when it settled, i.e. tilting under ~2 deg/s) and gyro quiet (under
  // STILL_DPS once the bias is known), for at least STILL_S.
  static constexpr float STILL_G = 0.02f;
  static constexpr float STILL_DPS = 4;
  static constexpr float STILL_S = 0.5f;

  // Gyro bias is the average over all still time so far, up to the latest
  // BIAS_TAU_S. It settles quickly after boot, and the start of a movement at
  // the end of a still spell barely moves it. Until it has BIAS_SETTLED_S of
  // still time behind it, the gyro only has to be under UNSETTLED_DPS, and
  // STILL_LONG_S of steady accelerometer counts as still whatever the gyro
  // says, because an MPU-6050's bias can be well over STILL_DPS.
  static constexpr float BIAS_TAU_S = 5;
  static constexpr float BIAS_SETTLED_S = 1;
  static constexpr float UNSETTLED_DPS = 15;
  static constexpr float STILL_LONG_S = 3;

  // Resting within this of a turning point counts as resting at it.
  static constexpr float REST_AT_TURN_DEG = 10;

  // Only learn the elbow axis from clear rotation, averaged over about a
  // second of movement.
  static constexpr float AXIS_MIN_DPS = 30;
  static constexpr float AXIS_TAU_S = 1;

  // a in g, w in deg/s straight from the IMU, dt in seconds since the last
  // call, stamp the caller's timestamp for this sample (only used to time
  // turning points). Returns true when a rep has just been counted.
  bool update(Vec3 a, Vec3 w, float dt, uint32_t stamp = 0) {
    if (dt <= 0) return false;
    now = stamp;
    if (!started) {
      aSmooth = steadyAt = a;
      started = true;
    }
    // Smoothing takes the edge off sensor noise and hand tremor.
    aSmooth = aSmooth + smoothing(dt, 0.1f) * (a - aSmooth);
    wSmooth = wSmooth + smoothing(dt, 0.1f) * (w - wSmooth);
    Vec3 wc = w - bias;

    if (norm(aSmooth - steadyAt) < STILL_G) {
      steadyFor += dt;
    } else {
      steadyAt = aSmooth;
      steadyFor = 0;
    }
    bool settled = stillTime >= BIAS_SETTLED_S;
    float quietDps = settled ? STILL_DPS : UNSETTLED_DPS;
    quietFor = steadyFor > 0 && norm(wSmooth - bias) < quietDps ? quietFor + dt : 0;
    still = quietFor >= STILL_S || (!settled && steadyFor >= STILL_LONG_S);
    if (still) {
      stillTime = fminf(stillTime + dt, BIAS_TAU_S);
      bias = bias + (dt / stillTime) * (w - bias);
      // Resting at the last extreme: the turning point is when it leaves.
      if (dir == 0) {
        turnStamp = stamp;
      } else if (fabsf(angle - peak) < REST_AT_TURN_DEG) {
        peakStamp = stamp;
      }
      return false;
    }

    if (norm(wc) > AXIS_MIN_DPS) learnAxis(wc, dt);
    angle += dot(wc, axis) * dt;
    return trackSwing();
  }

  // Reps counted since boot.
  uint16_t reps() const { return repCount; }
  // Range of motion of the latest rep, in degrees.
  float romDeg() const { return rom; }
  // Arm angle about the elbow axis, in degrees from wherever it was at boot.
  float angleDeg() const { return angle; }
  bool isStill() const { return still; }
  Vec3 gyroBias() const { return bias; }
  // Turning points confirmed since boot; check it after update() to spot a
  // new one, then read it with lastTurn().
  uint32_t turnCount() const { return turns; }
  TurningPoint lastTurn() const { return lastTurnPoint; }

 private:
  // Fraction to move a running average by, for time constant tau.
  static float smoothing(float dt, float tau) { return dt >= tau ? 1 : dt / tau; }

  void learnAxis(Vec3 w, float dt) {
    // Running average of w * w^T (symmetric, so six terms).
    float k = smoothing(dt, AXIS_TAU_S);
    cxx += k * (w.x * w.x - cxx);
    cyy += k * (w.y * w.y - cyy);
    czz += k * (w.z * w.z - czz);
    cxy += k * (w.x * w.y - cxy);
    cxz += k * (w.x * w.z - cxz);
    cyz += k * (w.y * w.z - cyz);
    // One step of power iteration per sample: the axis settles on the main
    // direction of rotation without flipping sign.
    Vec3 next = {cxx * axis.x + cxy * axis.y + cxz * axis.z,
                 cxy * axis.x + cyy * axis.y + cyz * axis.z,
                 cxz * axis.x + cyz * axis.y + czz * axis.z};
    float n = norm(next);
    if (n > 1e-6f) axis = (1 / n) * next;
  }

  bool trackSwing() {
    if (dir == 0) {
      // First swing since boot, either way. Where it started counts as a
      // turning point too.
      if (fabsf(angle - turn) < SWING_DEG) return false;
      dir = angle > turn ? 1 : -1;
      recordTurn(turnStamp, turn, -dir);
      peak = angle;
      peakStamp = now;
      swings++;
      return false;
    }
    if ((angle - peak) * dir > 0) {
      peak = angle;  // still swinging the same way
      peakStamp = now;
      return false;
    }
    if ((peak - angle) * dir < SWING_DEG) return false;

    // Swung back far enough that `peak` was a turning point.
    float swing = fabsf(peak - turn);
    recordTurn(peakStamp, peak, dir);
    turn = peak;
    turnStamp = peakStamp;
    peak = angle;
    peakStamp = now;
    dir = -dir;
    if (++swings % 2 != 0) return false;
    repCount++;
    rom = swing;
    return true;
  }

  void recordTurn(uint32_t stamp, float angleDeg, int8_t top) {
    lastTurnPoint = {stamp, angleDeg, top};
    turns++;
  }

  bool started = false;
  Vec3 aSmooth = {0, 0, 0};
  Vec3 steadyAt = {0, 0, 0};  // smoothed accel when it last settled
  Vec3 wSmooth = {0, 0, 0};
  Vec3 bias = {0, 0, 0};
  float steadyFor = 0;  // accel steady this long
  float quietFor = 0;   // accel steady and gyro quiet this long
  float stillTime = 0;  // total still time, capped at BIAS_TAU_S
  bool still = false;

  Vec3 axis = {1, 0, 0};
  float cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
  float angle = 0;

  int8_t dir = 0;  // which way the arm is swinging, 0 until the first swing
  float turn = 0;  // angle at the last turning point
  float peak = 0;  // furthest angle so far in this swing
  uint32_t swings = 0;
  uint16_t repCount = 0;
  float rom = 0;

  uint32_t now = 0;         // stamp of the current update()
  uint32_t turnStamp = 0;   // when the arm was at `turn`
  uint32_t peakStamp = 0;   // when the arm was at `peak`
  uint32_t turns = 0;
  TurningPoint lastTurnPoint = {0, 0, 0};
};
