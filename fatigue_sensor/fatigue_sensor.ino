#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <Wire.h>
#include <Adafruit_DRV2605.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "capture.h"
#include "emg_adc.h"
#include "fatigue_features.h"
#include "fatigue_model.h"
#include "rep_counter.h"

// 1 = binary USB capture for tools/record.py, 0 = normal text logs.
#define CAPTURE_MODE 0
#define CAPTURE_BAUD 921600

#if CAPTURE_MODE
#define LOGF(...) do {} while (0)
#define LOGLN(...) do {} while (0)
#else
#define LOGF(...) Serial.printf(__VA_ARGS__)
#define LOGLN(...) Serial.println(__VA_ARGS__)
#endif

#define DEVICE_NAME        "Team9-Fatigue-Sensor"
#define SERVICE_UUID       "7fc2f476-4996-49df-b915-54e84d61fedb"
#define IMU_CHAR_UUID      "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define EMG_CHAR_UUID      "53e27fb3-6de1-4f27-8303-d663c79b818b"
#define REP_CHAR_UUID      "65719722-b91f-43b3-adbc-7c455a80c50c"

#define EMG_PIN            1    // XIAO A0 / D0
#define IMU_SDA_PIN        5    // XIAO D4, shared with DRV2605L
#define IMU_SCL_PIN        6    // XIAO D5, shared with DRV2605L
#define SNAPSHOT_PERIOD_MS 100  // EMG: 10 snapshots/s
#define IMU_PERIOD_MS      20   // IMU: 50 Hz

// Alert (haptic, and a flag in the BLE rep packet) once the model thinks
// the set is this many reps or fewer from failure.
constexpr float ALERT_RIR = 2;

BLECharacteristic *imuChar;
BLECharacteristic *emgChar;
BLECharacteristic *repChar;
volatile bool connected = false;

Adafruit_MPU6050 mpu;
bool imuFound = false;
RepCounter repCounter;

EmgAdc emgAdc;
bool emgFound = false;
capture::Writer captureOut(Serial);
capture::EmgBlocks emgBlocks;

FatigueFeatures fatigueFeatures;
RepFeatures lastFeatures;
float lastRir = NAN;  // the model's latest estimate, NaN until it has one
bool alertedThisSet = false;

// The model's estimates this set, for the 'f' (failure) report.
constexpr int MAX_SET_REPS = 64;
uint16_t setRep[MAX_SET_REPS];
float setRir[MAX_SET_REPS];
int setEstimates = 0;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    connected = true;
    LOGLN("Central connected");
  }

  void onDisconnect(BLEServer *server) override {
    connected = false;
    LOGLN("Central disconnected, advertising again");
    BLEDevice::startAdvertising();
  }
};

void putU16(uint8_t *p, uint16_t v) {
  p[0] = v & 0xFF;
  p[1] = (v >> 8) & 0xFF;
}

void putI16(uint8_t *p, int16_t v) {
  putU16(p, (uint16_t)v);
}

void putU32(uint8_t *p, uint32_t v) {
  putU16(p, v & 0xFFFF);
  putU16(p + 2, v >> 16);
}

// ---- EMG ----

uint32_t count = 0;
uint32_t sum = 0;
uint64_t sumSq = 0;
uint16_t minMv = UINT16_MAX;
uint16_t maxMv = 0;

void addSample(uint16_t mv) {
  count++;
  sum += mv;
  sumSq += (uint32_t)mv * mv;
  if (mv < minMv) minMv = mv;
  if (mv > maxMv) maxMv = mv;
}

void sendSnapshot(uint32_t now) {
  if (count == 0) return;

  uint16_t mean = (sum + count / 2) / count;
  uint64_t n2Var = (uint64_t)count * sumSq - (uint64_t)sum * sum;
  uint16_t rms = lround(sqrt((double)n2Var) / count);

  // Timestamp, sample count, mean, RMS, min and max. Voltages are in mV.
  uint8_t pkt[14];
  putU32(pkt, now);
  putU16(pkt + 4, count);
  putU16(pkt + 6, mean);
  putU16(pkt + 8, rms);
  putU16(pkt + 10, minMv);
  putU16(pkt + 12, maxMv);

  emgChar->setValue(pkt, sizeof(pkt));
  if (connected) emgChar->notify();

  static uint32_t sent = 0;  // a uint8_t wrapped every 256 and logged twice
  if (++sent % (1000 / SNAPSHOT_PERIOD_MS) == 0) {
    LOGF("t=%lu emg: mean=%umV rms=%umV min=%umV max=%umV | ",
         now, mean, rms, minMv, maxMv);

    if (imuFound) {
      LOGF("imu: angle=%.0f reps=%u%s", repCounter.angleDeg(),
           repCounter.reps(), repCounter.isStill() ? " still" : "");
    } else {
      LOGF("imu: not found");
    }

    LOGLN(connected ? " [connected]" : "");
  }

  count = 0;
  sum = 0;
  sumSq = 0;
  minMv = UINT16_MAX;
  maxMv = 0;
}

void onEmgSample(float mv, uint8_t clipped) {
  addSample(lroundf(mv));
  fatigueFeatures.addEmg(emgAdc.samples() - 1, mv);

#if CAPTURE_MODE
  if (emgBlocks.add(emgAdc.samples() - 1, mv, clipped)) {
    captureOut.send(capture::EMG, emgBlocks.payload, Serial.availableForWrite());
  }
#endif
}

// ---- USB capture frames ----

#if CAPTURE_MODE
void sendStatus(uint32_t now) {
  capture::Payload p;
  p.u8(capture::VERSION);
  p.u16(EmgAdc::OUT_HZ);
  p.u16(EmgAdc::ADC_HZ);
  p.u16(1000 / IMU_PERIOD_MS);
  p.u16(capture::EMG_PER_MV);
  p.u32(emgAdc.samples());
  p.u32(now);
  p.u32(emgAdc.overflows());
  p.u32(captureOut.dropped);
  captureOut.send(capture::STATUS, p, Serial.availableForWrite());
}

void sendImuFrame(uint32_t stamp, Vec3 a, Vec3 wc) {
  capture::Payload p;
  p.u32(stamp);
  p.i16((int16_t)(a.x * 1000));
  p.i16((int16_t)(a.y * 1000));
  p.i16((int16_t)(a.z * 1000));
  p.i16((int16_t)(wc.x * 10));
  p.i16((int16_t)(wc.y * 10));
  p.i16((int16_t)(wc.z * 10));
  p.f32(repCounter.angleDeg());
  p.u8(repCounter.isStill());
  captureOut.send(capture::IMU, p, Serial.availableForWrite());
}

void sendTurnFrame(TurningPoint t) {
  capture::Payload p;
  p.u32(t.stamp);
  p.f32(t.angleDeg);
  p.i8(t.top);
  captureOut.send(capture::TURN, p, Serial.availableForWrite());
}

void sendRepFrame(uint32_t stamp) {
  capture::Payload p;
  p.u32(stamp);
  p.u16(repCounter.reps());
  p.u16(lroundf(repCounter.romDeg() * 10));
  captureOut.send(capture::REP, p, Serial.availableForWrite());
}

void sendFeatFrame(uint32_t stamp, const RepFeatures &f, bool alert) {
  capture::Payload p;
  p.u32(stamp);
  p.u16(f.repInSet);
  p.u8((f.emgOk ? 1 : 0) | (f.hasBaseline ? 2 : 0) | (alert ? 4 : 0));
  p.f32(f.peakDps);
  p.f32(f.movingS);
  p.f32(f.romDeg);
  p.f32(f.emgRmsMv);
  p.f32(f.mdfHz);
  p.f32(f.velocityLoss);
  p.f32(f.durationRatio);
  p.f32(f.romRatio);
  p.f32(f.emgRatio);
  p.f32(f.mdfRatio);
  p.f32(lastRir);
  captureOut.send(capture::FEAT, p, Serial.availableForWrite());
}
#endif

// ---- IMU and reps ----

bool startImu() {
  if (!mpu.begin(0x68, &Wire) && !mpu.begin(0x69, &Wire)) return false;

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  return true;
}

void sendReps(uint32_t now) {
  // Timestamp, rep count and range of motion in tenths of a degree, then
  // the fatigue model: reps in reserve in tenths (0xFFFF until it has an
  // estimate), how far through the set that is in % (0xFF until then), and
  // flags (bit 0: alert). Apps that read only the first 8 bytes still work.
  uint16_t rir10 = 0xFFFF;
  uint8_t fatiguePct = 0xFF;
  if (!isnan(lastRir)) {
    rir10 = lroundf(fminf(lastRir, 6000) * 10);
    uint16_t done = lastFeatures.repInSet;
    fatiguePct = lroundf(100.0f * done / (done + lastRir));
  }

  uint8_t pkt[12];
  putU32(pkt, now);
  putU16(pkt + 4, repCounter.reps());
  putU16(pkt + 6, lroundf(repCounter.romDeg() * 10));
  putU16(pkt + 8, rir10);
  pkt[10] = fatiguePct;
  pkt[11] = alertedThisSet ? 1 : 0;

  repChar->setValue(pkt, sizeof(pkt));
  if (connected) repChar->notify();
}

void startHapticAlert();

// Runs the fatigue model on a lift that's just been measured.
void onLift(uint32_t stamp, const RepFeatures &f) {
  if (f.repInSet == 1) {
    alertedThisSet = false;
    setEstimates = 0;
  }
  lastFeatures = f;

  uint32_t startUs = micros();
  lastRir = f.hasBaseline ? fatigue_model::predict(f) : NAN;
  [[maybe_unused]] uint32_t modelUs = micros() - startUs;
  if (!isnan(lastRir) && setEstimates < MAX_SET_REPS) {
    setRep[setEstimates] = f.repInSet;
    setRir[setEstimates++] = lastRir;
  }

  bool alert = !isnan(lastRir) && lastRir <= ALERT_RIR && !alertedThisSet;
  if (alert) {
    alertedThisSet = true;
    startHapticAlert();
  }

  LOGF("Set rep %u: peak %.0f deg/s, moving %.2f s, rom %.0f deg, emg %.1f mV, mdf %.0f Hz%s\n",
       f.repInSet, f.peakDps, f.movingS, f.romDeg, f.emgRmsMv, f.mdfHz,
       f.emgOk ? "" : " (no emg)");
  if (f.hasBaseline) {
    LOGF("  vs start of set: speed %+.0f%%, time x%.2f, rom x%.2f, emg x%.2f, mdf x%.2f\n",
         -f.velocityLoss * 100, f.durationRatio, f.romRatio, f.emgRatio, f.mdfRatio);
    LOGF("  model: %.1f reps in reserve (%lu us)%s\n", lastRir, modelUs,
         alert ? "  ** ALERT **" : "");
  } else {
    LOGF("  baseline rep %u of %d\n", f.repInSet, FatigueFeatures::BASELINE_REPS);
  }

#if CAPTURE_MODE
  sendFeatFrame(stamp, f, alert);
#endif
}

void readImu(uint32_t now) {
  static uint32_t lastReadUs = micros();
  uint32_t nowUs = micros();
  float dt = (nowUs - lastReadUs) / 1e6f;
  lastReadUs = nowUs;

  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  Vec3 a = {
    accel.acceleration.x / SENSORS_GRAVITY_STANDARD,
    accel.acceleration.y / SENSORS_GRAVITY_STANDARD,
    accel.acceleration.z / SENSORS_GRAVITY_STANDARD
  };

  const float degPerRad = RAD_TO_DEG;
  Vec3 w = {
    gyro.gyro.x * degPerRad,
    gyro.gyro.y * degPerRad,
    gyro.gyro.z * degPerRad
  };

  // EMG sample index, or 2 kHz from millis() if the ADC isn't running.
  uint32_t stamp = emgFound ? emgAdc.samples() : now * 2;
  uint32_t turnsBefore = repCounter.turnCount();
  bool rep = repCounter.update(a, w, dt, stamp);
  Vec3 wc = w - repCounter.gyroBias();

  fatigueFeatures.addImu(stamp, norm(wc));
  RepFeatures lift;
  bool lifted = false;
  if (repCounter.turnCount() != turnsBefore) {
    lifted = fatigueFeatures.addTurn(repCounter.lastTurn(), &lift);
#if CAPTURE_MODE
    sendTurnFrame(repCounter.lastTurn());
#endif
  }

  if (rep) {
#if CAPTURE_MODE
    sendRepFrame(stamp);
#endif
    LOGF("Rep %u, %.0f deg\n", repCounter.reps(), repCounter.romDeg());
  }
  if (lifted) onLift(stamp, lift);
  if (rep || lifted) sendReps(now);

#if CAPTURE_MODE
  sendImuFrame(stamp, a, wc);
#endif

  // Timestamp, acceleration in milli-g and gyro in tenths of a degree/s.
  uint8_t pkt[16];
  putU32(pkt, now);
  putI16(pkt + 4,  (int16_t)(a.x * 1000));
  putI16(pkt + 6,  (int16_t)(a.y * 1000));
  putI16(pkt + 8,  (int16_t)(a.z * 1000));
  putI16(pkt + 10, (int16_t)(wc.x * 10));
  putI16(pkt + 12, (int16_t)(wc.y * 10));
  putI16(pkt + 14, (int16_t)(wc.z * 10));

  imuChar->setValue(pkt, sizeof(pkt));
  if (connected) imuChar->notify();
}

// ---- DRV2605L haptic ----

// The fatigue alert is HAPTIC_ALERT_PULSES pulses, HAPTIC_ALERT_GAP_MS apart.
constexpr uint32_t HAPTIC_PULSE_MS = 300;
constexpr uint8_t HAPTIC_TEST_STRENGTH = 20;
constexpr uint8_t HAPTIC_ALERT_PULSES = 3;
constexpr uint32_t HAPTIC_ALERT_GAP_MS = 1000;

Adafruit_DRV2605 haptic;
bool hapticFound = false;
bool hapticActive = false;
uint32_t lastHapticStartMs = 0;
uint8_t hapticPulsesLeft = 0;

// Starts one buzz; updateHaptic() ends it after HAPTIC_PULSE_MS without
// blocking the loop.
void startHapticPulse() {
  if (!hapticFound) return;
  haptic.setRealtimeValue(HAPTIC_TEST_STRENGTH);
  lastHapticStartMs = millis();
  hapticActive = true;
}

void startHapticAlert() {
  hapticPulsesLeft = HAPTIC_ALERT_PULSES - 1;
  startHapticPulse();
}

void updateHaptic(uint32_t now) {
  if (hapticActive && now - lastHapticStartMs >= HAPTIC_PULSE_MS) {
    haptic.setRealtimeValue(0);
    hapticActive = false;
  } else if (!hapticActive && hapticPulsesLeft > 0 &&
             now - lastHapticStartMs >= HAPTIC_ALERT_GAP_MS) {
    hapticPulsesLeft--;
    startHapticPulse();
  }
}

// ---- Info and serial commands ----

void printInfo() {
  [[maybe_unused]] float error = fatigue_model::selfTestError();
  uint32_t startUs = micros();
  volatile float sink = 0;
  for (int i = 0; i < 1000; i++) {
    sink = sink + fatigue_model::predict(fatigue_model::TEST_IN[i % fatigue_model::N_TEST]);
  }
  [[maybe_unused]] float us = (micros() - startUs) / 1000.0f;

  LOGF("Fatigue model, trained on %s\n", fatigue_model::SOURCE);
  LOGF("  self-test %s: largest difference from Python %.6f reps; %.1f us per prediction\n",
       error < 0.01f ? "ok" : "FAILED", error, us);
  LOGF("  sensors: EMG %s, IMU %s, haptic %s. Alert at %.0f reps in reserve or fewer.\n",
       emgFound ? "ok" : "missing", imuFound ? "ok" : "missing",
       hapticFound ? "ok" : "missing", ALERT_RIR);
  LOGF("  this set: %u reps\n", fatigueFeatures.repsInSet());
  LOGLN("  serial commands: i = this info, h = test the haptic alert, n = start a new set,");
  LOGLN("                   f = I'm at failure (shows how far off the model was, starts a new set)");
}

void startNewSet() {
  fatigueFeatures.newSet();
  lastRir = NAN;
  alertedThisSet = false;
  setEstimates = 0;
}

// The set ended at failure: the last rep done was the failure rep, so the
// true reps in reserve at rep r was (failure rep - r).
void reportFailure() {
  uint16_t failure = fatigueFeatures.repsInSet();
  LOGF("Failure at rep %u. Model's estimates vs actual reps in reserve:\n", failure);
  float totalError = 0;
  for (int i = 0; i < setEstimates; i++) {
    int actual = (int)failure - setRep[i];
    float error = setRir[i] - actual;
    totalError += fabsf(error);
    LOGF("  rep %2u: model %4.1f, actual %2d, off by %+.1f\n", setRep[i], setRir[i], actual, error);
  }
  if (setEstimates > 0) {
    LOGF("  average error %.1f reps over %d estimates\n", totalError / setEstimates, setEstimates);
  } else {
    LOGF("  no estimates this set (they start at rep %d)\n", FatigueFeatures::BASELINE_REPS + 1);
  }
  startNewSet();
  LOGLN("New set");
}

#if !CAPTURE_MODE
void handleSerial() {
  while (Serial.available()) {
    switch (Serial.read()) {
      case 'i': printInfo(); break;
      case 'h': LOGLN("Haptic alert test"); startHapticAlert(); break;
      case 'n': startNewSet(); LOGLN("New set"); break;
      case 'f': reportFailure(); break;
    }
  }
}
#endif

// ---- Setup and main loop ----

uint32_t lastSendMs;
uint32_t lastImuMs;
uint32_t lastStatusMs;

void setup() {
#if CAPTURE_MODE
  Serial.setTxBufferSize(4096);
  Serial.begin(CAPTURE_BAUD);
#else
#if ARDUINO_USB_MODE && ARDUINO_USB_CDC_ON_BOOT
  // Over native USB, a print waits for the computer to read it. With USB
  // plugged in but no serial monitor open, the default wait held up every
  // loop. No wait at all lost lines even with the monitor open, since a
  // rep's log is bigger than the default 256-byte buffer.
  Serial.setTxBufferSize(2048);
  Serial.setTxTimeoutMs(5);
#endif
  Serial.begin(115200);
#endif

  emgFound = emgAdc.begin(EMG_PIN);
  if (!emgFound) LOGLN("EMG ADC failed to start. Is EMG_PIN an ADC pin?");

  // Both the MPU6050 and DRV2605L use this bus.
  Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);
  Wire.setClock(100000);

  imuFound = startImu();
  if (!imuFound) {
    LOGLN("MPU6050 not found. Check power, wiring and solder joints.");
  }

  hapticFound = haptic.begin(&Wire);
  if (hapticFound) {
    haptic.useLRA();
    haptic.setRealtimeValue(0);
    haptic.setMode(DRV2605_MODE_REALTIME);
    LOGLN("DRV2605L ready.");
  } else {
    LOGLN("DRV2605L not found at 0x5A. Haptic test disabled.");
  }

  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);
  imuChar = service->createCharacteristic(
      IMU_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  emgChar = service->createCharacteristic(
      EMG_CHAR_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  repChar = service->createCharacteristic(
      REP_CHAR_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);

  service->start();
  sendReps(millis());

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();

  printInfo();
  LOGLN("Advertising as " DEVICE_NAME);
  lastSendMs = lastImuMs = lastStatusMs = millis();
  startHapticPulse();  // one buzz: powered on and advertising
}

void loop() {
  emgAdc.poll(onEmgSample);

  uint32_t now = millis();

  // After a hold-up, skip ahead rather than run back to back to catch up.
  if (imuFound && now - lastImuMs >= IMU_PERIOD_MS) {
    lastImuMs = now - lastImuMs >= 5 * IMU_PERIOD_MS ? now : lastImuMs + IMU_PERIOD_MS;
    readImu(now);
  }

  if (now - lastSendMs >= SNAPSHOT_PERIOD_MS) {
    lastSendMs = now - lastSendMs >= 5 * SNAPSHOT_PERIOD_MS ? now : lastSendMs + SNAPSHOT_PERIOD_MS;
    sendSnapshot(now);
  }

  updateHaptic(millis());

#if !CAPTURE_MODE
  handleSerial();
#endif

#if CAPTURE_MODE
  if (now - lastStatusMs >= 1000) {
    lastStatusMs += 1000;
    sendStatus(now);
  }
#endif
}