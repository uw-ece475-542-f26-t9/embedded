// ESP32-S3 firmware for the Team 9 fatigue sensor: a MyoWare 2.0 EMG sensor
// on the biceps and an MPU-6050 IMU on the forearm.
//
// Advertises as "Team9-Fatigue-Sensor" with one service and three
// characteristics. All packets are little-endian and fit the default 20-byte
// BLE payload, so no MTU request is needed.
//
// IMU (notify), every 20 ms (50 Hz), 16 bytes:
//   [0..3]   uint32  timestamp in ms since boot
//   [4..9]   int16   ax, ay, az   in milli-g        (1000 = 1 g)
//   [10..15] int16   gx, gy, gz   in 0.1 deg/s      (10 = 1 deg/s), gyro bias removed
//
// EMG (notify), every 100 ms (10 Hz), 14 bytes. The MyoWare's RAW output is
// sampled at 8 kHz by the ADC in DMA mode and decimated to 2 kHz (see
// emg_adc.h), and each packet summarises the last 100 ms; sending every
// sample would need ~4 KB/s and a larger MTU. Voltages in mV at the ADC pin:
//   [0..3]   uint32  timestamp in ms since boot, at the end of the window
//   [4..5]   uint16  number of samples in the window (~200)
//   [6..7]   uint16  mean   DC level the RAW signal sits on
//   [8..9]   uint16  RMS    of the signal around the mean; rises when the muscle contracts
//   [10..11] uint16  min
//   [12..13] uint16  max
//
// Reps (read, notify), each time a rep is counted, 8 bytes:
//   [0..3]   uint32  timestamp in ms since boot
//   [4..5]   uint16  reps counted since boot
//   [6..7]   uint16  range of motion of the latest rep, in 0.1 deg
// See rep_counter.h for how reps are counted. The app reads this once when it
// connects, and counts a set's reps from where the count was at the start.
//
// The app keeps a copy of these constants in
// mobile-app/lib/wearable/protocol.dart; change both together.
//
// Capture mode (CAPTURE_MODE 1): also streams every EMG and IMU sample and
// the rep counter's turning points over USB serial as framed binary, for
// tools/record.py. Text logging is off so it doesn't corrupt the stream.
// BLE works the same either way. See capture.h for the format.

#include <BLEDevice.h>
#include <BLEServer.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

#include "capture.h"
#include "emg_adc.h"
#include "rep_counter.h"

// 1 = stream raw data over USB for tools/record.py; 0 = normal text logs.
#define CAPTURE_MODE 0
// The stream is ~6.5 KB/s, more than 115200 baud carries comfortably. (On
// the S3's native USB port the baud rate is ignored.)
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
#define EMG_PIN            4
#define IMU_SDA_PIN        5
#define IMU_SCL_PIN        6
#define SNAPSHOT_PERIOD_MS 100  // EMG: 10 snapshots/s
#define IMU_PERIOD_MS      20   // IMU: 50 Hz

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

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    connected = true;
    LOGLN("Central connected");
  }
  void onDisconnect(BLEServer *server) override {
    connected = false;
    LOGLN("Central disconnected, advertising again");
    BLEDevice::startAdvertising();  // ESP32 stops advertising after a connection; restart it
  }
};

void putU16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF; }
void putI16(uint8_t *p, int16_t v) { putU16(p, (uint16_t)v); }
void putU32(uint8_t *p, uint32_t v) { putU16(p, v & 0xFFFF); putU16(p + 2, v >> 16); }

// ---- EMG ----

// Running totals for the current snapshot window.
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
  // n * sum(x^2) - (sum x)^2 = n^2 * variance, exact in integer math
  uint64_t n2Var = (uint64_t)count * sumSq - (uint64_t)sum * sum;
  uint16_t rms = lround(sqrt((double)n2Var) / count);

  uint8_t pkt[14];
  putU32(pkt, now);
  putU16(pkt + 4, count);
  putU16(pkt + 6, mean);
  putU16(pkt + 8, rms);
  putU16(pkt + 10, minMv);
  putU16(pkt + 12, maxMv);

  emgChar->setValue(pkt, sizeof(pkt));
  if (connected) emgChar->notify();

  // Once a second, a status line for the serial monitor.
  static uint8_t sent = 0;
  if (++sent % (1000 / SNAPSHOT_PERIOD_MS) == 0) {
    LOGF("t=%lu emg: mean=%umV rms=%umV min=%umV max=%umV | ", now, mean, rms, minMv, maxMv);
    if (imuFound) {
      LOGF("imu: angle=%.0f reps=%u%s", repCounter.angleDeg(), repCounter.reps(),
           repCounter.isStill() ? " still" : "");
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
#if CAPTURE_MODE
  // The sample just produced has index samples() - 1.
  if (emgBlocks.add(emgAdc.samples() - 1, mv, clipped)) {
    captureOut.send(capture::EMG, emgBlocks.payload, Serial.availableForWrite());
  }
#endif
}

// ---- Capture frames ----

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
#endif

// ---- IMU and reps ----

bool startImu() {
  Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);
  Wire.setClock(100000);  // I2C 100 kHz; a read takes ~2 ms
  if (!mpu.begin(0x68, &Wire) && !mpu.begin(0x69, &Wire)) return false;
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  return true;
}

void sendReps(uint32_t now) {
  uint8_t pkt[8];
  putU32(pkt, now);
  putU16(pkt + 4, repCounter.reps());
  putU16(pkt + 6, lroundf(repCounter.romDeg() * 10));

  repChar->setValue(pkt, sizeof(pkt));
  if (connected) repChar->notify();
}

void readImu(uint32_t now) {
  static uint32_t lastReadUs = micros();
  uint32_t nowUs = micros();
  float dt = (nowUs - lastReadUs) / 1e6f;
  lastReadUs = nowUs;

  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);
  Vec3 a = {accel.acceleration.x / SENSORS_GRAVITY_STANDARD,
            accel.acceleration.y / SENSORS_GRAVITY_STANDARD,
            accel.acceleration.z / SENSORS_GRAVITY_STANDARD};
  const float degPerRad = RAD_TO_DEG;  // float, so the S3's FPU does the math
  Vec3 w = {gyro.gyro.x * degPerRad, gyro.gyro.y * degPerRad, gyro.gyro.z * degPerRad};

  // Stamped with the EMG sample index so capture lines the streams up.
  uint32_t stamp = emgAdc.samples();
  uint32_t turnsBefore = repCounter.turnCount();
  bool rep = repCounter.update(a, w, dt, stamp);
#if CAPTURE_MODE
  if (repCounter.turnCount() != turnsBefore) sendTurnFrame(repCounter.lastTurn());
  if (rep) sendRepFrame(stamp);
#else
  (void)turnsBefore;
#endif
  if (rep) {
    LOGF("Rep %u, %.0f deg\n", repCounter.reps(), repCounter.romDeg());
    sendReps(now);
  }

  Vec3 wc = w - repCounter.gyroBias();
#if CAPTURE_MODE
  sendImuFrame(stamp, a, wc);
#endif
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

uint32_t lastSendMs;
uint32_t lastImuMs;
uint32_t lastStatusMs;

void setup() {
#if CAPTURE_MODE
  Serial.setTxBufferSize(4096);  // ~0.6 s of stream, if the port is a UART
  Serial.begin(CAPTURE_BAUD);
#else
  Serial.begin(115200);
#endif

  emgFound = emgAdc.begin(EMG_PIN);
  if (!emgFound) LOGLN("EMG ADC failed to start. Is EMG_PIN an ADC pin?");

  imuFound = startImu();
  if (!imuFound) {
    // Keep going so EMG still works; the app just won't see motion or reps.
    LOGLN("MPU6050 not found. Check power, wiring and solder joints.");
  }

  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  // No BLE2902 descriptors needed: on the S3 the core uses NimBLE, which adds
  // the CCCD (the descriptor the phone writes to turn notifications on) by
  // itself.
  BLEService *service = server->createService(SERVICE_UUID);
  imuChar = service->createCharacteristic(IMU_CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  emgChar = service->createCharacteristic(
      EMG_CHAR_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  repChar = service->createCharacteristic(
      REP_CHAR_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  service->start();
  sendReps(millis());  // so a read before the first rep returns 0

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();

  LOGLN("Advertising as " DEVICE_NAME);
  lastSendMs = lastImuMs = lastStatusMs = millis();
}

void loop() {
  // The ADC samples in the background; collect what it has.
  emgAdc.poll(onEmgSample);

  // Advance by whole periods so the average rates hold even if a BLE call or
  // an IMU read holds up the loop for a moment.
  uint32_t now = millis();
  if (imuFound && now - lastImuMs >= IMU_PERIOD_MS) {
    lastImuMs += IMU_PERIOD_MS;
    readImu(now);
  }
  if (now - lastSendMs >= SNAPSHOT_PERIOD_MS) {
    lastSendMs += SNAPSHOT_PERIOD_MS;
    sendSnapshot(now);
  }
#if CAPTURE_MODE
  if (now - lastStatusMs >= 1000) {
    lastStatusMs += 1000;
    sendStatus(now);
  }
#endif
}
