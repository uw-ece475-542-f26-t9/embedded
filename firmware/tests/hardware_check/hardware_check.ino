// Hardware check for the wearable: every sensor, the haptic, and BLE.
//
// Flash it in place of the wearable firmware, open the Serial Monitor at
// 115200 and send "c" to run the guided check (about 30 s; wear the
// electrodes for the EMG part). It says what to do at each step, then gives
// each part PASS, FAIL or CHECK (look at it yourself):
//
//   IMU found    an MPU-6050 answers at 0x68 or 0x69
//   IMU still    gravity reads 0.85-1.15 g with the board held still
//   IMU motion   the gyro sees at least 45 deg/s when you twist the board
//   LRA driver   a DRV2605L answers at 0x5A, and its own diagnostic finds a
//                motor (a flag is only CHECK: the diagnostic can also flag a
//                working LRA its default settings don't suit). With a flag,
//                its overcurrent bit hints at a short; no overcurrent hints
//                at an open circuit (motor not connected, or a broken wire)
//   LRA buzz     you felt 3 buzzes (answer y or n)
//   EMG level    relaxed EMG sits mid-range (800-2400 mV), not near 0 V
//                (sensor unpowered, or RAW not on A0) or near the top
//   EMG rails    under 0.1% of readings at the ADC limits (a loose RAW wire
//                slams between them)
//   EMG flex     squeezing raises EMG RMS at least 5x over relaxed
//   BLE          the app is connected (optional)
//
// Commands (send in the Serial Monitor):
//   c  run the guided check
//   h  buzz 3 times
//   l  start live data: a status line every second, and the wearable
//      firmware's BLE EMG and IMU packets, so the app graphs them
//   s  stop live data (it starts stopped)
//   ?  list the commands
//
// It always advertises the wearable firmware's BLE service and
// characteristics, so the app connects (the rep packet stays at 0 reps, no
// estimate). Reflash fatigue_sensor when done.
//
// Thresholds come from sensor testing (emg-sensor-testing/): healthy RAW sat
// near 1.4 V at rest, a max squeeze was ~30x rest, and the MVP sanity check
// asked for at least 5x. EMG is read with plain analogRead at 2 kHz: enough
// for these checks, not for spectra (the wearable firmware uses DMA). mV are
// nominal (0-3100 mV for 0-4095), so they can be a few % off the wearable
// firmware's calibrated numbers.
//
// Pins match the wearable firmware: EMG (MyoWare RAW) on A0 (GPIO 1);
// MPU-6050 and DRV2605L on SDA D4 (GPIO 5) and SCL D5 (GPIO 6).

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <Wire.h>
#include <Adafruit_DRV2605.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

// Same as the wearable firmware, so the app connects to this too.
#define DEVICE_NAME        "Team9-Fatigue-Sensor"
#define SERVICE_UUID       "7fc2f476-4996-49df-b915-54e84d61fedb"
#define IMU_CHAR_UUID      "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define EMG_CHAR_UUID      "53e27fb3-6de1-4f27-8303-d663c79b818b"
#define REP_CHAR_UUID      "65719722-b91f-43b3-adbc-7c455a80c50c"

constexpr int EMG_PIN = 1;  // XIAO A0 / D0
constexpr int SDA_PIN = 5;  // XIAO D4
constexpr int SCL_PIN = 6;  // XIAO D5
constexpr uint8_t DRV_ADDR = 0x5A;

constexpr uint32_t EMG_PERIOD_US = 500;       // 2 kHz
constexpr uint32_t IMU_PERIOD_MS = 20;        // 50 Hz
constexpr uint32_t SNAPSHOT_PERIOD_MS = 100;  // EMG BLE packets, 10/s
constexpr float MV_PER_COUNT = 3100.0f / 4095;

// Pass rules (see the top of the file).
constexpr float GRAVITY_MIN_G = 0.85f;
constexpr float GRAVITY_MAX_G = 1.15f;
constexpr float MOTION_MIN_DPS = 45;
constexpr float EMG_LEVEL_MIN_MV = 800;
constexpr float EMG_LEVEL_MAX_MV = 2400;
constexpr float EMG_RAILS_MAX = 0.001f;  // fraction of readings
constexpr float EMG_FLEX_MIN_RATIO = 5;

// The wearable firmware's alert: 300 ms buzzes, starting 1 s apart.
constexpr uint8_t HAPTIC_STRENGTH = 20;
constexpr uint32_t HAPTIC_PULSE_MS = 300;
constexpr uint32_t HAPTIC_GAP_MS = 1000;

// The XIAO ESP32-S3's user LED (GPIO 21) is lit when the pin is LOW. It's
// lit while the check measures or buzzes.
constexpr uint8_t LED_ON = LOW;
constexpr uint8_t LED_OFF = HIGH;

struct Vec3 {
  float x, y, z;
};

struct ImuSample {
  Vec3 a;      // g
  Vec3 w;      // deg/s
  float tempC;
};

// Running EMG statistics, in ADC counts.
struct EmgStats {
  uint32_t n = 0;
  uint32_t rails = 0;
  uint64_t sum = 0;
  uint64_t sumSq = 0;
  uint16_t lo = UINT16_MAX;
  uint16_t hi = 0;

  void add(uint16_t c) {
    n++;
    sum += c;
    sumSq += (uint32_t)c * c;
    if (c < lo) lo = c;
    if (c > hi) hi = c;
    if (c <= 5 || c >= 4090) rails++;  // the rails rule from sensor testing
  }
  float meanMv() const { return n ? (float)sum / n * MV_PER_COUNT : 0; }
  // RMS around the mean, as in the wearable firmware's EMG packet.
  float rmsMv() const {
    if (!n) return 0;
    double m = (double)sum / n;
    double var = (double)sumSq / n - m * m;
    return sqrt(var > 0 ? var : 0) * MV_PER_COUNT;
  }
  float loMv() const { return n ? lo * MV_PER_COUNT : 0; }
  float hiMv() const { return n ? hi * MV_PER_COUNT : 0; }
  float railFrac() const { return n ? (float)rails / n : 0; }
};

enum Result : uint8_t { R_SKIP, R_PASS, R_CHECK, R_FAIL };  // FAIL alone clashes with ESP-IDF
const char *const RESULT_NAMES[] = {"skip", "PASS", "CHECK", "FAIL"};

struct Line {
  const char *name;
  Result result;
  char note[128];
};

constexpr int MAX_LINES = 12;
Line lines[MAX_LINES];
int nLines = 0;

Adafruit_MPU6050 mpu;
Adafruit_DRV2605 drv;
bool imuFound = false;
bool drvFound = false;
// What's known about the haptic, for the status line. The driver answering
// on I2C says nothing about the motor; only the check's buzz test does.
const char *hapticStatus = "missing";
Vec3 gyroBias = {0, 0, 0};
ImuSample lastImu = {{0, 0, 0}, {0, 0, 0}, 0};

BLECharacteristic *imuChar;
BLECharacteristic *emgChar;
BLECharacteristic *repChar;
volatile bool connected = false;

bool streaming = false;  // live data on (l) or off (s)
EmgStats live;
uint32_t nextEmgUs;
uint32_t lastImuMs;
uint32_t lastSnapshotMs;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    connected = true;
    Serial.println("App connected");
  }

  void onDisconnect(BLEServer *server) override {
    connected = false;
    Serial.println("App disconnected, advertising again");
    BLEDevice::startAdvertising();
  }
};

// ---- Helpers ----

float norm(Vec3 v) {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

Vec3 sub(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

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

void led(bool on) {
  digitalWrite(LED_BUILTIN, on ? LED_ON : LED_OFF);
}

// Prints a result now and keeps it for the summary.
void record(const char *name, Result result, const char *fmt, ...) {
  if (nLines >= MAX_LINES) return;
  Line &line = lines[nLines++];
  line.name = name;
  line.result = result;
  va_list args;
  va_start(args, fmt);
  vsnprintf(line.note, sizeof(line.note), fmt, args);
  va_end(args);
  Serial.printf("  %-5s  %-10s  %s\n", RESULT_NAMES[result], name, line.note);
}

// Waits for y or n from the Serial Monitor: 'y', 'n', or 0 if none came.
int waitForAnswer(uint32_t ms) {
  while (Serial.available()) Serial.read();  // drop anything typed earlier
  uint32_t start = millis();
  while (millis() - start < ms) {
    int c = Serial.read();
    if (c == 'y' || c == 'Y') return 'y';
    if (c == 'n' || c == 'N') return 'n';
    delay(10);
  }
  return 0;
}

// ---- Sensors ----

bool startImu() {
  if (!mpu.begin(0x68, &Wire) && !mpu.begin(0x69, &Wire)) return false;
  // The wearable firmware's settings.
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  return true;
}

ImuSample readImu() {
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);
  ImuSample s;
  s.a = {accel.acceleration.x / SENSORS_GRAVITY_STANDARD,
         accel.acceleration.y / SENSORS_GRAVITY_STANDARD,
         accel.acceleration.z / SENSORS_GRAVITY_STANDARD};
  s.w = {gyro.gyro.x * (float)RAD_TO_DEG,
         gyro.gyro.y * (float)RAD_TO_DEG,
         gyro.gyro.z * (float)RAD_TO_DEG};
  s.tempC = temp.temperature;
  return s;
}

bool startHaptic() {
  if (!drv.begin(&Wire)) return false;
  drv.useLRA();
  drv.setRealtimeValue(0);
  drv.setMode(DRV2605_MODE_REALTIME);
  return true;
}

void buzz(int times) {
  for (int i = 0; i < times; i++) {
    if (i > 0) delay(HAPTIC_GAP_MS - HAPTIC_PULSE_MS);
    led(true);
    drv.setRealtimeValue(HAPTIC_STRENGTH);
    delay(HAPTIC_PULSE_MS);
    drv.setRealtimeValue(0);
    led(false);
  }
}

// Runs the DRV2605L's actuator diagnostic, which drives the motor briefly.
// Returns false if it didn't finish; otherwise *status is the STATUS register
// after it: bits 7-5 chip ID (7 = DRV2605L), bit 3 DIAG_RESULT (set = motor
// flagged), bit 0 OC_DETECT (overcurrent seen). Read once, as reading it
// clears OC_DETECT.
bool runDiagnostic(uint8_t *status) {
  drv.readRegister8(DRV2605_REG_STATUS);  // clear an old overcurrent flag
  drv.setMode(DRV2605_MODE_DIAGNOS);
  drv.go();
  bool finished = true;
  uint32_t start = millis();
  while (drv.readRegister8(DRV2605_REG_GO) & 0x01) {
    if (millis() - start > 2000) {
      drv.writeRegister8(DRV2605_REG_GO, 0);  // cancel it
      finished = false;
      break;
    }
    delay(10);
  }
  if (finished) *status = drv.readRegister8(DRV2605_REG_STATUS);
  drv.setMode(DRV2605_MODE_REALTIME);
  drv.setRealtimeValue(0);
  return finished;
}

// Reads EMG at 2 kHz for ms milliseconds.
void sampleEmg(uint32_t ms, EmgStats &s) {
  uint32_t start = millis();
  uint32_t next = micros();
  while (millis() - start < ms) {
    if ((int32_t)(micros() - next) >= 0) {
      next += EMG_PERIOD_US;
      s.add(analogRead(EMG_PIN));
    }
  }
}

// ---- The guided check ----

void checkBus() {
  Serial.println("\nI2C bus (SDA D4, SCL D5):");
  int found = 0;
  for (uint8_t addr = 0x08; addr < 0x78; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() != 0) continue;
    const char *what = (addr == 0x68 || addr == 0x69) ? "MPU-6050"
                       : addr == DRV_ADDR             ? "DRV2605L"
                                                      : "unknown";
    Serial.printf("  0x%02X %s\n", addr, what);
    found++;
  }
  if (!found) Serial.println("  nothing answered: check 3V3, GND, SDA and SCL");
}

void checkImu() {
  Serial.println("\nIMU (MPU-6050)");
  imuFound = startImu();
  if (!imuFound) {
    record("IMU found", R_FAIL, "no MPU-6050 at 0x68 or 0x69: check power, SDA/SCL and solder joints");
    record("IMU still", R_SKIP, "no IMU");
    record("IMU motion", R_SKIP, "no IMU");
    return;
  }
  record("IMU found", R_PASS, "MPU-6050 answered");

  Serial.println("Keep the board still (set it down, or hold your arm still)...");
  delay(1500);
  led(true);
  float gSum = 0, tSum = 0;
  Vec3 wSum = {0, 0, 0};
  int n = 0;
  uint32_t start = millis();
  while (millis() - start < 2000) {
    ImuSample s = readImu();
    gSum += norm(s.a);
    wSum = {wSum.x + s.w.x, wSum.y + s.w.y, wSum.z + s.w.z};
    tSum += s.tempC;
    n++;
    delay(IMU_PERIOD_MS);
  }
  led(false);
  float g = gSum / n;
  gyroBias = {wSum.x / n, wSum.y / n, wSum.z / n};
  bool gravityOk = g >= GRAVITY_MIN_G && g <= GRAVITY_MAX_G;
  record("IMU still", gravityOk ? R_PASS : R_FAIL,
         "gravity %.2f g (expect 1.00)%s; gyro offset %.1f deg/s, %.1f C",
         g, gravityOk ? "" : ", accelerometer reads wrong", norm(gyroBias), tSum / n);

  Serial.println("Now twist the board back and forth (or curl your arm) until it says stop...");
  delay(500);
  led(true);
  float peak = 0;
  start = millis();
  while (millis() - start < 3000) {
    peak = fmaxf(peak, norm(sub(readImu().w, gyroBias)));
    delay(IMU_PERIOD_MS);
  }
  led(false);
  Serial.println("Stop.");
  bool moved = peak >= MOTION_MIN_DPS;
  record("IMU motion", moved ? R_PASS : R_FAIL, "peak turn %.0f deg/s (need %.0f)%s",
         peak, MOTION_MIN_DPS, moved ? "" : ": gyro not responding, or the board didn't move");
}

void checkHaptic() {
  Serial.println("\nHaptic (DRV2605L and LRA)");
  drvFound = startHaptic();
  if (!drvFound) {
    hapticStatus = "missing";
    record("LRA driver", R_FAIL, "no DRV2605L at 0x5A: check power, SDA/SCL and solder joints");
    record("LRA buzz", R_SKIP, "no driver");
    return;
  }

  uint8_t status = 0;
  bool diagRan = runDiagnostic(&status);
  bool flagged = diagRan && (status & 0x08);
  if (diagRan) {
    Serial.printf("  driver status 0x%02X: chip ID %u (DRV2605L = 7), motor %s, overcurrent %s\n",
                  status, status >> 5, flagged ? "flagged" : "ok", (status & 0x01) ? "yes" : "no");
  }
  if (!diagRan) {
    record("LRA driver", R_CHECK, "DRV2605L answered, but its diagnostic didn't finish");
  } else if (!flagged) {
    record("LRA driver", R_PASS, "DRV2605L answered; its diagnostic found the motor");
  } else if (status & 0x01) {
    record("LRA driver", R_CHECK, "diagnostic flagged the motor with overcurrent: likely a short in the motor or its wires");
  } else {
    record("LRA driver", R_CHECK, "diagnostic flagged the motor, no overcurrent: likely open (not connected, broken wire)");
  }

  Serial.println("Buzzing 3 times...");
  delay(500);
  buzz(3);
  Serial.println("Did you feel 3 buzzes? Send y or n.");
  int answer = waitForAnswer(20000);
  if (answer == 'y') {
    hapticStatus = "ok (felt)";
    record("LRA buzz", R_PASS, "felt");
  } else if (answer == 'n') {
    hapticStatus = "NOT FELT (driver answers, motor doesn't buzz)";
    record("LRA buzz", R_FAIL, "not felt: check the motor and its two wires to the DRV2605L");
  } else {
    hapticStatus = flagged ? "motor flagged, buzz not confirmed" : "driver ok, buzz not confirmed";
    record("LRA buzz", R_CHECK, "no answer in 20 s");
  }
}

void checkEmg() {
  Serial.println("\nEMG (MyoWare RAW on A0; electrodes on the biceps)");
  Serial.println("Relax your arm completely: forearm supported, hand limp...");
  delay(2000);
  EmgStats rest;
  led(true);
  sampleEmg(3000, rest);
  led(false);
  Serial.printf("  relaxed:  level %.0f mV, RMS %.1f mV, %.0f-%.0f mV\n",
                rest.meanMv(), rest.rmsMv(), rest.loMv(), rest.hiMv());

  Serial.println("Squeeze hard now: make a fist and tense your biceps...");
  delay(1000);
  EmgStats flex;
  led(true);
  sampleEmg(2000, flex);
  led(false);
  Serial.println("Relax.");
  Serial.printf("  squeezed: level %.0f mV, RMS %.1f mV, %.0f-%.0f mV\n",
                flex.meanMv(), flex.rmsMv(), flex.loMv(), flex.hiMv());

  float level = rest.meanMv();
  if (level < EMG_LEVEL_MIN_MV) {
    record("EMG level", R_FAIL, "%.0f mV, near 0 V: sensor unpowered, or RAW isn't on A0", level);
  } else if (level > EMG_LEVEL_MAX_MV) {
    record("EMG level", R_FAIL, "%.0f mV, near the top: check the sensor's power and RAW wire", level);
  } else {
    record("EMG level", R_PASS, "%.0f mV, mid-range", level);
  }

  float rails = fmaxf(rest.railFrac(), flex.railFrac());
  bool railsOk = rails < EMG_RAILS_MAX;
  record("EMG rails", railsOk ? R_PASS : R_FAIL, "%.2f%% of readings at the ADC limits%s",
         rails * 100, railsOk ? "" : ": loose RAW wire or electrode?");

  // Below one ADC count (0.76 mV) the RMS is quantization, so floor it there.
  float ratio = flex.rmsMv() / fmaxf(rest.rmsMv(), MV_PER_COUNT);
  bool flexOk = ratio >= EMG_FLEX_MIN_RATIO;
  record("EMG flex", flexOk ? R_PASS : R_FAIL, "squeeze RMS %.1fx relaxed (need %.0fx)%s",
         ratio, EMG_FLEX_MIN_RATIO, flexOk ? "" : ": check electrodes, or rerun fully relaxed");
}

void checkBle() {
  if (connected) {
    record("BLE", R_PASS, "the app is connected");
  } else {
    record("BLE", R_SKIP, "advertising as " DEVICE_NAME "; connect the app and rerun to test it");
  }
}

void runCheck() {
  nLines = 0;
  Serial.println("\n===== Hardware check =====");
  Serial.println("(BLE data pauses until it's done)");
  checkBus();
  checkImu();
  checkHaptic();
  checkEmg();
  checkBle();

  Serial.println("\n===== Summary =====");
  int fails = 0, checks = 0;
  for (int i = 0; i < nLines; i++) {
    Serial.printf("  %-5s  %-10s  %s\n", RESULT_NAMES[lines[i].result], lines[i].name, lines[i].note);
    if (lines[i].result == R_FAIL) fails++;
    if (lines[i].result == R_CHECK) checks++;
  }
  if (fails) {
    Serial.printf("%d FAILED.\n", fails);
  } else if (checks) {
    Serial.println("No failures, but look at the CHECK lines.");
  } else {
    Serial.println("Everything passed.");
  }
  Serial.println(streaming ? "Live data continues (s to stop).\n" : "Send c to run again, l to start live data, ? for commands.\n");
}

void printHelp() {
  Serial.println("Commands: c = run the hardware check, h = buzz 3 times,");
  Serial.println("          l = start live data, s = stop live data, ? = this list");
}

// ---- Live data (between checks) ----

void sendSnapshot(uint32_t now) {
  if (live.n == 0) return;

  // Timestamp, sample count, mean, RMS, min and max in mV: the wearable
  // firmware's EMG packet.
  uint8_t pkt[14];
  putU32(pkt, now);
  putU16(pkt + 4, live.n);
  putU16(pkt + 6, lroundf(live.meanMv()));
  putU16(pkt + 8, lroundf(live.rmsMv()));
  putU16(pkt + 10, lroundf(live.loMv()));
  putU16(pkt + 12, lroundf(live.hiMv()));
  emgChar->setValue(pkt, sizeof(pkt));
  if (connected) emgChar->notify();

  static uint32_t sent = 0;
  if (++sent % (1000 / SNAPSHOT_PERIOD_MS) == 0) {
    Serial.printf("t=%lus emg: level %.0f mV, rms %.1f mV, %.0f-%.0f mV | ",
                  (unsigned long)(now / 1000), live.meanMv(), live.rmsMv(), live.loMv(), live.hiMv());
    if (imuFound) {
      Serial.printf("imu: %.2f g, turning %.0f deg/s", norm(lastImu.a), norm(sub(lastImu.w, gyroBias)));
    } else {
      Serial.print("imu: missing");
    }
    Serial.printf(" | haptic: %s%s\n", hapticStatus, connected ? " | [connected]" : "");
    if (sent % (10 * 1000 / SNAPSHOT_PERIOD_MS) == 0) {
      Serial.println("(s to stop, c to run the hardware check, ? for commands)");
    }
  }

  live = EmgStats();
}

void sendImu(uint32_t now) {
  lastImu = readImu();
  Vec3 w = sub(lastImu.w, gyroBias);

  // Timestamp, acceleration in milli-g and gyro in tenths of a degree/s: the
  // wearable firmware's IMU packet.
  uint8_t pkt[16];
  putU32(pkt, now);
  putI16(pkt + 4, (int16_t)(lastImu.a.x * 1000));
  putI16(pkt + 6, (int16_t)(lastImu.a.y * 1000));
  putI16(pkt + 8, (int16_t)(lastImu.a.z * 1000));
  putI16(pkt + 10, (int16_t)(w.x * 10));
  putI16(pkt + 12, (int16_t)(w.y * 10));
  putI16(pkt + 14, (int16_t)(w.z * 10));
  imuChar->setValue(pkt, sizeof(pkt));
  if (connected) imuChar->notify();
}

// Restarts the live timers after something blocked the loop, so it doesn't
// try to catch up.
void resumeLive() {
  live = EmgStats();
  nextEmgUs = micros();
  lastImuMs = lastSnapshotMs = millis();
}

void startBle() {
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

  // The app needs a rep packet: 0 reps, 0 range of motion, no estimate
  // (0xFFFF reps in reserve, 0xFF %), no alert.
  uint8_t rep[12];
  putU32(rep, millis());
  putU16(rep + 4, 0);
  putU16(rep + 6, 0);
  putU16(rep + 8, 0xFFFF);
  rep[10] = 0xFF;
  rep[11] = 0;
  repChar->setValue(rep, sizeof(rep));

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();
}

void handleSerial() {
  while (Serial.available()) {
    switch (Serial.read()) {
      case 'c':
      case 'C':
        runCheck();
        resumeLive();
        break;
      case 'h':
      case 'H':
        if (drvFound) {
          Serial.println("Buzzing 3 times");
          buzz(3);
          resumeLive();
        } else {
          Serial.println("No DRV2605L found. Run the check (c) to look again.");
        }
        break;
      case 'l':
      case 'L':
        if (!streaming) {
          streaming = true;
          resumeLive();
          Serial.println("Live data started (s to stop).");
        }
        break;
      case 's':
      case 'S':
        if (streaming) {
          streaming = false;
          Serial.println("Live data stopped (l to start).");
        }
        break;
      case '?':
        printHelp();
        break;
    }
  }
}

// ---- Setup and main loop ----

void setup() {
#if ARDUINO_USB_MODE && ARDUINO_USB_CDC_ON_BOOT
  // Over native USB, a print waits for the computer to read it. A short wait
  // keeps the timing with no Serial Monitor open; the big buffer keeps the
  // summary whole with one open.
  Serial.setTxBufferSize(4096);
  Serial.setTxTimeoutMs(5);
#endif
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);
  led(false);

  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);  // ~0-3.1 V

  // Both the MPU-6050 and DRV2605L use this bus.
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);
  imuFound = startImu();
  drvFound = startHaptic();
  hapticStatus = drvFound ? "driver answers, motor untested (run c)" : "missing";
  startBle();

  delay(1000);
  Serial.println("\nTeam 9 hardware check");
  Serial.printf("MPU-6050 %s, DRV2605L %s. Advertising as " DEVICE_NAME ".\n",
                imuFound ? "found" : "missing", drvFound ? "found" : "missing");
  Serial.println("Send c to run the guided check (about 30 s).");
  printHelp();
  Serial.println();
}

void loop() {
  if (streaming) {
    uint32_t nowUs = micros();
    if ((int32_t)(nowUs - nextEmgUs) >= 0) {
      nextEmgUs += EMG_PERIOD_US;
      live.add(analogRead(EMG_PIN));
    }

    uint32_t now = millis();
    if (imuFound && now - lastImuMs >= IMU_PERIOD_MS) {
      lastImuMs = now;
      sendImu(now);
    }
    if (now - lastSnapshotMs >= SNAPSHOT_PERIOD_MS) {
      lastSnapshotMs = now;
      sendSnapshot(now);
    }
  }

  handleSerial();
}
