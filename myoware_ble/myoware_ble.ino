// ESP32-S3 BLE streamer for the MyoWare 2.0 EMG sensor.
//
// Samples the MyoWare's RAW output at 2 kHz and, every 100 ms, sends one
// notification that summarises the samples from that window (10 snapshots/s).
// Sending every raw sample would need ~4 KB/s and a larger MTU; a snapshot
// fits the default 20-byte BLE payload, like the IMU packet.
//
// Advertises as "Team9-Fatigue-Sensor". Packet = 14 bytes, little-endian, all voltages
// in mV as measured at the ESP32's ADC pin:
//   [0..3]   uint32  timestamp in ms since boot, at the end of the window
//   [4..5]   uint16  number of samples in the window (~200)
//   [6..7]   uint16  mean   DC level the RAW signal sits on
//   [8..9]   uint16  RMS    of the signal around the mean; rises when the muscle contracts
//   [10..11] uint16  min
//   [12..13] uint16  max
//
// Uses the same service as the class IMU firmware (esp32_imu_ble.ino) with its
// own characteristic, so the app finds it with the same scan filter and the two
// can later live in one sketch. The app keeps a copy of these constants in
// mobile-app/lib/wearable/protocol.dart; change both together.

#include <BLEDevice.h>
#include <BLEServer.h>

#define DEVICE_NAME        "Team9-Fatigue-Sensor"
#define SERVICE_UUID       "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define EMG_CHAR_UUID      "53e27fb3-6de1-4f27-8303-d663c79b818b"
#define EMG_PIN            4
#define SAMPLE_PERIOD_US   500  // 2 kHz
#define SNAPSHOT_PERIOD_MS 100  // 10 snapshots/s

BLECharacteristic *emgChar;
volatile bool connected = false;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) override {
    connected = true;
    Serial.println("Central connected");
  }
  void onDisconnect(BLEServer *server) override {
    connected = false;
    Serial.println("Central disconnected, advertising again");
    BLEDevice::startAdvertising();  // ESP32 stops advertising after a connection; restart it
  }
};

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

void putU16(uint8_t *p, uint16_t v) { p[0] = v & 0xFF; p[1] = (v >> 8) & 0xFF; }
void putU32(uint8_t *p, uint32_t v) { putU16(p, v & 0xFFFF); putU16(p + 2, v >> 16); }

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

  static uint8_t sent = 0;
  if (++sent % (1000 / SNAPSHOT_PERIOD_MS) == 0) {
    Serial.printf("t=%lu n=%lu mean=%umV rms=%umV min=%umV max=%umV %s\n",
                  now, count, mean, rms, minMv, maxMv, connected ? "[connected]" : "");
  }

  count = 0;
  sum = 0;
  sumSq = 0;
  minMv = UINT16_MAX;
  maxMv = 0;
}

uint32_t lastSampleUs;
uint32_t lastSendMs;

void setup() {
  Serial.begin(115200);

  analogReadResolution(12);
  analogSetPinAttenuation(EMG_PIN, ADC_11db);  // widest input range, ~0-3.1 V

  BLEDevice::init(DEVICE_NAME);
  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new ServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);
  emgChar = service->createCharacteristic(
      EMG_CHAR_UUID,
      BLECharacteristic::PROPERTY_READ | BLECharacteristic::PROPERTY_NOTIFY);
  // No BLE2902 needed: on the S3 the core uses NimBLE, which adds the CCCD
  // (the descriptor the phone writes to turn notifications on) by itself.
  service->start();

  BLEAdvertising *adv = BLEDevice::getAdvertising();
  adv->addServiceUUID(SERVICE_UUID);
  adv->setScanResponse(true);
  BLEDevice::startAdvertising();

  Serial.println("Advertising as " DEVICE_NAME);
  lastSampleUs = micros();
  lastSendMs = millis();
}

void loop() {
  // Advance by whole periods so the average rate stays at 2 kHz even if a BLE
  // call holds up the loop for a moment.
  if (micros() - lastSampleUs >= SAMPLE_PERIOD_US) {
    lastSampleUs += SAMPLE_PERIOD_US;
    addSample(analogReadMilliVolts(EMG_PIN));
  }

  uint32_t now = millis();
  if (now - lastSendMs >= SNAPSHOT_PERIOD_MS) {
    lastSendMs += SNAPSHOT_PERIOD_MS;
    sendSnapshot(now);
  }
}
