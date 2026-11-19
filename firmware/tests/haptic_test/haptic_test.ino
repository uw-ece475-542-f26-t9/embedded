// Haptic test: the DRV2605L driver and the LRA, nothing else.
//
// Repeats: 3 buzzes, 3 s apart (start to start), then 30 s of quiet.
// The XIAO's user LED is lit during each buzz, and the Serial Monitor
// (115200) logs every buzz. If the DRV2605L doesn't answer on I2C, the LED
// blinks fast and the Serial Monitor says so once a second.
//
// Wiring: DRV2605L SDA -> D4 (GPIO 5), SCL -> D5 (GPIO 6), the same bus as
// the wearable firmware.

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_DRV2605.h> //have to download the "Adafruit DRV2605 Library" from Library Manager

constexpr int SDA_PIN = 5;  // XIAO D4
constexpr int SCL_PIN = 6;  // XIAO D5
constexpr uint8_t TEST_STRENGTH = 20;  // real-time drive level; raise it if the buzz is hard to feel

constexpr int BUZZES = 3;
constexpr uint32_t BUZZ_MS = 200;       // length of each buzz
constexpr uint32_t BUZZ_GAP_MS = 3000;  // start of one buzz to the start of the next
constexpr uint32_t REST_MS = 30000;     // quiet after the last buzz, then it repeats

// The XIAO ESP32-S3's user LED (GPIO 21) is lit when the pin is LOW.
constexpr uint8_t LED_ON = LOW;
constexpr uint8_t LED_OFF = HIGH;

Adafruit_DRV2605 drv;

void setup() {
  Serial.begin(115200);
#if ARDUINO_USB_MODE && ARDUINO_USB_CDC_ON_BOOT
  // Over native USB, a print waits for the computer to read it. Keep that
  // wait short so the buzz timing holds with no Serial Monitor open.
  Serial.setTxTimeoutMs(5);
#endif
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LED_OFF);

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  if (!drv.begin(&Wire)) {
    // Fast blink forever: the DRV2605L didn't answer at 0x5A.
    for (uint32_t i = 0;; i++) {
      digitalWrite(LED_BUILTIN, i % 2 ? LED_ON : LED_OFF);
      if (i % 10 == 0) {
        Serial.println("DRV2605L not found at 0x5A. Check power, SDA/SCL wiring and solder joints.");
      }
      delay(100);
    }
  }

  drv.useLRA();
  drv.setRealtimeValue(0);
  drv.setMode(DRV2605_MODE_REALTIME);
  Serial.println("DRV2605L ready.");
}

void buzz() {
  digitalWrite(LED_BUILTIN, LED_ON);
  drv.setRealtimeValue(TEST_STRENGTH);
  delay(BUZZ_MS);
  drv.setRealtimeValue(0);
  digitalWrite(LED_BUILTIN, LED_OFF);
}

void loop() {
  for (int i = 1; i <= BUZZES; i++) {
    Serial.printf("Buzz %d of %d\n", i, BUZZES);
    buzz();
    if (i < BUZZES) delay(BUZZ_GAP_MS - BUZZ_MS);
  }
  Serial.printf("Resting %u s\n", (unsigned)(REST_MS / 1000));
  delay(REST_MS);
}
