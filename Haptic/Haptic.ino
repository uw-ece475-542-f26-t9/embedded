#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_DRV2605.h> //have to download the "Adafruit DRV2605 Library" from Library Manager

constexpr int SDA_PIN = 5;
constexpr int SCL_PIN = 6;
constexpr uint8_t TEST_STRENGTH = 20;

Adafruit_DRV2605 drv;

void setup() {
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  if (!drv.begin(&Wire)) {
    // Red light on: DRV initialized fail
    rgbLedWrite(RGB_BUILTIN, 20, 0, 0);
    while (true) {
      delay(1000);
    }
  }

  drv.useLRA();
  drv.setRealtimeValue(0);
  drv.setMode(DRV2605_MODE_REALTIME);
}

void loop() {
  // Green light on: start vibration
  rgbLedWrite(RGB_BUILTIN, 0, 20, 0);
  drv.setRealtimeValue(TEST_STRENGTH);
  delay(200);

  drv.setRealtimeValue(0);
  rgbLedWrite(RGB_BUILTIN, 0, 0, 0);
  delay(800);
}