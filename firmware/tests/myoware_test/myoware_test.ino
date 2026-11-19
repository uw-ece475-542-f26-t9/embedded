#include <Arduino.h>

const int EMG_PIN = 4;
const uint32_t SAMPLE_INTERVAL_US = 500;  

uint32_t lastSample = 0;

void setup() {
  Serial.begin(230400);

  pinMode(EMG_PIN, INPUT);
  analogReadResolution(12);  // ADC number 0～4095
  analogSetPinAttenuation(EMG_PIN, ADC_11db);

  delay(1500);
  lastSample = micros();
}

void loop() {
  uint32_t now = micros();

  if ((uint32_t)(now - lastSample) >= SAMPLE_INTERVAL_US) {
    lastSample = now;

    int raw = analogRead(EMG_PIN);
    Serial.println(raw);
  }
}
