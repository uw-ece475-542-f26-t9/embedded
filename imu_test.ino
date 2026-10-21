#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>

Adafruit_MPU6050 mpu;

void setup() {
  Serial.begin(115200);
  delay(2000);

  Wire.begin(5, 6);       // SDA、SCL
  Wire.setClock(100000);  // I2C 100 kHz

  Serial.println("Starting MPU6050 test...");

  
  bool found = mpu.begin(0x68, &Wire);

  if (!found) {
    Serial.println("Trying address 0x69...");
    found = mpu.begin(0x69, &Wire);
  }

  if (!found) {
    Serial.println("MPU6050 not found.");
    Serial.println("Check power, wiring and solder joints.");

    while (true) {
      delay(1000);
    }
  }

  Serial.println("MPU6050 connected!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

void loop() {
  sensors_event_t accel, gyro, temp;
  mpu.getEvent(&accel, &gyro, &temp);

  Serial.print("Accel (m/s^2): ");
  Serial.print(accel.acceleration.x, 2);
  Serial.print(", ");
  Serial.print(accel.acceleration.y, 2);
  Serial.print(", ");
  Serial.println(accel.acceleration.z, 2);

  Serial.print("Gyro (rad/s): ");
  Serial.print(gyro.gyro.x, 2);
  Serial.print(", ");
  Serial.print(gyro.gyro.y, 2);
  Serial.print(", ");
  Serial.println(gyro.gyro.z, 2);

  Serial.print("Temperature (C): ");
  Serial.println(temp.temperature, 2);
  Serial.println();

  delay(200);
}
