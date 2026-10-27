// Capture mode: streams every EMG sample, every IMU sample and the rep
// counter's turning points and reps over USB serial as framed binary, for
// tools/record.py to save and tools/fatigue_features.py to analyse.
//
// Everything is stamped with the EMG sample index (2 kHz, counting from 0
// at boot), so the streams line up exactly in the analysis.
//
// Frame, little-endian:
//   0xA5 0x5A | type u8 | length u16 | payload (length bytes) | crc u16
// crc is CRC-16/CCITT-FALSE over type, length and payload. A frame that
// doesn't fit in the serial TX buffer is dropped (and counted) rather than
// blocking, so a slow or absent reader never stalls the ADC.
//
// Types and payloads:
//   0x01 STATUS, once a second
//        u8 version, u16 emg_hz, u16 adc_hz, u16 imu_hz, u16 emg_per_mv,
//        u32 emg_index, u32 millis, u32 adc_overflows, u32 frames_dropped
//   0x02 EMG, every EMG_BLOCK samples
//        u32 index of the first sample, u8 n, u8 clipped,
//        n x u16 sample in 1/emg_per_mv mV (at the ADC pin)
//        clipped = raw 8 kHz ADC readings at 0 or 4095 behind these samples
//   0x03 IMU, 50 Hz
//        u32 emg_index, i16 ax ay az (milli-g), i16 gx gy gz (0.1 deg/s,
//        bias removed), f32 arm angle (deg), u8 still
//   0x04 TURN, when the rep counter confirms a turning point
//        u32 emg_index of the turning point, f32 angle (deg),
//        i8 +1 if it was the top of a swing up the angle, -1 if the bottom
//   0x05 REP, when a rep is counted
//        u32 emg_index, u16 reps since boot, u16 range of motion (0.1 deg)
//
// No Arduino code in here except the Print it writes to.

#pragma once

#include <Print.h>
#include <stdint.h>
#include <string.h>

namespace capture {

constexpr uint8_t VERSION = 1;
constexpr uint8_t STATUS = 0x01, EMG = 0x02, IMU = 0x03, TURN = 0x04, REP = 0x05;
constexpr uint16_t EMG_PER_MV = 10;  // samples are sent in 0.1 mV
constexpr uint8_t EMG_BLOCK = 32;    // 16 ms of EMG per frame

inline uint16_t crc16(const uint8_t *p, size_t n, uint16_t crc = 0xFFFF) {
  while (n--) {
    crc ^= (uint16_t)(*p++) << 8;
    for (int i = 0; i < 8; i++) crc = crc & 0x8000 ? (crc << 1) ^ 0x1021 : crc << 1;
  }
  return crc;
}

// Builds a payload field by field.
class Payload {
 public:
  void u8(uint8_t v) { buf[len++] = v; }
  void i8(int8_t v) { u8((uint8_t)v); }
  void u16(uint16_t v) { u8(v & 0xFF); u8(v >> 8); }
  void i16(int16_t v) { u16((uint16_t)v); }
  void u32(uint32_t v) { u16(v & 0xFFFF); u16(v >> 16); }
  void f32(float v) { uint32_t u; memcpy(&u, &v, 4); u32(u); }

  uint8_t buf[2 * EMG_BLOCK + 16];
  uint16_t len = 0;
};

class Writer {
 public:
  explicit Writer(Print &out) : out(out) {}

  // Sends a frame now, or drops it if the TX buffer can't take it whole.
  // `room` is how much the TX buffer can take; pass out.availableForWrite().
  void send(uint8_t type, const Payload &p, int room) {
    uint8_t frame[sizeof(p.buf) + 7];
    frame[0] = 0xA5;
    frame[1] = 0x5A;
    frame[2] = type;
    frame[3] = p.len & 0xFF;
    frame[4] = p.len >> 8;
    memcpy(frame + 5, p.buf, p.len);
    uint16_t crc = crc16(frame + 2, p.len + 3);
    frame[5 + p.len] = crc & 0xFF;
    frame[6 + p.len] = crc >> 8;
    size_t n = p.len + 7;
    if (room < (int)n) {
      dropped++;
      return;
    }
    out.write(frame, n);
  }

  uint32_t dropped = 0;

 private:
  Print &out;
};

// Collects EMG samples into EMG frames.
class EmgBlocks {
 public:
  // Returns true when a block is full and `payload` is ready to send.
  bool add(uint32_t index, float mv, uint8_t clipped) {
    if (n == 0) first = index;
    float v = mv * EMG_PER_MV + 0.5f;
    samples[n++] = v <= 0 ? 0 : v >= 65535 ? 65535 : (uint16_t)v;
    clips += clipped;
    if (n < EMG_BLOCK) return false;
    payload.len = 0;
    payload.u32(first);
    payload.u8(n);
    payload.u8(clips > 255 ? 255 : clips);
    for (uint8_t i = 0; i < n; i++) payload.u16(samples[i]);
    n = 0;
    clips = 0;
    return true;
  }

  Payload payload;

 private:
  uint16_t samples[EMG_BLOCK];
  uint8_t n = 0;
  uint32_t first = 0;
  uint32_t clips = 0;
};

}  // namespace capture
