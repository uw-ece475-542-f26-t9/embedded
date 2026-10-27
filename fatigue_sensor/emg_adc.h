// Samples the MyoWare's RAW output with the ADC in continuous (DMA) mode.
//
// The ADC runs at 8 kHz in the background, so nothing in loop() (an I2C
// read of the IMU, a BLE call) can delay a sample. Polled analogRead
// stalled ~2 ms on every IMU read, which put a 50 Hz artifact in the EMG
// spectrum. Each poll() decimates what has arrived to 2 kHz with a FIR
// low-pass, because the MyoWare's own first-order 498 Hz filter barely
// attenuates at 1 kHz and anything above it would alias into the band.
//
// Voltages are in mV at the ADC pin, from the chip's calibration curve.

#pragma once

#include <esp_adc/adc_cali.h>
#include <esp_adc/adc_cali_scheme.h>
#include <esp_adc/adc_continuous.h>
#include <esp_attr.h>

class EmgAdc {
 public:
  static constexpr uint32_t ADC_HZ = 8000;
  static constexpr uint32_t DECIMATION = 4;
  static constexpr uint32_t OUT_HZ = ADC_HZ / DECIMATION;

  // Hamming-windowed sinc, 31 taps, 1 kHz cutoff at 8 kHz: flat within
  // 0.05 dB to 500 Hz, -6 dB at 1 kHz, at least 50 dB down from 1.6 kHz
  // (what would alias into 0-400 Hz). Delay: 15 input samples (1.9 ms).
  static constexpr int TAPS = 31;
  static constexpr float FIR[TAPS] = {
      -0.001203880f, -0.002053361f, -0.002079629f, 0.000000000f,
      0.004764901f,  0.009896034f,  0.009978464f,  0.000000000f,
      -0.018963789f, -0.036293321f, -0.034762035f, 0.000000000f,
      0.068632282f,  0.153265764f,  0.223458463f,  0.250720214f,
      0.223458463f,  0.153265764f,  0.068632282f,  0.000000000f,
      -0.034762035f, -0.036293321f, -0.018963789f, 0.000000000f,
      0.009978464f,  0.009896034f,  0.004764901f,  0.000000000f,
      -0.002079629f, -0.002053361f, -0.001203880f};

  // Returns false if the pin isn't an ADC pin or the driver won't start.
  bool begin(int pin) {
    adc_unit_t unit;
    adc_channel_t channel;
    if (adc_continuous_io_to_channel(pin, &unit, &channel) != ESP_OK) return false;

    // 64 readings (8 ms) per DMA frame, and room for 128 ms in the driver's
    // pool, so loop() can be held up for a while without losing data.
    adc_continuous_handle_cfg_t handleCfg = {};
    handleCfg.max_store_buf_size = 1024 * SOC_ADC_DIGI_RESULT_BYTES;
    handleCfg.conv_frame_size = FRAME_READINGS * SOC_ADC_DIGI_RESULT_BYTES;
    handleCfg.flags.flush_pool = 1;  // if it does fill, drop old data and carry on
    if (adc_continuous_new_handle(&handleCfg, &handle) != ESP_OK) return false;

    adc_digi_pattern_config_t pattern = {};
    pattern.atten = ADC_ATTEN_DB_12;  // widest input range, ~0-3.1 V
    pattern.channel = channel;
    pattern.unit = unit;
    pattern.bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;
    adc_continuous_config_t cfg = {};
    cfg.pattern_num = 1;
    cfg.adc_pattern = &pattern;
    cfg.sample_freq_hz = ADC_HZ;
    cfg.conv_mode = unit == ADC_UNIT_1 ? ADC_CONV_SINGLE_UNIT_1 : ADC_CONV_SINGLE_UNIT_2;
    cfg.format = ADC_DIGI_OUTPUT_FORMAT_TYPE2;
    if (adc_continuous_config(handle, &cfg) != ESP_OK) return false;

    adc_continuous_evt_cbs_t cbs = {};
    cbs.on_pool_ovf = onOverflow;
    adc_continuous_register_event_callbacks(handle, &cbs, this);

    // Raw code -> mV table from the calibration curve, so converting a
    // sample is a lookup and an interpolation.
    adc_cali_handle_t cali = nullptr;
    adc_cali_curve_fitting_config_t caliCfg = {};
    caliCfg.unit_id = unit;
    caliCfg.chan = channel;
    caliCfg.atten = ADC_ATTEN_DB_12;
    caliCfg.bitwidth = ADC_BITWIDTH_12;
    bool calibrated = adc_cali_create_scheme_curve_fitting(&caliCfg, &cali) == ESP_OK;
    for (int raw = 0; raw < 4096; raw++) {
      int mv = 0;
      if (!calibrated || adc_cali_raw_to_voltage(cali, raw, &mv) != ESP_OK) {
        mv = raw * 3100 / 4095;  // nominal range if there's no calibration
      }
      mvTable[raw] = mv;
    }
    if (calibrated) adc_cali_delete_scheme_curve_fitting(cali);

    return adc_continuous_start(handle) == ESP_OK;
  }

  // Calls onSample(mv, clipped) for each new 2 kHz sample, where clipped
  // is how many of the 4 raw readings behind it were at 0 or 4095.
  template <typename F>
  void poll(F onSample) {
    if (!handle) return;
    uint32_t bytes = 0;
    while (adc_continuous_read(handle, readBuf, sizeof(readBuf), &bytes, 0) == ESP_OK) {
      uint32_t n = 0;
      if (adc_continuous_parse_data(handle, readBuf, bytes, parsed, &n) != ESP_OK) continue;
      for (uint32_t i = 0; i < n; i++) {
        if (!parsed[i].valid) continue;
        uint16_t raw = parsed[i].raw_data;
        history[pos] = raw;
        pos = (pos + 1) % TAPS;
        if (raw == 0 || raw >= 4095) clipped++;
        if (++phase < DECIMATION) continue;
        phase = 0;

        // Newest reading is at pos - 1; FIR is symmetric, so order is moot.
        float acc = 0;
        for (int k = 0; k < TAPS; k++) acc += FIR[k] * history[(pos + k) % TAPS];
        outCount++;
        onSample(toMv(acc), clipped);
        clipped = 0;
      }
    }
  }

  // 2 kHz samples produced so far, i.e. the index the next one will have.
  uint32_t samples() const { return outCount; }
  // Times the driver's pool filled up and data was lost.
  uint32_t overflows() const { return overflowCount; }

 private:
  static constexpr uint32_t FRAME_READINGS = 64;

  static bool IRAM_ATTR onOverflow(adc_continuous_handle_t, const adc_continuous_evt_data_t *, void *self) {
    static_cast<EmgAdc *>(self)->overflowCount++;
    return false;
  }

  float toMv(float raw) const {
    if (raw <= 0) return mvTable[0];
    if (raw >= 4095) return mvTable[4095];
    int i = (int)raw;
    return mvTable[i] + (raw - i) * (mvTable[i + 1] - mvTable[i]);
  }

  adc_continuous_handle_t handle = nullptr;
  uint8_t readBuf[FRAME_READINGS * SOC_ADC_DIGI_RESULT_BYTES];
  adc_continuous_data_t parsed[FRAME_READINGS];
  uint16_t mvTable[4096];
  float history[TAPS] = {};
  int pos = 0;
  uint32_t phase = 0;
  uint8_t clipped = 0;
  uint32_t outCount = 0;
  volatile uint32_t overflowCount = 0;
};
