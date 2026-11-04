// EMG recorder: streams MyoWare 2.0 RAW + ENV from an ESP32-S3 to the laptop.
//
// Flash through the board's "USB" port with USB CDC On Boot enabled, so Serial goes over that port:
//   arduino-cli compile --upload -p /dev/cu.usbmodemXXXX --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc firmware/emg_recorder
//
// Wiring: MyoWare + -> 3V3, - -> GND, ENV -> GPIO2,
//         RAW -> GPIO4. (An RC filter, 1 kOhm in series + 100 nF to GND, is recommended but not fitted;
//         the Test 2 spike check at 10000 vs 10500/s is run each session instead to catch aliasing.)
//
// The ADC's DMA engine sets the timing in hardware, so readings stay evenly spaced
// no matter what loop() is doing. Wi-Fi and Bluetooth are never started (they add
// ADC noise).
//
// Output, binary little-endian packets:
//   uint16 sync (0x5AA5) | uint32 board time (us) | uint16 pair count | uint32 overflow count
//   followed by <pair count> x (uint16 raw, uint16 env), each 0-4095.
// Input: "RATE <hz>\n" sets readings per second per pin (default 10000).

#include "esp_adc/adc_continuous.h"
#include "soc/soc_caps.h"

static const adc_channel_t RAW_CH = ADC_CHANNEL_3;  // GPIO4 (ADC1 channel 3 on the ESP32-S3)
static const adc_channel_t ENV_CH = ADC_CHANNEL_1;  // GPIO2
static const uint32_t DEFAULT_RATE_HZ = 10000;
static const uint32_t MAX_RATE_HZ = 15000;  // 921600 baud carries ~92 kB/s; 15k pairs/s is ~61 kB/s
// Conversions per DMA frame = 10 RAW/ENV pairs, so one packet per 10 readings. Each packet causes a tiny
// glitch on the ADC; at 1 packet per 10 readings every echo of that glitch lands on a multiple of
// rate/10 (1000 Hz at 10000/s), well outside the 20-450 Hz muscle band.
static const uint32_t FRAME_CONV = 20;
static const uint32_t FRAME_BYTES = FRAME_CONV * SOC_ADC_DIGI_RESULT_BYTES;
static const uint16_t SYNC = 0x5AA5;

struct __attribute__((packed)) Header {
  uint16_t sync;
  uint32_t t_us;
  uint16_t pairs;
  uint32_t overflows;
};

static adc_continuous_handle_t adc = nullptr;
static volatile uint32_t overflows = 0;  // DMA frames dropped because loop() fell behind
static uint8_t frame[FRAME_BYTES];
static uint16_t pairs_out[FRAME_CONV];   // raw, env, raw, env, ...
static int pending_raw = -1;             // RAW reading waiting for its ENV partner

static bool IRAM_ATTR on_overflow(adc_continuous_handle_t, const adc_continuous_evt_data_t *, void *) {
  overflows = overflows + 1;
  return false;
}

static void start_adc(uint32_t rate_hz) {
  if (adc) {
    adc_continuous_stop(adc);
    adc_continuous_deinit(adc);
    adc = nullptr;
  }
  pending_raw = -1;

  adc_continuous_handle_cfg_t handle_cfg = {};
  handle_cfg.max_store_buf_size = 32768;  // 8192 conversions, ~400 ms of slack if the USB link stalls
  handle_cfg.conv_frame_size = FRAME_BYTES;
  ESP_ERROR_CHECK(adc_continuous_new_handle(&handle_cfg, &adc));

  adc_digi_pattern_config_t pattern[2] = {};
  const adc_channel_t channels[2] = {RAW_CH, ENV_CH};
  for (int i = 0; i < 2; i++) {
    pattern[i].atten = ADC_ATTEN_DB_12;                // widest input range, ~0-3.1 V
    pattern[i].channel = channels[i];
    pattern[i].unit = ADC_UNIT_1;
    pattern[i].bit_width = SOC_ADC_DIGI_MAX_BITWIDTH;  // 12 bits -> 0-4095
  }

  adc_continuous_config_t cfg = {};
  cfg.pattern_num = 2;
  cfg.adc_pattern = pattern;
  cfg.sample_freq_hz = rate_hz * 2;  // conversions alternate RAW, ENV, RAW, ENV, ...
  cfg.conv_mode = ADC_CONV_SINGLE_UNIT_1;
  cfg.format = ADC_DIGI_OUTPUT_FORMAT_TYPE2;
  ESP_ERROR_CHECK(adc_continuous_config(adc, &cfg));

  adc_continuous_evt_cbs_t cbs = {};
  cbs.on_pool_ovf = on_overflow;
  ESP_ERROR_CHECK(adc_continuous_register_event_callbacks(adc, &cbs, nullptr));
  ESP_ERROR_CHECK(adc_continuous_start(adc));
}

static void handle_commands() {
  static char line[32];
  static size_t len = 0;
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n' || c == '\r') {
      line[len] = '\0';
      if (strncmp(line, "RATE ", 5) == 0) {
        uint32_t rate = strtoul(line + 5, nullptr, 10);
        if (rate >= 1000 && rate <= MAX_RATE_HZ) start_adc(rate);
      }
      len = 0;
    } else if (len < sizeof(line) - 1) {
      line[len++] = c;
    }
  }
}

void setup() {
  Serial.setTxBufferSize(8192);
  Serial.begin(921600);
  start_adc(DEFAULT_RATE_HZ);
}

void loop() {
  handle_commands();

  uint32_t n_bytes = 0;
  if (adc_continuous_read(adc, frame, FRAME_BYTES, &n_bytes, 20) != ESP_OK) return;

  uint16_t n_pairs = 0;
  for (uint32_t i = 0; i < n_bytes; i += SOC_ADC_DIGI_RESULT_BYTES) {
    const adc_digi_output_data_t *p = (const adc_digi_output_data_t *)&frame[i];
    if (p->type2.unit != 0) continue;  // not ADC1
    if (p->type2.channel == (uint32_t)RAW_CH) {
      pending_raw = p->type2.data;
    } else if (p->type2.channel == (uint32_t)ENV_CH && pending_raw >= 0) {
      pairs_out[2 * n_pairs] = pending_raw;
      pairs_out[2 * n_pairs + 1] = p->type2.data;
      n_pairs++;
      pending_raw = -1;
    }
  }
  if (n_pairs == 0) return;

  Header h = {SYNC, (uint32_t)micros(), n_pairs, overflows};
  Serial.write((const uint8_t *)&h, sizeof(h));
  Serial.write((const uint8_t *)pairs_out, n_pairs * 2 * sizeof(uint16_t));
}
