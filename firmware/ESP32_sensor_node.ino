/*
  ESP32 sensor node for ConveyorGuard
  ------------------------------------
  Sends JSON to  http://<PC-IP>:8000/api/sensors  once per second:
    {"temperature":54.2,"vibration":3.1,"load":420.5,"speed":2.4}

  Sensors (all optional - comment out what you don't have, the dashboard only uses what it receives):
    DS18B20  temperature  -> GPIO 4  (4.7k pull-up between DATA and 3V3)
    MPU6050  vibration    -> I2C SDA GPIO 21, SCL GPIO 22   (vibration = RMS of acceleration minus gravity)
    HX711    load cell    -> DT GPIO 16, SCK GPIO 17
    Hall     belt speed   -> GPIO 27 (one magnet per pulley revolution)

  Libraries (Arduino Library Manager): OneWire, DallasTemperature, Adafruit MPU6050, Adafruit Unified Sensor, HX711 (bogde)
*/
#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <HX711.h>

// ---------- EDIT THESE ----------
const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* SERVER_URL = "http://192.168.1.20:8000/api/sensors";   // your PC's IP
const float PULLEY_CIRCUMFERENCE_M = 0.50;                          // for belt speed
const float LOAD_CAL_FACTOR = 420.0;                                // calibrate with a known weight
// --------------------------------

#define PIN_DS18B20 4
#define PIN_HX_DT 16
#define PIN_HX_SCK 17
#define PIN_HALL 27

OneWire oneWire(PIN_DS18B20);
DallasTemperature ds(&oneWire);
Adafruit_MPU6050 mpu;
HX711 scale;
bool haveMpu = false;
volatile uint32_t pulses = 0;
void IRAM_ATTR onPulse() { pulses++; }

void setup() {
  Serial.begin(115200);
  ds.begin();
  Wire.begin(21, 22);
  haveMpu = mpu.begin();
  if (haveMpu) { mpu.setAccelerometerRange(MPU6050_RANGE_8_G); mpu.setFilterBandwidth(MPU6050_BAND_44_HZ); }
  scale.begin(PIN_HX_DT, PIN_HX_SCK);
  scale.set_scale(LOAD_CAL_FACTOR);
  scale.tare();
  pinMode(PIN_HALL, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_HALL), onPulse, FALLING);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(400); Serial.print("."); }
  Serial.println("\nWiFi OK: " + WiFi.localIP().toString());
}

// RMS of (|a| - g) over ~0.5 s  -> simple vibration intensity in m/s^2
float readVibrationRms() {
  if (!haveMpu) return NAN;
  const int N = 100;
  double sum = 0;
  for (int i = 0; i < N; i++) {
    sensors_event_t a, g, t;
    mpu.getEvent(&a, &g, &t);
    float mag = sqrtf(a.acceleration.x * a.acceleration.x + a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z);
    float d = mag - 9.81f;
    sum += d * d;
    delay(5);
  }
  return sqrt(sum / N);
}

void loop() {
  uint32_t t0 = millis();
  pulses = 0;

  ds.requestTemperatures();
  float temp = ds.getTempCByIndex(0);
  float vib = readVibrationRms();                       // takes ~0.5 s
  float load = scale.is_ready() ? scale.get_units(1) : NAN;

  float elapsed = (millis() - t0) / 1000.0f;
  float speed = (pulses / elapsed) * PULLEY_CIRCUMFERENCE_M;   // m/s (1 pulse per revolution)

  String json = "{";
  bool first = true;
  auto add = [&](const char* k, float v) {
    if (isnan(v) || v < -100) return;
    if (!first) json += ",";
    json += String("\"") + k + "\":" + String(v, 2);
    first = false;
  };
  add("temperature", temp); add("vibration", vib); add("load", load); add("speed", speed);
  json += "}";

  if (WiFi.status() == WL_CONNECTED && !first) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST(json);
    Serial.printf("%s -> %d\n", json.c_str(), code);
    http.end();
  }
  long wait = 1000 - (long)(millis() - t0);
  if (wait > 0) delay(wait);
}
