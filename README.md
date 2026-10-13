# ConveyorGuard - Conveyor Belt Health Monitoring & Predictive Maintenance

Phone/IP camera + computer vision + sensors + ML. Paste a camera URL in the web dashboard and get live
annotated video, belt-misalignment and dust estimates, a Belt Health Score, alerts and a failure-risk
prediction. Data collection, training and model hot-swap are built into the UI.

> **Status: working base (v0.1).** See `NEXT_STEPS.md` for what is done and what to build next.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # ultralytics pulls in PyTorch (large download)
uvicorn backend.main:app --reload
```
Open **http://localhost:8000**.

No phone handy? Generate a synthetic belt video and use its full path as the stream URL:
```bash
python scripts/make_test_video.py
```
Then **Settings -> Enable simulator** to get fake sensor data, and **Inject fault** to watch the
health score drop, alerts fire and the failure risk rise.

## Connect the phone camera
1. Install **IP Webcam** (Android) -> *Start server*. Phone and PC must be on the same Wi-Fi.
2. Note the URL it shows, e.g. `http://192.168.1.10:8080`.
3. In the dashboard paste `http://192.168.1.10:8080/video` and click **Connect**.
   (Try `/videofeed` if `/video` fails. Enter `0` to use the PC webcam.)
4. For less blur: lock exposure, add steady LED light, choose 720p/60 fps in the app, keep the phone plugged in.

## Real sensors (ESP32)
Flash `firmware/ESP32_sensor_node.ino`, set Wi-Fi + your PC's IP. It POSTs JSON each second:

| Sensor | Part | ESP32 pins |
|---|---|---|
| Temperature | DS18B20 (+4.7k pull-up) | GPIO 4 |
| Vibration | MPU6050 | SDA 21, SCL 22 |
| Load | HX711 + load cell | DT 16, SCK 17 |
| Belt speed | Hall sensor + magnet | GPIO 27 |

Any device can also `POST /api/sensors` or publish to MQTT topic `conveyor/sensors` (enable in settings).
Missing sensors are fine: the dashboard only uses what it receives.

## Workflow: collect -> label -> train -> deploy
1. **Dataset & Training -> Capture** frames (or auto-capture) with healthy belt and (staged) defects.
2. Label them (Roboflow / LabelImg), then **upload the YOLO-format zip**. (In-app labelling UI is the next phase;
   the label API already exists.)
3. Pick base model, epochs -> **Start training** (needs `ultralytics`; GPU recommended, CPU works for yolov8n).
4. When it finishes, **Deploy to live feed**. The "demo mode" banner disappears and detections appear.

## How the score and prediction work
* **Health score** = 100 - weighted penalties (defects, misalignment, dust, temperature, vibration, load, tension).
  Weights are re-normalised over the sources that currently have data, and the score is capped by the worst single
  component, so one critical reading always turns it red.
* **Anomaly detector** (Isolation Forest) learns "normal" from the first ~10 min of healthy live data.
* **Failure risk / RUL** (Random Forest) is bootstrapped on **synthetic** run-to-failure data. It demonstrates the
  pipeline; train it on your own labelled history for real predictions.

## Project layout
```
backend/   main.py (API) | video.py (stream+vision threads) | monitor.py (fusion loop, WebSocket)
           health.py | alerts.py | config.py | db.py | schemas.py
           vision/ (belt.py, detector.py) | sensors/ (simulator.py, mqtt_client.py)
           ml/predictor.py | training/trainer.py
frontend/  index.html, css/style.css, js/app.js   (plain JS + Chart.js, no build step)
firmware/  ESP32_sensor_node.ino
scripts/   make_test_video.py
```

## API (selected)
`POST /api/stream/connect {url}` | `GET /video_feed` | `WS /ws/live` | `POST /api/sensors` | `GET /api/sensors/history`
| `GET/POST /api/settings` | `GET /api/alerts` | `POST /api/alerts/{id}/ack` | `GET /api/predict/risk`
| `POST /api/dataset/capture` | `POST /api/dataset/upload` | `POST /api/train/start` | `GET /api/train/status`
| `POST /api/model/deploy` | `POST /api/simulator/toggle` | `POST /api/simulator/fault`.  Interactive docs: `/docs`.

## Known limits
Camera sees the surface only: internal cord damage, tension and overheating need the sensors. The belt-edge
tracker assumes two clear straight belt edges. Phones are fine for a prototype, not for a dusty mine.
