"""FastAPI entrypoint.   Run:  uvicorn backend.main:app --reload   then open http://localhost:8000"""
from __future__ import annotations

import asyncio
import csv
import io
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .alerts import AlertEngine
from .config import CLASSES, FRONTEND_DIR, IMAGES_DIR, MODELS_DIR, SNAP_DIR, settings_store
from .db import Alert, FeatureRow, SensorReading, session_scope
from .ml.predictor import Predictor
from .monitor import Monitor, WSManager
from .schemas import DeployIn, LabelsIn, SensorIn, StreamConnect, ToggleIn, TrainParams
from .sensors.mqtt_client import MqttBridge
from .sensors.simulator import Simulator
from .training import trainer
from .video import VideoService, make_placeholder
from .vision.detector import Detector
from sqlalchemy import select

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("main")

store = settings_store
detector = Detector()
video = VideoService(store, detector)
predictor = Predictor()
alert_engine = AlertEngine()
simulator = Simulator()
ws_manager = WSManager()
monitor = Monitor(store, video, predictor, alert_engine, simulator, ws_manager)
mqtt = MqttBridge(monitor.ingest_sensors)
training = trainer.TrainingManager()


def _activate_model(version: str) -> bool:
    path = MODELS_DIR / version / "best.pt"
    if path.exists() and detector.load(path):
        store.update({"active_model": version})
        return True
    return False


@asynccontextmanager
async def lifespan(_app: FastAPI):
    s = store.get()
    if s.active_model:
        _activate_model(s.active_model)
    threading.Thread(target=predictor.load_or_bootstrap, daemon=True).start()
    if s.mqtt_enabled:
        mqtt.start(s.mqtt_host, s.mqtt_port, s.mqtt_topic)
    task = asyncio.create_task(monitor.run())
    yield
    task.cancel()
    video.stop()
    mqtt.stop()


app = FastAPI(title="Conveyor Belt Health Monitor", lifespan=lifespan)


# ------------------------------------------------------------------ stream / video
@app.post("/api/stream/connect")
def stream_connect(body: StreamConnect):
    store.update({"stream_url": body.url})
    video.start(body.url)
    return video.status()


@app.post("/api/stream/disconnect")
def stream_disconnect():
    video.stop()
    return video.status()


@app.get("/api/stream/status")
def stream_status():
    return video.status()


@app.get("/video_feed")
async def video_feed():
    async def gen():
        last_text, placeholder = "", b""
        while True:
            st = video
            if st.latest_jpeg and st.connected and time.time() - st.last_frame_ts < 3:
                jpg = st.latest_jpeg
            else:
                text = ("Connecting..." if st.running and not st.last_error else
                        st.last_error if st.running else "No stream connected")
                if text != last_text:
                    last_text, placeholder = text, make_placeholder(text[:48])
                jpg = placeholder
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(0.04)
    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.get("/frame.jpg")
@app.get("/api/stream/frame")
def single_frame():
    st = video
    if st.latest_jpeg and st.connected and time.time() - st.last_frame_ts < 3:
        jpg = st.latest_jpeg
    else:
        text = ("Connecting..." if st.running and not st.last_error else
                st.last_error if st.running else "No stream connected")
        jpg = make_placeholder(text[:48])
    return Response(
        content=jpg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
    )


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        await ws.send_json(monitor.snapshot())
        while True:
            await ws.receive_text()      # keep-alive; client sends nothing important
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)
    except Exception:  # noqa: BLE001
        ws_manager.disconnect(ws)


# ------------------------------------------------------------------ sensors / simulator
@app.post("/api/sensors")
def post_sensors(body: SensorIn):
    data = body.model_dump(exclude_none=True)
    if not data:
        raise HTTPException(400, "No sensor values supplied")
    monitor.ingest_sensors(data, "http")
    return {"ok": True}


@app.get("/api/sensors/history")
def sensors_history(minutes: int = 30):
    cutoff = time.time() - max(1, min(minutes, 1440)) * 60
    with session_scope() as db:
        rows = db.execute(select(SensorReading).where(SensorReading.ts >= cutoff)
                          .order_by(SensorReading.ts)).scalars().all()
    stride = max(1, len(rows) // 600)
    rows = rows[::stride]
    return {k: [getattr(r, k) for r in rows] for k in ("ts", "temperature", "vibration", "load", "speed", "tension")}


@app.post("/api/simulator/toggle")
def simulator_toggle(body: ToggleIn):
    new = (not store.get().simulator_enabled) if body.enabled is None else body.enabled
    store.update({"simulator_enabled": new})
    if not new:
        simulator.inject = False
    return {"enabled": new}


@app.post("/api/simulator/fault")
def simulator_fault(body: ToggleIn):
    simulator.inject = (not simulator.inject) if body.enabled is None else body.enabled
    return {"inject": simulator.inject}


# ------------------------------------------------------------------ settings
@app.get("/api/settings")
def get_settings():
    return {**store.get().model_dump(), "classes": CLASSES}


@app.post("/api/settings")
def post_settings(patch: dict):
    before = store.get()
    try:
        new = store.update(patch)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False))
    if (before.mqtt_enabled, before.mqtt_host, before.mqtt_port, before.mqtt_topic) != \
            (new.mqtt_enabled, new.mqtt_host, new.mqtt_port, new.mqtt_topic):
        mqtt.start(new.mqtt_host, new.mqtt_port, new.mqtt_topic) if new.mqtt_enabled else mqtt.stop()
    return new


# ------------------------------------------------------------------ alerts
def _alert_dict(a: Alert) -> dict:
    return {"id": a.id, "ts": a.ts, "severity": a.severity, "kind": a.kind, "message": a.message,
            "snapshot": f"/snapshots/{a.snapshot}" if a.snapshot else None, "acknowledged": a.acknowledged}


@app.get("/api/alerts")
def get_alerts(limit: int = 100, severity: str = "", acknowledged: str = ""):
    q = select(Alert).order_by(Alert.ts.desc()).limit(min(limit, 1000))
    if severity:
        q = q.where(Alert.severity == severity)
    if acknowledged in ("true", "false"):
        q = q.where(Alert.acknowledged == (acknowledged == "true"))
    with session_scope() as db:
        return [_alert_dict(a) for a in db.execute(q).scalars().all()]


@app.post("/api/alerts/{alert_id}/ack")
def ack_alert(alert_id: int):
    with session_scope() as db:
        a = db.get(Alert, alert_id)
        if not a:
            raise HTTPException(404, "Alert not found")
        a.acknowledged = True
    return {"ok": True}


@app.get("/api/alerts/export.csv")
def export_alerts():
    with session_scope() as db:
        rows = db.execute(select(Alert).order_by(Alert.ts.desc())).scalars().all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "time", "severity", "kind", "message", "acknowledged", "snapshot"])
    for a in rows:
        w.writerow([a.id, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a.ts)), a.severity,
                    a.kind, a.message, a.acknowledged, a.snapshot or ""])
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=alerts.csv"})


# ------------------------------------------------------------------ prediction
@app.get("/api/predict/risk")
def predict_risk():
    return predictor.last


@app.post("/api/predict/retrain")
def predict_retrain():
    if predictor.training:
        raise HTTPException(409, "Already training")
    threading.Thread(target=predictor.retrain, daemon=True).start()
    return {"ok": True}


@app.get("/api/predict/history")
def predict_history(minutes: int = 60):
    cutoff = time.time() - max(1, min(minutes, 1440)) * 60
    with session_scope() as db:
        rows = db.execute(select(FeatureRow).where(FeatureRow.ts >= cutoff)
                          .order_by(FeatureRow.ts)).scalars().all()
    stride = max(1, len(rows) // 500)
    rows = rows[::stride]
    return {k: [getattr(r, k) for r in rows] for k in ("ts", "health", "risk", "anomaly")}


# ------------------------------------------------------------------ dataset / labels
@app.post("/api/dataset/capture")
def dataset_capture():
    frame = video.get_raw_frame()
    if frame is None:
        raise HTTPException(409, "No live frame - connect a stream first")
    return {"name": trainer.save_capture(frame)}


@app.get("/api/dataset/images")
def dataset_images():
    return {"images": trainer.list_images(), "uploads": trainer.list_uploads(),
            "labeled": sum(1 for i in trainer.list_images() if i["labeled"])}


@app.delete("/api/dataset/images/{name}")
def dataset_delete(name: str):
    try:
        trainer.delete_image(name)
    except FileNotFoundError:
        raise HTTPException(404, "Image not found")
    return {"ok": True}


@app.post("/api/dataset/upload")
async def dataset_upload(file: UploadFile = File(...)):
    try:
        return trainer.import_zip(await file.read(), file.filename or "upload.zip")
    except (ValueError, Exception) as e:  # noqa: BLE001
        raise HTTPException(400, str(e))


@app.get("/api/labels/{image}")
def get_labels(image: str):
    try:
        trainer.safe_image_path(image)
    except FileNotFoundError:
        raise HTTPException(404, "Image not found")
    return {"boxes": trainer.load_labels(image), "classes": CLASSES}


@app.post("/api/labels/{image}")
def post_labels(image: str, body: LabelsIn):
    try:
        trainer.save_labels(image, [b.model_dump() for b in body.boxes])
    except FileNotFoundError:
        raise HTTPException(404, "Image not found")
    return {"ok": True}


# ------------------------------------------------------------------ training / models
@app.post("/api/train/start")
def train_start(p: TrainParams):
    try:
        return training.start(p.model_dump())
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@app.get("/api/train/status")
def train_status():
    return training.status


@app.get("/api/models")
def models():
    return {"models": trainer.list_models(store.get().active_model),
            "demo_mode": detector.demo_mode, "error": detector.error}


@app.post("/api/model/deploy")
def model_deploy(body: DeployIn):
    if not _activate_model(body.version):
        raise HTTPException(400, detector.error or f"Model {body.version} not found")
    return {"active": body.version}


@app.post("/api/model/unload")
def model_unload():
    detector.unload()
    store.update({"active_model": ""})
    return {"ok": True}


@app.get("/api/state")
def state():
    return monitor.snapshot()


# ------------------------------------------------------------------ static
app.mount("/snapshots", StaticFiles(directory=SNAP_DIR), name="snapshots")
app.mount("/dataset-images", StaticFiles(directory=IMAGES_DIR), name="dataset-images")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
