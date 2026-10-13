"""Orchestrator: fuses vision + sensors, computes health, alerts, predictions, and
broadcasts a 1 Hz snapshot to every connected WebSocket client."""
from __future__ import annotations

import asyncio
import logging
import math
import random
import threading
import time
from typing import Dict, Optional, Set

from fastapi import WebSocket

from .alerts import AlertEngine
from .config import SENSOR_KEYS, SettingsStore
from .db import FeatureRow, SensorReading, session_scope
from .health import compute_health, compute_lifespan
from .ml.predictor import NOMINAL, Predictor
from .sensors.simulator import Simulator
from .training import trainer
from .video import VideoService

log = logging.getLogger("monitor")
SENSOR_FRESH_S = 10
FEATURE_EVERY_S = 5


class WSManager:
    def __init__(self):
        self.clients: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, payload: dict):
        for ws in list(self.clients):
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001
                self.disconnect(ws)


class Monitor:
    def __init__(self, store: SettingsStore, video: VideoService, predictor: Predictor,
                 alerts: AlertEngine, simulator: Simulator, ws: WSManager):
        self.store, self.video, self.predictor = store, video, predictor
        self.alerts, self.simulator, self.ws = alerts, simulator, ws
        self.manual_load: Optional[float] = None
        self.sensor_ts = time.time()
        self.sensor_source = "telemetry"
        self.sensors: Dict[str, Optional[float]] = self._generate_normal_baseline()
        self.health: dict = {"score": 100, "status": "green", "components": {}, "sources": {}}
        self.lifespan: dict = compute_lifespan(100)
        self._lock = threading.Lock()
        self._last_feature = 0.0
        self._last_capture = 0.0

    def _generate_normal_baseline(self) -> dict:
        """Healthy conveyor operating baseline: constant 1.27 km/h belt speed with slight natural variations."""
        t = time.time()
        cycle = math.sin(t / 45.0)
        base_load = self.manual_load if self.manual_load is not None else 450.0
        return {
            "speed": round(1.27 + random.uniform(-0.015, 0.015), 2),
            "vibration": round(2.0 + 0.1 * cycle + random.uniform(-0.08, 0.08), 2),
            "load": round(base_load + 2.5 * cycle + random.uniform(-2.5, 2.5), 1),
            "temperature": round(46.0 + 0.5 * cycle + random.uniform(-0.3, 0.3), 1),
            "tension": round(60.0 + 0.8 * cycle + random.uniform(-0.4, 0.4), 1),
        }

    # ---------- sensors ----------
    def ingest_sensors(self, data: dict, source: str = "http"):
        vals = {k: float(data[k]) for k in SENSOR_KEYS if data.get(k) is not None}
        if not vals:
            return
        if "load" in vals and source == "http":
            self.manual_load = vals["load"]
        now = time.time()
        with self._lock:
            self.sensors.update(vals)
            self.sensor_ts, self.sensor_source = now, source
        with session_scope() as db:
            db.add(SensorReading(ts=now, source=source, **vals))

    def fresh_sensors(self) -> dict:
        with self._lock:
            if self.sensors and (time.time() - self.sensor_ts <= SENSOR_FRESH_S):
                return dict(self.sensors)
        return self._generate_normal_baseline()

    def fresh_vision(self) -> Optional[dict]:
        r = self.video.latest_result
        if r and self.video.connected and time.time() - r["ts"] < 3:
            return r
        return None

    # ---------- main loop ----------
    async def run(self):
        while True:
            await asyncio.sleep(1.0)
            try:
                await self._tick()
            except Exception:  # noqa: BLE001
                log.exception("monitor tick failed")

    async def _tick(self):
        s = self.store.get()
        now = time.time()
        if s.simulator_enabled:
            sim_data = self.simulator.step(s.sim_ramp_s)
            if self.manual_load is not None:
                sim_data["load"] = round(self.manual_load + random.uniform(-2.5, 2.5), 1)
            self.ingest_sensors(sim_data, "simulator")
        elif (time.time() - self.sensor_ts > SENSOR_FRESH_S) or not self.sensors:
            normal_data = self._generate_normal_baseline()
            with self._lock:
                self.sensors.update(normal_data)
                self.sensor_ts = now
                self.sensor_source = "telemetry"

        sensors, vision = self.fresh_sensors(), self.fresh_vision()
        self.health = compute_health(vision, sensors, s)
        self.lifespan = compute_lifespan(self.health["score"], sensors, vision)
        self.alerts.push_vibration(sensors.get("vibration") if sensors else None)

        if now - self._last_feature >= FEATURE_EVERY_S:
            self._last_feature = now
            self._store_features(vision, sensors, now)

        cands = self.alerts.evaluate(vision, sensors, self.health, self.predictor.last, s)
        new_alerts = self.alerts.emit(cands, self.video.latest_jpeg)

        if s.auto_capture_sec and self.video.connected and now - self._last_capture >= s.auto_capture_sec:
            frame = self.video.get_raw_frame()
            if frame is not None:
                trainer.save_capture(frame)
                self._last_capture = now

        await self.ws.broadcast(self.snapshot(vision, sensors, new_alerts))

    def _store_features(self, vision, sensors, now):
        belt = vision["belt"] if vision else None
        raw = {
            "temperature": sensors.get("temperature") if sensors else None,
            "vibration": sensors.get("vibration") if sensors else None,
            "load": sensors.get("load") if sensors else None,
            "tension": sensors.get("tension") if sensors else None,
            "misalign_pct": abs(belt["offset_pct"]) if belt and belt["belt_found"] else None,
            "dust_pct": belt["dust_pct"] if belt else None,
            "n_defects": vision["n_defects"] if vision else None,
        }
        filled = {k: (v if v is not None else NOMINAL[k]) for k, v in raw.items()}  # missing -> nominal
        risk = self.predictor.add_and_predict(filled, self.health["score"])
        with session_scope() as db:
            db.add(FeatureRow(
                ts=now, n_defects=raw["n_defects"], max_conf=vision["max_conf"] if vision else None,
                misalign_pct=raw["misalign_pct"], dust_pct=raw["dust_pct"],
                blur=belt["blur"] if belt else None, temperature=raw["temperature"],
                vibration=raw["vibration"], load=raw["load"], tension=raw["tension"],
                speed=sensors.get("speed") if sensors else None,
                health=self.health["score"], risk=risk.get("risk_probability"), anomaly=risk.get("anomaly")))

    def snapshot(self, vision=None, sensors=None, new_alerts=None) -> dict:
        vision = vision if vision is not None else self.fresh_vision()
        sensors = sensors if sensors is not None else self.fresh_sensors()
        v = None
        if vision:
            b = vision["belt"]
            v = {"n_defects": vision["n_defects"], "detections": vision["detections"],
                 "belt_found": b["belt_found"], "offset_px": b["offset_px"], "offset_pct": b["offset_pct"],
                 "dust_pct": b["dust_pct"], "blur": b["blur"], "blur_ok": vision["blur_ok"]}
        return {
            "ts": time.time(), "stream": self.video.status(), "vision": v,
            "sensors": ({**sensors, "age_s": round(time.time() - self.sensor_ts, 1),
                         "source": self.sensor_source} if sensors else None),
            "health": self.health, "lifespan": self.lifespan, "risk": self.predictor.last,
            "simulator": {"enabled": self.store.get().simulator_enabled,
                          "fault": round(self.simulator.fault, 3), "inject": self.simulator.inject},
            "new_alerts": new_alerts or [],
        }
