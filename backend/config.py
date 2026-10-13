"""Central configuration: filesystem paths, constants and persisted runtime settings."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
UPLOADS_DIR = DATASET_DIR / "uploads"
YOLO_DIR = DATASET_DIR / "yolo"
SNAP_DIR = ROOT / "snapshots"
FRONTEND_DIR = ROOT / "frontend"

for _d in (DATA_DIR, MODELS_DIR, IMAGES_DIR, LABELS_DIR, UPLOADS_DIR, SNAP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Defect classes the custom YOLO model is expected to learn (order = class id).
CLASSES = ["crack", "worn_out", "tear", "joint_damage", "dust_spillage", "edge_fraying"]
# How much each defect class hurts the health score (0..1).
SEVERITY = {"tear": 1.0, "joint_damage": 1.0, "crack": 0.6, "worn_out": 0.4, "edge_fraying": 0.5, "dust_spillage": 0.3}
SENSOR_KEYS = ["temperature", "vibration", "load", "speed", "tension"]


class Settings(BaseModel):
    # --- video / vision ---
    stream_url: str = ""
    belt_orientation: str = Field("vertical", pattern="^(vertical|horizontal)$")
    roi: Optional[List[float]] = None          # [x, y, w, h] normalised 0..1, None = full frame
    conf_threshold: float = Field(0.4, ge=0.01, le=0.99)
    resize_width: int = Field(640, ge=160, le=1920)
    detect_every_n: int = Field(3, ge=1, le=30)  # run YOLO every Nth processed frame
    blur_min: float = 40.0                      # Laplacian variance below this = blurry
    # --- thresholds ---
    misalign_warn_pct: float = 8.0
    misalign_crit_pct: float = 15.0
    dust_warn_pct: float = 15.0
    dust_crit_pct: float = 30.0
    temp_warn: float = 65.0
    temp_crit: float = 80.0
    vib_warn: float = 5.0
    vib_crit: float = 8.0
    load_warn: float = 700.0
    load_crit: float = 900.0
    tension_warn: float = 75.0
    tension_crit: float = 85.0
    # --- health score weights (normalised internally) ---
    weights: Dict[str, float] = {
        "defects": 40, "misalignment": 25, "dust": 15,
        "vibration": 10, "load": 10,
    }
    # --- simulator ---
    simulator_enabled: bool = False
    sim_ramp_s: int = Field(180, ge=20, le=3600)   # seconds for an injected fault to reach 100%
    # --- dataset ---
    auto_capture_sec: int = Field(0, ge=0, le=3600)  # 0 = off
    # --- model ---
    active_model: str = ""                      # e.g. "v2"
    # --- MQTT ---
    mqtt_enabled: bool = False
    mqtt_host: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic: str = "conveyor/sensors"


class SettingsStore:
    """Thread-safe settings persisted to data/settings.json."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        try:
            self._s = Settings.model_validate_json(path.read_text())
        except Exception:
            self._s = Settings()

    def get(self) -> Settings:
        return self._s

    def update(self, patch: dict) -> Settings:
        """Merge `patch` into the settings. Raises pydantic.ValidationError on bad values."""
        with self._lock:
            cur = self._s.model_dump()
            for k, v in patch.items():
                if k not in cur:
                    continue
                if k == "weights" and isinstance(v, dict):
                    cur[k] = {**cur[k], **v}
                else:
                    cur[k] = v
            new = Settings(**cur)
            self._s = new
            self.path.write_text(new.model_dump_json(indent=2))
            return new


settings_store = SettingsStore(DATA_DIR / "settings.json")
