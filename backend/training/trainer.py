"""Dataset management, labelling storage, zip import and YOLO training.

NOTE: `TrainingManager` needs `ultralytics` (+ torch). It is wired end-to-end but must be
exercised on a machine with ultralytics installed.
"""
from __future__ import annotations

import json
import logging
import random
import shutil
import threading
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from ..config import CLASSES, IMAGES_DIR, LABELS_DIR, MODELS_DIR, UPLOADS_DIR, YOLO_DIR

log = logging.getLogger("trainer")
IMG_EXT = {".jpg", ".jpeg", ".png"}


# ------------------------------------------------------------------ dataset
def safe_image_path(name: str) -> Path:
    p = IMAGES_DIR / name
    if Path(name).name != name or p.suffix.lower() not in IMG_EXT or not p.exists():
        raise FileNotFoundError(name)
    return p


def label_path(name: str) -> Path:
    return LABELS_DIR / (Path(name).stem + ".txt")


def save_capture(frame: np.ndarray) -> str:
    name = f"frame_{int(time.time() * 1000)}.jpg"
    cv2.imwrite(str(IMAGES_DIR / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return name


def list_images(limit: int = 500) -> List[dict]:
    files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMG_EXT]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "url": f"/dataset-images/{p.name}", "labeled": label_path(p.name).exists()}
            for p in files[:limit]]


def delete_image(name: str):
    safe_image_path(name).unlink()
    label_path(name).unlink(missing_ok=True)


def save_labels(name: str, boxes: List[dict]):
    safe_image_path(name)
    lines = [f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}" for b in boxes]
    label_path(name).write_text("\n".join(lines))     # empty file = labelled "no defects"


def load_labels(name: str) -> List[dict]:
    p = label_path(name)
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            out.append({"cls": int(parts[0]), **dict(zip(("cx", "cy", "w", "h"), map(float, parts[1:])))})
    return out


def build_yolo_dataset(val_frac: float = 0.2) -> Path:
    """Copy labelled captures into dataset/yolo/{images,labels}/{train,val} and write data.yaml."""
    labeled = [p for p in IMAGES_DIR.iterdir()
               if p.suffix.lower() in IMG_EXT and label_path(p.name).exists()]
    if len(labeled) < 10:
        raise ValueError(f"Need at least 10 labelled images (have {len(labeled)})")
    shutil.rmtree(YOLO_DIR, ignore_errors=True)
    random.Random(0).shuffle(labeled)
    n_val = max(1, int(len(labeled) * val_frac))
    for i, img in enumerate(labeled):
        split = "val" if i < n_val else "train"
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, YOLO_DIR / "images" / split / img.name)
        shutil.copy(label_path(img.name), YOLO_DIR / "labels" / split / (img.stem + ".txt"))
    cfg = {"path": str(YOLO_DIR.resolve()), "train": "images/train", "val": "images/val",
           "names": {i: n for i, n in enumerate(CLASSES)}}
    out = YOLO_DIR / "data.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def import_zip(data: bytes, filename: str) -> dict:
    """Extract a YOLO-format zip (e.g. Roboflow export) and normalise its data.yaml."""
    stem = "".join(c for c in Path(filename).stem if c.isalnum() or c in "-_") or "upload"
    dest = UPLOADS_DIR / stem
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(BytesIO(data)) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if dest.resolve() not in target.parents and target != dest.resolve():
                raise ValueError("Unsafe path in zip")        # zip-slip protection
        z.extractall(dest)
    yamls = list(dest.rglob("data.yaml"))
    if not yamls:
        shutil.rmtree(dest, ignore_errors=True)
        raise ValueError("data.yaml not found in zip (expected a YOLO-format dataset)")
    ypath = yamls[0]
    cfg = yaml.safe_load(ypath.read_text())
    base = ypath.parent
    for key in ("train", "val", "test"):
        if key in cfg and cfg[key]:
            cand = [(base / cfg[key]).resolve(), (base / str(cfg[key]).lstrip("./").replace("../", "")).resolve()]
            found = next((c for c in cand if c.exists()), None)
            if found is None and key != "test":
                raise ValueError(f"'{key}' path from data.yaml not found: {cfg[key]}")
            if found:
                cfg[key] = str(found)
    cfg.pop("path", None)
    fixed = base / "data_fixed.yaml"
    fixed.write_text(yaml.safe_dump(cfg))
    n_img = sum(1 for p in dest.rglob("*") if p.suffix.lower() in IMG_EXT)
    return {"name": stem, "images": n_img, "classes": cfg.get("names"), "yaml": str(fixed)}


def list_uploads() -> List[str]:
    return sorted(p.name for p in UPLOADS_DIR.iterdir() if p.is_dir())


# ------------------------------------------------------------------ model versions
def list_models(active: str = "") -> List[dict]:
    out = []
    for d in sorted(MODELS_DIR.glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
        if (d / "best.pt").exists():
            meta = {}
            if (d / "meta.json").exists():
                meta = json.loads((d / "meta.json").read_text())
            out.append({"version": d.name, "active": d.name == active, **meta})
    return out


def next_version() -> str:
    nums = [int(d.name[1:]) for d in MODELS_DIR.glob("v*") if d.name[1:].isdigit()]
    return f"v{max(nums, default=0) + 1}"


# ------------------------------------------------------------------ training
class TrainingManager:
    def __init__(self):
        self.status: Dict = self._idle()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _idle() -> dict:
        return {"state": "idle", "epoch": 0, "epochs": 0, "message": "", "history": [],
                "metrics": None, "version": None}

    def start(self, params: dict) -> dict:
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Training already running")
        # Pre-validate so the HTTP caller gets an immediate, clear error.
        if params.get("data_source") == "labeled":
            labeled = [p for p in IMAGES_DIR.iterdir()
                       if p.suffix.lower() in IMG_EXT and label_path(p.name).exists()]
            if len(labeled) < 10:
                raise RuntimeError(
                    f"Need at least 10 labelled images to train (have {len(labeled)}). "
                    "Capture frames, label them, then retry.")
        self.status = {**self._idle(), "state": "preparing", "epochs": params["epochs"],
                       "message": "Preparing dataset..."}
        self._thread = threading.Thread(target=self._run, args=(params,), daemon=True)
        self._thread.start()
        return self.status

    def _run(self, p: dict):
        try:
            from ultralytics import YOLO
        except Exception as e:  # noqa: BLE001
            self.status.update(state="error", message=f"ultralytics not installed: {e}")
            return
        try:
            if p["data_source"] == "labeled":
                data = build_yolo_dataset()
            else:
                data = UPLOADS_DIR / p["data_source"]
                data = next(iter(data.rglob("data_fixed.yaml")))
            version = next_version()
            model = YOLO(p["base_model"] + ".pt")          # downloads pretrained weights on first use

            def on_epoch_end(tr):
                try:
                    m = {k: float(v) for k, v in tr.metrics.items()}
                    self.status["history"].append({
                        "epoch": tr.epoch + 1,
                        "box_loss": m.get("val/box_loss"), "cls_loss": m.get("val/cls_loss"),
                        "map50": m.get("metrics/mAP50(B)"),
                    })
                    self.status.update(epoch=tr.epoch + 1, message=f"Epoch {tr.epoch + 1}/{p['epochs']}")
                except Exception:  # noqa: BLE001
                    log.exception("progress callback failed")

            model.add_callback("on_fit_epoch_end", on_epoch_end)
            self.status.update(state="training", message="Training started...")
            results = model.train(data=str(data), epochs=p["epochs"], imgsz=p["imgsz"], batch=p["batch"],
                                  project=str((MODELS_DIR / "runs").resolve()), name=version,
                                  exist_ok=True, workers=0, verbose=False)
            best = Path(model.trainer.save_dir) / "weights" / "best.pt"
            vdir = MODELS_DIR / version
            vdir.mkdir(exist_ok=True)
            shutil.copy(best, vdir / "best.pt")
            rd = getattr(results, "results_dict", {}) or {}
            metrics = {"precision": rd.get("metrics/precision(B)"), "recall": rd.get("metrics/recall(B)"),
                       "map50": rd.get("metrics/mAP50(B)")}
            meta = {"trained_at": datetime.now().isoformat(timespec="seconds"), "base_model": p["base_model"],
                    "epochs": p["epochs"], "metrics": metrics}
            (vdir / "meta.json").write_text(json.dumps(meta, indent=2))
            cm = Path(model.trainer.save_dir) / "confusion_matrix.png"
            if cm.exists():
                shutil.copy(cm, vdir / "confusion_matrix.png")
            self.status.update(state="done", version=version, metrics=metrics,
                               message=f"Done. Saved as {version}")
        except Exception as e:  # noqa: BLE001
            log.exception("training failed")
            self.status.update(state="error", message=str(e))
