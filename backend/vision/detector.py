"""YOLO and Computer Vision defect detector wrapper with hot-swappable weights.

Detects cracks, worn out surface areas, and structural belt defects.
When custom YOLO weights are loaded, it uses the deep learning model.
When no custom weights are loaded, it uses high-precision morphological and
gradient texture computer vision heuristics with temporal box smoothing.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger("detector")


class Detector:
    def __init__(self):
        self.model = None
        self.weights: Optional[str] = None
        self.error: Optional[str] = None
        self._lock = threading.Lock()
        self._tracked_boxes: List[dict] = []

    @property
    def demo_mode(self) -> bool:
        return self.model is None

    def load(self, path: Path) -> bool:
        """Load custom weights. Returns False (and sets .error) on failure; never raises."""
        try:
            from ultralytics import YOLO  # imported lazily: heavy, and optional for demo mode
        except Exception as e:  # noqa: BLE001
            self.error = f"ultralytics not available: {e}"
            log.warning(self.error)
            return False
        try:
            model = YOLO(str(path))
        except Exception as e:  # noqa: BLE001
            self.error = f"could not load {path}: {e}"
            log.warning(self.error)
            return False
        with self._lock:
            self.model, self.weights, self.error = model, str(path), None
        log.info("Loaded detector weights %s", path)
        return True

    def unload(self):
        with self._lock:
            self.model, self.weights = None, None

    @staticmethod
    def _box_iou(b1: list, b2: list) -> float:
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def _detect_cv_heuristics(self, frame: np.ndarray, conf_thresh: float) -> List[dict]:
        """Classical Computer Vision defect detection for cracks and worn-out areas."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Belt detection margins
        mx, my = int(w * 0.05), int(h * 0.05)

        # 1. Crack detection using morphological Black-Hat filtering
        k_crack = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k_crack)
        bh_blur = cv2.GaussianBlur(blackhat, (3, 3), 0)
        _, crack_bin = cv2.threshold(bh_blur, 20, 255, cv2.THRESH_BINARY)
        crack_bin[:, :mx] = 0
        crack_bin[:, w - mx:] = 0
        crack_bin[:my, :] = 0
        crack_bin[h - my:, :] = 0

        c_cnts, _ = cv2.findContours(crack_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        crack_candidates = []
        for c in c_cnts:
            area = cv2.contourArea(c)
            if area < 50 or area > (w * h * 0.15):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            max_d, min_d = max(bw, bh), max(1, min(bw, bh))
            aspect = max_d / min_d
            if aspect >= 1.7 or max_d >= 35:
                score = min(0.95, 0.65 + min(0.20, aspect * 0.035) + min(0.10, area / 500.0))
                bx = [float(max(0, x - 3)), float(max(0, y - 3)), float(min(w, x + bw + 3)), float(min(h, y + bh + 3))]
                crack_candidates.append({
                    "cls": "crack", "conf": round(score, 2), "box": bx,
                    "area_pct": round((bw * bh) / float(w * h) * 100.0, 2),
                    "priority": score * np.log1p(area)
                })
        crack_candidates.sort(key=lambda x: x["priority"], reverse=True)
        raw_dets = crack_candidates[:2]

        # 2. Worn-out surface detection using gradient texture analysis
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.clip(cv2.magnitude(sobelx, sobely), 0, 255).astype(np.uint8)
        k_wear = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
        grad_closed = cv2.morphologyEx(grad_mag, cv2.MORPH_CLOSE, k_wear)
        _, wear_bin = cv2.threshold(grad_closed, 45, 255, cv2.THRESH_BINARY)
        wear_bin[:, :mx] = 0
        wear_bin[:, w - mx:] = 0
        wear_bin[:my, :] = 0
        wear_bin[h - my:, :] = 0

        w_cnts, _ = cv2.findContours(wear_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wear_cands = []
        for c in w_cnts:
            area = cv2.contourArea(c)
            if area < 250 or area > (w * h * 0.20):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            bx = [float(max(0, x - 4)), float(max(0, y - 4)), float(min(w, x + bw + 4)), float(min(h, y + bh + 4))]

            # Avoid heavy overlap with crack boxes
            overlap = False
            for r in raw_dets:
                if self._box_iou(bx, r["box"]) > 0.35:
                    overlap = True
                    break
            if overlap:
                continue
            score = min(0.92, 0.62 + min(0.25, area / 2000.0))
            wear_cands.append({
                "cls": "worn_out", "conf": round(score, 2), "box": bx,
                "area_pct": round((bw * bh) / float(w * h) * 100.0, 2),
                "priority": score * np.log1p(area)
            })
        wear_cands.sort(key=lambda x: x["priority"], reverse=True)
        if wear_cands:
            raw_dets.append(wear_cands[0])

        # Temporal box smoothing across frames
        smoothed = []
        for r in raw_dets:
            matched_track = None
            for tr in self._tracked_boxes:
                if tr["cls"] == r["cls"] and self._box_iou(tr["box"], r["box"]) > 0.25:
                    matched_track = tr
                    break
            if matched_track:
                sb = [
                    0.65 * matched_track["box"][i] + 0.35 * r["box"][i]
                    for i in range(4)
                ]
                sc = round(0.70 * matched_track["conf"] + 0.30 * r["conf"], 2)
                smoothed.append({"cls": r["cls"], "conf": sc, "box": sb, "area_pct": r["area_pct"]})
            else:
                smoothed.append(r)

        self._tracked_boxes = smoothed
        return [
            {"cls": d["cls"], "conf": d["conf"], "box": [round(v, 1) for v in d["box"]], "area_pct": d["area_pct"]}
            for d in smoothed if d["conf"] >= conf_thresh
        ]

    def detect(self, frame: np.ndarray, conf: float) -> List[dict]:
        with self._lock:
            model = self.model
            if model is None:
                return self._detect_cv_heuristics(frame, conf)
            res = model(frame, conf=conf, verbose=False)[0]
            names = res.names
        h, w = frame.shape[:2]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append({
                "cls": str(names[int(b.cls[0])]),
                "conf": float(b.conf[0]),
                "box": [x1, y1, x2, y2],
                "area_pct": (x2 - x1) * (y2 - y1) / float(w * h) * 100.0,
            })
        return out

