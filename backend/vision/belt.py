"""Classical computer vision for the belt: edge/misalignment, dust heuristic, blur check.

ASSUMPTIONS (documented so the next developer can improve them):
  * Belt boundaries are the two strongest straight edges across the frame
    (vertical belt = left/right edges; horizontal belt = top/bottom edges).
  * Dust % is "pixels that differ strongly from the belt's median colour". It is a
    fallback heuristic; a trained `dust_spillage` YOLO class should replace it.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import cv2
import numpy as np


def roi_box(shape, roi: Optional[Sequence[float]]) -> Tuple[int, int, int, int]:
    """Convert normalised [x, y, w, h] to pixel (x0, y0, x1, y1), clamped to the frame."""
    h, w = shape[:2]
    if not roi or len(roi) != 4:
        return 0, 0, w, h
    x, y, rw, rh = roi
    x0, y0 = int(max(0, x) * w), int(max(0, y) * h)
    x1, y1 = int(min(1, x + rw) * w), int(min(1, y + rh) * h)
    if x1 - x0 < 32 or y1 - y0 < 32:
        return 0, 0, w, h
    return x0, y0, x1, y1


def _dust_pct(belt_bgr: np.ndarray) -> float:
    if belt_bgr.size == 0:
        return 0.0
    small = cv2.resize(belt_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = np.median(lab.reshape(-1, 3), axis=0)
    delta = np.linalg.norm(lab - ref, axis=2)
    return float((delta > 30).mean() * 100.0)


def analyze_belt(frame: np.ndarray, orientation: str = "vertical",
                 roi: Optional[Sequence[float]] = None) -> dict:
    x0, y0, x1, y1 = roi_box(frame.shape, roi)
    crop = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    result = {
        "belt_found": False, "roi_box": (x0, y0, x1, y1), "left": None, "right": None,
        "offset_px": 0.0, "offset_pct": 0.0, "dust_pct": 0.0, "blur": blur,
    }

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    if orientation == "horizontal":
        edges = edges.T  # columns now correspond to original rows
    strength = edges.sum(axis=0).astype(np.float32) / 255.0
    strength = cv2.GaussianBlur(strength.reshape(1, -1), (0, 0), 4).ravel()
    n = strength.size
    if n < 20 or strength.max() < 3:
        result["dust_pct"] = _dust_pct(crop)
        return result

    half = n // 2
    a = int(np.argmax(strength[:half]))
    b = half + int(np.argmax(strength[half:]))
    if b - a < n * 0.2:                      # implausibly narrow belt
        result["dust_pct"] = _dust_pct(crop)
        return result

    offset_px = (a + b) / 2.0 - n / 2.0
    belt_region = crop[:, a:b] if orientation != "horizontal" else crop[a:b, :]
    origin = x0 if orientation != "horizontal" else y0
    result.update({
        "belt_found": True,
        "left": origin + a, "right": origin + b,      # in frame coordinates
        "offset_px": float(offset_px),
        "offset_pct": float(offset_px / n * 100.0),
        "dust_pct": _dust_pct(belt_region),
    })
    return result
