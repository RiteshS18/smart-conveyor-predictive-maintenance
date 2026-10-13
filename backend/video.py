"""Stream reader + vision processing threads + MJPEG frames.

Two threads:
  reader    - pulls frames from the URL as fast as possible, keeps ONLY the newest one
              (no latency build-up) and auto-reconnects when the stream drops.
  processor - runs belt analysis (+ YOLO every Nth frame), draws the overlay and
              publishes the annotated JPEG and a result dict for the monitor.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .config import SettingsStore
from .vision.belt import analyze_belt, roi_box
from .vision.detector import Detector

log = logging.getLogger("video")

CLASS_COLORS = {
    "crack": (0, 60, 255),          # High-contrast red-orange
    "worn_out": (0, 200, 255),       # Vibrant golden amber
    "tear": (0, 0, 255),
    "joint_damage": (255, 0, 160),
    "dust_spillage": (0, 220, 255),
    "edge_fraying": (255, 0, 255),
}


def make_placeholder(text: str, w: int = 640, h: int = 360) -> bytes:
    img = np.full((h, w, 3), 28, np.uint8)
    cv2.putText(img, text, (24, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def draw_overlay(frame, belt: dict, dets: list, s, demo: bool) -> np.ndarray:
    out = frame.copy()
    x0, y0, x1, y1 = belt["roi_box"]
    if s.roi:
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 200, 0), 1)
    if belt["belt_found"]:
        for v in (belt["left"], belt["right"]):
            if s.belt_orientation == "horizontal":
                cv2.line(out, (x0, int(v)), (x1, int(v)), (0, 255, 255), 2)
            else:
                cv2.line(out, (int(v), y0), (int(v), y1), (0, 255, 255), 2)
    for d in dets:
        bx = [int(v) for v in d["box"]]
        col = CLASS_COLORS.get(d["cls"], (0, 255, 0))
        # 1. Main bounding box
        cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), col, 2)
        # 2. Tech corner targeting accents
        clen = max(6, min(16, (bx[2] - bx[0]) // 5, (bx[3] - bx[1]) // 5))
        cv2.line(out, (bx[0], bx[1]), (bx[0] + clen, bx[1]), col, 3)
        cv2.line(out, (bx[0], bx[1]), (bx[0], bx[1] + clen), col, 3)
        cv2.line(out, (bx[2], bx[1]), (bx[2] - clen, bx[1]), col, 3)
        cv2.line(out, (bx[2], bx[1]), (bx[2], bx[1] + clen), col, 3)
        cv2.line(out, (bx[0], bx[3]), (bx[0] + clen, bx[3]), col, 3)
        cv2.line(out, (bx[0], bx[3]), (bx[0], bx[3] - clen), col, 3)
        cv2.line(out, (bx[2], bx[3]), (bx[2] - clen, bx[3]), col, 3)
        cv2.line(out, (bx[2], bx[3]), (bx[2] - clen, bx[3]), col, 3)

        # 3. Computer Vision filled label badge
        label = f"{d['cls'].replace('_', ' ').upper()} {int(d['conf'] * 100)}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        tag_h = th + 6
        if bx[1] - tag_h >= 0:
            cv2.rectangle(out, (bx[0], bx[1] - tag_h), (bx[0] + tw + 6, bx[1]), col, cv2.FILLED)
            cv2.putText(out, label, (bx[0] + 3, bx[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (bx[0], bx[1]), (bx[0] + tw + 6, bx[1] + tag_h), col, cv2.FILLED)
            cv2.putText(out, label, (bx[0] + 3, bx[1] + tag_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    off = belt["offset_pct"]
    bad = belt["belt_found"] and abs(off) >= s.misalign_warn_pct
    hud = f"offset {off:+.1f}%   dust {belt['dust_pct']:.0f}%   blur {belt['blur']:.0f}"
    if dets:
        hud += f"   defects {len(dets)}"
    cv2.putText(out, hud, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if bad else (80, 255, 80), 2, cv2.LINE_AA)
    return out


class VideoService:
    def __init__(self, store: SettingsStore, detector: Detector):
        self.store, self.detector = store, detector
        self._frame: Optional[np.ndarray] = None
        self._frame_id = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list = []
        self._url = ""
        self.connected = False
        self.last_error = ""
        self.last_frame_ts = 0.0
        self.fps = 0.0
        self.latest_jpeg: Optional[bytes] = None
        self.latest_result: Optional[dict] = None
        self.running = False

    # ---------- lifecycle ----------
    def start(self, url: str):
        self.stop()
        url = url.strip()
        # Auto-fix bare IP:port URLs → append /video (IP Webcam app MJPEG endpoint)
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https") and (parsed.path == "" or parsed.path == "/"):
            url = url.rstrip("/") + "/video"
        self._url = url
        self._stop = threading.Event()
        self.running = True
        self.last_error = ""
        self._threads = [
            threading.Thread(target=self._reader, args=(self._stop, self._url), daemon=True),
            threading.Thread(target=self._processor, args=(self._stop,), daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3)
        self._threads = []
        self.running = False
        self.connected = False
        self.last_error = ""
        with self._lock:
            self._frame = None
        self.latest_jpeg = None
        self.latest_result = None
        self.fps = 0.0

    def status(self) -> dict:
        h, w = (self._frame.shape[:2] if self._frame is not None else (0, 0))
        return {
            "running": self.running, "connected": self.connected, "url": self._url,
            "fps": round(self.fps, 1), "last_error": self.last_error,
            "stalled": self.running and time.time() - self.last_frame_ts > 3,
            "demo_mode": self.detector.demo_mode, "model": self.detector.weights,
            "frame_size": [w, h],
        }

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    # ---------- threads ----------
    def _reader(self, stop: threading.Event, url: str):
        is_file = Path(url).exists()

        # Use urllib for http/https streams (avoids FFmpeg TCP issues with IP Webcam)
        if not is_file and url.startswith(("http://", "https://")):
            self._reader_http(stop, url)
            return

        # Fallback: OpenCV VideoCapture (files, RTSP, numeric device)
        src = int(url) if url.isdigit() else url
        while not stop.is_set():
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                self.connected = False
                self.last_error = "Cannot open stream - check URL / same Wi-Fi / app running"
                stop.wait(2.0)
                continue
            self.connected, self.last_error = True, ""
            delay = 1.0 / max(1.0, cap.get(cv2.CAP_PROP_FPS) or 25.0) if is_file else 0.0
            fails = 0
            while not stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    fails += 1
                    if is_file and fails <= 3:       # loop video files for testing
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    if fails > 30:
                        break
                    time.sleep(0.03)
                    continue
                fails = 0
                with self._lock:
                    self._frame, self._frame_id = frame, self._frame_id + 1
                self.last_frame_ts = time.time()
                if delay:
                    time.sleep(delay)
            cap.release()
            self.connected = False
            self.last_error = "Stream lost - reconnecting..."
            stop.wait(1.0)

    def _reader_http(self, stop: threading.Event, url: str):
        """Read MJPEG stream over HTTP using urllib (works where OpenCV/FFmpeg fails)."""
        import urllib.request
        BOUNDARY_CHUNK = 65536
        while not stop.is_set():
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urllib.request.urlopen(req, timeout=5)
                self.connected, self.last_error = True, ""
                buf = b""
                while not stop.is_set():
                    chunk = resp.read(BOUNDARY_CHUNK)
                    if not chunk:
                        break
                    buf += chunk
                    # Parse MJPEG: find JPEG boundaries
                    while True:
                        start = buf.find(b"\xff\xd8")
                        end = buf.find(b"\xff\xd9", start + 2) if start != -1 else -1
                        if start == -1 or end == -1:
                            # Keep last partial frame in buffer, discard leading garbage
                            if start != -1:
                                buf = buf[start:]
                            elif len(buf) > BOUNDARY_CHUNK * 2:
                                buf = b""
                            break
                        jpg = buf[start: end + 2]
                        buf = buf[end + 2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                self._frame, self._frame_id = frame, self._frame_id + 1
                            self.last_frame_ts = time.time()
                resp.close()
            except Exception as exc:
                self.connected = False
                self.last_error = f"Stream error: {exc}"
                log.warning("HTTP reader error: %s", exc)
            if not stop.is_set():
                stop.wait(2.0)

    def _processor(self, stop: threading.Event):
        last_id, count, last_dets = -1, 0, []
        t_win, n_win = time.time(), 0
        while not stop.is_set():
            t0 = time.time()
            with self._lock:
                frame, fid = self._frame, self._frame_id
            if frame is None or fid == last_id:
                stop.wait(0.01)
                continue
            last_id = fid
            try:
                s = self.store.get()
                if frame.shape[1] > s.resize_width:
                    sc = s.resize_width / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                belt = analyze_belt(frame, s.belt_orientation, s.roi)
                count += 1
                if count % s.detect_every_n == 0:
                    last_dets = self.detector.detect(frame, s.conf_threshold)
                annotated = draw_overlay(frame, belt, last_dets, s, self.detector.demo_mode)
                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self.latest_jpeg = buf.tobytes()
                self.latest_result = {
                    "ts": time.time(), "belt": belt, "detections": list(last_dets),
                    "n_defects": len(last_dets),
                    "max_conf": max((d["conf"] for d in last_dets), default=0.0),
                    "blur_ok": belt["blur"] >= s.blur_min,
                }
            except Exception:  # noqa: BLE001 - never let the vision thread die
                log.exception("vision processing error")
            n_win += 1
            if time.time() - t_win >= 2:
                self.fps, t_win, n_win = n_win / (time.time() - t_win), time.time(), 0
            stop.wait(max(0.0, 1 / 15 - (time.time() - t0)))   # cap at ~15 fps
