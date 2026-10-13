"""Alert engine: turns the current vision/sensor/health state into de-duplicated alerts."""
from __future__ import annotations

import time
from collections import deque
from typing import List, Optional

import numpy as np

from .config import SNAP_DIR, Settings
from .db import Alert, session_scope

SENSOR_RULES = [  # key, label, unit, warn attr, crit attr
    ("vibration", "Vibration", "m/s²", "vib_warn", "vib_crit"),
    ("load", "Load", "kg", "load_warn", "load_crit"),
]
COOLDOWN = {"critical": 30, "warning": 60, "info": 120}   # seconds per alert key
RANK = {"critical": 0, "warning": 1, "info": 2}
MAX_PER_MINUTE = 12


class AlertEngine:
    def __init__(self):
        self._last: dict = {}
        self._emitted: deque = deque()
        self._vib: deque = deque(maxlen=60)   # last ~60 s of vibration for trend

    def push_vibration(self, v: Optional[float]):
        if v is not None:
            self._vib.append(v)

    def evaluate(self, vision: Optional[dict], sensors: Optional[dict], health: dict,
                 risk: Optional[dict], s: Settings) -> List[dict]:
        c: List[dict] = []

        def add(key, sev, kind, msg):
            c.append({"key": key, "severity": sev, "kind": kind, "message": msg})

        if vision:
            for d in vision["detections"]:
                sev = "critical" if d["cls"] in ("tear", "joint_damage") else "warning"
                add(f"defect:{d['cls']}", sev, "defect",
                    f"{d['cls'].replace('_', ' ').title()} detected (confidence {d['conf']:.0%})")
            belt = vision["belt"]
            if belt["belt_found"]:
                pct = belt["offset_pct"]
                side = "right" if pct > 0 else "left"
                if abs(pct) >= s.misalign_crit_pct:
                    add("misalign", "critical", "misalignment", f"Belt drifting {side} {abs(pct):.0f}% - critical")
                elif abs(pct) >= s.misalign_warn_pct:
                    add("misalign", "warning", "misalignment", f"Belt drifting {side} {abs(pct):.0f}%")
            dust = belt["dust_pct"]
            if dust >= s.dust_crit_pct:
                add("dust", "critical", "dust", f"Heavy dust/spillage on belt ({dust:.0f}%)")
            elif dust >= s.dust_warn_pct:
                add("dust", "warning", "dust", f"Dust/spillage building up ({dust:.0f}%)")
            if not vision["blur_ok"]:
                add("blur", "info", "image_quality", "Image blur high - detection reliability reduced")

        if sensors:
            for key, label, unit, wa, ca in SENSOR_RULES:
                v = sensors.get(key)
                if v is None:
                    continue
                if v >= getattr(s, ca):
                    add(f"sensor:{key}", "critical", key, f"{label} critical: {v:.1f} {unit}")
                elif v >= getattr(s, wa):
                    add(f"sensor:{key}", "warning", key, f"{label} above {getattr(s, wa):g} {unit} ({v:.1f})")
            if len(self._vib) >= 30:
                slope = float(np.polyfit(np.arange(len(self._vib)), list(self._vib), 1)[0])
                rise = slope * len(self._vib)
                if rise > 0.25 * s.vib_warn and self._vib[-1] > 0.5 * s.vib_warn:
                    add("vib_trend", "warning", "vibration", "Vibration trend rising")

        if health["status"] == "red":
            add("health_red", "critical", "health", f"Belt health critical ({health['score']:.0f}/100)")
        if risk and (risk.get("risk_probability") or 0) >= 0.7:
            add("risk_high", "critical", "prediction",
                f"High failure risk: {risk['risk_probability']:.0%} within {risk.get('horizon_min', '?')} min")
        return c

    def emit(self, candidates: List[dict], jpeg: Optional[bytes]) -> List[dict]:
        now = time.time()
        while self._emitted and now - self._emitted[0] > 60:
            self._emitted.popleft()
        out = []
        for c in sorted(candidates, key=lambda x: RANK[x["severity"]]):
            if len(self._emitted) >= MAX_PER_MINUTE:
                break
            if now - self._last.get(c["key"], 0) < COOLDOWN[c["severity"]]:
                continue
            self._last[c["key"]] = now
            self._emitted.append(now)
            snap = None
            if jpeg:
                snap = f"alert_{int(now * 1000)}_{c['kind']}.jpg"
                (SNAP_DIR / snap).write_bytes(jpeg)
            with session_scope() as db:
                row = Alert(ts=now, severity=c["severity"], kind=c["kind"],
                            message=c["message"], snapshot=snap, acknowledged=False)
                db.add(row)
                db.flush()
                rid = row.id
            out.append({"id": rid, "ts": now, "severity": c["severity"], "kind": c["kind"],
                        "message": c["message"], "snapshot": f"/snapshots/{snap}" if snap else None,
                        "acknowledged": False})
        return out
