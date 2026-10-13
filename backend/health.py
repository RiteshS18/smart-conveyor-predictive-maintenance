"""Belt Health Score (0-100): weighted penalties from vision + sensors."""
from __future__ import annotations

from typing import Optional

from .config import SEVERITY, Settings


def ramp(v: Optional[float], warn: float, crit: float) -> float:
    """0 well below warn, 0.5 at warn, 1.0 at/after crit."""
    if v is None:
        return 0.0
    if crit <= warn:
        return 1.0 if v >= warn else 0.0
    lo = warn * 0.85
    if v <= lo:
        return 0.0
    if v < warn:
        return 0.5 * (v - lo) / (warn - lo)
    if v < crit:
        return 0.5 + 0.5 * (v - warn) / (crit - warn)
    return 1.0


WORST_CAP = 55.0   # component at critical (penalty 1.0) caps score at 45 = red; at warn (0.5) -> 72.5 = yellow


def compute_health(vision: Optional[dict], sensors: Optional[dict], s: Settings) -> dict:
    """Score = 100 - weighted penalty over the components that currently HAVE data
    (weights are re-normalised, so a missing camera/sensor doesn't hide a critical reading),
    additionally capped by the single worst component."""
    p: dict = {}                                    # only components with live data
    if vision:
        sev = sum(d["conf"] * SEVERITY.get(d["cls"], 0.6) * (1 + min(d["area_pct"] / 5.0, 1.0))
                  for d in vision["detections"])
        p["defects"] = min(1.0, sev / 2.0)          # count x severity x size
        belt = vision["belt"]
        if belt["belt_found"]:
            p["misalignment"] = ramp(abs(belt["offset_pct"]), s.misalign_warn_pct, s.misalign_crit_pct)
        p["dust"] = ramp(belt["dust_pct"], s.dust_warn_pct, s.dust_crit_pct)
    if sensors:
        for key, warn, crit in (("vibration", s.vib_warn, s.vib_crit),
                                ("load", s.load_warn, s.load_crit)):
            if sensors.get(key) is not None:
                p[key] = ramp(sensors[key], warn, crit)

    avail_w = sum(s.weights.get(k, 0.0) for k in p) or 1.0
    penalty = sum(s.weights.get(k, 0.0) * v for k, v in p.items()) / avail_w * 100.0
    score = 100.0 - penalty
    if p:
        score = min(score, 100.0 - WORST_CAP * max(p.values()))
    score = max(0.0, score)
    status = "green" if score >= 75 else "yellow" if score >= 50 else "red"
    return {
        "score": round(score, 1), "status": status,
        "components": {k: {"penalty": round(p.get(k, 0.0), 3), "weight": s.weights.get(k, 0.0),
                           "active": k in p} for k in s.weights},
        "sources": {"vision": bool(vision), "sensors": bool(sensors)},
    }


def compute_lifespan(health_score: float, sensors: Optional[dict] = None, vision: Optional[dict] = None) -> dict:
    """Computes estimated conveyor belt operational lifespan in days, months, and years
    based on nominal design life (3.4 years / ~1250 operational days) adjusted by
    health score, load stress, vibration, and detected defect severity."""
    base_days = 1250.0
    h = max(1.0, min(100.0, float(health_score)))
    health_factor = (h / 100.0) ** 1.8

    load = float(sensors.get("load", 450.0)) if sensors and sensors.get("load") is not None else 450.0
    load_factor = max(0.6, min(2.5, (load / 450.0) ** 1.2))

    vib = float(sensors.get("vibration", 2.0)) if sensors and sensors.get("vibration") is not None else 2.0
    vib_factor = max(0.7, min(2.5, (vib / 2.0) ** 0.7))

    n_defects = len(vision.get("detections", [])) if vision else 0
    if n_defects > 0 and vision:
        defect_impact = sum(d.get("conf", 0.7) * SEVERITY.get(d.get("cls"), 0.5) * 0.3 for d in vision.get("detections", []))
        defect_factor = 1.0 / (1.0 + defect_impact)
    else:
        defect_factor = 1.0

    stress_multiplier = max(0.5, load_factor * 0.75 + vib_factor * 0.25)
    days = round(max(3.0, (base_days * health_factor * defect_factor) / stress_multiplier))

    years = round(days / 365.25, 1)
    months = round(days / 30.44, 1)

    y_int = int(days // 365.25)
    m_int = int((days % 365.25) // 30.44)
    d_int = int(days % 30.44)

    if y_int > 0:
        main_text = f"{y_int} Year{'s' if y_int > 1 else ''}, {m_int} Month{'s' if m_int != 1 else ''}"
    elif m_int > 0:
        main_text = f"{m_int} Month{'s' if m_int > 1 else ''}, {d_int} Day{'s' if d_int != 1 else ''}"
    else:
        main_text = f"{int(days)} Days"

    condition = "Optimal Condition" if h >= 80 else "Moderate Wear" if h >= 50 else "High Degradation" if h >= 25 else "Critical Replacement"

    return {
        "days": int(days),
        "months": months,
        "years": years,
        "text": main_text,
        "condition": condition,
        "load_applied": round(load, 1),
    }
