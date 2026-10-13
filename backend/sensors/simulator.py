"""Sensor simulator. `generate_reading` is pure so the predictor can reuse it to
create synthetic run-to-failure data."""
from __future__ import annotations

import math
import random


def generate_reading(t_seconds: float, fault: float, rng: random.Random) -> dict:
    """Healthy baseline + daily-like cycle + noise; `fault` (0..1) drifts values toward failure."""
    daily = math.sin(2 * math.pi * t_seconds / 900.0)
    return {
        "temperature": 46 + 4 * daily + rng.gauss(0, 0.4) + 38 * fault ** 1.2,
        "vibration": 2.0 + 0.3 * daily + abs(rng.gauss(0, 0.15)) + 6.5 * fault ** 1.5,
        "load": 420 + 50 * daily + rng.gauss(0, 8) + 60 * fault,
        "speed": 2.4 + rng.gauss(0, 0.02) - 0.3 * fault,
        "tension": 60 + 2 * daily + rng.gauss(0, 0.5) + 18 * fault,
    }


def generate_vision_features(fault: float, rng: random.Random) -> dict:
    return {
        "misalign_pct": abs(rng.gauss(1.5, 0.5)) + 10 * fault ** 2,
        "dust_pct": max(0.0, rng.gauss(3, 1) + 8 * fault),
        "n_defects": int(max(0, round(rng.gauss(4 * fault ** 2, 0.5)))),
    }


class Simulator:
    def __init__(self):
        self.t = 0
        self.fault = 0.0
        self.inject = False
        self.rng = random.Random()

    def step(self, ramp_s: int) -> dict:
        self.t += 1
        if self.inject:
            self.fault = min(1.0, self.fault + 1.0 / max(1, ramp_s))
        else:
            self.fault = max(0.0, self.fault - 1.0 / 60.0)
        return generate_reading(self.t, self.fault, self.rng)
