"""Predictive maintenance.

* Isolation Forest   - learns the "normal" baseline from the first BASELINE_N healthy
                       live feature rows, then flags anomalies.
* RandomForest clf   - P(failure within HORIZON) ; RandomForest reg - remaining useful life.
                       BOOTSTRAPPED ON SYNTHETIC run-to-failure data (see simulator.py).
                       Replace with real labelled run-to-failure logs when you have them.

Time base: one feature row = STEP_S seconds (the monitor writes one every 5 s).
"""
from __future__ import annotations

import logging
import random
import threading
from collections import deque
from datetime import datetime
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from ..config import MODELS_DIR
from ..sensors.simulator import generate_reading, generate_vision_features

log = logging.getLogger("predictor")

STEP_S = 5
WINDOW = 12                  # rows used for slope features (= 1 minute)
HORIZON_STEPS = 60           # "failure within next 5 minutes" (demo time scale)
RUL_CAP = 240                # steps
BASELINE_N = 120             # rows (~10 min) of normal data before anomaly detection starts
FEATURES = ["temperature", "vibration", "load", "tension", "misalign_pct", "dust_pct",
            "n_defects", "temp_slope", "vib_slope"]
NOMINAL = {"temperature": 46.0, "vibration": 2.0, "load": 420.0, "tension": 60.0,
           "misalign_pct": 1.5, "dust_pct": 3.0, "n_defects": 0.0}
MODEL_PATH = MODELS_DIR / "predictor.joblib"


def _slope(vals) -> float:
    """Linear trend per MINUTE."""
    if len(vals) < 3:
        return 0.0
    return float(np.polyfit(np.arange(len(vals)), vals, 1)[0]) * (60.0 / STEP_S)


def _vector(row: dict, temp_slope: float, vib_slope: float):
    return [row["temperature"], row["vibration"], row["load"], row["tension"],
            row["misalign_pct"], row["dust_pct"], row["n_defects"], temp_slope, vib_slope]


def _build_synthetic(n_runs: int, seed: int):
    rng = random.Random(seed)
    X, y_clf, y_rul, groups = [], [], [], []
    for run in range(n_runs):
        faulty = rng.random() < 0.7
        if faulty:
            start = rng.randint(60, 200)
            ramp = rng.randint(30, 200)
            length = start + ramp            # run ends at failure (fault reaches 1.0)
        else:
            start, ramp, length = 10 ** 9, 1, rng.randint(250, 450)
        win_t, win_v = deque(maxlen=WINDOW), deque(maxlen=WINDOW)
        for t in range(length):
            fault = min(1.0, max(0.0, (t - start) / ramp)) if faulty else 0.0
            row = {**generate_reading(t * STEP_S, fault, rng), **generate_vision_features(fault, rng)}
            win_t.append(row["temperature"])
            win_v.append(row["vibration"])
            X.append(_vector(row, _slope(win_t), _slope(win_v)))
            steps_left = (length - t) if faulty else None
            y_clf.append(int(faulty and steps_left <= HORIZON_STEPS))
            y_rul.append(min(steps_left, RUL_CAP) if faulty else RUL_CAP)
            groups.append(run)
    return np.array(X), np.array(y_clf), np.array(y_rul, dtype=float), np.array(groups)


class Predictor:
    def __init__(self):
        self.clf = None
        self.reg = None
        self.iso: Optional[IsolationForest] = None
        self.meta: dict = {}
        self.window: deque = deque(maxlen=WINDOW)
        self.normal_rows: list = []
        self.last: dict = {"ready": False}
        self.training = False
        self._lock = threading.Lock()

    # ---------- training ----------
    def load_or_bootstrap(self):
        try:
            blob = joblib.load(MODEL_PATH)
            with self._lock:
                self.clf, self.reg, self.meta = blob["clf"], blob["reg"], blob["meta"]
            log.info("Loaded predictor from %s", MODEL_PATH)
        except Exception:  # noqa: BLE001
            self.train_synthetic()

    def train_synthetic(self, n_runs: int = 80, seed: int = 42):
        self.training = True
        try:
            X, yc, yr, g = _build_synthetic(n_runs, seed)
            train, test = g % 5 != 0, g % 5 == 0          # split by RUN, not by row (no leakage)
            clf = RandomForestClassifier(n_estimators=80, max_depth=12, min_samples_leaf=3,
                                         class_weight="balanced", n_jobs=-1, random_state=seed)
            reg = RandomForestRegressor(n_estimators=60, max_depth=12, min_samples_leaf=3,
                                        n_jobs=-1, random_state=seed)
            clf.fit(X[train], yc[train])
            reg.fit(X[train], yr[train])
            meta = {
                "trained_at": datetime.now().isoformat(timespec="seconds"),
                "source": "synthetic", "n_rows": int(len(X)),
                "accuracy": round(float(accuracy_score(yc[test], clf.predict(X[test]))), 3),
                "f1": round(float(f1_score(yc[test], clf.predict(X[test]))), 3),
                "rul_mae_min": round(float(mean_absolute_error(yr[test], reg.predict(X[test]))) * STEP_S / 60, 2),
                "horizon_min": HORIZON_STEPS * STEP_S / 60,
                "features": FEATURES,
            }
            with self._lock:
                self.clf, self.reg, self.meta = clf, reg, meta
            joblib.dump({"clf": clf, "reg": reg, "meta": meta}, MODEL_PATH)
            log.info("Predictor trained on synthetic data: %s", meta)
        finally:
            self.training = False

    def retrain(self):
        """Re-train the risk model and restart baseline learning for anomaly detection."""
        with self._lock:
            self.iso, self.normal_rows = None, []
        self.train_synthetic(seed=random.randint(0, 10_000))

    # ---------- inference ----------
    def add_and_predict(self, row: dict, health_score: float) -> dict:
        """`row` must contain all base features (no None). Called once per STEP_S."""
        with self._lock:
            self.window.append(row)
            vec = _vector(row, _slope([r["temperature"] for r in self.window]),
                          _slope([r["vibration"] for r in self.window]))
            res = {"ready": self.clf is not None, "training": self.training,
                   "model": self.meta, "baseline_n": BASELINE_N,
                   "baseline_have": len(self.normal_rows), "anomaly": None, "anomaly_score": None,
                   "risk_probability": None, "rul_min": None,
                   "horizon_min": HORIZON_STEPS * STEP_S / 60,
                   "temp_slope": vec[7], "vib_slope": vec[8]}
            if self.clf is not None:
                proba = self.clf.predict_proba([vec])[0]
                res["risk_probability"] = float(proba[list(self.clf.classes_).index(1)]) \
                    if 1 in self.clf.classes_ else 0.0
                res["rul_min"] = float(self.reg.predict([vec])[0]) * STEP_S / 60.0
            if self.iso is None:
                if health_score >= 80:
                    self.normal_rows.append(vec)
                if len(self.normal_rows) >= BASELINE_N:
                    self.iso = IsolationForest(n_estimators=100, contamination=0.02, random_state=0)
                    self.iso.fit(np.array(self.normal_rows))
            if self.iso is not None:
                res["anomaly_score"] = float(self.iso.decision_function([vec])[0])
                res["anomaly"] = bool(self.iso.predict([vec])[0] == -1)
            self.last = res
            return res
