"""
Serve the dashboard and POST /api/predict using the trained sklearn pipeline.

Usage (from project root):
  pip install -r requirements.txt
  python train_and_save_model.py   # if model_artifact.joblib is missing
  python app.py

Open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

from water_ml.inference import inputs_to_raw_row, row_to_feature_vector

ROOT = Path(__file__).resolve().parent
ARTIFACT_PATH = ROOT / "water_ml" / "model_artifact.joblib"
META_PATH = ROOT / "water_ml" / "model_meta.json"
CSV_PATH = ROOT / "Modified_Campus_Water_Full_Feature_Set.csv"

app = Flask(__name__)
logger = logging.getLogger(__name__)

_artifact = None
_campus_zones_cache: list[str] | None = None
_win_train_lock = threading.Lock()


def _train_if_missing() -> None:
    """
    If the joblib artifact is absent but the CSV exists (e.g. Render build
    skipped training), train once. Uses a file lock on Linux so Gunicorn
    workers do not train in parallel.
    """
    global _artifact
    if ARTIFACT_PATH.is_file():
        return
    if not CSV_PATH.is_file():
        return

    lock_path = ROOT / "water_ml" / ".training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import fcntl  # noqa: PLC0415 — Unix only (Render)
    except ImportError:
        fcntl = None  # type: ignore[misc, assignment]

    try:
        if fcntl is not None:
            with open(lock_path, "a", encoding="utf-8") as lf:
                fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
                if not ARTIFACT_PATH.is_file():
                    from train_and_save_model import main as train_main  # noqa: PLC0415

                    train_main()
        else:
            with _win_train_lock:
                if not ARTIFACT_PATH.is_file():
                    from train_and_save_model import main as train_main  # noqa: PLC0415

                    train_main()
    except Exception:
        logger.exception("Automatic model training failed")
    finally:
        _artifact = None


def get_artifact():
    global _artifact
    _train_if_missing()
    if _artifact is None:
        if not ARTIFACT_PATH.is_file():
            return None
        _artifact = joblib.load(ARTIFACT_PATH)
    return _artifact


@app.route("/")
def index():
    return send_from_directory(ROOT, "index.html")


def _sustainability_from_meta(meta: dict) -> tuple[int | None, str]:
    """
    Single UI number derived from model test R² (0–1 → 0–100).
    This is a *model fit / prediction confidence* indicator, not a full
    environmental sustainability or LCA score.
    """
    r2 = meta.get("r2_test")
    if r2 is None:
        return None, "Model fit metrics are not available yet."
    try:
        r2 = float(r2)
    except (TypeError, ValueError):
        return None, "Invalid R² in model_meta.json."
    score = int(round(max(0.0, min(1.0, r2)) * 100))
    return score, "Based on test-set R² (model fit on your CSV). Not a campus lifecycle audit."


@app.route("/api/health", methods=["GET"])
def health():
    art = get_artifact()
    meta = {}
    if META_PATH.is_file():
        meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    score, note = _sustainability_from_meta(meta)
    return jsonify(
        {
            "ok": art is not None,
            "r2_test": meta.get("r2_test"),
            "rmse_test": meta.get("rmse_test"),
            "default_campus": meta.get("default_campus"),
            "sustainability_index": score,
            "sustainability_note": note,
        }
    )


@app.route("/api/campus_zones", methods=["GET"])
def campus_zones():
    """Unique zone names from the training CSV (for the Campus Zones page)."""
    global _campus_zones_cache
    if _campus_zones_cache is None:
        if not CSV_PATH.is_file():
            return jsonify({"zones": [], "error": "CSV not found"}), 404
        df = pd.read_csv(CSV_PATH, usecols=["Campus_Location"])
        _campus_zones_cache = sorted(df["Campus_Location"].unique().tolist())
    return jsonify({"zones": _campus_zones_cache})


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 204

    art = get_artifact()
    if art is None:
        return jsonify({"error": "unavailable"}), 503

    data = request.get_json(silent=True) or {}
    try:
        temp = float(data.get("ambient_temp_c", data.get("temperature", 28)))
        occ = float(data.get("occupancy_pct", data.get("occupancy", 75)))
        day_type = str(data.get("day_type", "Weekday"))
        campus = data.get("campus_location") or None
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid numeric fields"}), 400

    pipe = art["pipeline"]
    cols = art["feature_columns"]
    default_campus = art["default_campus"]

    raw = inputs_to_raw_row(temp, occ, day_type, campus, default_campus)
    X = row_to_feature_vector(raw, cols)
    pred_liters = float(pipe.predict(X)[0])

    return jsonify(
        {
            "predicted_demand_liters_per_day": round(pred_liters, 2),
            "predicted_demand_m3_per_day": round(pred_liters / 1000.0, 4),
            "default_campus_used": campus is None,
            "default_campus": default_campus,
            "campus_location_used": campus or default_campus,
            "model": "RandomForestRegressor + StandardScaler",
        }
    )


@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return resp


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    # Cloud hosts set PORT; bind 0.0.0.0 so the service is reachable.
    host = "0.0.0.0" if os.environ.get("PORT") else "127.0.0.1"
    app.run(host=host, port=port, debug=False)
