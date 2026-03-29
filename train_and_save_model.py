"""
Train RandomForest + StandardScaler (same as notebook) and save joblib artifact.
Run from project root: python train_and_save_model.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from water_ml.inference import inputs_to_raw_row, row_to_feature_vector
from water_ml.pipeline import preprocess_for_model

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "Modified_Campus_Water_Full_Feature_Set.csv"
ARTIFACT_PATH = ROOT / "water_ml" / "model_artifact.joblib"
META_PATH = ROOT / "water_ml" / "model_meta.json"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    X, y = preprocess_for_model(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    default_campus = df["Campus_Location"].mode().iloc[0]

    artifact = {
        "pipeline": pipe,
        "feature_columns": list(X.columns),
        "default_campus": str(default_campus),
    }
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT_PATH)

    meta = {
        "rmse_test": float(rmse),
        "r2_test": float(r2),
        "n_features": len(X.columns),
        "default_campus": str(default_campus),
        "artifact": str(ARTIFACT_PATH),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", ARTIFACT_PATH)
    print("Test RMSE:", round(rmse, 2), "R2:", round(r2, 4))

    # Sanity check inference path
    raw = inputs_to_raw_row(32.0, 75.0, "Weekday", None, default_campus)
    xv = row_to_feature_vector(raw, artifact["feature_columns"])
    pred = pipe.predict(xv)[0]
    print("Sanity prediction (Weekday, 32C, 75%):", round(pred, 2), "L/day")


if __name__ == "__main__":
    main()
