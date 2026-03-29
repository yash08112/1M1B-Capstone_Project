"""Build one-row feature frame aligned with training columns."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from water_ml.pipeline import preprocess_for_model


# Canonical dates so weekday / season features match expectations
DAY_TYPE_CANONICAL = {
    "Weekday": (datetime(2025, 6, 4, 12, 0, 0), "Regular"),  # Wednesday
    "Weekend": (datetime(2025, 6, 7, 12, 0, 0), "Regular"),  # Saturday
    "Holiday": (datetime(2025, 6, 8, 12, 0, 0), "Vacation"),  # Sunday
    "Exam Period": (datetime(2025, 6, 4, 12, 0, 0), "Exams"),  # Wednesday
}


def occupancy_to_activity_label(occupancy_pct: float) -> str:
    if occupancy_pct < 35:
        return "Low"
    if occupancy_pct < 70:
        return "Medium"
    return "High"


def inputs_to_raw_row(
    ambient_temp_c: float,
    occupancy_pct: float,
    day_type: str,
    campus_location: str | None,
    default_campus: str,
) -> pd.DataFrame:
    if day_type not in DAY_TYPE_CANONICAL:
        day_type = "Weekday"
    ts, academic = DAY_TYPE_CANONICAL[day_type]
    activity = occupancy_to_activity_label(occupancy_pct)
    campus = campus_location or default_campus

    row = {
        "Timestamp": ts,
        "Campus_Location": campus,
        "Ambient_Temp_C": float(ambient_temp_c),
        "Activity_Level": activity,  # Low / Medium / High
        "Total_Demand_Liters": 0.0,  # placeholder; dropped before X
        "Academic_Status": academic,
    }
    return pd.DataFrame([row])


def row_to_feature_vector(
    raw: pd.DataFrame,
    training_columns: list[str],
) -> pd.DataFrame:
    X, _ = preprocess_for_model(raw)
    # Align with training (missing dummies = 0)
    X = X.reindex(columns=training_columns, fill_value=0.0)
    return X.astype("float64")
