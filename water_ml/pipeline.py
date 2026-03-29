"""
Preprocessing aligned with 1M1B_AI_WATER_DEMAND_FORECASTING.ipynb
"""

from __future__ import annotations

import pandas as pd


def get_season(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Summer"
    if month in (6, 7, 8):
        return "Monsoon"
    return "Post-Monsoon"


def preprocess_for_model(wdf: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    wdf = wdf.copy()
    wdf.columns = wdf.columns.str.strip()

    wdf["Timestamp"] = pd.to_datetime(wdf["Timestamp"])
    wdf["Activity_Level"] = wdf["Activity_Level"].map({"Low": 0, "Medium": 1, "High": 2}).astype("int64")

    # Raw CSV columns superseded by Timestamp-based features
    for col in ("Day_of_Week", "Weekend"):
        if col in wdf.columns:
            wdf.drop(columns=col, inplace=True)

    wdf["Year"] = wdf["Timestamp"].dt.year
    wdf["Month"] = wdf["Timestamp"].dt.month
    wdf["Day"] = wdf["Timestamp"].dt.day
    wdf["Day_Of_week"] = wdf["Timestamp"].dt.day_name()
    wdf["Weekend"] = wdf["Day_Of_week"].isin(["Saturday", "Sunday"]).astype(int)
    wdf["Week"] = wdf["Timestamp"].dt.isocalendar().week.astype(int)

    wdf["Season"] = wdf["Month"].apply(get_season)

    wdf = pd.get_dummies(
        wdf,
        columns=["Campus_Location", "Day_Of_week", "Season", "Academic_Status"],
        drop_first=True,
    )

    bool_cols = wdf.select_dtypes(include=["bool"]).columns
    wdf[bool_cols] = wdf[bool_cols].astype(int)

    wdf.drop(columns="Timestamp", inplace=True)

    y = wdf["Total_Demand_Liters"]
    X = wdf.drop(columns="Total_Demand_Liters")

    # Match notebook: only numeric types for scaler (include uint8 from dummies)
    num_cols = X.select_dtypes(include=["number"]).columns
    X = X[num_cols].astype("float64")

    return X, y
