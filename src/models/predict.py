"""Prediction helpers."""
from __future__ import annotations

import pandas as pd

from src.features.engineering import build_features


def predict_risk(model, df: pd.DataFrame) -> pd.DataFrame:
    df = build_features(df)
    features = df[[
        "age",
        "height_cm",
        "weight_kg",
        "minutes_last_30d",
        "matches_last_30d",
        "sprint_distance_km",
        "total_distance_km",
        "previous_injuries",
        "days_since_last_injury",
        "muscle_fatigue_index",
        "sleep_quality",
        "travel_km_last_30d",
        "position",
        "dominant_foot",
        "acute_load",
        "chronic_load",
        "acute_chronic_ratio",
        "match_intensity",
        "travel_burden",
        "fatigue_score",
    ]]
    proba = model.predict_proba(features)[:, 1]
    df = df.copy()
    df["injury_risk_score"] = proba
    return df
