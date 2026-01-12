"""Feature engineering utilities."""
from __future__ import annotations

import pandas as pd


def add_workload_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["acute_load"] = df["minutes_last_30d"] / 30
    df["chronic_load"] = df["minutes_last_30d"] / 90
    df["acute_chronic_ratio"] = df["acute_load"] / df["chronic_load"].replace(0, pd.NA)
    df["match_intensity"] = df["minutes_last_30d"] / df["matches_last_30d"].replace(0, pd.NA)
    return df


def add_travel_burden(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["travel_burden"] = df["travel_km_last_30d"] / df["matches_last_30d"].replace(0, pd.NA)
    return df


def add_fatigue_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["fatigue_score"] = (
        df["muscle_fatigue_index"] * 0.6 + (10 - df["sleep_quality"]) * 0.4 / 10
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_workload_ratios(df)
    df = add_travel_burden(df)
    df = add_fatigue_score(df)
    return df
