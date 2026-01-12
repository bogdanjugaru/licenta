"""Biomechanics-inspired risk scoring."""
from __future__ import annotations

import pandas as pd


def compute_biomech_flags(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    df = df.copy()
    df["flag_acute_chronic"] = df["acute_chronic_ratio"] >= thresholds["acute_chronic_ratio_threshold"]
    df["flag_workload_spike"] = (
        df["sprint_distance_km"] / df["total_distance_km"].replace(0, pd.NA)
    ) >= thresholds["workload_spike_threshold"]
    df["flag_fatigue"] = df["fatigue_score"] >= thresholds["fatigue_threshold"]
    df["biomech_risk_score"] = (
        df[["flag_acute_chronic", "flag_workload_spike", "flag_fatigue"]].sum(axis=1) / 3
    )
    return df
