"""Synthetic data generator for demos."""
from __future__ import annotations

import numpy as np
import pandas as pd


POSITIONS = ["Goalkeeper", "Defender", "Midfielder", "Forward"]
FOOT = ["Right", "Left"]


def generate_synthetic_players(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "player_id": np.arange(1, n + 1),
        "name": [f"Player {i}" for i in range(1, n + 1)],
        "team": rng.choice(["FC Carpathia", "SC Dunarea", "AC Rasarit"], size=n),
        "age": rng.integers(18, 35, size=n),
        "height_cm": rng.integers(168, 195, size=n),
        "weight_kg": rng.integers(60, 90, size=n),
        "position": rng.choice(POSITIONS, size=n),
        "dominant_foot": rng.choice(FOOT, size=n),
        "minutes_last_30d": rng.integers(200, 900, size=n),
        "matches_last_30d": rng.integers(3, 10, size=n),
        "sprint_distance_km": rng.uniform(3, 14, size=n).round(2),
        "total_distance_km": rng.uniform(60, 110, size=n).round(2),
        "previous_injuries": rng.integers(0, 5, size=n),
        "days_since_last_injury": rng.integers(10, 400, size=n),
        "muscle_fatigue_index": rng.uniform(0.3, 0.95, size=n).round(2),
        "sleep_quality": rng.uniform(5.5, 8.5, size=n).round(1),
        "travel_km_last_30d": rng.integers(200, 3000, size=n),
    }
    df = pd.DataFrame(data)
    risk_score = (
        0.02 * df["previous_injuries"]
        + 0.015 * (df["minutes_last_30d"] / 90)
        + 0.2 * df["muscle_fatigue_index"]
        + 0.01 * (10 - df["sleep_quality"])
    )
    prob = 1 / (1 + np.exp(-risk_score))
    df["injury_next_30d"] = rng.binomial(1, p=np.clip(prob, 0.05, 0.8))
    return df
