"""Model training pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.engineering import build_features


@dataclass
class TrainingResult:
    model: Pipeline
    metrics: dict


def train_model(df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str], params: dict, random_state: int, test_size: float) -> TrainingResult:
    df = build_features(df)
    y = df[target]
    X = df[numeric + categorical + [
        "acute_load",
        "chronic_load",
        "acute_chronic_ratio",
        "match_intensity",
        "travel_burden",
        "fatigue_score",
    ]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric + [
                "acute_load",
                "chronic_load",
                "acute_chronic_ratio",
                "match_intensity",
                "travel_burden",
                "fatigue_score",
            ]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = GradientBoostingClassifier(**params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "report": classification_report(y_test, preds, output_dict=True),
    }

    return TrainingResult(model=pipeline, metrics=metrics)
