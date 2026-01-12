"""Streamlit application for injury analysis and prediction."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.biomech.risk import compute_biomech_flags
from src.data.loader import load_external_sources, load_local_data
from src.data.synthetic import generate_synthetic_players
from src.features.engineering import build_features
from src.models.predict import predict_risk
from src.models.train import train_model

CONFIG_PATH = Path("configs/config.yaml")


@st.cache_data
def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    st.set_page_config(page_title="Injury Analytics", layout="wide")
    st.title("Analiza si Predictia Accidentarilor Jucatorilor de Fotbal")

    config = load_config()
    data_cfg = config["data"]
    feature_cfg = config["features"]

    st.sidebar.header("Date")
    data_choice = st.sidebar.selectbox(
        "Sursa date",
        options=["Dataset local", "Surse externe (stub)", "Date sintetice"],
    )

    if data_choice == "Dataset local":
        df = load_local_data(Path(data_cfg["raw_path"]))
    elif data_choice == "Surse externe (stub)":
        df = load_external_sources()
    else:
        n_samples = st.sidebar.slider("Numar jucatori", 100, 1000, 300, step=50)
        df = generate_synthetic_players(n_samples, seed=data_cfg["random_state"])

    if df.empty:
        st.warning(
            "Sursele externe sunt in mod implicit goale. "
            "Incarca un dataset local sau foloseste date sintetice."
        )
        st.stop()

    st.subheader("Preview date")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Feature engineering")
    features_df = build_features(df)
    st.dataframe(features_df.head(10), use_container_width=True)

    st.subheader("Biomecanica si risc fiziologic")
    biomech_df = compute_biomech_flags(features_df, config["biomechanics"])
    st.dataframe(
        biomech_df[[
            "name",
            "acute_chronic_ratio",
            "fatigue_score",
            "biomech_risk_score",
        ]].head(10),
        use_container_width=True,
    )

    st.subheader("Antrenare ML")
    train_button = st.button("Antreneaza model")

    if train_button:
        result = train_model(
            df,
            target=data_cfg["target"],
            numeric=feature_cfg["numeric"],
            categorical=feature_cfg["categorical"],
            params=config["model"]["params"],
            random_state=data_cfg["random_state"],
            test_size=data_cfg["test_size"],
        )
        st.success(f"Model antrenat. ROC-AUC: {result.metrics['roc_auc']:.3f}")
        st.json(result.metrics["report"])

        st.subheader("Predictie risc accidentare")
        scored = predict_risk(result.model, df)
        st.dataframe(
            scored[["name", "team", "position", "injury_risk_score"]]
            .sort_values(by="injury_risk_score", ascending=False)
            .head(20),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
