"""Data loading helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.sources import (
    ClubMedicalStubSource,
    LocalCSVSource,
    TransfermarktStubSource,
    UEFAStubSource,
    combine_sources,
)


def load_local_data(path: Path) -> pd.DataFrame:
    return LocalCSVSource(name="local", description="Local CSV", path=path).fetch()


def load_external_sources() -> pd.DataFrame:
    sources = [
        TransfermarktStubSource(name="transfermarkt", description="Transfermarkt export"),
        UEFAStubSource(name="uefa", description="UEFA export"),
        ClubMedicalStubSource(name="club", description="Club medical export"),
    ]
    return combine_sources(sources)
