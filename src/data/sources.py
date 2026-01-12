"""Data source abstractions for football injury analytics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class DataSource:
    name: str
    description: str

    def fetch(self) -> pd.DataFrame:
        raise NotImplementedError


@dataclass
class LocalCSVSource(DataSource):
    path: Path

    def fetch(self) -> pd.DataFrame:
        return pd.read_csv(self.path)


@dataclass
class TransfermarktStubSource(DataSource):
    """Placeholder for Transfermarkt ingestion.

    This intentionally avoids automated scraping. Replace with licensed exports or
    curated datasets where permitted.
    """

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame()


@dataclass
class UEFAStubSource(DataSource):
    """Placeholder for UEFA data ingestion."""

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame()


@dataclass
class ClubMedicalStubSource(DataSource):
    """Placeholder for club medical data ingestion."""

    def fetch(self) -> pd.DataFrame:
        return pd.DataFrame()


def combine_sources(sources: Iterable[DataSource]) -> pd.DataFrame:
    frames = []
    for source in sources:
        frame = source.fetch()
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
