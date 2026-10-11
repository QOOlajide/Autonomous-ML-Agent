"""
FRED (Federal Reserve Economic Data) batch client.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Core macros for firm financial prediction experiments
DEFAULT_FRED_SERIES: Dict[str, str] = {
    "gdp": "GDPC1",          # Real GDP (quarterly)
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "fed_funds": "FEDFUNDS",
    "treasury_10y": "DGS10",
    "ind_prod": "INDPRO",
}

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"


class FredClient:
    """Download FRED series into a tidy DataFrame."""

    def __init__(self, api_key: Optional[str] = None, session: Optional[requests.Session] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        self.session = session or requests.Session()
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY is not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def fetch_series(
        self,
        series_id: str,
        observation_start: str = "2012-01-01",
        observation_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch one FRED series.

        Returns columns: observation_date, <series_id lowercase alias handled by caller>
        """
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type": "json",
            "observation_start": observation_start,
        }
        if observation_end:
            params["observation_end"] = observation_end

        logger.info("Fetching FRED series %s", series_id)
        resp = self.session.get(FRED_OBSERVATIONS_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        observations = payload.get("observations", [])
        if not observations:
            logger.warning("No observations returned for %s", series_id)
            return pd.DataFrame(columns=["observation_date", "value"])

        df = pd.DataFrame(observations)[["date", "value"]]
        df = df.rename(columns={"date": "observation_date"})
        df["observation_date"] = pd.to_datetime(df["observation_date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("observation_date")
        return df.reset_index(drop=True)

    def fetch_many(
        self,
        series_map: Optional[Dict[str, str]] = None,
        observation_start: str = "2012-01-01",
        observation_end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch multiple series and return a wide panel indexed by observation_date.

        Args:
            series_map: {column_name: FRED_SERIES_ID}
        """
        series_map = series_map or DEFAULT_FRED_SERIES
        frames: List[pd.DataFrame] = []

        for col_name, series_id in series_map.items():
            s = self.fetch_series(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            if s.empty:
                continue
            piece = s.rename(columns={"value": col_name})
            frames.append(piece)

        if not frames:
            raise RuntimeError("No FRED series data downloaded")

        out = frames[0]
        for piece in frames[1:]:
            out = out.merge(piece, on="observation_date", how="outer")

        return out.sort_values("observation_date").reset_index(drop=True)

    def save(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_parquet(path, index=False) if path.endswith(".parquet") else df.to_csv(path, index=False)
        logger.info("Wrote FRED data to %s (%s rows)", path, len(df))
