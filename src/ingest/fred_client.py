"""
FRED (Federal Reserve Economic Data) batch client.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)

# ==============================================================================
# MACROECONOMIC VARIABLE SELECTION & TRANSMISSION MAPPING
# Each variable maps to a corporate financial-statement transmission channel.
# ==============================================================================
PIPELINE_FEATURES: Dict[str, str] = {
    # 1. Aggregate demand & top-line growth (revenue / sales volume)
    "real_gdp": "GDPC1",
    "industrial_prod": "INDPRO",
    "real_pce": "PCEC96",
    # 2. Cost pressures & input inflation (COGS, SG&A, margins)
    "cpi": "CPIAUCSL",
    "ppi": "PPIFID",
    "employment_cost": "ECIALLCIV",
    "wti_oil": "DCOILWTICO",
    # 3. Cost of capital & debt service (interest, WACC, solvency)
    "fed_funds": "DFF",
    "treasury_10y": "DGS10",
    "yield_curve_10_2": "T10Y2Y",
    "high_yield_spread": "BAMLH0A0HYM2",
    # 4. Forward-looking leading indicators
    "unemployment": "UNRATE",
    "consumer_sentiment": "UMCSENT",
}


# Level indices → YoY % change (4 quarters after QE resample)
GROWTH_INDICES = [
    "real_gdp",
    "industrial_prod",
    "real_pce",
    "cpi",
    "ppi",
    "employment_cost",
    "wti_oil",
]

# Rates / spreads → YoY first difference (change over 4 quarters)
RATE_DIFFERENTIALS = [
    "fed_funds",
    "treasury_10y",
    "yield_curve_10_2",
    "high_yield_spread",
    "unemployment",
]


class FredClient:
    """Download FRED series, align to quarter-end, apply stationarity transforms."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FRED_API_KEY is not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self.fred = Fred(api_key=self.api_key)
        self._last_cadence_report: Dict[str, Dict[str, Any]] = {}

    def get_series_info(self, series_id: str) -> Any:
        return self.fred.get_series_info(series_id)

    def describe_series_cadences(
        self,
        series_map: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Print each series' native FRED frequency (mixed D/M/Q is expected)."""
        series_map = series_map or PIPELINE_FEATURES
        report: Dict[str, Dict[str, Any]] = {}
        for alias, series_id in series_map.items():
            info = self.fred.get_series_info(series_id)
            frequency = info["frequency"]
            print(f"Variable Cadence ({alias} / {series_id}): {frequency}")
            logger.info("Variable Cadence (%s / %s): %s", alias, series_id, frequency)
            report[alias] = {
                "series_id": series_id,
                "frequency": frequency,
                "frequency_short": info["frequency_short"] if "frequency_short" in info.index else None,
                "title": info["title"] if "title" in info.index else None,
            }
        self._last_cadence_report = report
        return report

    def fetch_series(
        self,
        series_id: str,
        observation_start: str = "2012-01-01",
        observation_end: Optional[str] = None,
    ) -> pd.Series:
        kwargs = {"observation_start": observation_start}
        if observation_end:
            kwargs["observation_end"] = observation_end
        logger.info("Fetching FRED series %s", series_id)
        series = self.fred.get_series(series_id, **kwargs)
        if series is None or len(series) == 0:
            logger.warning("No observations returned for %s", series_id)
            return pd.Series(dtype=float)
        series.index = pd.to_datetime(series.index)
        return pd.to_numeric(series, errors="coerce").dropna().sort_index()

    def fetch_many(
        self,
        series_map: Optional[Dict[str, str]] = None,
        observation_start: str = "2012-01-01",
        observation_end: Optional[str] = None,
        *,
        describe_cadence: bool = True,
        align_quarterly: bool = True,
        add_stationarity_transforms: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch macros into one frame.

        - Mixed cadences are expected (daily/monthly/quarterly).
        - If align_quarterly: resample to quarter-end ('QE') with .last().ffill()
          so rows line up with 10-Q/10-K style quarters.
        - If add_stationarity_transforms: add YoY growth / YoY rate differentials.
        """
        series_map = series_map or PIPELINE_FEATURES
        if describe_cadence:
            self.describe_series_cadences(series_map)

        raw: Dict[str, pd.Series] = {}
        for alias, series_id in series_map.items():
            s = self.fetch_series(
                series_id,
                observation_start=observation_start,
                observation_end=observation_end,
            )
            if not s.empty:
                raw[alias] = s

        if not raw:
            raise RuntimeError("No FRED series data downloaded")

        macro_df = pd.DataFrame(raw).sort_index()

        if align_quarterly:
            # Mixed D/M/Q → quarter-end; ffill covers gaps / non-trading days
            macro_df = macro_df.resample("QE").last().ffill()

        if add_stationarity_transforms:
            for col in GROWTH_INDICES:
                if col in macro_df.columns:
                    macro_df[f"{col}_yoy"] = macro_df[col].pct_change(4)
            for col in RATE_DIFFERENTIALS:
                if col in macro_df.columns:
                    macro_df[f"{col}_diff_yoy"] = macro_df[col] - macro_df[col].shift(4)

        out = macro_df.reset_index().rename(columns={"index": "observation_date"})
        if "observation_date" not in out.columns:
            # pandas may name the index column differently
            out = out.rename(columns={out.columns[0]: "observation_date"})
        out["observation_date"] = pd.to_datetime(out["observation_date"])
        return out.sort_values("observation_date").reset_index(drop=True)

    def save_cadence_report(self, path: str) -> None:
        report = self._last_cadence_report or {}
        if not report:
            logger.warning("No cadence report to save; call describe_series_cadences/fetch_many first")
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info("Wrote FRED cadence report to %s", path)

    def save(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info("Wrote FRED data to %s (%s rows)", path, len(df))
