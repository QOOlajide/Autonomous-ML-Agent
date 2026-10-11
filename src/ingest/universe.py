"""
S&P 500 study universe: fetch once from Wikipedia, cache Symbol+CIK to CSV.

Default path: data/raw/sp500_constituents.csv
Later builds reuse the cache so the universe stays a fixed freeze.
"""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "raw" / "sp500_constituents.csv"
# Wikipedia rejects bare urllib clients (403); identify the app politely.
_WIKI_HEADERS = {
    "User-Agent": "AutonomousMLAgent/0.1 (research; sp500 universe cache)",
}


def normalize_ticker(symbol: str) -> str:
    """SEC-style ticker: uppercase, dots → dashes (e.g. BRK.B → BRK-B)."""
    return str(symbol).strip().upper().replace(".", "-")


def _fetch_sp500_table() -> pd.DataFrame:
    resp = requests.get(WIKI_SP500_URL, headers=_WIKI_HEADERS, timeout=60)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    if not tables:
        raise RuntimeError("Wikipedia returned no tables for S&P 500 constituents")
    raw = tables[0]
    if "Symbol" not in raw.columns or "CIK" not in raw.columns:
        raise RuntimeError(
            f"Unexpected Wiki columns: {list(raw.columns)}. Expected Symbol and CIK."
        )
    out = raw[["Symbol", "CIK"]].copy() # Only keep Symbol and CIK columns
    out["Symbol"] = out["Symbol"].map(normalize_ticker) # Normalize Symbol to uppercase and replace dots with dashes
    out["CIK"] = pd.to_numeric(out["CIK"], errors="coerce") # Convert CIK to numeric
    out = out.dropna(subset=["Symbol", "CIK"]) # Drop rows with missing Symbol or CIK
    out["CIK"] = out["CIK"].astype(int) # Convert CIK to integer
    out = out.drop_duplicates(subset=["Symbol"], keep="first") # Drop duplicate Symbols
    return out.reset_index(drop=True) # Reset index and drop the index column


def _df_to_ticker_ciks(df: pd.DataFrame) -> Dict[str, int]:
    return {str(row.Symbol): int(row.CIK) for row in df.itertuples(index=False)}


def load_sp500_ticker_ciks(
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> Dict[str, int]:
    """
    Load ticker→CIK for the S&P 500 study universe.

    - If cache exists and refresh=False: read CSV (no network).
    - Else: fetch Wikipedia, print head(5), save CSV with utf-8-sig, return map.
    """
    path = Path(cache_path) if cache_path else DEFAULT_CACHE
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not refresh:
        logger.info("Loading cached S&P 500 universe from %s", path)
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "Symbol" not in df.columns or "CIK" not in df.columns:
            raise ValueError(f"Cache {path} missing Symbol/CIK columns")
        df["Symbol"] = df["Symbol"].map(normalize_ticker)
        df["CIK"] = pd.to_numeric(df["CIK"], errors="coerce")
        df = df.dropna(subset=["Symbol", "CIK"])
        df["CIK"] = df["CIK"].astype(int)
        mapping = _df_to_ticker_ciks(df)
        logger.info("Cached universe: %s tickers", len(mapping))
        return mapping

    logger.info("Fetching S&P 500 constituents from Wikipedia")
    df = _fetch_sp500_table()
    print("S&P 500 constituents (first 5 rows before cache write):")
    print(df.head(5).to_string(index=False))
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Wrote frozen universe (%s tickers) to %s", len(df), path)
    return _df_to_ticker_ciks(df)
