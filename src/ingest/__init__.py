"""
Batch ingest for the S&P-style macro vs firm financials case study.

Pulls FRED macros + SEC EDGAR company fundamentals and builds an as-of panel
for predicting revenue growth with and without macro features.
"""

from .edgar_client import (
    DEFAULT_TICKER_CIKS,
    FUNDAMENTAL_CONCEPTS,
    EdgarClient,
)
from .fred_client import DEFAULT_FRED_SERIES, PIPELINE_FEATURES, FredClient
from .panel_builder import MacroFirmPanelBuilder
from .universe import load_sp500_ticker_ciks, normalize_ticker

__all__ = [
    "EdgarClient",
    "FredClient",
    "MacroFirmPanelBuilder",
    "DEFAULT_TICKER_CIKS",
    "DEFAULT_FRED_SERIES",
    "PIPELINE_FEATURES",
    "FUNDAMENTAL_CONCEPTS",
    "load_sp500_ticker_ciks",
    "normalize_ticker",
]
