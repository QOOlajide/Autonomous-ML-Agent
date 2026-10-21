#!/usr/bin/env python3
"""
Step 1 — Build batch history panel: FRED macros + SEC EDGAR firm fundamentals.

Usage (from repo root):
  python scripts/build_macro_firm_panel.py
  python scripts/build_macro_firm_panel.py --refresh-universe
  python scripts/build_macro_firm_panel.py --max-tickers 50
  python scripts/build_macro_firm_panel.py --tickers AAPL,MSFT,JPM --start 2015-01-01

Requires:
  FRED_API_KEY in .env
  Optional: SEC_CONTACT_EMAIL / SEC_USER_AGENT for EDGAR fair-access header
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from ingest.edgar_client import DEFAULT_TICKER_CIKS, FUNDAMENTAL_CONCEPTS, EdgarClient
from ingest.fred_client import PIPELINE_FEATURES, FredClient
from ingest.panel_builder import MacroFirmPanelBuilder
from ingest.universe import load_sp500_ticker_ciks, normalize_ticker


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build FRED + EDGAR macro–firm panel")
    p.add_argument(
        "--tickers",
        type=str,
        default="",
        help="Optional comma-separated subset (must exist in the S&P 500 universe map)",
    )
    p.add_argument(
        "--refresh-universe",
        action="store_true",
        help="Re-fetch Wikipedia S&P 500 and overwrite data/raw/sp500_constituents.csv",
    )
    p.add_argument(
        "--max-tickers",
        type=int,
        default=0,
        help="Optional cap after universe load (smoke tests). 0 = no cap.",
    )
    p.add_argument("--start", type=str, default="2012-01-01", help="Panel/FRED start date (inclusive)")
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "data"),
        help="Output directory for raw + processed files",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("build_panel")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    processed_dir = out_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache_path = raw_dir / "sp500_constituents.csv"
    try:
        universe = load_sp500_ticker_ciks(cache_path=cache_path, refresh=args.refresh_universe)
    except Exception as exc:
        logger.warning("S&P 500 universe load failed (%s); falling back to DEFAULT_TICKER_CIKS", exc)
        universe = dict(DEFAULT_TICKER_CIKS)

    if args.tickers.strip():
        wanted = [normalize_ticker(t) for t in args.tickers.split(",") if t.strip()]
        ticker_ciks = {t: universe[t] for t in wanted if t in universe}
        missing = [t for t in wanted if t not in universe]
        if missing:
            logger.error("Tickers not in universe map: %s", missing)
            return 1
    else:
        ticker_ciks = dict(universe)

    if args.max_tickers and args.max_tickers > 0:
        kept = list(ticker_ciks.items())[: args.max_tickers]
        ticker_ciks = dict(kept)
        logger.info("Capped universe to --max-tickers=%s", args.max_tickers)

    if not ticker_ciks:
        logger.error("No tickers selected")
        return 1

    logger.info("Universe size for this run: %s tickers", len(ticker_ciks))

    logger.info("Fetching FRED series: %s", list(PIPELINE_FEATURES.values()))
    fred = FredClient()
    # Cadence print → fetch → QE align → YoY / YoY-diff transforms
    macro = fred.fetch_many(
        PIPELINE_FEATURES,
        observation_start=args.start,
        describe_cadence=True,
        align_quarterly=True,
        add_stationarity_transforms=True,
    )
    fred_path = raw_dir / "fred_macros.csv"
    fred.save(macro, str(fred_path))
    fred.save_cadence_report(str(raw_dir / "fred_series_cadence.json"))

    logger.info(
        "Fetching EDGAR fundamentals %s for %s tickers",
        list(FUNDAMENTAL_CONCEPTS.keys()),
        len(ticker_ciks),
    )
    edgar = EdgarClient()
    fundamentals = edgar.fetch_fundamentals_panel(ticker_ciks)
    edgar_path = raw_dir / "edgar_fundamentals.csv"
    edgar.save(fundamentals, str(edgar_path))

    builder = MacroFirmPanelBuilder()
    panel = builder.build(fundamentals, macro, min_date=args.start)
    feature_sets = builder.split_feature_sets(panel)

    panel_csv = processed_dir / "macro_firm_panel.csv"
    panel_parquet = processed_dir / "macro_firm_panel.parquet"
    meta_path = processed_dir / "feature_sets.json"

    builder.save(panel, str(panel_csv))
    try:
        builder.save(panel, str(panel_parquet))
    except Exception as exc:
        logger.warning("Parquet write skipped (%s). CSV is enough.", exc)

    meta_path.write_text(json.dumps(feature_sets, indent=2))
    logger.info("Feature sets written to %s", meta_path)
    logger.info(
        "Done. target=revenue_growth | tickers=%s | firm_only=%s | firm_plus_macro=%s",
        panel["ticker"].nunique() if "ticker" in panel.columns else "?",
        feature_sets["firm_only"],
        feature_sets["firm_plus_macro"],
    )
    logger.info("Panel head:\n%s", panel.head(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
