"""
SEC EDGAR company-facts client for quarterly firm fundamentals.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Small liquid universe for the case-study panel (ticker -> CIK int)
DEFAULT_TICKER_CIKS: Dict[str, int] = {
    "AAPL": 320193,
    "MSFT": 789019,
    "JPM": 19617,
    "XOM": 34088,
    "WMT": 104169,
    "JNJ": 200406,
    "PG": 80424,
    "CVX": 93410,
    "HD": 354950,
    "BAC": 70858,
}

# field_name -> ordered US-GAAP concept fallbacks (first hit with USD units wins)
FUNDAMENTAL_CONCEPTS: Dict[str, Sequence[str]] = {
    "revenue": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "SalesRevenueGoodsNet",
    ),
    "operating_income": (
        "OperatingIncomeLoss",
    ),
    "net_income": (
        "NetIncomeLoss",
        "ProfitLoss",
    ),
    "gross_profit": (
        "GrossProfit",
    ),
    "operating_cash_flow": (
        "NetCashProvidedByUsedInOperatingActivities",
    ),
    "total_assets": (
        "Assets",
    ),
    "total_liabilities": (
        "Liabilities",
    ),
    "stockholders_equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "eps_basic": (
        "EarningsPerShareBasic",
    ),
    "eps_diluted": (
        "EarningsPerShareDiluted",
    ),
}

# EPS is reported per-share; prefer USD/shares units when present
_EPS_UNIT_KEYS = ("USD/shares", "USD/share", "pure")

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"


class EdgarClient:
    """
    Download SEC company facts and extract a quarterly fundamentals panel.

    SEC requires a descriptive User-Agent with contact email.
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        session: Optional[requests.Session] = None,
        request_pause_s: float = 0.2,
        concepts: Optional[Dict[str, Sequence[str]]] = None,
    ):
        email = os.getenv("SEC_CONTACT_EMAIL", "ml-agent@example.com")
        self.user_agent = user_agent or os.getenv(
            "SEC_USER_AGENT",
            f"AutonomousMLAgent/0.1 ({email})",
        )
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self.request_pause_s = request_pause_s
        self.concepts = concepts or FUNDAMENTAL_CONCEPTS

    def _cik_str(self, cik: int) -> str:
        return f"{int(cik):010d}"

    def fetch_company_facts(self, cik: int) -> dict:
        url = COMPANY_FACTS_URL.format(cik=self._cik_str(cik))
        logger.info("Fetching EDGAR company facts CIK=%s", self._cik_str(cik))
        resp = self.session.get(url, timeout=60)
        resp.raise_for_status()
        time.sleep(self.request_pause_s)
        return resp.json()

    def _pick_unit_facts(self, node: dict, field_name: str) -> Tuple[Optional[str], List[dict]]:
        units = node.get("units", {}) or {}
        if field_name.startswith("eps"):
            for key in _EPS_UNIT_KEYS:
                if key in units and units[key]:
                    return key, units[key]
            # fall through to any available unit
        if "USD" in units and units["USD"]:
            return "USD", units["USD"]
        for key, vals in units.items():
            if vals:
                return key, vals
        return None, []

    def _extract_series(
        self,
        us_gaap: dict,
        field_name: str,
        concept_options: Sequence[str],
    ) -> pd.DataFrame:
        for concept in concept_options:
            node = us_gaap.get(concept)
            if not node:
                continue
            unit_key, facts = self._pick_unit_facts(node, field_name)
            if not facts:
                continue

            rows = []
            for item in facts:
                fp = item.get("fp")
                form = item.get("form")
                end = item.get("end")
                val = item.get("val")
                if fp not in {"Q1", "Q2", "Q3", "Q4"}:
                    continue
                if form not in {"10-Q", "10-K", "10-Q/A", "10-K/A"}:
                    continue
                if end is None or val is None:
                    continue
                rows.append(
                    {
                        "period_end": pd.to_datetime(end),
                        "fy": item.get("fy"),
                        "fp": fp,
                        "form": form,
                        "filed": pd.to_datetime(item.get("filed")) if item.get("filed") else pd.NaT,
                        field_name: float(val),
                        f"{field_name}_concept": concept,
                        f"{field_name}_unit": unit_key,
                    }
                )

            if not rows:
                continue

            df = pd.DataFrame(rows)
            df = df.sort_values(["period_end", "filed"])
            df = df.drop_duplicates(subset=["period_end", "fp"], keep="last")
            return df.reset_index(drop=True)

        return pd.DataFrame()

    def extract_quarterly_fundamentals(self, facts: dict, ticker: str, cik: int) -> pd.DataFrame:
        """
        Build a quarterly fundamentals table (one row per period_end).
        """
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        series_frames: List[pd.DataFrame] = []

        for field_name, concept_options in self.concepts.items():
            s = self._extract_series(us_gaap, field_name, concept_options)
            if s.empty:
                logger.warning("No %s facts for %s (CIK %s)", field_name, ticker, cik)
                continue
            series_frames.append(s)

        if not series_frames:
            logger.warning("No fundamentals extracted for %s (CIK %s)", ticker, cik)
            return pd.DataFrame()

        # Outer-join all metrics on period identity
        out = series_frames[0]
        key_cols = ["period_end", "fy", "fp", "form", "filed"]
        for frame in series_frames[1:]:
            # Prefer identity cols from left; merge on period_end + fp
            value_cols = [c for c in frame.columns if c not in key_cols]
            out = out.merge(
                frame[["period_end", "fp"] + value_cols],
                on=["period_end", "fp"],
                how="outer",
            )

        out["ticker"] = ticker
        out["cik"] = cik

        # Outer joins can create duplicate period_end rows; coalesce to one row per period
        out = out.sort_values(["period_end", "fp"])
        grouped = out.groupby(["ticker", "cik", "period_end", "fp"], as_index=False).agg(
            lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else pd.NA
        )
        return grouped.sort_values(["period_end", "fp"]).reset_index(drop=True)

    # Back-compat alias
    def extract_quarterly_revenue(self, facts: dict, ticker: str, cik: int) -> pd.DataFrame:
        return self.extract_quarterly_fundamentals(facts, ticker, cik)

    def fetch_fundamentals_panel(
        self,
        ticker_ciks: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        ticker_ciks = ticker_ciks or DEFAULT_TICKER_CIKS
        frames: List[pd.DataFrame] = []
        for ticker, cik in ticker_ciks.items():
            try:
                facts = self.fetch_company_facts(cik)
                fundamentals = self.extract_quarterly_fundamentals(facts, ticker=ticker, cik=cik)
                if not fundamentals.empty:
                    frames.append(fundamentals)
            except requests.HTTPError as exc:
                logger.error("EDGAR HTTP error for %s: %s", ticker, exc)
            except Exception as exc:
                logger.error("EDGAR error for %s: %s", ticker, exc)

        if not frames:
            raise RuntimeError("No EDGAR fundamentals downloaded for any ticker")

        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    # Back-compat alias used by older script versions
    def fetch_revenue_panel(
        self,
        ticker_ciks: Optional[Dict[str, int]] = None,
    ) -> pd.DataFrame:
        return self.fetch_fundamentals_panel(ticker_ciks)

    def save(self, df: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        logger.info("Wrote EDGAR fundamentals to %s (%s rows)", path, len(df))
