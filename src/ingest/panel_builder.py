"""
Build firm × quarter panel: revenue growth + firm fundamentals + as-of macros.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RAW_FIRM_LEVEL_COLS = [
    "revenue",
    "operating_income",
    "net_income",
    "gross_profit",
    "operating_cash_flow",
    "total_assets",
    "total_liabilities",
    "stockholders_equity",
    "eps_basic",
    "eps_diluted",
]

# Stationary macro features produced by FredClient (QE-aligned YoY transforms)
MACRO_MODEL_FEATURES = {
    "real_gdp_yoy",
    "industrial_prod_yoy",
    "real_pce_yoy",
    "cpi_yoy",
    "ppi_yoy",
    "employment_cost_yoy",
    "wti_oil_yoy",
    "fed_funds_diff_yoy",
    "treasury_10y_diff_yoy",
    "yield_curve_10_2_diff_yoy",
    "high_yield_spread_diff_yoy",
    "unemployment_diff_yoy",
    # consumer_sentiment kept as level (already survey index); include if present
    "consumer_sentiment",
}

# Legacy alias
BASE_MACROS = MACRO_MODEL_FEATURES


class MacroFirmPanelBuilder:
    """
    Join EDGAR quarterly fundamentals with FRED macros using as-of merge.

    As-of rule: for each firm quarter with period_end T, attach the latest
    macro observation with observation_date <= period_start, where
    period_start ≈ period_end - 1 quarter (no lookahead into the quarter).
    """

    def __init__(self, macro_cols: Optional[Sequence[str]] = None):
        self.macro_cols = list(macro_cols) if macro_cols else None

    @staticmethod
    def _quarter_start(period_end: pd.Series) -> pd.Series:
        return period_end - pd.Timedelta(days=91)

    def add_firm_features(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        df = fundamentals_df.copy()
        df = df.sort_values(["ticker", "period_end"])

        if "revenue" not in df.columns:
            raise ValueError("EDGAR panel must include revenue to define revenue_growth target")

        for col in RAW_FIRM_LEVEL_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["revenue_lag1"] = df.groupby("ticker")["revenue"].shift(1)
        df["revenue_growth"] = (df["revenue"] - df["revenue_lag1"]) / df["revenue_lag1"].abs()
        df["revenue_growth_lag1"] = df.groupby("ticker")["revenue_growth"].shift(1)
        df["log_revenue"] = np.log(df["revenue"].clip(lower=1.0))
        df.loc[df["revenue_growth"].abs() > 5, "revenue_growth"] = np.nan

        # Margins / returns (skip when denominator missing/zero)
        if "gross_profit" in df.columns:
            df["gross_margin"] = df["gross_profit"] / df["revenue"].replace(0, np.nan)
        if "operating_income" in df.columns:
            df["operating_margin"] = df["operating_income"] / df["revenue"].replace(0, np.nan)
        if "net_income" in df.columns:
            df["net_margin"] = df["net_income"] / df["revenue"].replace(0, np.nan)
        if "total_assets" in df.columns:
            df["log_assets"] = np.log(df["total_assets"].clip(lower=1.0))
            if "net_income" in df.columns:
                df["roa"] = df["net_income"] / df["total_assets"].replace(0, np.nan)
        if "stockholders_equity" in df.columns and "net_income" in df.columns:
            df["roe"] = df["net_income"] / df["stockholders_equity"].replace(0, np.nan)
        if "total_liabilities" in df.columns and "total_assets" in df.columns:
            df["leverage"] = df["total_liabilities"] / df["total_assets"].replace(0, np.nan)
        if "operating_cash_flow" in df.columns and "total_assets" in df.columns:
            df["ocf_to_assets"] = df["operating_cash_flow"] / df["total_assets"].replace(0, np.nan)

        # Lagged levels so models don't peek at contemporaneous accounting identity with target
        for col in [
            "operating_margin",
            "net_margin",
            "gross_margin",
            "roa",
            "roe",
            "leverage",
            "ocf_to_assets",
            "eps_basic",
            "eps_diluted",
            "log_assets",
        ]:
            if col in df.columns:
                df[f"{col}_lag1"] = df.groupby("ticker")[col].shift(1)

        return df

    def attach_macros(self, firm_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        firm = firm_df.copy()
        macro = macro_df.copy()

        firm["period_end"] = pd.to_datetime(firm["period_end"])
        firm["period_start"] = self._quarter_start(firm["period_end"])
        macro["observation_date"] = pd.to_datetime(macro["observation_date"])
        macro = macro.sort_values("observation_date")

        value_cols = [c for c in macro.columns if c != "observation_date"]
        if self.macro_cols:
            value_cols = [c for c in self.macro_cols if c in value_cols]

        firm = firm.sort_values("period_start")
        macro_levels = macro[["observation_date"] + value_cols].sort_values("observation_date")
        # Prefer model features (YoY / diffs); fall back to whatever columns exist
        prefer = [c for c in value_cols if c in MACRO_MODEL_FEATURES or c.endswith("_yoy") or c.endswith("_diff_yoy")]
        attach_cols = prefer if prefer else value_cols
        macro_levels[attach_cols] = macro_levels[attach_cols].ffill()
        merged = pd.merge_asof(
            firm,
            macro_levels[["observation_date"] + attach_cols],
            left_on="period_start",
            right_on="observation_date",
            direction="backward",
        )
        if "observation_date" in merged.columns:
            merged = merged.rename(columns={"observation_date": "macro_asof_date"})

        return merged.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    def _firm_feature_columns(self, panel: pd.DataFrame) -> List[str]:
        preferred = [
            "log_revenue",
            "revenue_growth_lag1",
            "gross_margin_lag1",
            "operating_margin_lag1",
            "net_margin_lag1",
            "roa_lag1",
            "roe_lag1",
            "leverage_lag1",
            "ocf_to_assets_lag1",
            "log_assets_lag1",
            "eps_basic_lag1",
            "eps_diluted_lag1",
        ]
        return [c for c in preferred if c in panel.columns]

    def _macro_feature_columns(self, panel: pd.DataFrame) -> List[str]:
        preferred = [c for c in MACRO_MODEL_FEATURES if c in panel.columns]
        if preferred:
            return preferred
        # Fallback for older panels that used *_chg naming
        return [c for c in panel.columns if c.endswith("_yoy") or c.endswith("_diff_yoy") or c.endswith("_chg")]

    def build(
        self,
        fundamentals_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        min_date: str = "2012-01-01",
    ) -> pd.DataFrame:
        logger.info("Building macro–firm panel")
        firm = self.add_firm_features(fundamentals_df)
        panel = self.attach_macros(firm, macro_df)
        panel = panel[panel["period_end"] >= pd.to_datetime(min_date)].copy()
        panel = panel.dropna(subset=["revenue_growth"])
        panel["target"] = panel["revenue_growth"]

        id_cols = [
            "ticker",
            "cik",
            "period_end",
            "period_start",
            "macro_asof_date",
            "fy",
            "fp",
            "form",
        ]
        firm_cols = self._firm_feature_columns(panel)
        macro_cols = self._macro_feature_columns(panel)

        # Keep raw levels for inspection, but feature_sets will use engineered cols
        meta_raw = [c for c in RAW_FIRM_LEVEL_COLS if c in panel.columns]
        concept_cols = [c for c in panel.columns if c.endswith("_concept") or c.endswith("_unit")]

        ordered = (
            [c for c in id_cols if c in panel.columns]
            + ["target", "revenue_growth"]
            + firm_cols
            + macro_cols
            + meta_raw
            + concept_cols
        )
        seen = set()
        final_cols = []
        for c in ordered:
            if c not in seen and c in panel.columns:
                seen.add(c)
                final_cols.append(c)
        # append any remaining columns
        for c in panel.columns:
            if c not in seen:
                final_cols.append(c)

        panel = panel[final_cols]
        logger.info(
            "Panel ready: %s rows, %s tickers, %s columns",
            len(panel),
            panel["ticker"].nunique(),
            panel.shape[1],
        )
        return panel

    def split_feature_sets(self, panel: pd.DataFrame) -> dict:
        """Column lists for with-macro vs without-macro experiments."""
        firm_cols = self._firm_feature_columns(panel)
        macro_cols = self._macro_feature_columns(panel)
        return {
            "target": "target",
            "firm_only": firm_cols,
            "firm_plus_macro": firm_cols + macro_cols,
            "macro_only": macro_cols,
        }

    def save(self, panel: pd.DataFrame, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if path.endswith(".parquet"):
            panel.to_parquet(path, index=False)
        else:
            panel.to_csv(path, index=False)
        logger.info("Wrote panel to %s", path)
