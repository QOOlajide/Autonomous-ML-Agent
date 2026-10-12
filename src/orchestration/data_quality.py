"""
Data-quality profiling and policy application for the agent.

The LLM judges missing/ugly columns; this module profiles and applies decisions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns the agent must never drop casually
PROTECTED = {
    "ticker",
    "cik",
    "period_end",
    "period_start",
    "macro_asof_date",
    "fy",
    "fp",
    "form",
    "target",
    "revenue_growth",
}

ALLOWED_ACTIONS = {
    "keep",
    "drop_column",
    "impute_median",
    "impute_mean",
    "impute_zero",
    "drop_rows_if_missing",
}


def profile_data_quality(
    df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    target_col: str = "target",
) -> Dict[str, Any]:
    """
    Build a missingness / dtype report the LLM can reason over.
    """
    cols = list(feature_cols) if feature_cols else [
        c for c in df.columns if c not in PROTECTED or c == target_col
    ]
    # Always include target + feature set columns when provided
    report_cols = []
    for c in list(cols) + ([target_col] if target_col in df.columns else []):
        if c in df.columns and c not in report_cols:
            report_cols.append(c)

    n = len(df)
    columns: List[Dict[str, Any]] = []
    for c in report_cols:
        s = df[c]
        miss = int(s.isna().sum())
        miss_pct = float(miss / n) if n else 0.0
        entry: Dict[str, Any] = {
            "column": c,
            "dtype": str(s.dtype),
            "missing_count": miss,
            "missing_pct": round(miss_pct, 4),
            "nunique": int(s.nunique(dropna=True)),
            "protected": c in PROTECTED,
        }
        if pd.api.types.is_numeric_dtype(s):
            entry["mean"] = float(s.mean()) if s.notna().any() else None
            entry["std"] = float(s.std()) if s.notna().sum() > 1 else None
            entry["min"] = float(s.min()) if s.notna().any() else None
            entry["max"] = float(s.max()) if s.notna().any() else None
        columns.append(entry)

    columns_sorted = sorted(columns, key=lambda x: x["missing_pct"], reverse=True)
    high_missing = [c for c in columns_sorted if c["missing_pct"] >= 0.2]
    any_missing = [c for c in columns_sorted if c["missing_pct"] > 0]

    hints = []
    if high_missing:
        hints.append(
            "Columns with >=20% missing often warrant drop_column or careful imputation; "
            "explain likely EDGAR concept gaps (banks lack gross_profit, etc.)."
        )
    if any_missing:
        hints.append(
            "Prefer impute_median for numeric firm/macro features used in training; "
            "use drop_rows_if_missing only for target or critical keys."
        )
    else:
        hints.append("No missing values in profiled columns.")

    return {
        "ok": True,
        "n_rows": n,
        "n_profiled_columns": len(columns_sorted),
        "n_with_missing": len(any_missing),
        "high_missing_columns": high_missing[:25],
        "missing_columns": any_missing[:40],
        "all_columns_preview": columns_sorted[:40],
        "hints": hints,
        "allowed_actions": sorted(ALLOWED_ACTIONS),
        "decision_schema": {
            "column": "column name",
            "action": "one of allowed_actions",
            "reason": "why this action given missingness / domain",
        },
    }


def apply_quality_decisions(
    df: pd.DataFrame,
    decisions: List[Dict[str, Any]],
    feature_sets: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Apply LLM decisions to a copy of the panel.

    Returns updated dataframe info + possibly updated feature_sets (dropped cols removed).
    """
    out = df.copy()
    applied: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    dropped_features: List[str] = []

    for raw in decisions or []:
        col = raw.get("column")
        action = (raw.get("action") or "").strip()
        reason = raw.get("reason", "")
        if not col or col not in out.columns:
            skipped.append({"column": col, "action": action, "error": "missing column"})
            continue
        if action not in ALLOWED_ACTIONS:
            skipped.append({"column": col, "action": action, "error": "invalid action"})
            continue
        if col in PROTECTED and action in {"drop_column", "drop_rows_if_missing"} and col == "target":
            # allow drop_rows_if_missing on target; forbid drop_column on protected ids/target
            if action == "drop_column":
                skipped.append({"column": col, "action": action, "error": "protected column"})
                continue
        if col in PROTECTED and action == "drop_column":
            skipped.append({"column": col, "action": action, "error": "protected column"})
            continue

        before_miss = int(out[col].isna().sum())

        if action == "keep":
            pass
        elif action == "drop_column":
            out = out.drop(columns=[col])
            dropped_features.append(col)
        elif action == "impute_median":
            if not pd.api.types.is_numeric_dtype(out[col]):
                skipped.append({"column": col, "action": action, "error": "not numeric"})
                continue
            out[col] = out[col].fillna(out[col].median())
        elif action == "impute_mean":
            if not pd.api.types.is_numeric_dtype(out[col]):
                skipped.append({"column": col, "action": action, "error": "not numeric"})
                continue
            out[col] = out[col].fillna(out[col].mean())
        elif action == "impute_zero":
            out[col] = out[col].fillna(0)
        elif action == "drop_rows_if_missing":
            out = out[out[col].notna()].copy()

        applied.append(
            {
                "column": col,
                "action": action,
                "reason": reason,
                "missing_before": before_miss,
                "missing_after": int(out[col].isna().sum()) if col in out.columns else 0,
            }
        )

    updated_fs = None
    if feature_sets is not None:
        updated_fs = dict(feature_sets)
        for key in ("firm_only", "firm_plus_macro", "macro_only"):
            if key in updated_fs and isinstance(updated_fs[key], list):
                updated_fs[key] = [c for c in updated_fs[key] if c in out.columns]

    return {
        "ok": True,
        "rows_before": len(df),
        "rows_after": len(out),
        "cols_before": df.shape[1],
        "cols_after": out.shape[1],
        "applied": applied,
        "skipped": skipped,
        "dropped_features": dropped_features,
        "dataframe": out,
        "feature_sets": updated_fs,
    }
