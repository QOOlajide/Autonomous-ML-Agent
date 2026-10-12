"""
Tools the LLM orchestrator can call for the macro vs firm case study.

Flow: load → profile ugly data → apply LLM quality decisions → bake-off → export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from models.trainer import ModelTrainer
from ui.leaderboard import LeaderboardUI

from .data_quality import apply_quality_decisions, profile_data_quality

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = ROOT / "data" / "processed" / "macro_firm_panel.csv"
DEFAULT_FEATURE_SETS = ROOT / "data" / "processed" / "feature_sets.json"
DEFAULT_ARTIFACTS = ROOT / "artifacts" / "macro_bakeoff"


class AgentState:
    """Shared state across tool calls in one agent run."""

    def __init__(self) -> None:
        self.panel: Optional[pd.DataFrame] = None
        self.feature_sets: Optional[Dict[str, Any]] = None
        self.trainer: Optional[ModelTrainer] = None
        self.results: List[Any] = []
        self.summary: Optional[Dict[str, Any]] = None
        self.export_dir: Optional[str] = None
        self.finished: bool = False
        self.history: List[Dict[str, Any]] = []
        self.quality_profile: Optional[Dict[str, Any]] = None
        self.quality_decisions: List[Dict[str, Any]] = []


def load_panel(
    state: AgentState,
    panel_path: str = "",
    feature_sets_path: str = "",
) -> Dict[str, Any]:
    """Load the processed macro–firm panel and feature set definitions."""
    panel_file = Path(panel_path) if panel_path else DEFAULT_PANEL
    fs_file = Path(feature_sets_path) if feature_sets_path else DEFAULT_FEATURE_SETS
    if not panel_file.exists():
        return {
            "ok": False,
            "error": f"Panel not found: {panel_file}. Run scripts/build_macro_firm_panel.py first.",
        }
    if not fs_file.exists():
        return {"ok": False, "error": f"feature_sets.json not found: {fs_file}"}

    state.panel = pd.read_csv(panel_file)
    state.feature_sets = json.loads(fs_file.read_text())
    state.quality_profile = None
    state.quality_decisions = []
    return {
        "ok": True,
        "rows": int(state.panel.shape[0]),
        "cols": int(state.panel.shape[1]),
        "tickers": sorted(state.panel["ticker"].dropna().unique().tolist())
        if "ticker" in state.panel.columns
        else [],
        "target": state.feature_sets.get("target", "target"),
        "firm_only": state.feature_sets.get("firm_only", []),
        "firm_plus_macro": state.feature_sets.get("firm_plus_macro", []),
        "panel_path": str(panel_file),
    }


def inspect_panel(state: AgentState) -> Dict[str, Any]:
    """Summarize the loaded panel for the LLM."""
    if state.panel is None or state.feature_sets is None:
        return {"ok": False, "error": "No panel loaded. Call load_panel first."}
    df = state.panel
    target = state.feature_sets.get("target", "target")
    out: Dict[str, Any] = {
        "ok": True,
        "rows": len(df),
        "columns": list(df.columns)[:40],
        "firm_only_count": len(state.feature_sets.get("firm_only") or []),
        "firm_plus_macro_count": len(state.feature_sets.get("firm_plus_macro") or []),
    }
    if target in df.columns:
        out["target_describe"] = {
            k: float(v) if hasattr(v, "item") else v
            for k, v in df[target].describe().to_dict().items()
        }
    return out


def profile_missingness(state: AgentState) -> Dict[str, Any]:
    """
    Profile missing / ugly columns for LLM judgment.

    Call this before apply_data_quality_decisions.
    """
    if state.panel is None or state.feature_sets is None:
        return {"ok": False, "error": "No panel loaded. Call load_panel first."}

    feature_cols = list(
        dict.fromkeys(
            (state.feature_sets.get("firm_plus_macro") or [])
            + (state.feature_sets.get("firm_only") or [])
        )
    )
    target = state.feature_sets.get("target", "target")
    profile = profile_data_quality(state.panel, feature_cols=feature_cols, target_col=target)
    state.quality_profile = profile

    # Persist for resume/demo evidence
    out_dir = DEFAULT_ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_quality_profile.json").write_text(json.dumps(profile, indent=2, default=str))

    return {
        "ok": True,
        "n_rows": profile["n_rows"],
        "n_with_missing": profile["n_with_missing"],
        "high_missing_columns": profile["high_missing_columns"],
        "missing_columns": profile["missing_columns"],
        "hints": profile["hints"],
        "allowed_actions": profile["allowed_actions"],
        "decision_schema": profile["decision_schema"],
        "next_step": (
            "Call apply_data_quality_decisions with a decisions list. "
            "Each item: {column, action, reason}. "
            "Reason about EDGAR gaps (e.g. banks often lack gross_profit) and macro sparsity."
        ),
    }


def apply_data_quality_decisions(
    state: AgentState,
    decisions: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Apply LLM-chosen cleaning actions to the in-memory panel before training.
    """
    if state.panel is None or state.feature_sets is None:
        return {"ok": False, "error": "No panel loaded. Call load_panel first."}
    if not decisions:
        return {
            "ok": False,
            "error": "Pass decisions=[{column, action, reason}, ...]. Run profile_missingness first.",
        }

    result = apply_quality_decisions(state.panel, decisions, state.feature_sets)
    state.panel = result.pop("dataframe")
    if result.get("feature_sets"):
        state.feature_sets = result["feature_sets"]
    state.quality_decisions = list(decisions)

    out_dir = DEFAULT_ARTIFACTS
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_quality_decisions.json").write_text(
        json.dumps(
            {"decisions": decisions, "applied": result.get("applied"), "skipped": result.get("skipped")},
            indent=2,
            default=str,
        )
    )

    return {
        "ok": True,
        "rows_before": result["rows_before"],
        "rows_after": result["rows_after"],
        "cols_before": result["cols_before"],
        "cols_after": result["cols_after"],
        "applied": result["applied"],
        "skipped": result["skipped"],
        "dropped_features": result["dropped_features"],
        "firm_only": state.feature_sets.get("firm_only"),
        "firm_plus_macro": state.feature_sets.get("firm_plus_macro"),
    }


def run_bakeoff(
    state: AgentState,
    metric: str = "rmse",
    max_models: int = 4,
    compute_budget: int = 300,
) -> Dict[str, Any]:
    """Train firm_only vs firm_plus_macro models and store results on state."""
    if state.panel is None or state.feature_sets is None:
        return {"ok": False, "error": "No panel loaded. Call load_panel first."}
    if metric not in {"rmse", "mae", "r2"}:
        metric = "rmse"

    state.trainer = ModelTrainer(max_models=max_models, compute_budget=compute_budget)
    state.results = state.trainer.run_bakeoff(
        state.panel,
        state.feature_sets,
        primary_metric=metric,
    )
    state.summary = state.trainer.compare_macro_lift(primary_metric=metric)
    if state.summary is not None:
        state.summary["data_quality_decisions_applied"] = len(state.quality_decisions)

    leaderboard = [
        {
            "model": r.display_name,
            "feature_set": r.feature_set,
            "metrics": r.metrics,
        }
        for r in state.results[:12]
    ]
    return {
        "ok": True,
        "n_models": len(state.results),
        "metric": metric,
        "leaderboard_top": leaderboard,
        "summary": state.summary,
        "quality_decisions_count": len(state.quality_decisions),
    }


def compare_macro_lift(state: AgentState, metric: str = "rmse") -> Dict[str, Any]:
    """Return the with-macro vs without-macro verdict."""
    if state.trainer is None or not state.results:
        return {"ok": False, "error": "No bake-off results. Call run_bakeoff first."}
    state.summary = state.trainer.compare_macro_lift(primary_metric=metric)
    return {"ok": True, "summary": state.summary}


def export_best_model(state: AgentState, output_dir: str = "") -> Dict[str, Any]:
    """Export best model + leaderboard JSON to artifacts/."""
    if state.trainer is None or not state.results:
        return {"ok": False, "error": "No bake-off results. Call run_bakeoff first."}
    out = output_dir or str(DEFAULT_ARTIFACTS)
    best = state.results[0]
    summary = dict(state.summary or {})
    summary["data_quality_decisions"] = state.quality_decisions
    state.trainer.export_pipeline(
        best,
        output_dir=out,
        feature_sets=state.feature_sets,
        summary=summary,
    )
    state.export_dir = out
    LeaderboardUI().display_leaderboard(state.results, primary_metric="rmse")
    if state.summary:
        LeaderboardUI().display_summary(state.summary)
    return {
        "ok": True,
        "best_model": best.display_name,
        "metrics": best.metrics,
        "output_dir": out,
        "quality_decisions_count": len(state.quality_decisions),
    }


def stop(state: AgentState, reason: str = "done") -> Dict[str, Any]:
    """End the agent loop."""
    state.finished = True
    return {"ok": True, "stopped": True, "reason": reason}


TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "name": "load_panel",
        "description": "Load macro_firm_panel.csv and feature_sets.json from disk.",
        "args": {
            "panel_path": "optional path to panel csv",
            "feature_sets_path": "optional path to feature_sets.json",
        },
    },
    {
        "name": "inspect_panel",
        "description": "Summarize loaded panel shape, target, and feature counts.",
        "args": {},
    },
    {
        "name": "profile_missingness",
        "description": (
            "Profile missing/ugly columns (counts, pct, dtype). "
            "Required before apply_data_quality_decisions. "
            "Use results to reason why fields are missing (EDGAR concept gaps, etc.)."
        ),
        "args": {},
    },
    {
        "name": "apply_data_quality_decisions",
        "description": (
            "Apply per-column cleaning decisions before training. "
            "Args.decisions is a list of {column, action, reason}. "
            "Actions: keep|drop_column|impute_median|impute_mean|impute_zero|drop_rows_if_missing."
        ),
        "args": {
            "decisions": "list of {column, action, reason}",
        },
    },
    {
        "name": "run_bakeoff",
        "description": "Train models on firm_only and firm_plus_macro; return leaderboard.",
        "args": {
            "metric": "rmse|mae|r2",
            "max_models": "int, default 4",
            "compute_budget": "seconds, default 300",
        },
    },
    {
        "name": "compare_macro_lift",
        "description": "State whether macro features improved the best model.",
        "args": {"metric": "rmse|mae|r2"},
    },
    {
        "name": "export_best_model",
        "description": "Write best model pickle + leaderboard JSON under artifacts/.",
        "args": {"output_dir": "optional output directory"},
    },
    {
        "name": "stop",
        "description": "Finish the agent run.",
        "args": {"reason": "short reason"},
    },
]


def get_tool_registry(state: AgentState) -> Dict[str, Callable[..., Dict[str, Any]]]:
    return {
        "load_panel": lambda **kwargs: load_panel(state, **kwargs),
        "inspect_panel": lambda **kwargs: inspect_panel(state),
        "profile_missingness": lambda **kwargs: profile_missingness(state),
        "apply_data_quality_decisions": lambda **kwargs: apply_data_quality_decisions(state, **kwargs),
        "run_bakeoff": lambda **kwargs: run_bakeoff(state, **kwargs),
        "compare_macro_lift": lambda **kwargs: compare_macro_lift(state, **kwargs),
        "export_best_model": lambda **kwargs: export_best_model(state, **kwargs),
        "stop": lambda **kwargs: stop(state, **kwargs),
    }
