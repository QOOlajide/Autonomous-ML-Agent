"""
CLI / lightweight leaderboard display.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class LeaderboardUI:
    """Print ranked model results (CLI)."""

    def __init__(self, ui_type: str = "cli"):
        self.ui_type = ui_type

    def display_leaderboard(self, trained_models: List[Any], primary_metric: str = "rmse") -> None:
        if not trained_models:
            print("No trained models to display.")
            return

        higher = primary_metric == "r2"
        ranked = sorted(
            trained_models,
            key=lambda m: m.metrics.get(primary_metric, float("inf")),
            reverse=higher,
        )

        print("\n=== Model Leaderboard ===")
        print(f"{'rank':<5} {'model':<40} {'rmse':>10} {'mae':>10} {'r2':>10}")
        for i, m in enumerate(ranked, 1):
            name = getattr(m, "display_name", f"{m.name}[{m.feature_set}]")
            metrics = m.metrics
            print(
                f"{i:<5} {name:<40} "
                f"{metrics.get('rmse', float('nan')):>10.4f} "
                f"{metrics.get('mae', float('nan')):>10.4f} "
                f"{metrics.get('r2', float('nan')):>10.4f}"
            )
        best = ranked[0]
        print(
            f"\nBest by {primary_metric}: "
            f"{getattr(best, 'display_name', best.name)} "
            f"-> {best.metrics.get(primary_metric):.4f}"
        )

    def display_summary(self, summary: Optional[dict]) -> None:
        if not summary:
            return
        print("\n=== Macro lift summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")
