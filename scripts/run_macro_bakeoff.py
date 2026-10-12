#!/usr/bin/env python3
"""
Step 2 — Autonomous bake-off: firm-only vs firm+macro models.

Reads the panel from Step 1 and trains the curated model suite on both
feature sets, then exports the best pipeline + leaderboard.

Usage (from repo root):
  python scripts/run_macro_bakeoff.py
  python scripts/run_macro_bakeoff.py --metric r2 --max-models 4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from models.trainer import ModelTrainer
from ui.leaderboard import LeaderboardUI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run firm-only vs firm+macro bake-off")
    p.add_argument(
        "--panel",
        type=str,
        default=str(ROOT / "data" / "processed" / "macro_firm_panel.csv"),
    )
    p.add_argument(
        "--feature-sets",
        type=str,
        default=str(ROOT / "data" / "processed" / "feature_sets.json"),
    )
    p.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mae", "r2"])
    p.add_argument("--max-models", type=int, default=6)
    p.add_argument("--compute-budget", type=int, default=600)
    p.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "artifacts" / "macro_bakeoff"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("bakeoff")

    panel_path = Path(args.panel)
    fs_path = Path(args.feature_sets)
    if not panel_path.exists():
        logger.error("Panel not found: %s — run scripts/build_macro_firm_panel.py first", panel_path)
        return 1
    if not fs_path.exists():
        logger.error("feature_sets.json not found: %s", fs_path)
        return 1

    panel = pd.read_csv(panel_path)
    feature_sets = json.loads(fs_path.read_text())
    logger.info("Loaded panel %s rows x %s cols", panel.shape[0], panel.shape[1])
    logger.info("Feature sets: %s", feature_sets)

    trainer = ModelTrainer(max_models=args.max_models, compute_budget=args.compute_budget)
    results = trainer.run_bakeoff(panel, feature_sets, primary_metric=args.metric)
    summary = trainer.compare_macro_lift(primary_metric=args.metric)

    ui = LeaderboardUI(ui_type="cli")
    ui.display_leaderboard(results, primary_metric=args.metric)
    ui.display_summary(summary)

    if not results:
        logger.error("No models trained successfully")
        return 1

    best = results[0]
    trainer.export_pipeline(
        best,
        output_dir=args.output_dir,
        feature_sets=feature_sets,
        summary=summary,
    )
    logger.info("Best model: %s", best.display_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
