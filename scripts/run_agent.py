#!/usr/bin/env python3
"""
LLM-first agent loop for the macro vs firm revenue-growth case study.

Requires:
  - data/processed/macro_firm_panel.csv (from build_macro_firm_panel.py)
  - OPENROUTER_API_KEY in .env

Usage:
  python scripts/run_agent.py
  python scripts/run_agent.py --max-steps 6
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

from orchestration.llm_orchestrator import LLMOrchestrator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LLM tool-calling agent on macro bake-off")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument(
        "--goal",
        type=str,
        default=(
            "Load the macro-firm panel, profile missingness, apply per-column data-quality "
            "decisions with reasons (impute or drop as appropriate for EDGAR/macro gaps), "
            "then run firm_only vs firm_plus_macro bake-off with rmse, compare macro lift, "
            "export best model, stop."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("run_agent")

    panel = ROOT / "data" / "processed" / "macro_firm_panel.csv"
    if not panel.exists():
        logger.error("Missing %s — run: python scripts/build_macro_firm_panel.py", panel)
        return 1

    orch = LLMOrchestrator()
    logger.info("OpenRouter ping: %s", orch.ping())
    state = orch.run_agent_loop(goal=args.goal, max_steps=args.max_steps)

    history_path = ROOT / "artifacts" / "macro_bakeoff" / "agent_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(state.history, indent=2, default=str))
    logger.info("Wrote agent history to %s", history_path)
    logger.info("Finished=%s export_dir=%s", state.finished, state.export_dir)
    if state.quality_decisions:
        logger.info("Quality decisions applied: %s", len(state.quality_decisions))
    if state.summary:
        logger.info("Verdict: %s", state.summary.get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
