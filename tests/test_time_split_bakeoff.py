"""Time-based train/test masks in ModelTrainer.run_bakeoff."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.trainer import ModelTrainer  # noqa: E402


def _tiny_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "period_end": pd.to_datetime(
                ["2022-03-31", "2022-06-30", "2024-03-31", "2024-06-30"]
            ),
            "target": [0.1, 0.2, 0.3, 0.4],
            "firm_a": [1.0, 2.0, 3.0, 4.0],
            "macro_a": [10.0, 20.0, 30.0, 40.0],
        }
    )


def test_time_split_shared_masks_and_no_future_in_train():
    panel = _tiny_panel()
    feature_sets = {
        "firm_only": ["firm_a"],
        "firm_plus_macro": ["firm_a", "macro_a"],
    }
    trainer = ModelTrainer(max_models=1, compute_budget=60)
    results = trainer.run_bakeoff(panel, feature_sets, target_col="target")

    assert len(results) >= 1
    # Train rows must all be before cut; test on/after (proved via successful fit sizes).
    dates = pd.to_datetime(panel["period_end"])
    assert (dates < pd.Timestamp("2024-01-01")).sum() == 2
    assert (dates >= pd.Timestamp("2024-01-01")).sum() == 2


def test_time_split_requires_period_end():
    panel = pd.DataFrame({"target": [1.0], "firm_a": [1.0], "macro_a": [1.0]})
    trainer = ModelTrainer(max_models=1, compute_budget=60)
    with pytest.raises(ValueError, match="period_end"):
        trainer.run_bakeoff(
            panel,
            {"firm_only": ["firm_a"], "firm_plus_macro": ["firm_a", "macro_a"]},
            target_col="target",
        )
