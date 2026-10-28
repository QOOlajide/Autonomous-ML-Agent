"""
Model trainer: bake-off firm-only vs firm+macro feature sets.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import config

logger = logging.getLogger(__name__)


@dataclass
class TrainedModel:
    name: str
    feature_set: str
    metrics: Dict[str, float]
    params: Dict[str, Any] = field(default_factory=dict)
    feature_names: List[str] = field(default_factory=list)
    importances: Dict[str, float] = field(default_factory=dict)
    estimator: Any = field(default=None, repr=False)

    @property
    def display_name(self) -> str:
        return f"{self.name}[{self.feature_set}]"


class ModelTrainer:
    """Train curated regressors; compare firm_only vs firm_plus_macro."""

    # Holdout: train on period_end before this date; test on or after.
    DEFAULT_TEST_START = "2024-01-01"

    def __init__(
        self,
        max_models: Optional[int] = None,
        compute_budget: Optional[int] = None,
        random_state: int = 42,
    ):
        self.max_models = max_models or config.model.max_models_to_train
        self.compute_budget = compute_budget or config.model.max_compute_budget
        self.random_state = random_state
        self.test_start = pd.Timestamp(self.DEFAULT_TEST_START)
        self.results: List[TrainedModel] = []

    def _model_zoo(self) -> Dict[str, Any]:
        return {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0),
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": GradientBoostingRegressor(random_state=self.random_state),
            "knn": KNeighborsRegressor(n_neighbors=7),
            "mlp": MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=400,
                random_state=self.random_state,
            ),
        }

    @staticmethod
    def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }

    @staticmethod
    def _importances(estimator: Pipeline, feature_names: Sequence[str]) -> Dict[str, float]:
        model = estimator.named_steps.get("model")
        if model is None:
            return {}
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
        elif hasattr(model, "coef_"):
            vals = np.abs(np.asarray(model.coef_).ravel())
        else:
            return {}
        return {
            name: float(v)
            for name, v in sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)
        }

    def _fit_one(
        self,
        name: str,
        feature_set: str,
        estimator: Any,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ) -> TrainedModel:
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics = self._scores(y_test.to_numpy(), pred)
        return TrainedModel(
            name=name,
            feature_set=feature_set,
            metrics=metrics,
            params=getattr(estimator, "get_params", lambda: {})(),
            feature_names=list(X_train.columns),
            importances=self._importances(pipe, list(X_train.columns)),
            estimator=pipe,
        )

    def run_bakeoff(
        self,
        panel: pd.DataFrame,
        feature_sets: Dict[str, Any],
        target_col: str = "target",
        primary_metric: str = "rmse",
    ) -> List[TrainedModel]:
        if target_col not in panel.columns:
            raise ValueError(f"Missing target column: {target_col}")
        if "period_end" not in panel.columns:
            raise ValueError("Missing period_end column for time-based split")

        sets_to_run = {
            "firm_only": list(feature_sets.get("firm_only") or []),
            "firm_plus_macro": list(feature_sets.get("firm_plus_macro") or []),
        }
        for label, cols in sets_to_run.items():
            missing = [c for c in cols if c not in panel.columns]
            if missing:
                raise ValueError(f"Feature set {label} missing columns: {missing}")
            if not cols:
                raise ValueError(f"Feature set {label} is empty")

        cut = self.test_start
        dates = pd.to_datetime(panel["period_end"])
        train_mask = dates < cut
        test_mask = dates >= cut
        if not train_mask.any():
            raise ValueError(f"No train rows with period_end < {cut.date()}")
        if not test_mask.any():
            raise ValueError(f"No test rows with period_end >= {cut.date()}")

        y = panel[target_col]
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]
        logger.info(
            "Time split period_end | cut=%s | train=%s test=%s",
            cut.date(),
            int(train_mask.sum()),
            int(test_mask.sum()),
        )

        start = time.time()
        model_names = list(self._model_zoo().keys())[: max(1, self.max_models)]
        results: List[TrainedModel] = []

        for feature_set, cols in sets_to_run.items():
            X = panel[cols]
            X_train = X.loc[train_mask]
            X_test = X.loc[test_mask]
            logger.info(
                "Feature set=%s | features=%s | train=%s test=%s",
                feature_set,
                len(cols),
                len(X_train),
                len(X_test),
            )
            for name in model_names:
                if time.time() - start > self.compute_budget:
                    logger.warning("Compute budget hit; stopping bake-off early")
                    break
                logger.info("Training %s on %s", name, feature_set)
                try:
                    est = self._model_zoo()[name]
                    tm = self._fit_one(name, feature_set, est, X_train, X_test, y_train, y_test)
                    results.append(tm)
                    logger.info(
                        "  %s rmse=%.4f mae=%.4f r2=%.4f",
                        tm.display_name,
                        tm.metrics["rmse"],
                        tm.metrics["mae"],
                        tm.metrics["r2"],
                    )
                except Exception as exc:
                    logger.error("Failed %s on %s: %s", name, feature_set, exc)

        higher_is_better = primary_metric == "r2"
        results.sort(
            key=lambda m: m.metrics.get(primary_metric, float("inf")),
            reverse=higher_is_better,
        )
        self.results = results
        return results

    def compare_macro_lift(self, primary_metric: str = "rmse") -> Dict[str, Any]:
        if not self.results:
            return {"error": "No results; run run_bakeoff first"}

        by_set: Dict[str, List[TrainedModel]] = {"firm_only": [], "firm_plus_macro": []}
        for m in self.results:
            by_set.setdefault(m.feature_set, []).append(m)

        def best(ms: List[TrainedModel]) -> Optional[TrainedModel]:
            if not ms:
                return None
            higher = primary_metric == "r2"
            return sorted(ms, key=lambda x: x.metrics[primary_metric], reverse=higher)[0]

        best_firm = best(by_set.get("firm_only", []))
        best_macro = best(by_set.get("firm_plus_macro", []))

        def avg(ms: List[TrainedModel]) -> Optional[float]:
            if not ms:
                return None
            return float(np.mean([m.metrics[primary_metric] for m in ms]))

        summary: Dict[str, Any] = {
            "primary_metric": primary_metric,
            "best_firm_only": None
            if not best_firm
            else {"model": best_firm.display_name, "metrics": best_firm.metrics},
            "best_firm_plus_macro": None
            if not best_macro
            else {"model": best_macro.display_name, "metrics": best_macro.metrics},
            "avg_firm_only": avg(by_set.get("firm_only", [])),
            "avg_firm_plus_macro": avg(by_set.get("firm_plus_macro", [])),
        }

        if best_firm and best_macro:
            if primary_metric == "r2":
                lift = best_macro.metrics[primary_metric] - best_firm.metrics[primary_metric]
                helps = lift > 0
            else:
                lift = best_firm.metrics[primary_metric] - best_macro.metrics[primary_metric]
                helps = lift > 0
            summary["macro_helps_best_model"] = bool(helps)
            summary["best_model_lift"] = float(lift)
            summary["verdict"] = (
                "Macro features improved the best-model score."
                if helps
                else "Macro features did not improve the best-model score."
            )
        return summary

    def export_pipeline(
        self,
        model: TrainedModel,
        output_dir: str,
        data_pipeline: Any = None,
        feature_sets: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> Path:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "best_model.pkl", "wb") as f:
            pickle.dump(model.estimator, f)

        meta = {
            "name": model.name,
            "feature_set": model.feature_set,
            "metrics": model.metrics,
            "feature_names": model.feature_names,
            "importances": model.importances,
            "feature_sets": feature_sets,
            "summary": summary,
        }
        (out / "best_model_meta.json").write_text(json.dumps(meta, indent=2))
        leaderboard = [
            {"name": m.name, "feature_set": m.feature_set, "metrics": m.metrics}
            for m in self.results
        ]
        (out / "leaderboard.json").write_text(json.dumps(leaderboard, indent=2))
        logger.info("Exported best model to %s", out)
        return out

    def train_models(
        self,
        data: Any,
        orchestrator: Any = None,
        target_metric: str = "rmse",
        feature_sets: Optional[Dict[str, Any]] = None,
    ) -> List[TrainedModel]:
        if isinstance(data, pd.DataFrame):
            if not feature_sets:
                raise ValueError("feature_sets required when data is a panel DataFrame")
            metric = target_metric if target_metric in {"rmse", "mae", "r2"} else "rmse"
            return self.run_bakeoff(data, feature_sets, primary_metric=metric)

        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]
        results: List[TrainedModel] = []
        for name, est in list(self._model_zoo().items())[: self.max_models]:
            tm = self._fit_one(name, "legacy", est, X_train, X_test, y_train, y_test)
            results.append(tm)
        self.results = results
        return results
