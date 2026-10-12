# Known limitations (intentional MVP)

This document lists evaluation and engineering gaps we **know about** and are **not claiming as solved**. Shipping them undocumented would look careless; documenting them is the honest MVP bar.

## ML evaluation

### 1. Random train/test split on a panel
`ModelTrainer.run_bakeoff` uses `sklearn.model_selection.train_test_split` on firm×quarter rows.

**Why that’s weak:** future quarters can land in train while past quarters land in test; the same firm can appear on both sides. That is not a time-safe backtest.

**Intended follow-up:** time-based split (e.g. last N years as test) and/or group-aware split by `ticker`.

### 2. Pre-split imputation from LLM data-quality step
`apply_quality_decisions` can median/mean-impute using the **full** in-memory panel **before** the bake-off split.

**Why that’s weak:** test-fold information can leak into impute statistics.

**Mitigation already present:** the sklearn `Pipeline` also fits `SimpleImputer` on the **train** fold only during `fit`. Prefer relying on that for training; treat LLM impute as demo/quality UX until we move imputation inside the train fold only.

### 3. Feature-set splits are not identical row sets
Firm-only and firm-plus-macro each call `train_test_split` on their own `X`. Same `random_state` does **not** guarantee the same test row indices across different column sets.

**Intended follow-up:** draw indices once, reuse for both feature sets.

### 4. No hyperparameter search yet
Model zoo uses fixed hyperparameters and a compute-budget cutoff. The Headstarter-style HPO loop is **not** implemented.

### 5. Claim scope
Results are conditional on the frozen study universe (S&P 500 cache / whatever panel was built) and the QoQ `revenue_growth` target. Do not read a single bake-off as a universal “macros always help” result.

## Agent / SWE

### 6. Fallback path is not autonomy
If the LLM returns unparseable JSON, `_fallback_plan` runs scripted tools with `reason: "fallback"`. Always check `artifacts/macro_bakeoff/agent_history.json` before claiming the LLM drove the run.

### 7. Logging ≠ monitoring
We persist traces and metrics to disk. There is no drift dashboard, alerting, or continuous eval harness yet.

### 8. Streamlit UI
The public Streamlit app still reflects the older generic CSV flow; it is **not** yet wired to the one-button agent + bake-off demo described in planning notes.

## What we do claim
- Deterministic tools for ingest → quality profile/decisions → firm_only vs firm_plus_macro bake-off.
- Leakage-aware **macro as-of join** and mostly **lagged** firm features in panel construction (see `src/ingest/panel_builder.py`).
- Explicit allowlisted agent tools and persisted run history for auditability.
