# Known limitations (intentional MVP)

This document lists evaluation and engineering gaps we **know about** and are **not claiming as solved**. Shipping them undocumented would look careless; documenting them is the honest MVP bar.

## ML evaluation

### 1. Time holdout (bake-off) — partially addressed
`ModelTrainer.run_bakeoff` splits on `period_end`: **train** `< 2024-01-01`, **test** `≥ 2024-01-01` (fixed `DEFAULT_TEST_START`). Same masks for both feature sets.

**Still weak / follow-up:** no validation fold; no group-aware split by `ticker` (same firm can still appear in train and test across different quarters). Generic CSV `DataPreprocessor` still uses random `train_test_split`.

### 2. Pre-split imputation from LLM data-quality step
`apply_quality_decisions` can median/mean-impute using the **full** in-memory panel **before** the bake-off split.

**Why that’s weak:** test-fold information can leak into impute statistics.

**Mitigation already present:** the sklearn `Pipeline` also fits `SimpleImputer` on the **train** fold only during `fit`. Prefer relying on that for training; treat LLM impute as demo/quality UX until we move imputation inside the train fold only.

### 3. Shared row indices across feature sets — addressed for bake-off
Firm-only and firm-plus-macro now share one `train_mask` / `test_mask` from `period_end`.

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

## FRED macros (current design)
- Cadence via `fredapi.Fred.get_series_info` → print `info['frequency']` (mixed D/M/Q expected).
- Ingest uses a **transmission-mapped** feature set (`PIPELINE_FEATURES` in `fred_client.py`).
- Mixed frequencies are **resampled to quarter-end (`QE`).last().ffill()** before modeling features.
- Stationarity: level indices → `*_yoy` (`pct_change(4)`); rates/spreads → `*_diff_yoy` (4Q difference).
- Firm panel still attaches with as-of on `period_start`.

## What we do claim
- Deterministic tools for ingest → quality profile/decisions → firm_only vs firm_plus_macro bake-off.
- Leakage-aware **macro as-of join** and mostly **lagged** firm features in panel construction (see `src/ingest/panel_builder.py`).
- Explicit allowlisted agent tools and persisted run history for auditability.
- Macro cadences come from FRED via `fredapi` + env `FRED_API_KEY` (not hardcoded).
