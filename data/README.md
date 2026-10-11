# Data directories

- `raw/` — FRED macros + EDGAR fundamentals from `scripts/build_macro_firm_panel.py`
- `raw/sp500_constituents.csv` — frozen S&P 500 `Symbol`,`CIK` universe (Wikipedia, cached once)
- `processed/` — as-of panel (`target` = revenue growth) + `feature_sets.json`

## Universe (S&P 500)

First time (or to freeze a new list):

```bash
source .venv/bin/activate
python scripts/build_macro_firm_panel.py --refresh-universe --max-tickers 5
```

That prints the first 5 rows, writes `data/raw/sp500_constituents.csv` (`utf-8-sig`), then can continue the build. Later runs **reuse the CSV** (no Wikipedia request) unless you pass `--refresh-universe`.

Dotted share classes are normalized (`BRK.B` → `BRK-B`) for SEC-style tickers.

## Rebuild panel
```bash
source .venv/bin/activate
python scripts/build_macro_firm_panel.py --max-tickers 50   # smoke
python scripts/build_macro_firm_panel.py                   # full cached S&P 500
```

## Run bake-off (firm-only vs firm+macro)
```bash
python scripts/run_macro_bakeoff.py
```

Outputs land in `artifacts/macro_bakeoff/`.

## LLM agent (tool loop)
```bash
python scripts/run_agent.py
```

See `docs.md` for the reframed platform problem (S3 / Databricks / Spark / K8s).
