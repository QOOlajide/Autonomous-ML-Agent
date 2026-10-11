# Design tradeoffs

## Universe source: Wikipedia S&P 500 vs iShares S&P 500 ETF holdings CSV

### What we chose
Fetch (once) the [Wikipedia S&P 500 constituents table](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies), keep `Symbol` + `CIK`, freeze to `data/raw/sp500_constituents.csv` (`utf-8-sig`). Later builds reuse the cache (`src/ingest/universe.py`).

### Alternative considered
Download the iShares S&P 500 ETF holdings CSV (e.g. IVV on the BlackRock/iShares site)—a real CSV that tracks the S&P 500—and load that as the ticker list.

### Why Wikipedia (for this pipeline)
The next step is **SEC EDGAR company facts**, which need a **CIK**, not only a ticker.

Wikipedia’s table already provides **Symbol + CIK** in one step:

```text
Wiki → {ticker: cik} → EdgarClient.fetch_fundamentals_panel
```

That matched the existing ingest path with fewer moving parts.

### Why iShares CSV can still be “better” as a file
- Cleaner story: download a CSV, don’t scrape HTML.
- Closer to a traded product that seeks to track the index.
- No Wikipedia 403 / `read_html` fragility.

### Tradeoffs of *not* using the iShares CSV
| Topic | iShares holdings CSV | Wikipedia (what we did) |
|---|---|---|
| Format | Real CSV download | HTML table parse |
| CIK | Usually **not** included → still need SEC `company_tickers.json` (or similar) | **Included** in the table |
| Tickers | Often non-SEC style (dots, share classes); extra normalize/filter (cash, etc.) | Still need normalize (`BRK.B` → `BRK-B`) |
| Universe meaning | Fund holdings (≈ index, small differences) | Index constituent list |
| Official-ness | Fund product file | Community-maintained Wiki page |

### Honest one-liner (interview)
> iShares would’ve been the cleaner *file* download for tickers. We used Wikipedia because it already gave SEC-ready CIKs for EDGAR in one step. If we already had a CIK resolver, freezing an iShares/index CSV and skipping the scrape would be preferable.

### Cache tradeoff (related)
We intentionally **do not** re-download the universe every build. Upside: fixed, reproducible study set. Downside: stale vs live index membership until `--refresh-universe`.
