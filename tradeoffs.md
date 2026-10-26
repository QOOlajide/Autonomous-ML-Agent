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

---

## Agent execution: allowlisted tools vs sandbox codegen (Daytona / Gemini code execution)

### What we chose (for now)
**Allowlisted tool-calling** over local Python (`LLMOrchestrator` → `tools.py` → deterministic helpers like `load_panel`, `data_quality.apply_quality_decisions`). The LLM picks `tool` + `args` (+ `reason`); it does **not** write/run arbitrary training or preprocessing code.

### Alternatives considered
1. **Daytona sandboxes** — run AI-generated code in an isolated VM/container ([isolation](https://www.daytona.io/docs/en/isolation/), [process/code execution](https://www.daytona.io/docs/en/process-code-execution/), [security guidance](https://www.mintlify.com/daytonaio/daytona/concepts/security)).
2. **Gemini code execution** — model generates and runs Python inside Google’s managed sandbox ([Gemini code execution docs](https://ai.google.dev/gemini-api/docs/code-execution); ~30s runtime, limited environment).

### Why allowlisted tools for this product
| Concern | Allowlisted tools (chosen) | Daytona / Gemini code exec |
|---|---|---|
| Demo lifecycle | Fixed: load → quality → bake-off → export | Open-ended scripts each run |
| Audit / UI “proof” | Stable JSON `result` shapes per tool | Harder to map to fixed Insights panes |
| Reproducibility | Same functions every time | Generated code can drift |
| Security surface | No arbitrary code on host | Sandbox required precisely *because* code is untrusted |
| Infra / time | No sandbox fleet | Extra service, latency, ops |

Daytona/Gemini are the right hammer when the agent **must** invent and execute code. Our case study is a **known pipeline**; we need a **glanceable proof trail** in Streamlit (right column) more than open-ended codegen.

### UI product rule (locked with this choice)
- Left column shows a `st.status` **only for tools the agent actually called** (history-driven; no pre-drawn fake checklist).
- A status becomes **complete** only when that tool’s `result` carries **physical proof** (e.g. `load_panel` → `preview` / `panel_path` shown as `df.head()` in Insights). Marking complete from “tool name appeared” alone is insufficient.

### Honest one-liner (interview)
> We use tool-calling, not Daytona/Gemini code execution, so every step has a deterministic function and a stable proof payload for the UI. Sandboxes matter when you run untrusted generated code; here the LLM only chooses among audited tools.
