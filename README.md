# ClosePilot — agentic month-end close copilot

Multi-agent finance-ops system on a synthetic Indian mid-market ledger (Acme Components Pvt Ltd, FY 2025-26, INR, GST/TDS).
Built to show Office-of-CFO style AI work: reconciliation, AP controls, GRC, cash forecasting, and ROI reporting.

**Design rule:** deterministic rules do the bulk of the work (cheap, auditable). The LLM (Gemini) handles only leftover
exceptions, forecast commentary, text-to-SQL and policy answers. Every agent proposal goes to a human approval queue with an
append-only audit log, and each queue item cites the policy clause it breaches (RAG over the policy manual and vendor contract).

```
raw CSVs -> DuckDB -> LangGraph: recon -> ap -> grc -> forecast -> approval queue -> policy citations -> ROI
              Streamlit UI  |  FastAPI  |  MCP server (Claude)  |  CFO chat (text-to-SQL, guarded)  |  Policy Q&A (RAG)
```

| Agent | What it does |
|---|---|
| Recon | Bank-to-GL cascade: exact ref+amount, amount+date window, fuzzy narration, many-to-one subset-sum, then LLM for leftovers |
| AP | PO/GRN/invoice 3-way match, duplicate invoices, GST arithmetic |
| GRC | Split-under-limit, segregation of duties, weekend postings, round amounts, vendor bank-account change, Benford and IsolationForest (advisory) |
| Forecast | 13-week receipts/payments via statsforecast, backtested; open AR/AP due dates overlaid |
| Copilot | NL question -> DuckDB SELECT (keyword guard, no access to label/audit tables), SQL always shown |
| Policy RAG | 30 clause-level chunks from `docs/policies/` (accounting policy, delegation of authority, vendor contract). TF-IDF + Gemini embeddings fused by reciprocal rank; vectors stored and searched in DuckDB. Cites a clause for every queue item and answers policy questions using only retrieved clauses |
| MCP server | 7 tools over stdio (`run_close`, `get_kpis`, `list_queue`, `explain_item`, `ledger_schema`, `query_ledger`, `search_policy`) so Claude can run and question the close. No approve tool: approval stays with a human |

## Run (conda)

```bash
conda env create -f environment.yml        # or: conda create -n closepilot python=3.11 && pip install -r requirements.txt
conda activate closepilot
cp .env.example .env                       # add GOOGLE_API_KEY to enable Gemini features
python data/generate.py --seed 42 --out data/raw
streamlit run app/Home.py                  # click "Run month-end close"
pytest -q                                  # 52 tests, offline: LLM is mocked, temp DB, no quota used
python eval/run_eval.py --no-llm           # writes eval/report.md (drop --no-llm to score text-to-SQL and hybrid retrieval)
uvicorn closepilot.api:app                 # optional API
python -m closepilot.mcp_server            # optional MCP server (stdio)
```

### Use from Claude (MCP)

```bash
claude mcp add closepilot -- /path/to/envs/closepilot/bin/python -m closepilot.mcp_server   # run inside the repo
```

For Claude Desktop add the same command to `claude_desktop_config.json` under `mcpServers`, with `cwd` set to the repo.
Then ask, for example, "Run the close and explain the three largest exceptions with the policy clause for each".
DuckDB allows one writing process per file, so stop the Streamlit app first or point `CLOSEPILOT_DB` at another file.

The pipeline, tests and eval run fully offline without an API key. Gemini features need `GOOGLE_API_KEY`.

## Results (seed 42, see eval/report.md)

- Bank-to-GL auto-match **98.0%**, pair precision/recall 99.9% vs ground truth
- AP checks and controls: 100% recall and precision on all 9 injected anomaly types. Earlier one duplicate invoice was missed (17/18); the cause was a generator bug (a duplicate also got a later anomaly, so it no longer matched its original), fixed in `data/generate.py`, not a detector change.
- Forecast backtest WAPE ≈ 17-21% (receipts, payments), model chosen on the same holdout
- Policy RAG, 26 gold questions over 30 clauses: TF-IDF alone hit@1 18/26, hit@3 23/26, MRR 0.77. Hybrid (TF-IDF + Gemini embeddings) hit@1 **22/26**, hit@3 **25/26**, MRR **0.91**. Correct clause cited for 9/14 queue finding kinds with TF-IDF alone and **14/14** with hybrid.
- CFO chat text-to-SQL (`gemini-3.5-flash`, 25 gold questions): **21/25 strict, 24/25 when extra columns are allowed.** The free tier allows about 20 requests a day, so the 25 questions were scored in two runs: questions 1-15 on 2026-10-02 (12/15 strict, 14/15 lenient) and 16-25 on 2026-10-09 (9/10 strict, 10/10 lenient). The one full miss was an ambiguous gold question (vendor name vs `vendor_id`), reworded since. Results vary between runs: a rerun of questions 1-9 on 2026-10-09 scored 6/9 strict, 8/9 lenient before the quota ended it (it read "in the year" as calendar 2026). `eval/report.md` holds the 16-25 run. Model names and free quotas change, so set `GEMINI_MODEL` in `.env`.
- Estimated ~112 analyst-hours saved per close (≈81% of manual effort) under **editable assumptions** (2 min per bank line, 6 min per invoice, 10 min per exception, 5 min per review item). This is an estimate, not a measurement.

## Limitations (say these in the interview)

- Data is synthetic and the anomaly checks were written knowing the injection patterns. High recall shows the logic works, not that it catches unseen fraud. IsolationForest (unsupervised) finds ~82% injected anomalies among its flags.
- The fuzzy-narration matching stage matches 0 lines on this dataset (the rules stage catches every equal-amount pair first). It is covered by a unit test with crafted data. Writing that test also exposed a bug: matching was case-sensitive, so bank narrations in UPPERCASE never matched GL text; fixed.
- Real ERP exports (Tally/SAP) are messier: partial payments, credit notes, multi-currency, TDS sections. Not modelled.
- Forecast is statistical with an AR/AP overlay, not a full direct-method model.
- Policy documents are synthetic and I wrote both the clauses and the gold questions, so the retrieval scores are a sanity check on 30 chunks, not a benchmark. Without an API key citations use TF-IDF only and 5 of 14 finding kinds get the wrong clause; the UI shows which retrieval mode produced each citation.
- The MCP server has no authentication and runs locally over stdio. `query_ledger` relies on the same keyword guard as the CFO chat.
- LLM paths (exception notes, commentary, text-to-SQL, policy answers) are guarded and optional; their wiring, validation and fallbacks are tested with a mocked LLM, but real-model accuracy is only measured by `eval/run_eval.py` with an API key, and the free tier rate-limits the full 25-question eval.

## Resume bullet (fill in your own run's numbers)

> **ClosePilot — Agentic Month-End Close Copilot.** Built a LangGraph multi-agent system (Gemini, DuckDB, Streamlit) automating bank-to-GL reconciliation, AP 3-way matching, GRC control testing and 13-week cash forecasting on a synthetic Indian ledger (GST/TDS). Rules → fuzzy → LLM cascade auto-matched 98% of bank lines at 99.9% precision, with human-in-the-loop approval and audit trail. Added a RAG layer over accounting policies and the vendor contract (TF-IDF + embeddings in DuckDB, hit@3 25/26) so every exception cites its policy clause, a guarded text-to-SQL chat (21/25 on gold questions), and an MCP server exposing the agents as tools to Claude. ROI dashboard estimating ~110 analyst-hours saved per close.

## Interview pitch (60 seconds)

Month-end close is slow because analysts manually tick bank lines to the ledger and check invoices. I built agents that clear the easy 98% with rules, send only exceptions to an LLM, and queue every proposal for human approval with an audit trail — because finance cannot accept unexplained AI decisions. Each flagged item cites the policy clause it breaches, retrieved from the policy manual, and the whole system is exposed over MCP so Claude can run the close and answer questions about it, but cannot approve anything. The output is framed as ROI: hours saved and INR exposure flagged, with the assumptions editable. For a Practus Office-of-CFO engagement the same skeleton plugs into a client's Tally/SAP export.
