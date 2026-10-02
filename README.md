# ClosePilot — agentic month-end close copilot

Multi-agent finance-ops system on a synthetic Indian mid-market ledger (Acme Components Pvt Ltd, FY 2025-26, INR, GST/TDS).
Built to show Office-of-CFO style AI work: reconciliation, AP controls, GRC, cash forecasting, and ROI reporting.

**Design rule:** deterministic rules do the bulk of the work (cheap, auditable). The LLM (Gemini) handles only leftover
exceptions, forecast commentary and text-to-SQL. Every agent proposal goes to a human approval queue with an append-only audit log.

```
raw CSVs -> DuckDB -> LangGraph: recon -> ap -> grc -> forecast -> approval queue -> ROI
                                                         Streamlit UI  |  FastAPI  |  CFO chat (text-to-SQL, guarded)
```

| Agent | What it does |
|---|---|
| Recon | Bank-to-GL cascade: exact ref+amount, amount+date window, fuzzy narration, many-to-one subset-sum, then LLM for leftovers |
| AP | PO/GRN/invoice 3-way match, duplicate invoices, GST arithmetic |
| GRC | Split-under-limit, segregation of duties, weekend postings, round amounts, vendor bank-account change, Benford and IsolationForest (advisory) |
| Forecast | 13-week receipts/payments via statsforecast, backtested; open AR/AP due dates overlaid |
| Copilot | NL question -> DuckDB SELECT (keyword guard, no access to label/audit tables), SQL always shown |

## Run (conda)

```bash
conda env create -f environment.yml        # or: conda create -n closepilot python=3.11 && pip install -r requirements.txt
conda activate closepilot
cp .env.example .env                       # add GOOGLE_API_KEY to enable Gemini features
python data/generate.py --seed 42 --out data/raw
streamlit run app/Home.py                  # click "Run month-end close"
pytest -q                                  # 42 tests, offline: LLM is mocked, temp DB, no quota used
python eval/run_eval.py --no-llm           # writes eval/report.md (drop --no-llm to score text-to-SQL)
uvicorn closepilot.api:app                 # optional API
```

The pipeline, tests and eval run fully offline without an API key. Gemini features need `GOOGLE_API_KEY`.

## Results (seed 42, see eval/report.md)

- Bank-to-GL auto-match **98.0%**, pair precision/recall 99.9% vs ground truth
- AP checks and controls: 100% recall and precision on all 9 injected anomaly types. Earlier one duplicate invoice was missed (17/18); the cause was a generator bug (a duplicate also got a later anomaly, so it no longer matched its original), fixed in `data/generate.py`, not a detector change.
- Forecast backtest WAPE ≈ 17-21% (receipts, payments), model chosen on the same holdout
- CFO chat text-to-SQL (`gemini-3.5-flash`, 25 gold questions): **partial run, 15 answered before the free-tier quota (429) stopped it.** 12/15 strict, 14/15 when extra columns are allowed. The one strict-and-lenient miss was an ambiguous gold question (vendor name vs `vendor_id`); the gold wording is now clarified, so that run used the old wording. The other 10 questions are not yet scored. Model names and free quotas change, so set `GEMINI_MODEL` in `.env`.
- Estimated ~112 analyst-hours saved per close (≈81% of manual effort) under **editable assumptions** (2 min per bank line, 6 min per invoice, 10 min per exception, 5 min per review item). This is an estimate, not a measurement.

## Limitations (say these in the interview)

- Data is synthetic and the anomaly checks were written knowing the injection patterns. High recall shows the logic works, not that it catches unseen fraud. IsolationForest (unsupervised) finds ~82% injected anomalies among its flags.
- The fuzzy-narration matching stage matches 0 lines on this dataset (the rules stage catches every equal-amount pair first). It is covered by a unit test with crafted data. Writing that test also exposed a bug: matching was case-sensitive, so bank narrations in UPPERCASE never matched GL text; fixed.
- Real ERP exports (Tally/SAP) are messier: partial payments, credit notes, multi-currency, TDS sections. Not modelled.
- Forecast is statistical with an AR/AP overlay, not a full direct-method model.
- LLM paths (exception notes, commentary, text-to-SQL) are guarded and optional; their wiring, validation and fallbacks are tested with a mocked LLM, but real-model accuracy is only measured by `eval/run_eval.py` with an API key, and the free tier rate-limits the full 25-question eval.

## Resume bullet (fill in your own run's numbers)

> **ClosePilot — Agentic Month-End Close Copilot.** Built a LangGraph multi-agent system (Gemini, DuckDB, Streamlit) automating bank-to-GL reconciliation, AP 3-way matching, GRC control testing and 13-week cash forecasting on a synthetic Indian ledger (GST/TDS). Rules → fuzzy → LLM cascade auto-matched 98% of bank lines at 99.9% precision; controls flagged injected frauds (split invoices, SoD, duplicate and price-mismatch invoices) with human-in-the-loop approval and audit trail; ROI dashboard estimating ~110 analyst-hours saved per close.

## Interview pitch (60 seconds)

Month-end close is slow because analysts manually tick bank lines to the ledger and check invoices. I built agents that clear the easy 98% with rules, send only exceptions to an LLM, and queue every proposal for human approval with an audit trail — because finance cannot accept unexplained AI decisions. The output is framed as ROI: hours saved and INR exposure flagged, with the assumptions editable. For a Practus Office-of-CFO engagement the same skeleton plugs into a client's Tally/SAP export.
