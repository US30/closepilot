# ClosePilot evaluation (seed 42)

## Reconciliation (deterministic cascade, LLM off)

- Auto-match rate (bank lines): **98.0%** (1311/1338)
- Pair precision **99.9%**, recall **99.9%** vs ground truth (1321 true pairs)
- By method: exact=829, many_to_one=10, rules=472
- Exceptions: 55

## Anomaly detection vs injected labels

| anomaly           |   injected |   flagged |   true_pos |   precision |   recall |
|:------------------|-----------:|----------:|-----------:|------------:|---------:|
| price_mismatch    |         25 |        25 |         25 |           1 |        1 |
| no_grn            |         15 |        15 |         15 |           1 |        1 |
| gst_error         |         20 |        20 |         20 |           1 |        1 |
| duplicate_invoice |         18 |        18 |         18 |           1 |        1 |
| split_invoice     |         16 |        16 |         16 |           1 |        1 |
| sod_breach        |          3 |         3 |          3 |           1 |        1 |
| weekend_posting   |         20 |        20 |         20 |           1 |        1 |
| round_amount      |         15 |        15 |         15 |           1 |        1 |
| bank_change       |          4 |         4 |          4 |           1 |        1 |

IsolationForest (advisory): 28 flagged, 23 are injected anomalies (82%). Rule checks were written with knowledge of the injection patterns, so these recall numbers show the checks work, not that they generalise to unseen fraud.

## Cash forecast backtest (last 13 weeks, WAPE, lower is better)

| series   | model         |   wape |
|:---------|:--------------|-------:|
| payments | AutoETS       | 0.2009 |
| payments | Naive         | 0.6358 |
| payments | WindowAverage | 0.1722 |
| receipts | AutoETS       | 0.2059 |
| receipts | Naive         | 0.2531 |
| receipts | WindowAverage | 0.4145 |

Selected: receipts=AutoETS, payments=WindowAverage. Selection uses the same holdout, so the selected WAPE is optimistic.

## Policy RAG (30 clauses, 26 gold questions)

| retrieval   |   questions | hit_at_1   | hit_at_3   |   mrr | queue_kinds_cited_correctly   |
|:------------|------------:|:-----------|:-----------|------:|:------------------------------|
| lexical     |          26 | 18/26      | 23/26      | 0.769 | 9/14                          |
| hybrid      |          26 | 22/26      | 25/26      | 0.913 | 14/14                         |

lexical = TF-IDF only (offline). hybrid = TF-IDF + Gemini embeddings, reciprocal rank fusion. A row labelled lexical in a run with the LLM on means the embedding API failed. The policy documents and the gold questions were written by the same author, so this is a sanity check.

## Text-to-SQL (25 gold questions)

- Reference SQL executes: 25/25
- Model: `gemini-3.5-flash`; questions 16-25; answered by API: 10/10 (0 API errors)
- Strict result-set accuracy (answered): **9/10**
- Lenient (extra columns allowed): **10/10**
