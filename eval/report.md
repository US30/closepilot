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

## Text-to-SQL (25 gold questions)

- Reference SQL executes: 25/25
- LLM accuracy: not run (no GOOGLE_API_KEY or --no-llm)
