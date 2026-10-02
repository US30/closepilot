"""CFO copilot: natural-language question -> guarded DuckDB SQL -> answer. SQL is always shown."""
from __future__ import annotations

import re

import pandas as pd
from pydantic import BaseModel

from closepilot import llm

# Ground truth / internal tables are never exposed to the model.
EXPOSED = ["vendors", "pos", "grns", "invoices", "payments", "ar_invoices", "gl", "bank"]
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|pragma|install|load|export|"
                       r"call|set|read_\w+|glob|labels|recon_truth|audit_log|approval_queue)\b|https?:|;", re.I)
MAX_ROWS = 200

FEW_SHOT = """
Q: Total unpaid payables?
SQL: SELECT ROUND(SUM(net_payable),2) AS unpaid FROM invoices WHERE invoice_id NOT IN (SELECT invoice_id FROM payments)
Q: Top 5 vendors by spend?
SQL: SELECT v.name, ROUND(SUM(p.amount),2) AS spend FROM payments p JOIN vendors v USING (vendor_id) GROUP BY 1 ORDER BY 2 DESC LIMIT 5
Q: Which vendors had payments over 5 lakh without a GRN?
SQL: SELECT DISTINCT v.name, p.amount FROM payments p JOIN invoices i USING (invoice_id) JOIN vendors v ON v.vendor_id = p.vendor_id LEFT JOIN grns g ON g.po_id = i.po_id WHERE p.amount > 500000 AND g.grn_id IS NULL
"""


class SQLQuery(BaseModel):
    sql: str


def schema_text(con) -> str:
    rows = con.execute("""SELECT table_name, column_name, data_type FROM information_schema.columns
                          WHERE table_name IN (%s) ORDER BY table_name, ordinal_position"""
                       % ",".join(f"'{t}'" for t in EXPOSED)).fetchall()
    tables: dict[str, list[str]] = {}
    for t, c, d in rows:
        tables.setdefault(t, []).append(f"{c} {d}")
    return "\n".join(f"{t}({', '.join(cols)})" for t, cols in tables.items())


def guard(sql: str) -> str:
    s = sql.strip().rstrip(";").strip()
    if not re.match(r"^(select|with)\b", s, re.I):
        raise ValueError("Only SELECT queries are allowed")
    if FORBIDDEN.search(s):
        raise ValueError("Query uses a forbidden keyword, table or function")
    if not re.search(r"\blimit\s+\d+\s*$", s, re.I):
        s = f"SELECT * FROM ({s}) LIMIT {MAX_ROWS}"
    return s


def ask(con, question: str, summarize: bool = True) -> dict:
    if not llm.available():
        return dict(ok=False, error="Set GOOGLE_API_KEY to enable the copilot.", sql=None, df=None, answer=None)
    as_of = con.execute("SELECT MAX(CAST(txn_date AS DATE)) FROM bank").fetchone()[0]
    prompt = (f"Write one DuckDB SELECT query answering the question. Amounts are INR. Treat {as_of} "
              f"(last date in the data) as today. Tables:\n{schema_text(con)}\nExamples:{FEW_SHOT}\nQ: {question}")
    err = None
    for _ in range(2):
        q = llm.structured(prompt + (f"\nPrevious attempt failed: {err}. Fix it." if err else ""), SQLQuery)
        if q is None:
            return dict(ok=False, error="LLM unavailable", sql=None, df=None, answer=None)
        try:
            safe = guard(q.sql)
            df = con.execute(safe).df()
            break
        except Exception as e:  # guard rejection or SQL error -> one retry with the message
            err = str(e)
    else:
        return dict(ok=False, error=err, sql=q.sql, df=None, answer=None)
    if not summarize:
        return dict(ok=True, error=None, sql=safe, df=df, answer="")
    ans = llm.text(f"Answer the CFO's question in 2 sentences using only this result (INR).\nQ: {question}\n"
                   f"Result:\n{df.head(20).to_string(index=False)}") or ""
    return dict(ok=True, error=None, sql=safe, df=df, answer=ans)


def run_sql(con, sql: str) -> pd.DataFrame:
    return con.execute(guard(sql)).df()
