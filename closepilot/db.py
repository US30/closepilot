"""DuckDB access: load raw CSVs, approval queue, append-only audit log."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "closepilot.duckdb"
RAW = ROOT / "data" / "raw"
TABLES = ["vendors", "pos", "grns", "invoices", "payments", "ar_invoices", "gl", "bank", "recon_truth", "labels"]


def connect(path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Default path is data/closepilot.duckdb, overridable with CLOSEPILOT_DB (tests use a temp file)."""
    return duckdb.connect(str(path or os.getenv("CLOSEPILOT_DB") or DB_PATH))


def init_db(con: duckdb.DuckDBPyConnection, raw: Path = RAW) -> None:
    for t in TABLES:
        con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_csv_auto('{raw / (t + '.csv')}')")
    con.execute("""
        CREATE TABLE IF NOT EXISTS approval_queue (
            item_id VARCHAR PRIMARY KEY, agent VARCHAR, kind VARCHAR, summary VARCHAR,
            payload VARCHAR, exposure_inr DOUBLE, confidence DOUBLE,
            status VARCHAR DEFAULT 'pending', created_at TIMESTAMP)""")
    con.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            ts TIMESTAMP, actor VARCHAR, action VARCHAR, item_id VARCHAR, before VARCHAR, after VARCHAR)""")


def audit(con, actor: str, action: str, item_id: str, before: dict | None, after: dict | None) -> None:
    con.execute("INSERT INTO audit_log VALUES (?,?,?,?,?,?)",
                [datetime.now(), actor, action, item_id, json.dumps(before, default=str), json.dumps(after, default=str)])


def enqueue(con, item_id: str, agent: str, kind: str, summary: str, payload: dict,
            exposure_inr: float = 0.0, confidence: float = 1.0) -> None:
    con.execute("INSERT OR REPLACE INTO approval_queue VALUES (?,?,?,?,?,?,?, 'pending', ?)",
                [item_id, agent, kind, summary, json.dumps(payload, default=str), exposure_inr, confidence, datetime.now()])
    audit(con, agent, "proposed", item_id, None, {"kind": kind, "summary": summary})


def decide(con, item_id: str, user: str, decision: str) -> None:
    assert decision in ("approved", "rejected")
    row = con.execute("SELECT status FROM approval_queue WHERE item_id=?", [item_id]).fetchone()
    con.execute("UPDATE approval_queue SET status=? WHERE item_id=?", [decision, item_id])
    audit(con, user, decision, item_id, {"status": row[0] if row else None}, {"status": decision})


if __name__ == "__main__":
    c = connect()
    init_db(c)
    print({t: c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in TABLES})
