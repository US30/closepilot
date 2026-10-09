"""MCP server: exposes ClosePilot to MCP clients (Claude Desktop, Claude Code) over stdio.

Run: python -m closepilot.mcp_server

The client's model can run the close, read results, query the ledger and look up policy.
There is deliberately no approve/reject tool: approval stays with a human in the UI.
"""
from __future__ import annotations

import contextlib
import json
import sys
from typing import Any

import duckdb
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from closepilot import db, graph, rag
from closepilot.agents import copilot

mcp = MCPServer("closepilot", instructions=(
    "Month-end close copilot on a synthetic Indian ledger (INR). Call run_close first if get_kpis says there is no run. "
    "Use ledger_schema before query_ledger. Cite policy clause ids from search_policy or explain_item. "
    "You cannot approve or reject queue items; tell the user to do that in the ClosePilot UI."))
_con = None


def con():
    global _con
    if _con is None:
        _con = db.connect()
        db.init_db(_con)
    return _con


def _records(df) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


@mcp.tool()
def run_close(use_llm: bool = False) -> dict[str, Any]:
    """Run the month-end close: reconciliation, AP checks, controls, cash forecast. Fills the approval queue."""
    with contextlib.redirect_stdout(sys.stderr):  # stdout is the MCP transport
        s = graph.run_close(con(), use_llm)
    return dict(queued=s["queued"], policy_citations=s["cited"], kpis=s["kpis"])


@mcp.tool()
def get_kpis() -> dict[str, Any]:
    """KPIs of the last close run: auto-match rate, exceptions, flagged exposure, estimated hours saved."""
    try:
        r = con().execute("SELECT kpis FROM run_summary").fetchone()
    except duckdb.CatalogException:
        r = None
    if not r:
        raise ToolError("No run yet. Call run_close first.")
    return json.loads(r[0])


@mcp.tool()
def list_queue(status: str = "pending", agent: str | None = None, limit: int = 20) -> list[dict]:
    """Approval-queue items by exposure, each with its cited policy clause. agent is recon, ap or grc."""
    return _records(con().execute(
        "SELECT q.item_id, agent, kind, summary, exposure_inr, confidence, status, c.clause AS policy_clause "
        "FROM approval_queue q LEFT JOIN policy_citations c USING (item_id) "
        "WHERE status = ? AND (? IS NULL OR agent = ?) ORDER BY exposure_inr DESC LIMIT ?",
        [status, agent, agent, min(limit, 200)]).df())


@mcp.tool()
def explain_item(item_id: str) -> dict[str, Any]:
    """One approval-queue item with its evidence and the full text of the policy clause it breaches."""
    rows = _records(con().execute(
        "SELECT q.*, c.clause, c.doc, c.title FROM approval_queue q LEFT JOIN policy_citations c USING (item_id) "
        "WHERE q.item_id = ?", [item_id]).df())
    if not rows:
        raise ToolError(f"No queue item {item_id}")
    r = rows[0]
    r["payload"] = json.loads(r["payload"])
    r["policy_text"] = next((c["text"] for c in rag.chunks() if c["clause"] == r["clause"]), None)
    return r


@mcp.tool()
def ledger_schema() -> str:
    """Tables and columns that query_ledger can read."""
    return copilot.schema_text(con())


@mcp.tool()
def query_ledger(sql: str) -> list[dict]:
    """Run one read-only DuckDB SELECT on the ledger. Writes, file access and internal tables are blocked."""
    try:
        return _records(copilot.run_sql(con(), sql))
    except (ValueError, duckdb.Error) as e:  # guard rejection or SQL error: let the model read it and retry
        raise ToolError(str(e)) from e


@mcp.tool()
def search_policy(question: str, k: int = 3) -> list[dict]:
    """Find the accounting-policy, delegation-of-authority or vendor-contract clauses relevant to a question."""
    return rag.search(question, min(k, 10), con(), hybrid=True)


if __name__ == "__main__":
    mcp.run()
