"""FastAPI wrapper. Run: uvicorn closepilot.api:app --reload"""
import json

import duckdb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from closepilot import db, graph, rag
from closepilot.agents import copilot

app = FastAPI(title="ClosePilot")
_con = None


def con():
    global _con
    if _con is None:
        _con = db.connect()
        db.init_db(_con)
    return _con


class Decision(BaseModel):
    user: str
    decision: str


class Question(BaseModel):
    question: str


@app.post("/run-close")
def run_close(use_llm: bool = False):
    s = graph.run_close(con(), use_llm)
    return dict(queued=s["queued"], kpis=s["kpis"])


@app.get("/kpis")
def kpis():
    try:
        r = con().execute("SELECT kpis FROM run_summary").fetchone()
    except duckdb.CatalogException:  # table is created by the first /run-close
        r = None
    if not r:
        raise HTTPException(404, "No run yet")
    return json.loads(r[0])


@app.get("/queue")
def queue(status: str = "pending", limit: int = 100):
    return con().execute("SELECT * FROM approval_queue WHERE status=? ORDER BY exposure_inr DESC LIMIT ?",
                         [status, limit]).df().to_dict("records")


@app.post("/approve/{item_id}")
def approve(item_id: str, d: Decision):
    if d.decision not in ("approved", "rejected"):
        raise HTTPException(422, "decision must be approved or rejected")
    db.decide(con(), item_id, d.user, d.decision)
    return {"ok": True}


@app.post("/ask")
def ask(q: Question):
    r = copilot.ask(con(), q.question)
    return dict(ok=r["ok"], error=r["error"], sql=r["sql"], answer=r["answer"],
                rows=None if r["df"] is None else r["df"].to_dict("records"))


@app.post("/policy")
def policy(q: Question):
    return rag.answer(con(), q.question)
