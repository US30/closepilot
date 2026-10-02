"""LangGraph orchestrator: recon -> ap -> grc -> forecast -> queue -> roi. Results persist to DuckDB."""
from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from closepilot import db, roi
from closepilot.agents import ap, forecast, grc, recon


class State(TypedDict, total=False):
    recon: dict
    ap: Any
    grc: Any
    forecast: dict
    queued: dict
    kpis: dict


def build_graph(con, use_llm: bool = True):
    def n_recon(s: State):
        return {"recon": recon.run_recon(con, use_llm=use_llm)}

    def n_ap(s: State):
        return {"ap": ap.run_ap(con)}

    def n_grc(s: State):
        return {"grc": grc.run_grc(con)}

    def n_forecast(s: State):
        return {"forecast": forecast.run_forecast(con)}

    def n_queue(s: State):
        con.execute("DELETE FROM approval_queue WHERE status = 'pending'")  # fresh run replaces stale proposals
        q = dict(recon=recon.enqueue_exceptions(con, s["recon"]["exceptions"]),
                 ap=ap.enqueue_findings(con, s["ap"]), grc=grc.enqueue_flags(con, s["grc"]))
        return {"queued": q}

    def n_roi(s: State):
        k = roi.compute(con, s["recon"], s["ap"], s["grc"], s["forecast"])
        _persist(con, s, k)
        return {"kpis": k}

    g = StateGraph(State)
    for name, fn in [("recon", n_recon), ("ap", n_ap), ("grc", n_grc), ("forecast", n_forecast),
                     ("queue", n_queue), ("roi", n_roi)]:
        g.add_node(name, fn)
    g.add_edge(START, "recon")
    for a, b in [("recon", "ap"), ("ap", "grc"), ("grc", "forecast"), ("forecast", "queue"), ("queue", "roi")]:
        g.add_edge(a, b)
    g.add_edge("roi", END)
    return g.compile()


def _persist(con, s: State, kpis: dict) -> None:
    for name, df in [("recon_matches", s["recon"]["matches"]), ("recon_exceptions", s["recon"]["exceptions"]),
                     ("ap_findings", s["ap"]), ("grc_flags", s["grc"]), ("forecast_weekly", s["forecast"]["forecast"]),
                     ("forecast_backtest", s["forecast"]["backtest"])]:
        con.register("_df", df)
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    con.execute("CREATE OR REPLACE TABLE run_summary AS SELECT ? AS kpis, ? AS commentary",
                [json.dumps(kpis, default=str), s["forecast"]["commentary"]])


def run_close(con, use_llm: bool = True) -> State:
    return build_graph(con, use_llm).invoke({})
