"""ClosePilot Streamlit UI. Run: streamlit run app/Home.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from closepilot import db, graph, llm, rag, roi  # noqa: E402
from closepilot.agents import copilot  # noqa: E402
from data.generate import generate  # noqa: E402

st.set_page_config(page_title="ClosePilot", layout="wide")


@st.cache_resource
def get_con():
    if not (db.RAW / "gl.csv").exists():
        generate(42, db.RAW)
    con = db.connect()
    if "gl" not in {r[0] for r in con.execute("SHOW TABLES").fetchall()}:
        db.init_db(con)
    return con


con = get_con()
tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
has_run = "run_summary" in tables
if has_run and "policy_citations" not in tables:  # database from a run made before the policy layer existed
    rag.cite_queue(con, hybrid=True)
inr = lambda x: f"₹{x/1e5:,.1f} L" if abs(x) < 1e7 else f"₹{x/1e7:,.2f} Cr"  # noqa: E731

st.title("ClosePilot — agentic month-end close copilot")
st.caption("Synthetic ledger for Acme Components Pvt Ltd · FY 2025-26 · INR · rules first, LLM for exceptions, human approves")

with st.sidebar:
    user = st.text_input("Reviewer", "cfo.arun")
    use_llm = st.toggle("Use Gemini for exceptions + commentary", value=llm.available(), disabled=not llm.available())
    if not llm.available():
        st.info("Set GOOGLE_API_KEY to enable Gemini features (exception notes, commentary, CFO chat).")
    if st.button("Run month-end close", type="primary"):
        with st.spinner("Running agents…"):
            graph.run_close(con, use_llm=use_llm)
        st.rerun()
    st.markdown("**ROI assumptions** (editable)")
    a = {k: st.number_input(k, value=float(v)) for k, v in roi.ASSUMPTIONS.items()}

if not has_run:
    st.warning("No run yet. Click **Run month-end close** in the sidebar.")
    st.stop()

kp = json.loads(con.execute("SELECT kpis FROM run_summary").fetchone()[0])
t_over, t_rec, t_ap, t_grc, t_cash, t_chat, t_pol, t_q = st.tabs(
    ["Overview", "Reconciliation", "AP checks", "Controls", "Cash forecast", "CFO copilot", "Policy Q&A", "Approval queue"])

with t_over:
    recon_df = con.execute("SELECT * FROM recon_matches").df()
    k = roi.with_assumptions(kp, a)
    c = st.columns(5)
    c[0].metric("Bank lines auto-matched", f"{k['auto_match_rate']:.1%}")
    c[1].metric("Exceptions left", k["exceptions"])
    c[2].metric("Invoices flagged", f"{k['flagged_invoices']} / {k['invoices_checked']}")
    c[3].metric("Exposure flagged", inr(k["flagged_exposure_inr"]))
    c[4].metric("Analyst hours saved (est.)", f"{k['hours_saved']:.0f} h", f"{k['effort_reduction']:.0%} of manual effort")
    st.caption(f"Hours are estimates from the editable assumptions in the sidebar. Baseline manual effort "
               f"{k['baseline_hours']} h; remaining {k['remaining_hours']} h; ≈ ₹{k['cost_saved_inr']:,} saved per close. "
               f"Vendor-level exposure (SoD / bank-change) of {inr(k['vendor_level_exposure_inr'])} is reported separately.")
    st.subheader("Matches by method")
    st.plotly_chart(px.bar(recon_df.groupby("method").bank_id.nunique().reset_index(name="bank lines"),
                           x="method", y="bank lines"), use_container_width=True)

with t_rec:
    st.dataframe(con.execute("SELECT * FROM recon_exceptions ORDER BY ABS(amount) DESC").df(), use_container_width=True)
    with st.expander("All matches"):
        st.dataframe(con.execute("SELECT * FROM recon_matches").df(), use_container_width=True)

with t_ap:
    st.dataframe(con.execute("SELECT * FROM ap_findings ORDER BY exposure_inr DESC").df(), use_container_width=True)

with t_grc:
    st.dataframe(con.execute("SELECT * FROM grc_flags ORDER BY control_id, exposure_inr DESC").df(), use_container_width=True)
    st.caption("C06 Benford and C07 IsolationForest are advisory and are not sent to the approval queue.")

with t_cash:
    fc = con.execute("SELECT * FROM forecast_weekly").df()
    st.plotly_chart(px.line(fc, x="ds", y=["receipts", "payments", "cumulative_net"], markers=True), use_container_width=True)
    st.write(con.execute("SELECT commentary FROM run_summary").fetchone()[0])
    st.markdown("**Backtest (last 13 weeks, WAPE — lower is better)**")
    st.dataframe(con.execute("SELECT * FROM forecast_backtest").df(), use_container_width=True)
    st.caption("Model per series is picked on the same holdout, so the reported WAPE is slightly optimistic.")

with t_chat:
    q = st.text_input("Ask about the ledger", "Which vendors had payments over 5 lakh without a GRN?")
    if st.button("Ask") and q:
        r = copilot.ask(con, q)
        if r["ok"]:
            st.write(r["answer"])
            st.code(r["sql"], language="sql")
            st.dataframe(r["df"], use_container_width=True)
        else:
            st.error(r["error"])
            if r["sql"]:
                st.code(r["sql"], language="sql")

with t_pol:
    pq = st.text_input("Ask about policy or the vendor contract", "Can we pay an invoice before the goods are received?")
    if st.button("Search policy") and pq:
        r = rag.answer(con, pq)
        if r["answer"]:
            st.write(r["answer"])
            st.caption("Cited: " + (", ".join(r["citations"]) or "none"))
        for s in r["sources"]:
            with st.expander(f"{s['clause']} {s['title']} · {s['doc']} ({s['retrieval']} retrieval)"):
                st.write(s["text"])

with t_q:
    qdf = con.execute("SELECT q.item_id, agent, kind, summary, c.clause || ' ' || c.title AS policy, exposure_inr, "
                      "confidence, status FROM approval_queue q LEFT JOIN policy_citations c USING (item_id) "
                      "ORDER BY status='pending' DESC, exposure_inr DESC").df()
    f1, f2 = st.columns(2)
    agent = f1.multiselect("Agent", sorted(qdf.agent.unique()), default=list(qdf.agent.unique()))
    status = f2.multiselect("Status", ["pending", "approved", "rejected"], default=["pending"])
    view = qdf[qdf.agent.isin(agent) & qdf.status.isin(status)]
    st.dataframe(view, use_container_width=True, height=320)
    ids = st.multiselect("Select items", view.item_id.tolist())
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Approve") and ids:
        [db.decide(con, i, user, "approved") for i in ids]
        st.rerun()
    if b2.button("Reject") and ids:
        [db.decide(con, i, user, "rejected") for i in ids]
        st.rerun()
    with st.expander("Audit log (append-only)"):
        st.dataframe(con.execute("SELECT * FROM audit_log ORDER BY ts DESC LIMIT 500").df(), use_container_width=True)
