"""Headless Streamlit tests. Temp DB, LLM mocked: no network, no quota, real data file untouched."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from closepilot import db, llm

APP = str(Path(__file__).resolve().parent.parent / "app" / "Home.py")


@pytest.fixture()
def app(tmp_path, monkeypatch):
    import streamlit as st
    monkeypatch.setenv("CLOSEPILOT_DB", str(tmp_path / "t.duckdb"))
    st.cache_resource.clear()  # get_con() is cached per process
    monkeypatch.setattr(llm, "available", lambda: False)
    yield AppTest.from_file(APP, default_timeout=180)
    st.cache_resource.clear()


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_run_close_and_render(app):
    at = app.run()
    assert not at.exception
    at.sidebar.button[0].click().run()
    assert not at.exception
    assert len(at.metric) == 5 and at.metric[0].value.endswith("%")


def test_approve_and_reject_from_ui_write_audit_log(app):
    at = app.run()
    at.sidebar.button[0].click().run()
    ms = next(m for m in at.multiselect if m.label == "Select items")
    ids = ms.options[:3]
    ms.set_value(ids[:2]).run()
    _button(at, "Approve").click().run()
    con = db.connect()
    st = dict(con.execute("SELECT item_id, status FROM approval_queue WHERE item_id IN (?,?)", ids[:2]).fetchall())
    assert set(st.values()) == {"approved"}
    n = con.execute("SELECT count(*) FROM audit_log WHERE action='approved'").fetchone()[0]
    assert n == 2
    con.close()
    # reject path: pick one still-pending item
    at = at.run()
    ms = next(m for m in at.multiselect if m.label == "Select items")
    ms.set_value(ms.options[:1]).run()
    _button(at, "Reject").click().run()
    con = db.connect()
    assert con.execute("SELECT count(*) FROM audit_log WHERE action='rejected'").fetchone()[0] == 1


def test_chat_tab_with_mocked_gemini(app, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: True)
    from closepilot.agents.copilot import SQLQuery
    from closepilot.agents.recon import ExceptionVerdict

    def fake_structured(prompt, schema, retries=3):  # recon asks for a verdict, copilot asks for SQL
        if schema is SQLQuery:
            return SQLQuery(sql="SELECT count(*) AS n FROM vendors")
        return ExceptionVerdict(category="unrecorded_receipt", suggested_action="Book it", reason="mock")

    monkeypatch.setattr(llm, "structured", fake_structured)
    monkeypatch.setattr(llm, "text", lambda p, retries=3: "Mocked: 40 vendors.")
    at = app.run()
    assert at.sidebar.toggle[0].value is True  # Gemini toggle on when a key is present
    at.sidebar.button[0].click().run()  # full run with use_llm=True, LLM mocked
    assert not at.exception
    _button(at, "Ask").click().run()
    assert not at.exception
    assert any("SELECT" in c.value for c in at.code)
    assert any("Mocked: 40 vendors." in m.value for m in at.markdown)
