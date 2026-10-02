"""Gemini-dependent code paths with the LLM mocked (no network, no quota).

These prove the wiring, validation and fallbacks. They do not measure real model quality.
"""
import pandas as pd
import pytest

from closepilot import llm
from closepilot.agents import copilot, forecast, recon


@pytest.fixture()
def fake_llm(monkeypatch):
    calls = {"structured": [], "text": []}
    monkeypatch.setattr(llm, "available", lambda: True)

    def install(structured=None, text="mock text"):
        def s(prompt, schema, retries=3):
            calls["structured"].append(prompt)
            return structured(prompt, schema) if callable(structured) else structured

        def t(prompt, retries=3):
            calls["text"].append(prompt)
            return text

        monkeypatch.setattr(llm, "structured", s)
        monkeypatch.setattr(llm, "text", t)
        return calls

    return install


def _mini_con(con):
    """Replace ledger with 1 unknown bank credit and 2 GL candidates near its amount."""
    con.execute("DELETE FROM gl")
    con.execute("DELETE FROM bank")
    con.execute("INSERT INTO gl VALUES ('G1','2026-01-05','Bank','Receipt maybe Acme',1000,0,'','u'),"
                "('G2','2026-01-06','Bank','Receipt other',1010,0,'','u')")
    con.execute("INSERT INTO bank VALUES ('B1','2026-02-20','UNIDENTIFIED CREDIT 1234',0,1005,'UNK')")
    return con


def test_recon_llm_verdict_used_when_valid(con, fake_llm):
    calls = fake_llm(structured=recon.ExceptionVerdict(category="unrecorded_receipt", suggested_action="Book it",
                                                       reason="matches G1", matched_gl_id="G1"))
    res = recon.run_recon(_mini_con(con), use_llm=True)
    row = res["exceptions"][res["exceptions"].item_id == "B1"].iloc[0]
    assert row.reason == "matches G1" and row.confidence == 0.6
    assert len(calls["structured"]) == 1


def test_recon_llm_hallucinated_gl_id_rejected(con, fake_llm):
    fake_llm(structured=recon.ExceptionVerdict(category="x", suggested_action="y", reason="z", matched_gl_id="G999"))
    res = recon.run_recon(_mini_con(con), use_llm=True)
    row = res["exceptions"][res["exceptions"].item_id == "B1"].iloc[0]
    assert row.confidence == 0.5 and "No GL entry" in row.reason  # fell back to deterministic text


def test_recon_llm_none_falls_back(con, fake_llm):
    fake_llm(structured=None)
    res = recon.run_recon(_mini_con(con), use_llm=True)
    assert res["exceptions"][res["exceptions"].item_id == "B1"].iloc[0].confidence == 0.5


def test_forecast_commentary_uses_llm_text(con, fake_llm):
    calls = fake_llm(text="CFO says all fine.")
    r = forecast.run_forecast(con)
    assert r["commentary"] == "CFO says all fine."
    assert "net_13w" in calls["text"][0]


def test_copilot_happy_path(con, fake_llm):
    fake_llm(structured=lambda p, schema: schema(sql="SELECT count(*) AS n FROM vendors"), text="There are 40 vendors.")
    r = copilot.ask(con, "How many vendors?")
    assert r["ok"] and r["df"].n[0] == 40 and r["answer"] == "There are 40 vendors."


def test_copilot_retries_after_guard_rejection(con, fake_llm):
    seq = iter(["DROP TABLE vendors", "SELECT count(*) AS n FROM vendors"])
    calls = fake_llm(structured=lambda p, schema: schema(sql=next(seq)))
    r = copilot.ask(con, "q", summarize=False)
    assert r["ok"] and "Previous attempt failed" in calls["structured"][1]
    assert con.execute("SELECT count(*) FROM vendors").fetchone()[0] == 40  # DROP never ran


def test_copilot_gives_up_after_two_bad_queries(con, fake_llm):
    fake_llm(structured=lambda p, schema: schema(sql="SELECT * FROM labels"))
    r = copilot.ask(con, "q")
    assert not r["ok"] and "forbidden" in r["error"]


def test_copilot_prompt_hides_truth_tables_and_uses_data_date(con, fake_llm):
    calls = fake_llm(structured=lambda p, schema: schema(sql="SELECT 1"))
    copilot.ask(con, "q", summarize=False)
    prompt = calls["structured"][0]
    assert "recon_truth" not in prompt and "labels" not in prompt and "audit_log" not in prompt
    as_of = con.execute("SELECT MAX(CAST(txn_date AS DATE)) FROM bank").fetchone()[0]
    assert str(as_of) in prompt  # as-of date comes from the data, not hardcoded


def test_copilot_without_key(con, monkeypatch):
    monkeypatch.setattr(llm, "available", lambda: False)
    assert copilot.ask(con, "q")["ok"] is False


def test_fuzzy_stage_matches_near_amount_similar_narration(con):
    con.execute("DELETE FROM gl")
    con.execute("DELETE FROM bank")
    con.execute("INSERT INTO gl VALUES ('G1','2026-01-05','Bank','Payment to Sharma Steel Works inv 12',0,100000,'','u')")
    con.execute("INSERT INTO bank VALUES ('B1','2026-01-06','NEFT SHARMA STEEL WORKS',100000.0,0,'')")
    # 0.3% amount difference (bank fee netted) so exact and rules stages cannot match
    con.execute("UPDATE bank SET debit = 100300 WHERE bank_id='B1'")
    res = recon.run_recon(con, use_llm=False)
    m = res["matches"]
    assert list(m.method) == ["fuzzy"] and m.gl_id[0] == "G1"
