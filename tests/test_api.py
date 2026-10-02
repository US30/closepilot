import pytest
from fastapi.testclient import TestClient

from closepilot import api, llm


@pytest.fixture()
def client(con, monkeypatch):
    monkeypatch.setattr(api, "_con", con)
    monkeypatch.setattr(llm, "available", lambda: False)
    return TestClient(api.app)


def test_kpis_404_before_run(client):
    assert client.get("/kpis").status_code == 404


def test_run_queue_approve_flow(client, con):
    r = client.post("/run-close")
    assert r.status_code == 200 and r.json()["kpis"]["auto_match_rate"] > 0.9
    assert client.get("/kpis").json()["bank_lines"] > 0
    q = client.get("/queue", params={"limit": 5}).json()
    assert len(q) == 5
    ok = client.post(f"/approve/{q[0]['item_id']}", json={"user": "cfo.arun", "decision": "approved"})
    assert ok.json() == {"ok": True}
    assert con.execute("SELECT status FROM approval_queue WHERE item_id=?", [q[0]["item_id"]]).fetchone()[0] == "approved"
    assert client.post(f"/approve/{q[1]['item_id']}", json={"user": "u", "decision": "maybe"}).status_code == 422


def test_ask_without_key_reports_error(client):
    r = client.post("/ask", json={"question": "How many vendors?"}).json()
    assert r["ok"] is False and "GOOGLE_API_KEY" in r["error"]
