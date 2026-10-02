import pytest

from closepilot import db
from closepilot.agents import ap, forecast, grc


def _recall(con, found_ids, anomaly):
    true = set(con.execute("SELECT entity_id FROM labels WHERE anomaly_type=?", [anomaly]).df().entity_id)
    return len(true & set(found_ids)) / len(true), len(set(found_ids) & true) / max(len(set(found_ids)), 1)


@pytest.mark.parametrize("check", ["price_mismatch", "no_grn", "gst_error", "duplicate_invoice"])
def test_ap_checks(con, check):
    f = ap.run_ap(con)
    rec, prec = _recall(con, f[f.check == check].invoice_id, check)
    assert rec >= 0.9 and prec >= 0.9


@pytest.mark.parametrize("control", ["split_invoice", "sod_breach", "weekend_posting", "round_amount", "bank_change"])
def test_grc_controls(con, control):
    f = grc.run_grc(con)
    rec, prec = _recall(con, f[f.control == control].entity_id, control)
    assert rec == 1.0 and prec >= 0.9


def test_forecast_shape(con):
    r = forecast.run_forecast(con)
    assert len(r["forecast"]) == 13
    assert r["facts"]["wape_receipts"] < 0.5 and r["facts"]["wape_payments"] < 0.5


def test_queue_and_audit(con):
    db.enqueue(con, "T-1", "ap", "price_mismatch", "x", {}, 100.0)
    db.decide(con, "T-1", "cfo.arun", "approved")
    assert con.execute("SELECT status FROM approval_queue WHERE item_id='T-1'").fetchone()[0] == "approved"
    assert con.execute("SELECT count(*) FROM audit_log WHERE item_id='T-1'").fetchone()[0] == 2
