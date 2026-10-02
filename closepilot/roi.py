"""ROI KPIs. Hours saved rest on editable assumptions; they are estimates, not measurements."""
from __future__ import annotations

import pandas as pd

ASSUMPTIONS = dict(
    min_per_recon_line=2.0,       # manual bank-to-GL tick-and-tie per line
    min_per_invoice_check=6.0,    # manual 3-way match + GST check per invoice
    min_per_exception=10.0,       # analyst time to resolve one remaining exception
    min_per_review_item=5.0,      # reviewer time per approval-queue item
    loaded_cost_inr_per_hour=900.0,
)


def with_assumptions(k: dict, a: dict | None = None) -> dict:
    """Recompute effort numbers from stored counts, so the UI can edit assumptions without re-running agents."""
    a = {**ASSUMPTIONS, **(a or {})}
    baseline_h = (k["bank_lines"] * a["min_per_recon_line"] + k["invoices_checked"] * a["min_per_invoice_check"]) / 60
    remaining_h = (k["exceptions"] * a["min_per_exception"] + k["queue_items"] * a["min_per_review_item"]) / 60
    saved_h = max(baseline_h - remaining_h, 0.0)
    return k | dict(baseline_hours=round(baseline_h, 1), remaining_hours=round(remaining_h, 1),
                    hours_saved=round(saved_h, 1), effort_reduction=round(saved_h / baseline_h, 3) if baseline_h else 0.0,
                    cost_saved_inr=round(saved_h * a["loaded_cost_inr_per_hour"]), assumptions=a)


def compute(con, recon: dict, ap: pd.DataFrame, grc: pd.DataFrame, forecast: dict, a: dict | None = None) -> dict:
    stats = recon["stats"].iloc[0]
    hard = grc[~grc.control_id.isin(["C06", "C07"])]
    inv_exp = pd.concat([ap[["invoice_id", "exposure_inr"]].rename(columns={"invoice_id": "e"}),
                         hard[hard.entity_type == "invoice"][["entity_id", "exposure_inr"]].rename(columns={"entity_id": "e"})])
    k = dict(
        auto_match_rate=float(stats.auto_match_rate), exceptions=int(stats.exceptions), bank_lines=int(stats.bank_lines),
        invoices_checked=int(con.execute("SELECT count(*) FROM invoices").fetchone()[0]),
        queue_items=int(con.execute("SELECT count(*) FROM approval_queue WHERE status='pending'").fetchone()[0]),
        flagged_invoices=int(inv_exp.e.nunique()),
        flagged_exposure_inr=float(inv_exp.groupby("e").exposure_inr.max().sum()),  # no double counting per invoice
        vendor_level_exposure_inr=float(hard[hard.entity_type == "vendor"].exposure_inr.sum()),
        forecast_wape_receipts=forecast["facts"]["wape_receipts"], forecast_wape_payments=forecast["facts"]["wape_payments"])
    return with_assumptions(k, a)
