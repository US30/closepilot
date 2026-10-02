"""AP agent: 3-way match, duplicate invoices, GST arithmetic. All deterministic."""
from __future__ import annotations

import pandas as pd
from rapidfuzz import fuzz

from closepilot import db

PRICE_TOL = 0.02
GST_TOL = 1.0
DUP_DAYS = 10


def run_ap(con) -> pd.DataFrame:
    inv = con.execute("""
        SELECT i.*, p.qty AS po_qty, p.unit_price AS po_price, g.qty_received,
               (SELECT count(*) FROM payments y WHERE y.invoice_id = i.invoice_id) > 0 AS paid
        FROM invoices i
        LEFT JOIN pos p USING (po_id)
        LEFT JOIN grns g USING (po_id)""").df()
    out: list[dict] = []

    def flag(r, check, severity, detail):
        out.append(dict(invoice_id=r.invoice_id, vendor_id=r.vendor_id, check=check, severity=severity,
                        exposure_inr=float(r.total), detail=detail, paid=bool(r.paid)))

    for r in inv.itertuples():
        if pd.isna(r.po_qty):
            flag(r, "no_po", "high", "Invoice has no purchase order")
        else:
            expected = r.po_qty * r.po_price
            if abs(r.base_amount - expected) / expected > PRICE_TOL:
                pct = (r.base_amount / expected - 1) * 100
                flag(r, "price_mismatch", "high", f"Billed {r.base_amount:,.0f} vs PO {expected:,.0f} ({pct:+.1f}%)")
            if pd.isna(r.qty_received):
                flag(r, "no_grn", "high", "No goods receipt recorded for PO")
            elif r.qty_received < r.po_qty:
                flag(r, "short_receipt", "medium", f"Received {r.qty_received} of {r.po_qty}")
        tax = r.cgst + r.sgst + r.igst
        if abs(r.base_amount * r.gst_rate - tax) > GST_TOL:
            flag(r, "gst_error", "medium", f"Tax {tax:,.2f} vs expected {r.base_amount * r.gst_rate:,.2f}")

    # duplicates: same vendor, near-equal total, dates close; flag the later posting
    for _, grp in inv.sort_values("posted_at").groupby("vendor_id"):
        rows = list(grp.itertuples())
        for j, b in enumerate(rows):
            for a in rows[:j]:
                if (abs(a.total - b.total) < 1.0 and abs((b.invoice_date - a.invoice_date).days) <= DUP_DAYS
                        and fuzz.ratio(a.invoice_no, b.invoice_no) >= 60):
                    flag(b, "duplicate_invoice", "high", f"Duplicate of {a.invoice_id} (same vendor, amount {b.total:,.2f})")
                    break
    return pd.DataFrame(out, columns=["invoice_id", "vendor_id", "check", "severity", "exposure_inr", "detail", "paid"])


def enqueue_findings(con, findings: pd.DataFrame) -> int:
    for r in findings.itertuples():
        action = "Recover / debit-note" if r.paid else "Hold payment"
        db.enqueue(con, f"AP-{r.invoice_id}-{r.check}", "ap", r.check, f"{r.invoice_id}: {r.detail}. {action}.",
                   {"invoice_id": r.invoice_id, "paid": r.paid}, r.exposure_inr, 0.9 if r.severity == "high" else 0.75)
    return len(findings)
