"""GRC / controls agent. Rule-based control tests (auditable) + Benford + IsolationForest (advisory)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

from closepilot import db

APPROVAL_LIMIT = 500_000
CONTROLS = {
    "C01": "split_invoice", "C02": "sod_breach", "C03": "weekend_posting",
    "C04": "round_amount", "C05": "bank_change", "C06": "benford", "C07": "ml_outlier",
}


def run_grc(con) -> pd.DataFrame:
    inv = con.execute("SELECT * FROM invoices").df()
    pay = con.execute("SELECT * FROM payments").df()
    ven = con.execute("SELECT * FROM vendors").df()
    out: list[dict] = []

    def flag(cid, etype, eid, exposure, evidence):
        out.append(dict(control_id=cid, control=CONTROLS[cid], entity_type=etype, entity_id=eid,
                        exposure_inr=float(exposure), evidence=evidence))

    # C01 split invoices just under approval limit
    near = inv[(inv.total < APPROVAL_LIMIT) & (inv.total >= 0.8 * APPROVAL_LIMIT)]
    for (vid, d), g in near.groupby(["vendor_id", "invoice_date"]):
        if len(g) >= 2 and g.total.sum() > APPROVAL_LIMIT:
            for r in g.itertuples():
                flag("C01", "invoice", r.invoice_id, r.total,
                     f"{len(g)} invoices from {vid} on {d}, each < limit, total {g.total.sum():,.0f}")

    # C02 segregation of duties: vendor creator == payment approver
    m = pay.merge(ven[["vendor_id", "created_by"]], on="vendor_id")
    for vid, g in m[m.approved_by == m.created_by].groupby("vendor_id"):
        flag("C02", "vendor", vid, g.amount.sum(), f"{g.approved_by.iloc[0]} created vendor and approved {len(g)} payments")

    # C03 weekend postings
    for r in inv[pd.to_datetime(inv.posted_at).dt.weekday >= 5].itertuples():
        flag("C03", "invoice", r.invoice_id, r.total, f"Posted {r.posted_at} (weekend) by {r.posted_by}")

    # C04 round-amount invoices
    for r in inv[(inv.base_amount >= 10_000) & (inv.base_amount % 10_000 == 0)].itertuples():
        flag("C04", "invoice", r.invoice_id, r.total, f"Base amount {r.base_amount:,.0f} is a round number")

    # C05 vendor bank account differs from master at payment time
    m2 = pay.merge(ven[["vendor_id", "bank_account"]], on="vendor_id")
    chg = m2[m2.vendor_bank_account.astype(str) != m2.bank_account.astype(str)]
    for vid, g in chg.groupby("vendor_id"):
        flag("C05", "vendor", vid, g.amount.sum(), f"{len(g)} payments to an account not on vendor master")

    # C06 Benford (advisory): vendor-level chi-square on first digits
    inv["d1"] = inv.total.abs().astype(str).str.lstrip("0.").str[0].astype(int)
    expect = np.log10(1 + 1 / np.arange(1, 10))
    for vid, g in inv.groupby("vendor_id"):
        if len(g) < 25:
            continue
        obs = np.bincount(g.d1, minlength=10)[1:]
        p = stats.chisquare(obs, expect * obs.sum() + 1e-9).pvalue
        if p < 0.01:
            flag("C06", "vendor", vid, g.total.sum(), f"First-digit distribution deviates from Benford (p={p:.4f})")

    # C07 IsolationForest on invoice features (advisory)
    feats = pd.DataFrame({
        "log_total": np.log1p(inv.total),
        "hour": pd.to_datetime(inv.posted_at).dt.hour,
        "weekday": pd.to_datetime(inv.posted_at).dt.weekday,
        "lag_days": (pd.to_datetime(inv.posted_at) - pd.to_datetime(inv.invoice_date)).dt.days,
        "round": ((inv.base_amount % 10_000) == 0).astype(int),
        "vs_vendor_median": inv.total / inv.groupby("vendor_id").total.transform("median"),
    })
    iso = IsolationForest(contamination=0.03, random_state=0).fit(feats)
    inv["score"] = -iso.score_samples(feats)
    for r in inv[iso.predict(feats) == -1].itertuples():
        flag("C07", "invoice", r.invoice_id, r.total, f"IsolationForest anomaly score {r.score:.3f}")
    return pd.DataFrame(out)


def enqueue_flags(con, flags: pd.DataFrame, advisory: tuple[str, ...] = ("C06", "C07")) -> int:
    n = 0
    for r in flags[~flags.control_id.isin(advisory)].itertuples():
        db.enqueue(con, f"GRC-{r.control_id}-{r.entity_id}", "grc", r.control,
                   f"[{r.control_id}] {r.entity_type} {r.entity_id}: {r.evidence}", {"entity": r.entity_id},
                   r.exposure_inr, 0.85)
        n += 1
    return n
