"""Synthetic ledger for Acme Components Pvt Ltd (12 months, INR, GST/TDS).

Writes CSVs plus a ground-truth `labels.csv` of injected anomalies so the
agents can be scored in eval/run_eval.py.

Usage: python data/generate.py --seed 42 --out data/raw
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

START = date(2025, 4, 1)
END = date(2026, 3, 31)
APPROVAL_LIMIT = 500_000
GST_RATES = [0.05, 0.12, 0.18, 0.28]
USERS_AP = ["priya.s", "rahul.m", "neha.k"]
USERS_APPROVER = ["cfo.arun", "fc.meera"]
USERS_MASTER = ["vm.sanjay", "priya.s"]  # priya.s can also create vendors -> SoD breach
ITEMS = ["Steel coils", "PCB assemblies", "Fasteners", "Packaging", "Logistics",
         "Copper wire", "Machine parts", "Lubricants", "IT services", "Consulting"]


def _gstin(fake: Faker) -> str:
    return f"27{fake.bothify('?????####?').upper()}1Z{fake.random_element('ABCDEFGHJK')}"


def _workday(d: date) -> date:
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def generate(seed: int = 42, out: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    fake = Faker("en_IN")
    Faker.seed(seed)
    labels: list[dict] = []

    # ---- vendors ---------------------------------------------------------
    n_vendors = 40
    vendors = pd.DataFrame({
        "vendor_id": [f"V{i:03d}" for i in range(1, n_vendors + 1)],
        "name": [fake.company().replace(",", "") for _ in range(n_vendors)],
        "gstin": [_gstin(fake) for _ in range(n_vendors)],
        "bank_account": [fake.numerify("##########") for _ in range(n_vendors)],
        "created_by": rng.choice(USERS_MASTER, n_vendors),
        "created_on": [START - timedelta(days=int(x)) for x in rng.integers(30, 900, n_vendors)],
        "tds_rate": rng.choice([0.0, 0.01, 0.02, 0.10], n_vendors, p=[0.3, 0.3, 0.3, 0.1]),
        "intra_state": rng.random(n_vendors) < 0.7,
    })
    vendor_price = {v: rng.uniform(0.7, 1.3) for v in vendors.vendor_id}

    # ---- PO -> GRN -> invoice -> payment --------------------------------
    pos, grns, invoices, payments = [], [], [], []
    po_n = inv_n = pay_n = 0
    days = (END - START).days
    for _ in range(900):
        po_n += 1
        v = vendors.iloc[rng.integers(n_vendors)]
        po_date = _workday(START + timedelta(days=int(rng.integers(0, days - 3))))
        qty = int(rng.integers(10, 400))
        unit = round(float(rng.uniform(200, 4000) * vendor_price[v.vendor_id]), 2)
        po_id = f"PO{po_n:05d}"
        pos.append(dict(po_id=po_id, vendor_id=v.vendor_id, po_date=po_date,
                        item=rng.choice(ITEMS), qty=qty, unit_price=unit,
                        created_by=rng.choice(USERS_AP)))
        grn_date = po_date + timedelta(days=int(rng.integers(3, 15)))
        grn_id = f"GRN{po_n:05d}"
        grns.append(dict(grn_id=grn_id, po_id=po_id, grn_date=grn_date, qty_received=qty))

        inv_n += 1
        inv_id = f"INV{inv_n:05d}"
        inv_date = grn_date + timedelta(days=int(rng.integers(1, 6)))
        base = round(qty * unit, 2)
        rate = float(rng.choice(GST_RATES, p=[0.1, 0.2, 0.6, 0.1]))
        gst = round(base * rate, 2)
        cgst = sgst = round(gst / 2, 2) if v.intra_state else 0.0
        igst = 0.0 if v.intra_state else gst
        total = round(base + cgst + sgst + igst, 2)
        tds = round(base * v.tds_rate, 2)
        posted_by = rng.choice(USERS_AP)
        posted_dt = datetime.combine(_workday(inv_date), time(int(rng.integers(9, 18)), int(rng.integers(0, 60))))
        invoices.append(dict(
            invoice_id=inv_id, vendor_id=v.vendor_id, po_id=po_id, invoice_no=f"{v.vendor_id}/{inv_n:05d}",
            invoice_date=inv_date, base_amount=base, gst_rate=rate, cgst=cgst, sgst=sgst, igst=igst,
            total=total, tds=tds, net_payable=round(total - tds, 2), posted_by=posted_by,
            posted_at=posted_dt, due_date=inv_date + timedelta(days=int(rng.choice([30, 45, 60])))))

    pos_df, grn_df, inv_df = map(pd.DataFrame, (pos, grns, invoices))
    inv_df = inv_df.set_index("invoice_id", drop=False)

    # ---- inject AP anomalies --------------------------------------------
    pick = lambda n: list(rng.choice(inv_df.index, n, replace=False))  # noqa: E731
    used: set[str] = set()

    def take(n: int) -> list[str]:
        ids = [i for i in pick(n * 3) if i not in used][:n]
        used.update(ids)
        return ids

    # duplicate invoices: re-keyed with slightly different invoice_no
    dup_rows = []
    for iid in take(18):
        r = inv_df.loc[iid].copy()
        inv_n += 1
        new_id = f"INV{inv_n:05d}"
        r["invoice_id"] = new_id
        r["invoice_no"] = r["invoice_no"] + rng.choice(["", "-A", " ", "/1"])
        r["invoice_date"] = r["invoice_date"] + timedelta(days=int(rng.integers(0, 4)))
        shifted = r["posted_at"] + timedelta(days=int(rng.integers(1, 6)))
        r["posted_at"] = datetime.combine(_workday(shifted.date()), shifted.time())  # stay off weekends
        dup_rows.append(r)
        used.add(new_id)  # a duplicate must not also receive a later anomaly, or it stops matching its original
        labels.append(dict(entity="invoice", entity_id=new_id, anomaly_type="duplicate_invoice"))
    inv_df = pd.concat([inv_df, pd.DataFrame(dup_rows).set_index("invoice_id", drop=False)])

    # price mismatch vs PO (3-way match failure): billed 8-25% above PO price
    for iid in take(25):
        f = float(rng.uniform(1.08, 1.25))
        r = inv_df.loc[iid]
        base = round(r.base_amount * f, 2)
        gst = round(base * r.gst_rate, 2)
        half = round(gst / 2, 2)
        cg, sg, ig = (half, half, 0.0) if r.igst == 0 else (0.0, 0.0, gst)
        total = round(base + cg + sg + ig, 2)
        inv_df.loc[iid, ["base_amount", "cgst", "sgst", "igst", "total"]] = [base, cg, sg, ig, total]
        inv_df.loc[iid, "net_payable"] = round(total - r.tds, 2)
        labels.append(dict(entity="invoice", entity_id=iid, anomaly_type="price_mismatch"))

    # GST arithmetic error: tax understated
    for iid in take(20):
        r = inv_df.loc[iid]
        off = round(r.base_amount * 0.03, 2)
        col = "igst" if r.igst > 0 else "cgst"
        inv_df.loc[iid, col] = max(r[col] - off, 0)
        inv_df.loc[iid, "total"] = round(inv_df.loc[iid, ["base_amount", "cgst", "sgst", "igst"]].sum(), 2)
        labels.append(dict(entity="invoice", entity_id=iid, anomaly_type="gst_error"))

    # weekend postings
    for iid in take(20):
        d = inv_df.loc[iid, "posted_at"]
        shift = (5 - d.weekday()) % 7 or 7
        inv_df.loc[iid, "posted_at"] = d + timedelta(days=shift)
        labels.append(dict(entity="invoice", entity_id=iid, anomaly_type="weekend_posting"))

    # round-amount invoices
    for iid in take(15):
        r = inv_df.loc[iid]
        base = float(round(r.base_amount, -4)) or 10000.0
        # keep the PO consistent so round amounts are flagged as round, not as price mismatches
        pq = pos_df.loc[pos_df.po_id == r.po_id, "qty"].iloc[0]
        pos_df.loc[pos_df.po_id == r.po_id, "unit_price"] = round(base / pq, 6)
        gst = round(base * r.gst_rate, 2)
        half = round(gst / 2, 2)
        cg, sg, ig = (half, half, 0.0) if r.igst == 0 else (0.0, 0.0, gst)
        total = round(base + cg + sg + ig, 2)
        inv_df.loc[iid, ["base_amount", "cgst", "sgst", "igst", "total"]] = [base, cg, sg, ig, total]
        inv_df.loc[iid, "net_payable"] = round(total - r.tds, 2)
        labels.append(dict(entity="invoice", entity_id=iid, anomaly_type="round_amount"))

    # split invoices: one large purchase billed as 2 invoices just under approval limit, same day
    split_rows = []
    for _ in range(8):
        v = vendors.iloc[rng.integers(n_vendors)]
        d = _workday(START + timedelta(days=int(rng.integers(30, days - 30))))
        for k in range(2):
            inv_n += 1
            iid = f"INV{inv_n:05d}"
            base = round(float(rng.uniform(0.9, 0.98) * APPROVAL_LIMIT / 1.18), 2)
            gst = round(base * 0.18, 2)
            total = round(base + gst, 2)
            split_rows.append(dict(
                invoice_id=iid, vendor_id=v.vendor_id, po_id=None, invoice_no=f"{v.vendor_id}/S{inv_n:05d}",
                invoice_date=d, base_amount=base, gst_rate=0.18, cgst=round(gst / 2, 2), sgst=round(gst / 2, 2),
                igst=0.0, total=total, tds=0.0, net_payable=total, posted_by=rng.choice(USERS_AP),
                posted_at=datetime.combine(d, time(15, 0)), due_date=d + timedelta(days=30)))
            labels.append(dict(entity="invoice", entity_id=iid, anomaly_type="split_invoice"))
    inv_df = pd.concat([inv_df, pd.DataFrame(split_rows).set_index("invoice_id", drop=False)])

    # payments without GRN: drop GRN for some POs whose invoice will be paid
    nogrn_pos = []
    for iid in take(15):
        po = inv_df.loc[iid, "po_id"]
        if po is not None:
            nogrn_pos.append(po)
            labels.append(dict(entity="invoice", entity_id=iid, anomaly_type="no_grn"))
    grn_df = grn_df[~grn_df.po_id.isin(nogrn_pos)]

    # ---- payments --------------------------------------------------------
    inv_df = inv_df.reset_index(drop=True).sort_values("invoice_date").reset_index(drop=True)
    vmap = vendors.set_index("vendor_id")
    bank_changed_vendors = set(rng.choice(vendors.vendor_id, 4, replace=False))
    for v in bank_changed_vendors:
        labels.append(dict(entity="vendor", entity_id=v, anomaly_type="bank_change"))
    pay_rows = []
    for r in inv_df.itertuples():
        if rng.random() < 0.06:  # still unpaid
            continue
        pay_n += 1
        pay_date = _workday(r.due_date - timedelta(days=int(rng.integers(0, 8))))
        if pay_date > END:
            continue
        acct = vmap.loc[r.vendor_id, "bank_account"]
        if r.vendor_id in bank_changed_vendors and pay_date > START + timedelta(days=200):
            acct = f"{int(rng.integers(10**9, 10**10))}"
        pay_rows.append(dict(
            payment_id=f"PAY{pay_n:05d}", invoice_id=r.invoice_id, vendor_id=r.vendor_id, pay_date=pay_date,
            amount=r.net_payable, approved_by=rng.choice(USERS_APPROVER),
            bank_ref=f"NEFT{rng.integers(10**8, 10**9)}", vendor_bank_account=acct))
    pay_df = pd.DataFrame(pay_rows)

    # SoD breach: approver also created vendor -> make a few vendors created_by an approver
    sod_vendors = list(rng.choice(vendors.vendor_id, 3, replace=False))
    vendors.loc[vendors.vendor_id.isin(sod_vendors), "created_by"] = "cfo.arun"
    pay_df.loc[pay_df.vendor_id.isin(sod_vendors), "approved_by"] = "cfo.arun"
    for v in sod_vendors:
        labels.append(dict(entity="vendor", entity_id=v, anomaly_type="sod_breach"))

    # ---- AR / receipts ---------------------------------------------------
    customers = [fake.company().replace(",", "") for _ in range(25)]
    ar_rows = []
    for i in range(700):
        d = _workday(START + timedelta(days=int(rng.integers(0, days - 1))))
        amt = round(float(rng.lognormal(13.2, 0.8)), 2)
        terms = int(rng.choice([30, 45, 60]))
        late = int(max(0, rng.normal(8, 12)))
        paid = d + timedelta(days=terms + late - 5) if rng.random() < 0.97 else None
        if paid is not None and paid > END:
            paid = None
        ar_rows.append(dict(ar_id=f"AR{i+1:05d}", customer=rng.choice(customers), invoice_date=d,
                            due_date=d + timedelta(days=terms), amount=amt,
                            paid_date=_workday(paid) if paid else None))
    ar_df = pd.DataFrame(ar_rows)

    # ---- GL cash ledger + bank statement ---------------------------------
    gl, bank = [], []
    gl_n = bank_n = 0

    def add_gl(d, narr, debit, credit, ref, user):
        nonlocal gl_n
        gl_n += 1
        gl.append(dict(gl_id=f"GL{gl_n:06d}", posting_date=d, account="Bank", narration=narr, debit=debit,
                       credit=credit, ref=ref, posted_by=user))
        return f"GL{gl_n:06d}"

    def add_bank(d, narr, debit, credit, ref):
        nonlocal bank_n
        bank_n += 1
        bank.append(dict(bank_id=f"BK{bank_n:06d}", txn_date=d, narration=narr, debit=debit, credit=credit, ref=ref))
        return f"BK{bank_n:06d}"

    truth = []  # gl_id <-> bank_id ground truth
    vname = vendors.set_index("vendor_id").name
    split_pay_pool: dict[str, list] = {}
    for p in pay_df.itertuples():
        vn = vname[p.vendor_id]
        gid = add_gl(p.pay_date, f"Payment to {vn} inv {p.invoice_id}", 0.0, p.amount, p.bank_ref, p.approved_by)
        r = rng.random()
        if r < 0.04:  # outstanding cheque: in GL, not yet in bank (timing)
            truth.append((gid, None, "outstanding"))
            continue
        lag = int(rng.choice([0, 1, 1, 2, 3]))
        narr = f"NEFT-{p.bank_ref}-{vn.upper()[:22]}" if rng.random() < 0.8 else f"NEFT/{vn.upper()[:14]}/{p.bank_ref[-6:]}"
        bref = p.bank_ref if rng.random() < 0.65 else ""  # bank often drops the reference
        bid = add_bank(_workday(p.pay_date + timedelta(days=lag)), narr, p.amount, 0.0, bref)
        truth.append((gid, bid, "1to1"))

    # many-to-one: bundled vendor payment (2 GL lines -> 1 bank debit)
    for k in range(10):
        v = vendors.iloc[rng.integers(n_vendors)]
        d = _workday(START + timedelta(days=int(rng.integers(20, days - 10))))
        a, b = (round(float(rng.uniform(20_000, 300_000)), 2) for _ in range(2))
        g1 = add_gl(d, f"Payment to {v['name']} adv A", 0.0, a, f"BND{k}A", "rahul.m")
        g2 = add_gl(d, f"Payment to {v['name']} adv B", 0.0, b, f"BND{k}B", "rahul.m")
        bid = add_bank(d + timedelta(days=1), f"NEFT-BULK-{v['name'].upper()[:20]}", round(a + b, 2), 0.0, f"BLK{k}")
        truth += [(g1, bid, "many_to_one"), (g2, bid, "many_to_one")]

    for a in ar_df.dropna(subset=["paid_date"]).itertuples():
        gid = add_gl(a.paid_date, f"Receipt from {a.customer} {a.ar_id}", a.amount, 0.0, a.ar_id, "rahul.m")
        lag = int(rng.choice([0, 0, 1]))
        narr = f"RTGS-{a.customer.upper()[:24]}-{a.ar_id}" if rng.random() < 0.75 else f"IMPS/{a.customer.upper()[:12]}"
        bref = a.ar_id if rng.random() < 0.65 else ""
        bid = add_bank(_workday(a.paid_date + timedelta(days=lag)), narr, 0.0, a.amount, bref)
        truth.append((gid, bid, "1to1"))

    # bank-only items: charges, interest, unrecorded receipts
    cur = START
    while cur <= END:
        add_bank(cur, "BANK CHARGES + GST", round(float(rng.uniform(300, 1800)), 2), 0.0, "CHG")
        cur += timedelta(days=30)
        if rng.random() < 0.3:
            add_bank(cur, "INTEREST CREDIT", 0.0, round(float(rng.uniform(800, 9000)), 2), "INT")
    for _ in range(12):
        d = _workday(START + timedelta(days=int(rng.integers(0, days))))
        add_bank(d, f"UNIDENTIFIED CREDIT {fake.numerify('####')}", 0.0, round(float(rng.uniform(10_000, 250_000)), 2), "UNK")

    gl_df, bank_df = pd.DataFrame(gl), pd.DataFrame(bank)
    truth_df = pd.DataFrame(truth, columns=["gl_id", "bank_id", "kind"])

    # GL timing: bank debits in GL table use 'credit' convention for cash outflow already.
    # Weekend GL postings are only labelled via invoices; nothing else injected here.

    labels_df = pd.DataFrame(labels).drop_duplicates()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tables = dict(vendors=vendors, pos=pos_df, grns=grn_df, invoices=inv_df, payments=pay_df,
                  ar_invoices=ar_df, gl=gl_df, bank=bank_df, recon_truth=truth_df, labels=labels_df)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/raw")
    args = ap.parse_args()
    t = generate(args.seed, args.out)
    for k, v in t.items():
        print(f"{k:12s} {len(v):6d} rows")
    print(t["labels"].anomaly_type.value_counts().to_string())
