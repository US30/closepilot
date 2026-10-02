"""Bank-to-GL reconciliation agent.

Cascade, cheapest and most auditable first:
  1 exact      same ref + same signed amount
  2 rules      same amount, date within window, closest date wins
  3 fuzzy      amount within 0.5% and narration similarity (rapidfuzz)
  4 many_to_one  one bank debit == sum of 2-3 GL payments (subset-sum)
  5 LLM        only leftover bank-only items, structured JSON, validated
Leftovers become exceptions and go to the approval queue.
"""
from __future__ import annotations

import re
from itertools import combinations

import pandas as pd
from pydantic import BaseModel
from rapidfuzz import fuzz, utils

from closepilot import db, llm

WINDOW_DAYS = 5
FUZZY_MIN = 70


class ExceptionVerdict(BaseModel):
    category: str  # unrecorded_receipt | bank_charge | interest | unknown
    suggested_action: str
    reason: str
    matched_gl_id: str | None = None


def _load(con):
    gl = con.execute("""SELECT gl_id, CAST(posting_date AS DATE) AS date, narration,
                        ROUND(debit - credit, 2) AS amt, COALESCE(ref, '') AS ref
                        FROM gl WHERE account = 'Bank'""").df()
    bank = con.execute("""SELECT bank_id, CAST(txn_date AS DATE) AS date, narration,
                          ROUND(credit - debit, 2) AS amt, COALESCE(ref, '') AS ref FROM bank""").df()
    for d in (gl, bank):
        d["date"] = pd.to_datetime(d["date"])
    return gl, bank


def _days(a, b) -> int:
    return abs((a - b).days)


def run_recon(con, use_llm: bool = True) -> dict[str, pd.DataFrame]:
    gl, bank = _load(con)
    matches: list[dict] = []
    gl_left = gl.set_index("gl_id")
    bk_left = bank.set_index("bank_id")

    def commit(gid, bid, method, conf, note=""):
        matches.append(dict(gl_id=gid, bank_id=bid, method=method, confidence=conf, note=note))

    # 1 exact: ref + amount
    by_key: dict[tuple, list[str]] = {}
    for gid, r in gl_left.iterrows():
        if r.ref:
            by_key.setdefault((r.ref, r.amt), []).append(gid)
    used_gl: set[str] = set()
    used_bk: set[str] = set()
    for bid, r in bk_left.iterrows():
        if not r.ref:
            continue
        cands = [g for g in by_key.get((r.ref, r.amt), []) if g not in used_gl]
        if cands:
            commit(cands[0], bid, "exact", 1.0)
            used_gl.add(cands[0]); used_bk.add(bid)

    # 2 rules: same amount, closest date within window
    gl_by_amt: dict[float, list[str]] = {}
    for gid, r in gl_left.iterrows():
        if gid not in used_gl:
            gl_by_amt.setdefault(r.amt, []).append(gid)
    for bid, r in bk_left.sort_values("date").iterrows():
        if bid in used_bk:
            continue
        cands = [g for g in gl_by_amt.get(r.amt, []) if g not in used_gl
                 and _days(gl_left.at[g, "date"], r.date) <= WINDOW_DAYS]
        if cands:
            cands.sort(key=lambda g: _days(gl_left.at[g, "date"], r.date))
            conf = 0.95 if len(cands) == 1 else 0.85
            commit(cands[0], bid, "rules", conf, f"{len(cands)} candidate(s) in window")
            used_gl.add(cands[0]); used_bk.add(bid)

    # 3 fuzzy: near amount + similar narration
    gl_rem = [g for g in gl_left.index if g not in used_gl]
    for bid, r in bk_left.iterrows():
        if bid in used_bk:
            continue
        best, best_s = None, 0.0
        for g in gl_rem:
            if g in used_gl:
                continue
            gr = gl_left.loc[g]
            if r.amt == 0 or abs(gr.amt - r.amt) / abs(r.amt) > 0.005 or _days(gr.date, r.date) > WINDOW_DAYS:
                continue
            s = fuzz.token_set_ratio(gr.narration, r.narration, processor=utils.default_process)  # case/punctuation-insensitive
            if s > best_s:
                best, best_s = g, s
        if best and best_s >= FUZZY_MIN:
            commit(best, bid, "fuzzy", round(0.6 + 0.3 * best_s / 100, 2), f"narration sim {best_s:.0f}")
            used_gl.add(best); used_bk.add(bid)

    # 4 many-to-one: bank debit equals sum of 2-3 GL payments within window
    for bid, r in bk_left.iterrows():
        if bid in used_bk or r.amt >= 0:
            continue
        cands = [g for g in gl_left.index if g not in used_gl and gl_left.at[g, "amt"] < 0
                 and _days(gl_left.at[g, "date"], r.date) <= WINDOW_DAYS][:14]
        found = None
        for k in (2, 3):
            for combo in combinations(cands, k):
                if abs(sum(gl_left.at[g, "amt"] for g in combo) - r.amt) < 0.01:
                    found = combo
                    break
            if found:
                break
        if found:
            for g in found:
                commit(g, bid, "many_to_one", 0.8, f"{len(found)} GL lines -> 1 bank debit")
                used_gl.add(g)
            used_bk.add(bid)

    # exceptions
    exc: list[dict] = []
    for bid in bk_left.index:
        if bid in used_bk:
            continue
        r = bk_left.loc[bid]
        exc.append(_classify_bank_only(bid, r, gl_left, used_gl, use_llm))
    last = max(gl["date"].max(), bank["date"].max())
    for gid in gl_left.index:
        if gid in used_gl:
            continue
        r = gl_left.loc[gid]
        age = (last - r.date).days
        exc.append(dict(item_id=gid, side="gl_only", amount=float(r.amt), category="outstanding_item",
                        suggested_action="Chase payee / confirm cheque presented" if age > 30 else "Timing - monitor",
                        reason=f"In GL, not on bank statement; aged {age} days.", confidence=0.9, aged_days=age))

    m = pd.DataFrame(matches, columns=["gl_id", "bank_id", "method", "confidence", "note"])
    e = pd.DataFrame(exc)
    return dict(matches=m, exceptions=e, stats=_stats(gl, bank, m, e))


def _classify_bank_only(bid, r, gl_left, used_gl, use_llm) -> dict:
    n = r.narration.upper()
    base = dict(item_id=bid, side="bank_only", amount=float(r.amt), aged_days=0)
    if "BANK CHARGES" in n:
        return base | dict(category="bank_charge", suggested_action="Post JE: Dr Bank charges, Cr Bank (incl. GST)",
                           reason="Recurring bank charge not recorded in GL.", confidence=0.98)
    if "INTEREST" in n:
        return base | dict(category="interest", suggested_action="Post JE: Dr Bank, Cr Interest income",
                           reason="Interest credit not recorded in GL.", confidence=0.98)
    if use_llm and llm.available():
        near = [(g, gl_left.at[g, "amt"], gl_left.at[g, "narration"]) for g in gl_left.index
                if g not in used_gl and abs(gl_left.at[g, "amt"] - r.amt) / max(abs(r.amt), 1) < 0.05][:5]
        prompt = (f"Bank-only line: {r.date.date()} '{r.narration}' amount {r.amt}.\n"
                  f"Unmatched GL candidates: {near}.\nClassify. matched_gl_id only if one candidate clearly fits, else null.")
        v = llm.structured(prompt, ExceptionVerdict)
        if v and (v.matched_gl_id is None or v.matched_gl_id in {g for g, _, _ in near}):
            return base | dict(category=v.category, suggested_action=v.suggested_action, reason=v.reason,
                               confidence=0.6, llm_gl_id=v.matched_gl_id)
    return base | dict(category="unrecorded_receipt" if r.amt > 0 else "unrecorded_payment",
                       suggested_action="Identify counterparty, then book receipt/payment",
                       reason="No GL entry within tolerance.", confidence=0.5)


def _stats(gl, bank, m, e) -> pd.DataFrame:
    n_bank = len(bank)
    matched_bank = m.bank_id.nunique()
    rows = dict(
        gl_lines=len(gl), bank_lines=n_bank, matched_bank_lines=matched_bank,
        auto_match_rate=round(matched_bank / n_bank, 4),
        exceptions=len(e), **{f"by_{k}": int(v) for k, v in m.groupby("method").bank_id.nunique().items()})
    return pd.DataFrame([rows])


def enqueue_exceptions(con, exceptions: pd.DataFrame) -> int:
    n = 0
    for r in exceptions.itertuples():
        if r.category == "outstanding_item" and r.aged_days <= 30:
            continue  # normal timing difference, no action
        db.enqueue(con, f"RECON-{r.item_id}", "recon", r.category,
                   f"{r.side} {r.item_id}: {r.suggested_action}",
                   {"amount": r.amount, "reason": r.reason}, abs(r.amount), r.confidence)
        n += 1
    return n


def score_against_truth(con, matches: pd.DataFrame) -> dict:
    """Pair precision/recall vs generator ground truth (bank-linked rows only)."""
    truth = con.execute("SELECT gl_id, bank_id FROM recon_truth WHERE bank_id IS NOT NULL").df()
    t = set(zip(truth.gl_id, truth.bank_id))
    p = set(zip(matches.gl_id, matches.bank_id))
    tp = len(t & p)
    return dict(precision=round(tp / max(len(p), 1), 4), recall=round(tp / max(len(t), 1), 4), true_pairs=len(t),
                predicted_pairs=len(p))
