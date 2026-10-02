"""Score every agent against the generator's ground truth. Writes eval/report.md.

Usage: python eval/run_eval.py [--seed 42] [--no-llm]
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from closepilot import db, llm  # noqa: E402
from closepilot.agents import ap, copilot, forecast, grc, recon  # noqa: E402
from data.generate import generate  # noqa: E402
from eval.gold_sql import GOLD  # noqa: E402


def prf(con, found: set, anomaly: str):
    true = set(con.execute("SELECT entity_id FROM labels WHERE anomaly_type=?", [anomaly]).df().entity_id)
    tp = len(found & true)
    return dict(anomaly=anomaly, injected=len(true), flagged=len(found), true_pos=tp,
                precision=round(tp / max(len(found), 1), 3), recall=round(tp / max(len(true), 1), 3))


def norm(df: pd.DataFrame):
    return sorted(tuple(round(v, 2) if isinstance(v, float) else str(v) for v in row) for row in df.itertuples(index=False))


def main(seed: int, use_llm: bool):
    raw = ROOT / "data" / "raw_eval"
    generate(seed, raw)
    con = db.connect(":memory:")
    db.init_db(con, raw)
    out = [f"# ClosePilot evaluation (seed {seed})\n"]

    r = recon.run_recon(con, use_llm=False)
    s = recon.score_against_truth(con, r["matches"])
    st = r["stats"].iloc[0]
    out += ["## Reconciliation (deterministic cascade, LLM off)\n",
            f"- Auto-match rate (bank lines): **{st.auto_match_rate:.1%}** ({int(st.matched_bank_lines)}/{int(st.bank_lines)})",
            f"- Pair precision **{s['precision']:.1%}**, recall **{s['recall']:.1%}** vs ground truth ({s['true_pairs']} true pairs)",
            f"- By method: " + ", ".join(f"{c[3:]}={int(st[c])}" for c in st.index if c.startswith("by_")),
            f"- Exceptions: {int(st.exceptions)}\n"]

    a = ap.run_ap(con)
    g = grc.run_grc(con)
    rows = [prf(con, set(a[a.check == t].invoice_id), t) for t in ("price_mismatch", "no_grn", "gst_error", "duplicate_invoice")]
    rows += [prf(con, set(g[g.control == t].entity_id), t) for t in
             ("split_invoice", "sod_breach", "weekend_posting", "round_amount", "bank_change")]
    out += ["## Anomaly detection vs injected labels\n", pd.DataFrame(rows).to_markdown(index=False), ""]
    labelled = set(con.execute("SELECT entity_id FROM labels").df().entity_id)
    ml = set(g[g.control == "ml_outlier"].entity_id)
    out += [f"IsolationForest (advisory): {len(ml)} flagged, {len(ml & labelled)} are injected anomalies "
            f"({len(ml & labelled) / max(len(ml), 1):.0%}). Rule checks were written with knowledge of the injection "
            f"patterns, so these recall numbers show the checks work, not that they generalise to unseen fraud.\n"]

    f = forecast.run_forecast(con)
    out += ["## Cash forecast backtest (last 13 weeks, WAPE, lower is better)\n", f["backtest"].to_markdown(index=False),
            f"\nSelected: receipts={f['facts']['model_receipts']}, payments={f['facts']['model_payments']}. "
            "Selection uses the same holdout, so the selected WAPE is optimistic.\n"]

    out.append("## Text-to-SQL (25 gold questions)\n")
    bad_ref = [q for q, sql in GOLD if _fails(con, sql)]
    out.append(f"- Reference SQL executes: {len(GOLD) - len(bad_ref)}/{len(GOLD)}")
    if use_llm and llm.available():
        strict = lenient = errored = streak = 0
        for n, (q, sql) in enumerate(GOLD, 1):
            res = copilot.ask(con, q, summarize=False)
            if not res["ok"] and res["error"] == "LLM unavailable":  # API failure, not a wrong answer
                errored += 1
                streak += 1
                print(f"[{n:02d}/{len(GOLD)}] API-ERROR {q}", flush=True)
                if streak >= 3:
                    print("3 API errors in a row (quota?), stopping early", flush=True)
                    break
                continue
            streak = 0
            gold = con.execute(sql).df()
            s_hit = bool(res["ok"] and norm(res["df"]) == norm(gold))
            # lenient: model may add helpful extra columns after the asked-for ones
            l_hit = bool(res["ok"] and norm(res["df"].iloc[:, :gold.shape[1]]) == norm(gold))
            strict += s_hit
            lenient += l_hit
            print(f"[{n:02d}/{len(GOLD)}] {'PASS' if s_hit else ('PASS(lenient)' if l_hit else 'FAIL')} {q}"
                  + ("" if l_hit else f"  -> {res['error'] or res['sql']}"), flush=True)
            time.sleep(1)
        answered = n - errored if n else 0
        out.append(f"- Model: `{llm._model().model}`; questions answered by API: {answered}/{len(GOLD)} ({errored} API errors)")
        out.append(f"- Strict result-set accuracy (answered): **{strict}/{answered}**")
        out.append(f"- Lenient (extra columns allowed): **{lenient}/{answered}**")
        if llm.last_error:
            out.append(f"- Last LLM error seen: `{llm.last_error}`")
    else:
        out.append("- LLM accuracy: not run (no GOOGLE_API_KEY or --no-llm)")
    (ROOT / "eval" / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def _fails(con, sql):
    try:
        con.execute(sql).fetchall()
        return False
    except Exception:
        return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-llm", action="store_true")
    a = p.parse_args()
    main(a.seed, not a.no_llm)
