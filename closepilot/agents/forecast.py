"""13-week cash forecast: statsforecast baselines + backtest + CFO commentary."""
from __future__ import annotations

import logging

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, Naive, WindowAverage

from closepilot import llm

logging.getLogger("statsforecast").setLevel(logging.ERROR)
H = 13


def _weekly(con) -> pd.DataFrame:
    df = con.execute("""
        SELECT CAST(posting_date AS DATE) AS d, debit AS receipts, credit AS payments
        FROM gl WHERE account = 'Bank'""").df()
    df["d"] = pd.to_datetime(df["d"])
    w = df.set_index("d")[["receipts", "payments"]].resample("W").sum()
    return w.iloc[1:-1]  # drop partial first/last weeks


def _long(w: pd.DataFrame, col: str) -> pd.DataFrame:
    return pd.DataFrame({"unique_id": col, "ds": w.index, "y": w[col].values})


def _models():
    return [AutoETS(season_length=4), Naive(), WindowAverage(window_size=4)]


def _fit_predict(train: pd.DataFrame, h: int) -> pd.DataFrame:
    sf = StatsForecast(models=_models(), freq="W", n_jobs=1)
    return sf.forecast(df=train, h=h).reset_index()


def backtest(w: pd.DataFrame) -> pd.DataFrame:
    train = pd.concat([_long(w.iloc[:-H], c) for c in ("receipts", "payments")])
    fc = _fit_predict(train, H)
    actual = pd.concat([_long(w.iloc[-H:], c) for c in ("receipts", "payments")])
    m = fc.merge(actual, on=["unique_id", "ds"])
    rows = []
    for uid, g in m.groupby("unique_id"):
        for model in ("AutoETS", "Naive", "WindowAverage"):
            rows.append(dict(series=uid, model=model, wape=round((g[model] - g.y).abs().sum() / g.y.abs().sum(), 4)))
    return pd.DataFrame(rows)


def _scheduled(con, last: pd.Timestamp) -> pd.DataFrame:
    """Open AP/AR already due inside the horizon, bucketed to the same weekly grid (direct-method overlay)."""
    ap = con.execute("""SELECT CAST(due_date AS DATE) AS d, net_payable AS amt FROM invoices
                        WHERE invoice_id NOT IN (SELECT invoice_id FROM payments)""").df()
    ar = con.execute("SELECT CAST(due_date AS DATE) AS d, amount AS amt FROM ar_invoices WHERE paid_date IS NULL").df()
    out = {}
    for name, df in (("scheduled_ap", ap), ("scheduled_ar", ar)):
        df["d"] = pd.to_datetime(df["d"])
        df.loc[df.d <= last, "d"] = last + pd.Timedelta(days=1)  # overdue items land in week 1
        out[name] = df.set_index("d").amt.resample("W").sum()
    return pd.DataFrame(out).fillna(0.0)


def run_forecast(con) -> dict:
    w = _weekly(con)
    bt = backtest(w)
    best = bt.loc[bt.groupby("series").wape.idxmin()].set_index("series")  # selected on holdout: slightly optimistic
    train = pd.concat([_long(w, c) for c in ("receipts", "payments")])
    fc = _fit_predict(train, H)
    pv = pd.DataFrame({s: fc[fc.unique_id == s].set_index("ds")[best.at[s, "model"]] for s in ("receipts", "payments")})
    pv["net"] = pv["receipts"] - pv["payments"]
    pv["cumulative_net"] = pv["net"].cumsum()
    pv = pv.join(_scheduled(con, w.index[-1]), how="left").fillna(0.0)
    open_ar = float(pv.scheduled_ar.sum())
    open_ap = float(pv.scheduled_ap.sum())
    facts = dict(horizon_weeks=H, net_13w=float(pv.net.sum()), min_cumulative=float(pv.cumulative_net.min()),
                 open_ar_due_in_horizon=open_ar, open_ap_due_in_horizon=open_ap,
                 model_receipts=best.at["receipts", "model"], wape_receipts=float(best.at["receipts", "wape"]),
                 model_payments=best.at["payments", "model"], wape_payments=float(best.at["payments", "wape"]))
    return dict(forecast=pv.reset_index(), backtest=bt, facts=facts, commentary=commentary(facts))


def commentary(f: dict) -> str:
    fallback = (f"Expected net cash over the next {f['horizon_weeks']} weeks is INR {f['net_13w']:,.0f}. "
                f"Open receivables due in the window: INR {f['open_ar_due_in_horizon']:,.0f}; unpaid payables due: INR {f['open_ap_due_in_horizon']:,.0f}. "
                f"Backtest WAPE: receipts {f['wape_receipts']:.0%}, payments {f['wape_payments']:.0%}.")
    out = llm.text("You are a finance analyst. Write 3 concise sentences of CFO commentary on this 13-week cash "
                   f"forecast. Use only these numbers (INR): {f}. Mention forecast uncertainty from the WAPE.")
    return out or fallback
