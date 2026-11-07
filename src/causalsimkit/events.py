"""Financial event study: market-model abnormal returns and CARs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .ols import ols


def market_model_abnormal_returns(
    returns: pd.DataFrame,
    market: pd.Series,
    event_day: pd.Series,
    estimation_window: tuple[int, int] = (-250, -30),
    event_window: tuple[int, int] = (-5, 10),
) -> pd.DataFrame:
    """For each firm: fit r_it = a + b * r_mt on the estimation window (relative to its
    own event day), then AR_it = r_it - (a + b r_mt) inside the event window.
    The gap between windows keeps pre-event leakage out of the benchmark."""
    rows = []
    for firm, d0 in event_day.items():
        e0, e1 = d0 + estimation_window[0], d0 + estimation_window[1]
        w0, w1 = d0 + event_window[0], d0 + event_window[1]
        if e0 < returns.index.min() or w1 > returns.index.max():
            continue
        fit = ols(returns.loc[e0:e1, firm], market.loc[e0:e1].to_frame("mkt"), cov_type="nonrobust")
        a, b = fit.params["const"], fit.params["mkt"]
        sigma = float(np.std(fit.resid, ddof=2))
        ar = returns.loc[w0:w1, firm] - (a + b * market.loc[w0:w1])
        rows.append(pd.DataFrame({"firm": firm, "rel_day": ar.index - d0, "ar": ar.to_numpy(),
                                  "sigma": sigma, "beta": b}))
    return pd.concat(rows, ignore_index=True)


def aggregate_abnormal_returns(ar: pd.DataFrame) -> pd.DataFrame:
    """Average AR and cumulative AR by event day with cross-sectional t-stats
    (robust to event-induced variance changes, unlike tests using estimation-window sigma)."""
    ar = ar.sort_values(["firm", "rel_day"]).copy()
    ar["car"] = ar.groupby("firm")["ar"].cumsum()
    g = ar.groupby("rel_day")
    out = pd.DataFrame({"aar": g["ar"].mean(), "aar_se": g["ar"].sem(),
                        "caar": g["car"].mean(), "caar_se": g["car"].sem(), "n": g.size()})
    out["aar_t"] = out["aar"] / out["aar_se"]
    out["caar_t"] = out["caar"] / out["caar_se"]
    out["caar_p"] = 2 * stats.t.sf(np.abs(out["caar_t"]), out["n"] - 1)
    return out


def car(ar: pd.DataFrame, start: int, end: int) -> pd.Series:
    """Per-firm CAR over [start, end]."""
    m = ar["rel_day"].between(start, end)
    return ar[m].groupby("firm")["ar"].sum()
