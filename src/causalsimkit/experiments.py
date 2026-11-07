"""Market-level designs: matched-pair geo experiments (TBR + randomization inference)
and switchback tests."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

from .ols import fe_dummies, ols


# ---------------------------------------------------------------- geo experiments
def matched_pairs(volume: pd.Series, seed: int = 0) -> pd.DataFrame:
    """Rank geos by pre-period volume, pair neighbours, randomize within each pair.
    Pairing on a strong predictor of the outcome is what buys precision with few units."""
    rng = np.random.default_rng(seed)
    ordered = volume.sort_values(ascending=False).index.to_list()
    if len(ordered) % 2:
        ordered = ordered[:-1]  # drop the smallest geo to keep pairs complete
    rows = []
    for p in range(len(ordered) // 2):
        a, b = ordered[2 * p], ordered[2 * p + 1]
        flip = rng.random() < 0.5
        rows += [{"geo": a, "pair": p, "treated": int(flip)},
                 {"geo": b, "pair": p, "treated": int(not flip)}]
    return pd.DataFrame(rows)


def tbr(df: pd.DataFrame, treated_geos, test_start: int, geo: str = "geo", time: str = "day",
        y: str = "sales", alpha: float = 0.10) -> tuple[pd.DataFrame, dict]:
    """Time-Based Regression (Kerman, Wang & Vaver, 2017).

    Aggregate treatment and control groups per day, fit treat_t = a + b*control_t on the
    pre-period, predict the post-period counterfactual. The interval on the cumulative
    effect includes both parameter uncertainty and residual noise.
    """
    grp = df.assign(_g=np.where(df[geo].isin(set(treated_geos)), "treatment", "control"))
    agg = grp.pivot_table(index=time, columns="_g", values=y, aggfunc="sum")
    pre, post = agg.index < test_start, agg.index >= test_start
    fit = ols(agg.loc[pre, "treatment"], agg.loc[pre, ["control"]], cov_type="nonrobust")
    Xp = np.column_stack([np.ones(post.sum()), agg.loc[post, "control"].to_numpy()])
    pred = Xp @ fit.params.to_numpy()
    actual = agg.loc[post, "treatment"].to_numpy()
    sigma2 = float(np.sum(fit.resid**2) / fit.df_resid)
    S = np.cumsum(Xp, axis=0)
    k = np.arange(1, post.sum() + 1)
    var_cum = np.einsum("ij,jk,ik->i", S, fit.cov.to_numpy(), S) + k * sigma2
    q = stats.t.ppf(1 - alpha / 2, fit.df_resid)
    cum = np.cumsum(actual - pred)
    out = pd.DataFrame({"actual": actual, "counterfactual": pred, "incremental": actual - pred,
                        "cum_incremental": cum, "cum_lower": cum - q * np.sqrt(var_cum),
                        "cum_upper": cum + q * np.sqrt(var_cum)}, index=agg.index[post])
    summary = {"cum_incremental": float(cum[-1]), "lower": float(out["cum_lower"].iloc[-1]),
               "upper": float(out["cum_upper"].iloc[-1]),
               "relative_lift": float(cum[-1] / pred.sum()),
               "pre_r2": float(1 - np.sum(fit.resid**2) /
                               np.sum((agg.loc[pre, 'treatment'] - agg.loc[pre, 'treatment'].mean())**2))}
    return out, summary


def pair_randomization_test(pairs: pd.DataFrame, statistic: Callable[[set], float],
                            n_perm: int = 2000, seed: int = 0) -> dict:
    """Re-randomize treatment within pairs (the actual design) and recompute the statistic.
    Exact-in-design inference that needs no distributional assumptions."""
    rng = np.random.default_rng(seed)
    observed = statistic(set(pairs.loc[pairs.treated == 1, "geo"]))
    by_pair = pairs.sort_values(["pair", "treated"]).groupby("pair")["geo"].apply(list)
    null = np.empty(n_perm)
    for i in range(n_perm):
        flips = rng.random(len(by_pair)) < 0.5
        geos = {g[1] if f else g[0] for g, f in zip(by_pair, flips)}
        null[i] = statistic(geos)
    return {"observed": float(observed), "null": null,
            "p": float((np.abs(null) >= abs(observed)).mean())}


def log_ratio_did(df: pd.DataFrame, test_start: int, geo="geo", time="day", y="sales") -> Callable[[set], float]:
    """Return statistic(treated_geos) = log(T_post/T_pre) - log(C_post/C_pre)."""
    tot = df.assign(post=df[time] >= test_start).pivot_table(index=geo, columns="post", values=y, aggfunc="sum")

    def stat(treated: set) -> float:
        m = tot.index.isin(treated)
        t, c = tot[m].sum(), tot[~m].sum()
        return float(np.log(t[True] / t[False]) - np.log(c[True] / c[False]))
    return stat


# ---------------------------------------------------------------------- switchback
def switchback_assignment(n_periods: int, block_len: int, seed: int = 0) -> pd.DataFrame:
    """Randomize treatment over consecutive time blocks. Also returns `since_switch`
    (periods since the assignment last changed) for burn-in trimming."""
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n_periods / block_len))
    block = np.repeat(np.arange(n_blocks), block_len)[:n_periods]
    a = rng.integers(0, 2, n_blocks)[block]
    since = np.zeros(n_periods, dtype=int)
    for i in range(1, n_periods):
        since[i] = 0 if a[i] != a[i - 1] else since[i - 1] + 1
    return pd.DataFrame({"block": block, "assignment": a, "since_switch": since})


def analyze_switchback(df: pd.DataFrame, y: str = "y", treat: str = "treated", block: str = "block",
                       burn_in: int = 0, fe: tuple[str, ...] = ("hour", "day")) -> dict:
    """Regress outcome on treatment with time-of-day/day FEs, clustering by block (the
    unit of randomization). Periods within `burn_in` of a switch are dropped so carryover
    from the previous regime does not contaminate the comparison."""
    d = df[df["since_switch"] >= burn_in]
    X = pd.concat([d[[treat]].astype(float), fe_dummies(d, list(fe))], axis=1)
    r = ols(d[y], X, cluster=d[block])
    ci = r.conf_int().loc[treat]
    return {"estimate": float(r.params[treat]), "se": float(r.bse[treat]),
            "ci": (float(ci["lower"]), float(ci["upper"])), "n_used": int(len(d))}
