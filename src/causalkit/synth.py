"""Synthetic control (Abadie et al.) plus a DiD counterfactual with the same interface,
so the two can be compared head-to-head in a model-selection workflow."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class CounterfactualResult:
    method: str
    weights: pd.Series
    actual: pd.Series
    counterfactual: pd.Series
    treat_time: int

    @property
    def gap(self) -> pd.Series:
        return self.actual - self.counterfactual

    @property
    def pre(self) -> np.ndarray:
        return (self.actual.index < self.treat_time)

    @property
    def pre_rmspe(self) -> float:
        return float(np.sqrt(np.mean(self.gap[self.pre] ** 2)))

    @property
    def post_rmspe(self) -> float:
        return float(np.sqrt(np.mean(self.gap[~self.pre] ** 2)))

    @property
    def att(self) -> float:
        return float(self.gap[~self.pre].mean())


def simplex_weights(y: np.ndarray, Y: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """argmin ||y - Y w||^2 + ridge ||w||^2  s.t. w >= 0, sum(w) = 1."""
    J = Y.shape[1]
    obj = lambda w: float(np.sum((y - Y @ w) ** 2) + ridge * w @ w)
    jac = lambda w: -2 * Y.T @ (y - Y @ w) + 2 * ridge * w
    res = minimize(obj, np.full(J, 1 / J), jac=jac, method="SLSQP",
                   bounds=[(0, 1)] * J,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1,
                                 "jac": lambda w: np.ones(J)}],
                   options={"maxiter": 2000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def synthetic_control(wide: pd.DataFrame, treated: str, treat_time: int,
                      ridge: float = 0.0) -> CounterfactualResult:
    """`wide`: index = time, columns = units (treated + donor pool), outcome values."""
    pre = wide.index < treat_time
    donors = wide.drop(columns=treated)
    w = simplex_weights(wide.loc[pre, treated].to_numpy(), donors.loc[pre].to_numpy(), ridge)
    cf = pd.Series(donors.to_numpy() @ w, index=wide.index)
    return CounterfactualResult("synthetic_control", pd.Series(w, index=donors.columns),
                                wide[treated], cf, treat_time)


def did_counterfactual(wide: pd.DataFrame, treated: str, treat_time: int) -> CounterfactualResult:
    """DiD as a counterfactual: equal-weight donors, shifted to match the treated pre-mean."""
    pre = wide.index < treat_time
    donors = wide.drop(columns=treated)
    avg = donors.mean(axis=1)
    cf = avg + (wide.loc[pre, treated].mean() - avg[pre].mean())
    w = pd.Series(1 / donors.shape[1], index=donors.columns)
    return CounterfactualResult("did", w, wide[treated], cf, treat_time)


Method = Callable[[pd.DataFrame, str, int], CounterfactualResult]


def placebo_in_space(wide: pd.DataFrame, treated: str, treat_time: int,
                     method: Method = synthetic_control) -> pd.DataFrame:
    """Re-run `method` pretending each donor was treated (real treated unit excluded).

    Returns per-unit pre/post RMSPE and their ratio; the permutation p-value is the
    treated unit's rank of the ratio among all units.
    """
    rows = []
    pool = wide.drop(columns=treated)
    for u in [treated, *pool.columns]:
        sub = wide if u == treated else pool
        r = method(sub, u, treat_time)
        rows.append({"unit": u, "pre_rmspe": r.pre_rmspe, "post_rmspe": r.post_rmspe,
                     "att": r.att, "gap": r.gap})
    out = pd.DataFrame(rows).set_index("unit")
    out["ratio"] = out["post_rmspe"] / out["pre_rmspe"]
    return out


def permutation_pvalue(placebos: pd.DataFrame, treated: str) -> float:
    return float((placebos["ratio"] >= placebos.loc[treated, "ratio"]).mean())


def backtest(wide: pd.DataFrame, treated: str, treat_time: int, holdout: int,
             method: Method) -> float:
    """In-time placebo: use pre-period only, pretend treatment began `holdout` periods
    early, and score out-of-sample RMSPE on the pseudo-post window (true effect = 0)."""
    pre_only = wide[wide.index < treat_time]
    return method(pre_only, treated, treat_time - holdout).post_rmspe
