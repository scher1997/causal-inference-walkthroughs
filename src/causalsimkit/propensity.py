"""Propensity-score estimators: IPW, AIPW (doubly robust, cross-fitted), matching, balance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors


@dataclass
class Estimate:
    name: str
    estimate: float
    se: float | None = None

    @property
    def ci(self):
        return None if self.se is None else (self.estimate - 1.96 * self.se, self.estimate + 1.96 * self.se)


def fit_propensity(X: pd.DataFrame, t, model=None) -> np.ndarray:
    model = LogisticRegression(C=1e6, max_iter=5000) if model is None else model
    return clone(model).fit(X, t).predict_proba(X)[:, 1]


def _wmean_var(x, w):
    m = np.average(x, weights=w)
    return m, np.average((x - m) ** 2, weights=w)


def standardized_mean_diff(X: pd.DataFrame, t, weights=None) -> pd.Series:
    """SMD per covariate (pooled unweighted SD in the denominator, so weighted and
    unweighted SMDs are on the same scale). Rule of thumb: |SMD| < 0.1 is balanced."""
    t = np.asarray(t)
    w = np.ones(len(t)) if weights is None else np.asarray(weights)
    out = {}
    for c in X.columns:
        x = X[c].to_numpy(float)
        m1, _ = _wmean_var(x[t == 1], w[t == 1])
        m0, _ = _wmean_var(x[t == 0], w[t == 0])
        sd = np.sqrt((x[t == 1].var() + x[t == 0].var()) / 2)
        out[c] = (m1 - m0) / sd
    return pd.Series(out, name="smd")


def ipw_weights(t, ps, estimand: str = "ATE") -> np.ndarray:
    t, ps = np.asarray(t), np.asarray(ps)
    if estimand == "ATE":
        return np.where(t == 1, 1 / ps, 1 / (1 - ps))
    if estimand == "ATT":
        return np.where(t == 1, 1.0, ps / (1 - ps))
    raise ValueError(estimand)


def ipw(y, t, ps, estimand: str = "ATE") -> Estimate:
    """Normalized (Hajek) IPW: weighted mean difference. Use bootstrap for SEs."""
    y, t = np.asarray(y, float), np.asarray(t)
    w = ipw_weights(t, ps, estimand)
    est = np.average(y[t == 1], weights=w[t == 1]) - np.average(y[t == 0], weights=w[t == 0])
    return Estimate(f"IPW ({estimand})", float(est))


def aipw(y, t, ps, mu0, mu1) -> Estimate:
    """Augmented IPW for the ATE with influence-function SE. Consistent if EITHER the
    propensity model OR the outcome model is correct (double robustness)."""
    y, t, ps, mu0, mu1 = map(lambda a: np.asarray(a, float), (y, t, ps, mu0, mu1))
    psi = mu1 - mu0 + t * (y - mu1) / ps - (1 - t) * (y - mu0) / (1 - ps)
    return Estimate("AIPW (ATE)", float(psi.mean()), float(psi.std(ddof=1) / np.sqrt(len(y))))


def cross_fit_aipw(X: pd.DataFrame, y, t, outcome_model=None, ps_model=None,
                   n_splits: int = 5, clip: float = 0.01, seed: int = 0) -> Estimate:
    """Cross-fitted AIPW (the DML recipe): nuisance models are fit out-of-fold so flexible
    ML learners do not overfit their way into bias."""
    # `is None` checks, not `or`: sklearn ensembles define __len__ and raise before fitting
    outcome_model = LinearRegression() if outcome_model is None else outcome_model
    ps_model = LogisticRegression(C=1e6, max_iter=5000) if ps_model is None else ps_model
    y, t = np.asarray(y, float), np.asarray(t)
    n = len(y)
    ps, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ps[te] = clone(ps_model).fit(Xtr, t[tr]).predict_proba(Xte)[:, 1]
        m0 = clone(outcome_model).fit(Xtr[t[tr] == 0], y[tr][t[tr] == 0])
        m1 = clone(outcome_model).fit(Xtr[t[tr] == 1], y[tr][t[tr] == 1])
        mu0[te], mu1[te] = m0.predict(Xte), m1.predict(Xte)
    est = aipw(y, t, np.clip(ps, clip, 1 - clip), mu0, mu1)
    est.name = "Cross-fit AIPW (ATE)"
    return est


def match_att(y, t, ps, caliper: float | None = 0.05, logit: bool = True) -> tuple[Estimate, np.ndarray]:
    """1:1 nearest-neighbour matching on the (logit) propensity, with replacement.
    Caliper is in SDs of the matching score. Returns ATT and the matched control index
    per treated unit (-1 if no match within caliper)."""
    y, t, ps = np.asarray(y, float), np.asarray(t), np.asarray(ps, float)
    s = np.log(ps / (1 - ps)) if logit else ps
    ti, ci = np.where(t == 1)[0], np.where(t == 0)[0]
    nn = NearestNeighbors(n_neighbors=1).fit(s[ci, None])
    dist, j = nn.kneighbors(s[ti, None])
    match = ci[j[:, 0]]
    if caliper is not None:
        match = np.where(dist[:, 0] <= caliper * s.std(), match, -1)
    ok = match >= 0
    diffs = y[ti[ok]] - y[match[ok]]
    return Estimate(f"PS matching (ATT, {ok.mean():.0%} matched)", float(diffs.mean())), match
