"""Ordinary / weighted least squares with HC1 or cluster-robust (CR1) covariance."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class OLSResult:
    params: pd.Series
    cov: pd.DataFrame
    resid: np.ndarray
    nobs: int
    df_resid: int
    cov_type: str
    n_clusters: int | None = None

    @property
    def bse(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.cov.to_numpy())), index=self.params.index)

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.bse

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series(2 * stats.t.sf(np.abs(self.tvalues), self.df_resid), index=self.params.index)

    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        q = stats.t.ppf(1 - alpha / 2, self.df_resid)
        return pd.DataFrame({"lower": self.params - q * self.bse, "upper": self.params + q * self.bse})

    def summary(self, names: Sequence[str] | None = None, alpha: float = 0.05) -> pd.DataFrame:
        ci = self.conf_int(alpha)
        out = pd.DataFrame(
            {"coef": self.params, "se": self.bse, "t": self.tvalues, "p": self.pvalues,
             "ci_lower": ci["lower"], "ci_upper": ci["upper"]}
        )
        return out.loc[list(names)] if names is not None else out

    def wald_test(self, names: Iterable[str]) -> dict:
        """Joint F-test that all listed coefficients equal zero (e.g. pre-trend leads)."""
        names = list(names)
        b = self.params.loc[names].to_numpy()
        V = self.cov.loc[names, names].to_numpy()
        q = len(names)
        f_stat = float(b @ np.linalg.solve(V, b)) / q
        return {"F": f_stat, "df_num": q, "df_den": self.df_resid,
                "p": float(stats.f.sf(f_stat, q, self.df_resid))}


def fe_dummies(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """One-hot fixed effects (first level dropped) as float columns."""
    return pd.concat(
        [pd.get_dummies(df[c], prefix=c, drop_first=True, dtype=float) for c in cols], axis=1
    )


def ols(
    y,
    X,
    *,
    cluster=None,
    weights=None,
    add_const: bool = True,
    cov_type: str = "HC1",
) -> OLSResult:
    """Fit (weighted) least squares.

    cov_type: "HC1" (default), "nonrobust", or ignored when `cluster` is given (CR1).
    Degrees of freedom for inference are G-1 under clustering, n-k otherwise.
    """
    X = pd.DataFrame(X).copy()
    if not all(isinstance(c, str) for c in X.columns):
        X.columns = [str(c) for c in X.columns]
    X = X.astype(float)
    if add_const:
        X.insert(0, "const", 1.0)
    y_arr = np.asarray(y, dtype=float)
    Xm = X.to_numpy()
    if np.isnan(y_arr).any() or np.isnan(Xm).any():
        raise ValueError("NaNs in y or X; drop or impute before fitting.")

    if weights is not None:
        sw = np.sqrt(np.asarray(weights, dtype=float))
        Xw, yw = Xm * sw[:, None], y_arr * sw
    else:
        Xw, yw = Xm, y_arr

    n = Xw.shape[0]
    rank = np.linalg.matrix_rank(Xw)
    bread = np.linalg.pinv(Xw.T @ Xw)
    beta = bread @ (Xw.T @ yw)
    u = yw - Xw @ beta

    n_clusters = None
    if cluster is not None:
        codes = pd.factorize(np.asarray(cluster))[0]
        G = codes.max() + 1
        S = np.zeros((G, Xw.shape[1]))
        np.add.at(S, codes, Xw * u[:, None])
        meat = S.T @ S
        scale = G / (G - 1) * (n - 1) / (n - rank)
        cov = scale * bread @ meat @ bread
        df_resid, kind, n_clusters = G - 1, "cluster", G
    elif cov_type == "HC1":
        Xu = Xw * u[:, None]
        cov = n / (n - rank) * bread @ (Xu.T @ Xu) @ bread
        df_resid, kind = n - rank, "HC1"
    elif cov_type == "nonrobust":
        cov = (u @ u) / (n - rank) * bread
        df_resid, kind = n - rank, "nonrobust"
    else:
        raise ValueError(f"unknown cov_type {cov_type!r}")

    names = list(X.columns)
    return OLSResult(
        params=pd.Series(beta, index=names),
        cov=pd.DataFrame(cov, index=names, columns=names),
        resid=y_arr - Xm @ beta,
        nobs=n,
        df_resid=int(df_resid),
        cov_type=kind,
        n_clusters=n_clusters,
    )
