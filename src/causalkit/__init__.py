"""causalkit: small, dependency-light estimators backing the example notebooks.

Every estimator here is deliberately transparent (numpy/scipy/sklearn only) so the
notebooks can show the mechanics. For production work, cross-check against the
mature packages listed in the README (statsmodels, rdrobust, pysyncon, csdid, ...).
"""
from .ols import OLSResult, ols, fe_dummies

__all__ = ["OLSResult", "ols", "fe_dummies"]
__version__ = "0.1.0"
