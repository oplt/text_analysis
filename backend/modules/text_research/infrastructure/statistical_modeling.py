"""User-configured statistical models for research outcomes (§50).

Fits OLS or logistic regression when the user supplies an explicit outcome and
independent variables. Never auto-discovers models or causal claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SUPPORTED_MODELS = ("ols", "logistic")


def _design_matrix(
    rows: list[dict[str, Any]],
    *,
    independent_vars: list[str],
    add_intercept: bool = True,
) -> tuple[np.ndarray, list[str]]:
    if not independent_vars:
        raise ValueError("independent_vars must be non-empty")
    data = []
    for row in rows:
        data.append([float(row[name]) for name in independent_vars])
    x = np.asarray(data, dtype=float)
    names = list(independent_vars)
    if add_intercept:
        x = np.column_stack([np.ones(len(rows)), x])
        names = ["Intercept", *names]
    return x, names


def _outcome_vector(rows: list[dict[str, Any]], dependent_var: str) -> np.ndarray:
    return np.asarray([float(row[dependent_var]) for row in rows], dtype=float)


def fit_ols(
    rows: list[dict[str, Any]],
    *,
    dependent_var: str,
    independent_vars: list[str],
    add_intercept: bool = True,
) -> dict[str, Any]:
    """Ordinary least squares with coefficient SE, CI, and p-values."""
    from scipy import stats

    if len(rows) < 3:
        raise ValueError("OLS requires at least 3 observations")
    y = _outcome_vector(rows, dependent_var)
    x, names = _design_matrix(rows, independent_vars=independent_vars, add_intercept=add_intercept)
    n, k = x.shape
    if n <= k:
        raise ValueError(f"OLS needs n > k (got n={n}, k={k})")

    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Design matrix is singular; check collinear predictors") from exc

    beta = xtx_inv @ (x.T @ y)
    fitted = x @ beta
    resid = y - fitted
    df_resid = n - k
    sigma2 = float(resid @ resid / df_resid)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    t_stats = beta / se
    p_values = 2 * stats.t.sf(np.abs(t_stats), df_resid)
    t_crit = float(stats.t.ppf(0.975, df_resid))
    ci_low = beta - t_crit * se
    ci_high = beta + t_crit * se
    ss_tot = float(((y - y.mean()) ** 2).sum())
    ss_res = float((resid**2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / df_resid if df_resid else r2

    coefficients = []
    for i, name in enumerate(names):
        coefficients.append(
            {
                "term": name,
                "coefficient": float(beta[i]),
                "std_error": float(se[i]),
                "t": float(t_stats[i]),
                "p_value": float(p_values[i]),
                "ci_low": float(ci_low[i]),
                "ci_high": float(ci_high[i]),
            }
        )

    return {
        "model": "ols",
        "dependent_var": dependent_var,
        "independent_vars": list(independent_vars),
        "add_intercept": add_intercept,
        "n_observations": n,
        "n_parameters": k,
        "df_resid": df_resid,
        "r_squared": r2,
        "adj_r_squared": adj_r2,
        "sigma2": sigma2,
        "coefficients": coefficients,
        "causal_claim": False,
        "notes": [
            "User-specified specification only; no automatic model search.",
            "Do not interpret coefficients as causal effects without a design.",
        ],
    }


def fit_logistic(
    rows: list[dict[str, Any]],
    *,
    dependent_var: str,
    independent_vars: list[str],
    add_intercept: bool = True,
) -> dict[str, Any]:
    """Binary logistic regression via statsmodels when available."""
    try:
        import statsmodels.api as sm
    except ImportError as exc:
        raise ValueError(
            "Logistic regression requires the optional statsmodels package. "
            "Install statsmodels or use model='ols'."
        ) from exc

    y = _outcome_vector(rows, dependent_var)
    unique = set(np.unique(y).tolist())
    if not unique.issubset({0.0, 1.0}):
        raise ValueError("Logistic dependent_var must be binary coded as 0/1")
    x, names = _design_matrix(rows, independent_vars=independent_vars, add_intercept=add_intercept)
    if x.shape[0] < x.shape[1] + 1:
        raise ValueError("Not enough observations for logistic regression")

    model = sm.Logit(y, x)
    result = model.fit(disp=False)
    conf = result.conf_int(alpha=0.05)
    coefficients = []
    for i, name in enumerate(names):
        coefficients.append(
            {
                "term": name,
                "coefficient": float(result.params[i]),
                "std_error": float(result.bse[i]),
                "z": float(result.tvalues[i]),
                "p_value": float(result.pvalues[i]),
                "ci_low": float(conf[i, 0]),
                "ci_high": float(conf[i, 1]),
            }
        )
    return {
        "model": "logistic",
        "dependent_var": dependent_var,
        "independent_vars": list(independent_vars),
        "add_intercept": add_intercept,
        "n_observations": int(x.shape[0]),
        "n_parameters": int(x.shape[1]),
        "pseudo_r_squared": float(result.prsquared),
        "llf": float(result.llf),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "coefficients": coefficients,
        "causal_claim": False,
        "notes": [
            "User-specified specification only; no automatic model search.",
            "Do not interpret coefficients as causal effects without a design.",
        ],
    }


def fit_statistical_model(
    rows: list[dict[str, Any]],
    *,
    model: str,
    dependent_var: str,
    independent_vars: list[str],
    add_intercept: bool = True,
) -> dict[str, Any]:
    model_key = (model or "").strip().lower()
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model!r}; use one of {SUPPORTED_MODELS}")
    if model_key == "ols":
        return fit_ols(
            rows,
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            add_intercept=add_intercept,
        )
    return fit_logistic(
        rows,
        dependent_var=dependent_var,
        independent_vars=independent_vars,
        add_intercept=add_intercept,
    )
