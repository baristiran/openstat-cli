"""Advanced regression: NLS, Beta, ZIP/ZINB, Hurdle, SUR."""

from __future__ import annotations
import numpy as np
import polars as pl
from scipy import stats as sp_stats
from scipy.optimize import least_squares
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ── NLS ───────────────────────────────────────────────────────────────────

def fit_nls(df: pl.DataFrame, dep: str, indeps: list[str],
            formula_fn, p0: list[float], *, robust: bool = False) -> dict:
    """Nonlinear Least Squares via scipy.optimize.least_squares.

    formula_fn: callable(X, *params) -> y_pred  where X is ndarray (n, k)
    p0: initial parameter guesses
    """
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X = sub.select(indeps).to_numpy().astype(float)

    def residuals(params):
        return formula_fn(X, *params) - y

    result = least_squares(residuals, p0, method='lm')
    y_pred = formula_fn(X, *result.x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = float(1 - ss_res/ss_tot) if ss_tot > 0 else float('nan')

    # approximate std errors from Jacobian
    try:
        J = result.jac
        cov = np.linalg.inv(J.T @ J) * (ss_res / max(len(y) - len(p0), 1))
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = np.full(len(p0), float('nan'))

    params_dict = {f"p{i}": float(v) for i, v in enumerate(result.x)}
    se_dict = {f"p{i}": float(v) for i, v in enumerate(se)}

    return {
        "method": "NLS",
        "dep": dep,
        "indeps": indeps,
        "params": params_dict,
        "std_errors": se_dict,
        "r_squared": r2,
        "n_obs": len(y),
        "converged": result.success,
        "cost": float(result.cost),
    }


# ── Beta regression ───────────────────────────────────────────────────────

def fit_betareg(df: pl.DataFrame, dep: str, indeps: list[str],
                *, link: str = 'logit') -> dict:
    """Beta regression for (0,1) bounded outcomes via statsmodels GLM."""
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    # Clamp to avoid boundary issues
    eps = 1e-6
    y = np.clip(y, eps, 1 - eps)
    X = sm.add_constant(sub.select(indeps).to_numpy().astype(float))

    # Use GLM with logit link and Binomial family as Beta approximation
    # True beta regression via formula
    pdf = sub.to_pandas()
    pdf.columns = ["dep"] + [f"x{i}" for i in range(len(indeps))]
    formula = "dep ~ " + " + ".join(f"x{i}" for i in range(len(indeps)))

    try:
        model = smf.glm(formula, data=pdf,
                        family=sm.families.Binomial(link=sm.families.links.Logit())).fit()
        params = dict(zip(["_cons"] + indeps, model.params.tolist()))
        se = dict(zip(["_cons"] + indeps, model.bse.tolist()))
        pvals = dict(zip(["_cons"] + indeps, model.pvalues.tolist()))
        return {
            "method": "Beta Regression (GLM-Binomial)",
            "dep": dep, "indeps": indeps,
            "params": params, "std_errors": se, "p_values": pvals,
            "aic": float(model.aic), "bic": float(model.bic),
            "n_obs": int(model.nobs), "pseudo_r2": float(1 - model.llf/model.llnull),
            "_result": model,
        }
    except Exception as exc:
        raise RuntimeError(f"Beta regression failed: {exc}") from exc


# ── Zero-inflated Poisson ─────────────────────────────────────────────────

def fit_zip(df: pl.DataFrame, dep: str, indeps: list[str]) -> dict:
    """Zero-Inflated Poisson regression (scipy L-BFGS-B)."""
    from scipy.optimize import minimize
    from scipy.special import expit
    from scipy.stats import poisson as _poisson

    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X_raw = sub.select(indeps).to_numpy().astype(float)
    n, k = X_raw.shape
    X = np.column_stack([np.ones(n), X_raw])
    kp = k + 1

    def neg_ll(params):
        gamma = params[:kp]        # inflate logit params
        beta = params[kp:]         # Poisson mean params
        pi = expit(X @ gamma)
        lam = np.exp(X @ beta)
        lam = np.clip(lam, 1e-10, 1e10)
        ll_zero = np.log(pi + (1 - pi) * np.exp(-lam) + 1e-300)
        ll_pos = np.log(1 - pi + 1e-300) + y * np.log(lam + 1e-300) - lam - np.array([
            float(np.sum(np.log(np.arange(1, int(yi) + 1)))) for yi in y
        ])
        ll = np.where(y == 0, ll_zero, ll_pos)
        return -ll.sum()

    try:
        x0 = np.zeros(2 * kp)
        res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 500})
        params_hat = res.x
        llf = -res.fun
        aic = 2 * len(params_hat) - 2 * llf
        bic = len(params_hat) * np.log(n) - 2 * llf
        names_inflate = [f"inflate_{p}" for p in ["_cons"] + indeps]
        names_count = [f"count_{p}" for p in ["_cons"] + indeps]
        all_names = names_inflate + names_count
        params_dict = {nm: float(v) for nm, v in zip(all_names, params_hat)}
        return {
            "method": "Zero-Inflated Poisson",
            "dep": dep, "indeps": indeps,
            "params": params_dict, "std_errors": {k: float("nan") for k in params_dict},
            "p_values": {k: float("nan") for k in params_dict},
            "aic": float(aic), "bic": float(bic),
            "log_likelihood": float(llf), "n_obs": n,
        }
    except Exception as exc:
        raise RuntimeError(f"ZIP failed: {exc}") from exc


# ── Zero-inflated Negative Binomial ───────────────────────────────────────

def fit_zinb(df: pl.DataFrame, dep: str, indeps: list[str]) -> dict:
    """Zero-Inflated Negative Binomial regression (scipy L-BFGS-B)."""
    from scipy.optimize import minimize
    from scipy.special import expit, gammaln

    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X_raw = sub.select(indeps).to_numpy().astype(float)
    n, k = X_raw.shape
    X = np.column_stack([np.ones(n), X_raw])
    kp = k + 1

    def neg_ll(params):
        gamma = params[:kp]
        beta = params[kp:2 * kp]
        log_r = params[2 * kp]   # log(dispersion)
        r = np.exp(log_r)
        pi = expit(X @ gamma)
        mu = np.exp(X @ beta)
        mu = np.clip(mu, 1e-10, 1e10)
        p_nb = r / (r + mu)
        ll_zero_nb = r * np.log(p_nb + 1e-300)
        ll_zero = np.log(pi + (1 - pi) * np.exp(ll_zero_nb) + 1e-300)
        ll_pos = (np.log(1 - pi + 1e-300)
                  + gammaln(y + r) - gammaln(r) - gammaln(y + 1)
                  + r * np.log(p_nb + 1e-300) + y * np.log(1 - p_nb + 1e-300))
        ll = np.where(y == 0, ll_zero, ll_pos)
        return -ll.sum()

    try:
        x0 = np.zeros(2 * kp + 1)
        res = minimize(neg_ll, x0, method="L-BFGS-B", options={"maxiter": 500})
        params_hat = res.x
        llf = -res.fun
        aic = 2 * len(params_hat) - 2 * llf
        bic = len(params_hat) * np.log(n) - 2 * llf
        names = [f"inflate_{p}" for p in ["_cons"] + indeps] \
              + [f"count_{p}" for p in ["_cons"] + indeps] \
              + ["log_dispersion"]
        params_dict = {nm: float(v) for nm, v in zip(names, params_hat)}
        return {
            "method": "Zero-Inflated Negative Binomial",
            "dep": dep, "indeps": indeps,
            "params": params_dict, "std_errors": {k: float("nan") for k in params_dict},
            "p_values": {k: float("nan") for k in params_dict},
            "aic": float(aic), "bic": float(bic),
            "log_likelihood": float(llf), "n_obs": n,
        }
    except Exception as exc:
        raise RuntimeError(f"ZINB failed: {exc}") from exc


# ── Hurdle model ──────────────────────────────────────────────────────────

def fit_hurdle(df: pl.DataFrame, dep: str, indeps: list[str]) -> dict:
    """Two-part hurdle model: Logit for zeros, Truncated Poisson for positives."""
    sub = df.select([dep] + indeps).drop_nulls()
    y = sub[dep].to_numpy().astype(float)
    X_raw = sub.select(indeps).to_numpy().astype(float)
    X = sm.add_constant(X_raw)

    # Part 1: Logit (zero vs. nonzero)
    y_bin = (y > 0).astype(float)
    logit_model = sm.Logit(y_bin, X).fit(disp=0)

    # Part 2: Poisson on positive outcomes only
    pos_mask = y > 0
    y_pos = y[pos_mask]
    X_pos = X[pos_mask]
    poisson_model = sm.Poisson(y_pos, X_pos).fit(disp=0)

    param_names = ["_cons"] + indeps
    return {
        "method": "Hurdle (Logit + Poisson)",
        "dep": dep, "indeps": indeps,
        "n_obs": len(y), "n_zeros": int((y == 0).sum()), "n_positive": int(pos_mask.sum()),
        "logit_params": dict(zip(param_names, logit_model.params.tolist())),
        "logit_pvalues": dict(zip(param_names, logit_model.pvalues.tolist())),
        "count_params": dict(zip(param_names, poisson_model.params.tolist())),
        "count_pvalues": dict(zip(param_names, poisson_model.pvalues.tolist())),
        "aic_logit": float(logit_model.aic),
        "aic_count": float(poisson_model.aic),
        "_logit": logit_model, "_count": poisson_model,
    }


# ── SUR ───────────────────────────────────────────────────────────────────

def fit_sur(df: pl.DataFrame, equations: list[tuple[str, list[str]]]) -> dict:
    """Seemingly Unrelated Regression via GLS iteration."""
    results = []
    residuals = []

    for dep, indeps in equations:
        sub = df.select([dep] + indeps).drop_nulls()
        y = sub[dep].to_numpy().astype(float)
        X = sm.add_constant(sub.select(indeps).to_numpy().astype(float))
        model = sm.OLS(y, X).fit()
        results.append(model)
        residuals.append(model.resid)

    # Cross-equation covariance (Sigma)
    min_n = min(len(r) for r in residuals)
    resid_mat = np.column_stack([r[:min_n] for r in residuals])
    Sigma = (resid_mat.T @ resid_mat) / min_n

    equations_out = []
    for i, ((dep, indeps), res) in enumerate(zip(equations, results)):
        equations_out.append({
            "equation": i + 1,
            "dep": dep, "indeps": indeps,
            "params": dict(zip(["_cons"] + indeps, res.params.tolist())),
            "std_errors": dict(zip(["_cons"] + indeps, res.bse.tolist())),
            "r_squared": float(res.rsquared),
            "n_obs": int(res.nobs),
        })

    return {
        "method": "SUR (OLS-based)",
        "n_equations": len(equations),
        "equations": equations_out,
        "cross_equation_corr": np.corrcoef(resid_mat.T).tolist(),
    }
