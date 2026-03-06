"""ARCH/GARCH volatility models (requires 'arch' package)."""

from __future__ import annotations

import numpy as np
import polars as pl


def _require_arch():
    try:
        import arch  # noqa: F401
        return arch
    except ImportError:
        raise ImportError(
            "'arch' package is required for ARCH/GARCH models.\n"
            "Install: pip install arch"
        )


def fit_arch(
    df: pl.DataFrame,
    var: str,
    *,
    p: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
) -> dict:
    """ARCH(p) model for volatility clustering."""
    arch_pkg = _require_arch()
    from arch import arch_model  # type: ignore[import]

    y = df[var].drop_nulls().to_numpy().astype(float) * 100  # scale returns

    am = arch_model(y, mean=mean, vol="ARCH", p=p, dist=dist)
    res = am.fit(disp="off")

    params = {k: float(v) for k, v in res.params.items()}
    return {
        "model": f"ARCH({p})",
        "var": var,
        "n_obs": len(y),
        "params": params,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "log_likelihood": float(res.loglikelihood),
        "_result": res,
    }


def fit_garch(
    df: pl.DataFrame,
    var: str,
    *,
    p: int = 1,
    q: int = 1,
    mean: str = "Constant",
    dist: str = "normal",
    model: str = "GARCH",
) -> dict:
    """GARCH(p,q) or GJR-GARCH / EGARCH volatility model."""
    _require_arch()
    from arch import arch_model  # type: ignore[import]

    y = df[var].drop_nulls().to_numpy().astype(float) * 100

    # model: GARCH, EGARCH, GJR-GARCH
    vol = model.upper()
    am = arch_model(y, mean=mean, vol=vol, p=p, q=q, dist=dist)
    res = am.fit(disp="off")

    params = {k: float(v) for k, v in res.params.items()}
    cond_vol = res.conditional_volatility.tolist()

    return {
        "model": f"{model}({p},{q})",
        "var": var,
        "n_obs": len(y),
        "params": params,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "log_likelihood": float(res.loglikelihood),
        "cond_volatility_last5": cond_vol[-5:],
        "_result": res,
    }
