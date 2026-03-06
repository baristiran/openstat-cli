"""Time series plots: ACF, PACF, forecast."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from openstat.config import get_config


def plot_acf(series: np.ndarray, var_name: str, output_dir: Path, lags: int = 40) -> Path:
    """Plot autocorrelation function."""
    from statsmodels.graphics.tsaplots import plot_acf as sm_plot_acf

    cfg = get_config()
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))
    sm_plot_acf(series, lags=min(lags, len(series) // 2 - 1), ax=ax)
    ax.set_title(f"ACF: {var_name}")

    path = output_dir / f"acf_{var_name}.png"
    fig.savefig(path, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pacf(series: np.ndarray, var_name: str, output_dir: Path, lags: int = 40) -> Path:
    """Plot partial autocorrelation function."""
    from statsmodels.graphics.tsaplots import plot_pacf as sm_plot_pacf

    cfg = get_config()
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))
    sm_plot_pacf(series, lags=min(lags, len(series) // 2 - 1), ax=ax, method="ywm")
    ax.set_title(f"PACF: {var_name}")

    path = output_dir / f"pacf_{var_name}.png"
    fig.savefig(path, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_forecast(actual: np.ndarray, forecast: np.ndarray, var_name: str, output_dir: Path) -> Path:
    """Plot actual values with forecast extension."""
    cfg = get_config()
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))

    n_actual = len(actual)
    n_fc = len(forecast)
    ax.plot(range(n_actual), actual, label="Actual", color="blue")
    ax.plot(range(n_actual, n_actual + n_fc), forecast, label="Forecast", color="red", linestyle="--")
    ax.axvline(x=n_actual, color="gray", linestyle=":", alpha=0.5)
    ax.set_title(f"Forecast: {var_name}")
    ax.legend()

    path = output_dir / f"forecast_{var_name}.png"
    fig.savefig(path, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return path
