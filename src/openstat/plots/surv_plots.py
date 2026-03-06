"""Survival analysis plots: Kaplan-Meier curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from openstat.config import get_config


def plot_km(kmf_objects, output_dir: Path, group_var: str | None = None) -> Path:
    """Plot Kaplan-Meier survival curve(s)."""
    cfg = get_config()
    fig, ax = plt.subplots(figsize=(cfg.plot_figsize_w, cfg.plot_figsize_h))

    if isinstance(kmf_objects, list):
        for kmf in kmf_objects:
            kmf.plot_survival_function(ax=ax)
    else:
        kmf_objects.plot_survival_function(ax=ax)

    ax.set_title("Kaplan-Meier Survival Estimate" + (f" by {group_var}" if group_var else ""))
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.set_ylim(0, 1)

    name = f"km_{group_var}" if group_var else "km"
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=cfg.plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return path
