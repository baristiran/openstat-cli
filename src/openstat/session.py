"""Session state: holds the active dataset, command history, and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from openstat.config import get_config


@dataclass
class ModelResult:
    """Stores a fitted model's summary."""

    name: str  # e.g. "OLS", "Logit"
    formula: str  # e.g. "y ~ x1 + x2"
    table: str  # formatted text table
    details: dict  # r2, n, etc.


@dataclass
class Session:
    """Holds all state for a single analysis session."""

    df: pl.DataFrame | None = None
    dataset_path: str | None = None
    dataset_name: str | None = None
    history: list[str] = field(default_factory=list)
    results: list[ModelResult] = field(default_factory=list)
    plot_paths: list[str] = field(default_factory=list)
    _last_model: object = field(default=None, repr=False)  # last fitted statsmodels result
    _last_model_vars: tuple | None = field(default=None, repr=False)  # (dep, indeps)
    _last_fit_result: object = field(default=None, repr=False)  # last FitResult for latex export
    _last_fit_kwargs: dict = field(default_factory=dict, repr=False)  # model-specific kwargs for bootstrap
    output_dir: Path = field(default=None)  # type: ignore[assignment]
    _undo_stack: list[pl.DataFrame] = field(default_factory=list)

    def __post_init__(self) -> None:
        cfg = get_config()
        if self.output_dir is None:
            self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(self, command: str) -> None:
        """Record a command in history."""
        self.history.append(command)

    def require_data(self) -> pl.DataFrame:
        """Return the active DataFrame or raise."""
        if self.df is None:
            raise RuntimeError("No dataset loaded. Use: load <path>")
        return self.df

    def snapshot(self) -> None:
        """Save current DataFrame to undo stack (call before mutations).

        Respects max_undo_stack and max_undo_memory_mb from config.
        """
        if self.df is not None:
            cfg = get_config()
            # Memory check: estimate DataFrame size
            df_size_mb = self.df.estimated_size("mb")
            stack_size_mb = sum(d.estimated_size("mb") for d in self._undo_stack)
            if stack_size_mb + df_size_mb > cfg.max_undo_memory_mb and self._undo_stack:
                # Drop oldest snapshots to stay within budget
                while (self._undo_stack
                       and stack_size_mb + df_size_mb > cfg.max_undo_memory_mb):
                    removed = self._undo_stack.pop(0)
                    stack_size_mb -= removed.estimated_size("mb")

            self._undo_stack.append(self.df.clone())
            # Keep stack bounded by count too
            if len(self._undo_stack) > cfg.max_undo_stack:
                self._undo_stack.pop(0)

    def undo(self) -> bool:
        """Restore the previous DataFrame. Returns True if successful."""
        if not self._undo_stack:
            return False
        self.df = self._undo_stack.pop()
        return True

    @property
    def undo_depth(self) -> int:
        return len(self._undo_stack)

    @property
    def shape_str(self) -> str:
        if self.df is None:
            return "No data"
        r, c = self.df.shape
        return f"{r:,} rows x {c} columns"
