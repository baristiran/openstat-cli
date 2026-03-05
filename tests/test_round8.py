"""Round 8 tests: new features, fixes, and infrastructure.

- Welch t-test df (Satterthwaite)
- Adjusted R², F-stat, AIC/BIC
- Tokenizer unmatched character detection
- Studentized residuals
- Config system
- CommandArgs helper
- Logging setup
- Script strict mode
"""

import math

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.config import Config, get_config
from openstat.dsl.tokenizer import tokenize
from openstat.dsl.parser import parse_expression, ParseError
from openstat.stats.models import (
    fit_ols, fit_logit, fit_probit, run_ttest, compute_residuals, compute_vif,
)
from openstat.commands.base import CommandArgs, rich_to_str
from openstat.logging_config import setup_logging


# ── Welch df fix ───────────────────────────────────────────────────────

class TestWelchDfFix:
    def _make_df(self, n1, n2, seed=42):
        rng = np.random.default_rng(seed)
        g1 = rng.normal(10, 2, n1)
        g2 = rng.normal(12, 5, n2)
        vals = np.concatenate([g1, g2]).tolist()
        groups = (["A"] * n1 + ["B"] * n2)
        return pl.DataFrame({"val": vals, "grp": groups})

    def test_welch_df_not_pooled(self):
        """Welch df should differ from pooled df when variances differ."""
        df = self._make_df(30, 30)
        result = run_ttest(df, "val", by="grp")
        pooled_df = 30 + 30 - 2  # 58
        # Welch df should be < pooled df when variances differ
        assert result.df != pooled_df

    def test_welch_df_reasonable(self):
        """Welch df should be between 0 and n1+n2-2."""
        df = self._make_df(20, 50)
        result = run_ttest(df, "val", by="grp")
        assert 1 < result.df < 20 + 50 - 2

    def test_welch_df_equal_variance(self):
        """When variances are equal, Welch df ~ pooled df."""
        rng = np.random.default_rng(42)
        n = 100
        vals = rng.normal(0, 1, 2 * n).tolist()
        df = pl.DataFrame({"val": vals, "grp": ["A"] * n + ["B"] * n})
        result = run_ttest(df, "val", by="grp")
        pooled_df = 2 * n - 2
        # Should be close to pooled df (within ~5%)
        assert abs(result.df - pooled_df) < pooled_df * 0.1


# ── Adjusted R², F-stat, AIC/BIC ──────────────────────────────────────

class TestExtendedOLSOutput:
    @pytest.fixture
    def ols_df(self):
        rng = np.random.default_rng(42)
        n = 100
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        y = 2 + 3 * x1 + 0.5 * x2 + rng.normal(0, 0.5, n)
        return pl.DataFrame({
            "y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist(),
        })

    def test_adj_r_squared_present(self, ols_df):
        result, _ = fit_ols(ols_df, "y", ["x1", "x2"])
        assert result.adj_r_squared is not None
        assert 0 < result.adj_r_squared <= 1

    def test_adj_r_squared_less_than_r_squared(self, ols_df):
        result, _ = fit_ols(ols_df, "y", ["x1", "x2"])
        assert result.adj_r_squared <= result.r_squared

    def test_f_statistic_present(self, ols_df):
        result, _ = fit_ols(ols_df, "y", ["x1", "x2"])
        assert result.f_statistic is not None
        assert result.f_statistic > 0
        assert result.f_pvalue is not None
        assert result.f_pvalue < 0.05  # strong relationship

    def test_aic_bic_present(self, ols_df):
        result, _ = fit_ols(ols_df, "y", ["x1", "x2"])
        assert result.aic is not None
        assert result.bic is not None

    def test_logit_aic_bic(self):
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        p = 1 / (1 + np.exp(-(x * 2)))
        y = (rng.random(n) < p).astype(float)
        df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        result, _ = fit_logit(df, "y", ["x"])
        assert result.aic is not None
        assert result.bic is not None

    def test_summary_table_contains_new_fields(self, ols_df):
        result, _ = fit_ols(ols_df, "y", ["x1", "x2"])
        summary = result.summary_table()
        assert "Adj.R²" in summary
        assert "AIC" in summary
        assert "BIC" in summary
        assert "F(" in summary

    def test_to_markdown_contains_new_fields(self, ols_df):
        result, _ = fit_ols(ols_df, "y", ["x1", "x2"])
        md = result.to_markdown()
        assert "Adj. R²" in md
        assert "AIC" in md
        assert "F-statistic" in md

    def test_to_latex_contains_new_fields(self, ols_df):
        result, _ = fit_ols(ols_df, "y", ["x1", "x2"])
        latex = result.to_latex()
        assert "Adj.$R^2$" in latex
        assert "AIC" in latex


# ── Tokenizer unmatched character detection ────────────────────────────

class TestTokenizerValidation:
    def test_unmatched_character_raises(self):
        with pytest.raises(ValueError, match="Unexpected character"):
            tokenize("x ¶ 5")

    def test_unmatched_at_sign(self):
        with pytest.raises(ValueError, match="Unexpected character"):
            tokenize("x @ 5")

    def test_unmatched_hash(self):
        with pytest.raises(ValueError, match="Unexpected character"):
            tokenize("x # 5")

    def test_valid_expression_still_works(self):
        tokens = tokenize("x > 5 and y < 10")
        names = [t.type.name for t in tokens]
        assert "IDENT" in names
        assert "AND" in names

    def test_trailing_whitespace_ok(self):
        tokens = tokenize("x > 5   ")
        assert tokens[-1].type.name == "EOF"

    def test_unmatched_unicode(self):
        with pytest.raises(ValueError, match="Unexpected character"):
            tokenize("x → 5")


# ── Studentized residuals ──────────────────────────────────────────────

class TestStudentizedResiduals:
    def test_residuals_have_leverage(self):
        rng = np.random.default_rng(42)
        n = 50
        x = rng.normal(0, 1, n)
        y = 2 + 3 * x + rng.normal(0, 0.5, n)
        df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        _, model = fit_ols(df, "y", ["x"])
        diag = compute_residuals(model, df, "y", ["x"])
        assert "leverage" in diag
        assert len(diag["leverage"]) == n

    def test_leverage_values_reasonable(self):
        rng = np.random.default_rng(42)
        n = 50
        x = rng.normal(0, 1, n)
        y = 2 + 3 * x + rng.normal(0, 0.5, n)
        df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        _, model = fit_ols(df, "y", ["x"])
        diag = compute_residuals(model, df, "y", ["x"])
        # Leverage should be between 0 and 1
        assert np.all(diag["leverage"] >= 0)
        assert np.all(diag["leverage"] <= 1)
        # Average leverage should be ~ p/n
        p = 2  # constant + x
        assert abs(np.mean(diag["leverage"]) - p / n) < 0.05

    def test_studentized_residuals_approx_standard_normal(self):
        """For well-specified model, studentized residuals ~ N(0,1)."""
        rng = np.random.default_rng(42)
        n = 200
        x = rng.normal(0, 1, n)
        y = 2 + 3 * x + rng.normal(0, 1, n)
        df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        _, model = fit_ols(df, "y", ["x"])
        diag = compute_residuals(model, df, "y", ["x"])
        std_r = diag["std_residuals"]
        # Mean should be close to 0, std close to 1
        assert abs(np.mean(std_r)) < 0.2
        assert abs(np.std(std_r) - 1.0) < 0.2


# ── Config system ─────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.max_undo_stack == 20
        assert cfg.plot_dpi == 150
        assert cfg.tabulate_limit == 50

    def test_get_config_returns_instance(self):
        cfg = get_config()
        assert isinstance(cfg, Config)
        assert cfg.max_undo_stack > 0

    def test_config_memory_limit(self):
        cfg = Config()
        assert cfg.max_undo_memory_mb == 500


# ── CommandArgs helper ─────────────────────────────────────────────────

class TestCommandArgs:
    def test_positional(self):
        ca = CommandArgs("col1 col2 col3")
        assert ca.positional == ["col1", "col2", "col3"]

    def test_flags(self):
        ca = CommandArgs("y ~ x1 + x2 --robust")
        assert ca.has_flag("--robust")
        assert not ca.has_flag("--verbose")

    def test_options(self):
        ca = CommandArgs("file.csv on key how=left")
        assert ca.get_option("how") == "left"
        assert ca.get_option("missing", "default") == "default"

    def test_rest_after(self):
        ca = CommandArgs("score by region")
        rest = ca.rest_after("by")
        assert rest == "region"

    def test_rest_after_missing(self):
        ca = CommandArgs("score region")
        assert ca.rest_after("by") is None

    def test_strip_flags_and_options(self):
        ca = CommandArgs("y ~ x1 + x2 --robust how=left")
        stripped = ca.strip_flags_and_options()
        assert "--robust" not in stripped
        assert "how=left" not in stripped
        assert "y ~ x1 + x2" in stripped

    def test_bool_empty(self):
        assert not CommandArgs("")
        assert CommandArgs("something")

    def test_option_float(self):
        ca = CommandArgs("--p_enter=0.05")
        assert ca.get_option_float("p_enter", 0.1) == 0.05
        assert ca.get_option_float("missing", 0.1) == 0.1


# ── Logging ────────────────────────────────────────────────────────────

class TestLogging:
    def test_setup_logging_no_crash(self):
        """Logging setup should not crash even if called multiple times."""
        # Reset to allow re-setup
        import openstat.logging_config as lc
        lc._configured = False
        setup_logging(verbose=False, debug=False)
        # Should be idempotent
        setup_logging(verbose=True, debug=True)


# ── Memory-safe undo ──────────────────────────────────────────────────

class TestMemorySafeUndo:
    def test_undo_respects_count_limit(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        cfg = get_config()
        for _ in range(cfg.max_undo_stack + 5):
            s.snapshot()
        assert s.undo_depth <= cfg.max_undo_stack
