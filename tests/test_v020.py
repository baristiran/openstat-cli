"""Tests for v0.2.0 features: interaction terms, cluster-robust SE,
Poisson, NegBin, quantile regression, marginal effects, bootstrap CI.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.config import reset_config
from openstat.dsl.parser import parse_formula, ParseError
from openstat.stats.models import (
    fit_ols,
    fit_logit,
    fit_probit,
    fit_poisson,
    fit_negbin,
    fit_quantreg,
    compute_margins,
    bootstrap_model,
)
from openstat.session import Session
from openstat.commands.stat_cmds import (
    cmd_ols,
    cmd_logit,
    cmd_probit,
    cmd_poisson,
    cmd_negbin,
    cmd_quantreg,
    cmd_margins,
    cmd_bootstrap,
)


@pytest.fixture(autouse=True)
def _reset():
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Helper DGPs
# ---------------------------------------------------------------------------

def _make_ols_data(n: int = 200, seed: int = 42) -> pl.DataFrame:
    """y = 1 + 2*x1 + 3*x2 + 4*x1*x2 + noise."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1 + 2 * x1 + 3 * x2 + 4 * x1 * x2 + rng.normal(0, 0.5, n)
    group = np.array(["A", "B", "C", "D"] * (n // 4))[:n]
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})


def _make_binary_data(n: int = 500, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    z = -1 + 0.5 * x1 + 0.3 * x2
    prob = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < prob).astype(float)
    group = np.array(["A", "B"] * (n // 2))[:n]
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2, "group": group})


def _make_count_data(n: int = 500, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 0.5, n)
    x2 = rng.normal(0, 0.5, n)
    mu = np.exp(0.5 + 0.3 * x1 + 0.2 * x2)
    y = rng.poisson(mu)
    exposure = rng.uniform(0.5, 2.0, n)
    return pl.DataFrame({
        "y": y.astype(float),
        "x1": x1,
        "x2": x2,
        "exposure": exposure,
    })


def _make_overdispersed_count_data(n: int = 500, seed: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 0.5, n)
    mu = np.exp(1.0 + 0.5 * x1)
    # NegBin via gamma-Poisson mixture
    alpha = 0.5  # dispersion
    p = alpha / (alpha + mu)
    y = rng.negative_binomial(alpha, p)
    return pl.DataFrame({"y": y.astype(float), "x1": x1})


# ===========================================================================
# 1. Interaction Terms
# ===========================================================================

class TestInteractionParsing:
    def test_colon_interaction(self):
        dep, indeps = parse_formula("y ~ x1 + x2 + x1:x2")
        assert dep == "y"
        assert indeps == ["x1", "x2", "x1:x2"]

    def test_star_expansion(self):
        dep, indeps = parse_formula("y ~ x1*x2")
        assert dep == "y"
        assert "x1" in indeps
        assert "x2" in indeps
        assert "x1:x2" in indeps
        assert len(indeps) == 3

    def test_star_no_duplicates(self):
        dep, indeps = parse_formula("y ~ x1 + x1*x2")
        assert indeps.count("x1") == 1
        assert "x2" in indeps
        assert "x1:x2" in indeps

    def test_interaction_only(self):
        dep, indeps = parse_formula("y ~ x1:x2")
        assert indeps == ["x1:x2"]


class TestInteractionOLS:
    def test_interaction_coefficient(self):
        df = _make_ols_data()
        result, _ = fit_ols(df, "y", ["x1", "x2", "x1:x2"])
        # True interaction coef is 4
        assert abs(result.params["x1:x2"] - 4.0) < 0.5

    def test_star_formula_via_parser(self):
        df = _make_ols_data()
        dep, indeps = parse_formula("y ~ x1*x2")
        result, _ = fit_ols(df, dep, indeps)
        assert "x1:x2" in result.params
        assert abs(result.params["x1:x2"] - 4.0) < 0.5

    def test_interaction_only_term(self):
        df = _make_ols_data()
        result, _ = fit_ols(df, "y", ["x1:x2"])
        assert "x1:x2" in result.params

    def test_interaction_command(self):
        session = Session()
        session.df = _make_ols_data()
        output = cmd_ols(session, "y ~ x1 + x2 + x1:x2")
        assert "x1:x2" in output


# ===========================================================================
# 2. Cluster-Robust Standard Errors
# ===========================================================================

class TestClusterRobust:
    def test_ols_cluster(self):
        df = _make_ols_data()
        result_plain, _ = fit_ols(df, "y", ["x1", "x2"])
        result_cluster, _ = fit_ols(df, "y", ["x1", "x2"], cluster_col="group")
        # Clustered SE should differ from plain
        assert result_cluster.model_type == "OLS (cluster-robust)"
        assert result_cluster.std_errors["x1"] != result_plain.std_errors["x1"]

    def test_logit_cluster(self):
        df = _make_binary_data()
        result, _ = fit_logit(df, "y", ["x1", "x2"], cluster_col="group")
        assert "cluster-robust" in result.model_type

    def test_cluster_col_not_found(self):
        df = _make_ols_data()
        with pytest.raises(ValueError, match="Cluster column not found"):
            fit_ols(df, "y", ["x1", "x2"], cluster_col="nonexistent")

    def test_cluster_command(self):
        session = Session()
        session.df = _make_ols_data()
        output = cmd_ols(session, "y ~ x1 + x2 --cluster=group")
        assert "cluster-robust" in output

    def test_cluster_overrides_robust(self):
        df = _make_ols_data()
        result, _ = fit_ols(df, "y", ["x1", "x2"], robust=True, cluster_col="group")
        assert "cluster-robust" in result.model_type


# ===========================================================================
# 3. Poisson & Negative Binomial
# ===========================================================================

class TestPoisson:
    def test_basic_poisson(self):
        df = _make_count_data()
        result, _ = fit_poisson(df, "y", ["x1", "x2"])
        assert result.model_type == "Poisson"
        assert result.log_likelihood is not None
        assert result.n_obs > 0

    def test_poisson_robust(self):
        df = _make_count_data()
        result, _ = fit_poisson(df, "y", ["x1", "x2"], robust=True)
        assert "robust" in result.model_type

    def test_poisson_exposure(self):
        df = _make_count_data()
        result, _ = fit_poisson(df, "y", ["x1", "x2"], exposure_col="exposure")
        assert result.n_obs > 0

    def test_poisson_command(self):
        session = Session()
        session.df = _make_count_data()
        output = cmd_poisson(session, "y ~ x1 + x2")
        assert "Poisson" in output

    def test_poisson_exposure_command(self):
        session = Session()
        session.df = _make_count_data()
        output = cmd_poisson(session, "y ~ x1 + x2 --exposure=exposure")
        assert "Poisson" in output


class TestNegBin:
    def test_basic_negbin(self):
        df = _make_overdispersed_count_data()
        result, _ = fit_negbin(df, "y", ["x1"])
        assert result.model_type == "NegBin"
        assert result.dispersion is not None
        assert result.log_likelihood is not None

    def test_negbin_robust(self):
        df = _make_overdispersed_count_data()
        result, _ = fit_negbin(df, "y", ["x1"], robust=True)
        assert "robust" in result.model_type

    def test_negbin_command(self):
        session = Session()
        session.df = _make_overdispersed_count_data()
        output = cmd_negbin(session, "y ~ x1")
        assert "NegBin" in output
        assert "alpha" in output


# ===========================================================================
# 4. Quantile Regression
# ===========================================================================

class TestQuantReg:
    def test_median_regression(self):
        df = _make_ols_data()
        result_qr, _ = fit_quantreg(df, "y", ["x1", "x2"], tau=0.5)
        result_ols, _ = fit_ols(df, "y", ["x1", "x2"])
        # Median and mean regression should give similar coefs for symmetric data
        assert abs(result_qr.params["x1"] - result_ols.params["x1"]) < 2.0
        assert "QuantReg" in result_qr.model_type

    def test_different_quantiles(self):
        df = _make_ols_data()
        result_25, _ = fit_quantreg(df, "y", ["x1", "x2"], tau=0.25)
        result_75, _ = fit_quantreg(df, "y", ["x1", "x2"], tau=0.75)
        # Different quantiles should give different intercepts
        assert result_25.params["_cons"] != result_75.params["_cons"]

    def test_quantreg_command(self):
        session = Session()
        session.df = _make_ols_data()
        output = cmd_quantreg(session, "y ~ x1 + x2 tau=0.5")
        assert "QuantReg" in output

    def test_quantreg_custom_tau(self):
        session = Session()
        session.df = _make_ols_data()
        output = cmd_quantreg(session, "y ~ x1 + x2 tau=0.25")
        assert "tau=0.25" in output


# ===========================================================================
# 5. Marginal Effects
# ===========================================================================

class TestMarginalEffects:
    def test_margins_after_logit(self):
        df = _make_binary_data()
        result, raw_model = fit_logit(df, "y", ["x1", "x2"])
        var_names = list(result.params.keys())
        mfx = compute_margins(raw_model, var_names, "average")
        assert "x1" in mfx.effects
        assert "x2" in mfx.effects
        # dy/dx should be positive for x1 (positive coefficient)
        assert mfx.effects["x1"] > 0

    def test_margins_at_means(self):
        df = _make_binary_data()
        result, raw_model = fit_logit(df, "y", ["x1", "x2"])
        var_names = list(result.params.keys())
        mfx_avg = compute_margins(raw_model, var_names, "average")
        mfx_means = compute_margins(raw_model, var_names, "means")
        # Average vs at-means should give different values
        assert mfx_avg.effects["x1"] != mfx_means.effects["x1"]

    def test_margins_command(self):
        session = Session()
        session.df = _make_binary_data()
        cmd_logit(session, "y ~ x1 + x2")
        output = cmd_margins(session, "")
        assert "dy/dx" in output

    def test_margins_no_model(self):
        session = Session()
        session.df = _make_binary_data()
        output = cmd_margins(session, "")
        assert "No model fitted" in output

    def test_margins_after_probit(self):
        session = Session()
        session.df = _make_binary_data()
        cmd_probit(session, "y ~ x1 + x2")
        output = cmd_margins(session, "")
        assert "dy/dx" in output

    def test_margins_after_ols_rejected(self):
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_margins(session, "")
        assert "only available for logit/probit" in output


# ===========================================================================
# 6. Bootstrap Confidence Intervals
# ===========================================================================

class TestBootstrap:
    def test_bootstrap_ols(self):
        df = _make_ols_data()
        result = bootstrap_model(df, "y", ["x1", "x2"], fit_ols, n_boot=50, ci=95.0)
        # True x1 coef is ~2; CI should contain it
        assert result.ci_low["x1"] < 2.0 < result.ci_high["x1"]
        assert result.n_boot == 50

    def test_bootstrap_logit(self):
        df = _make_binary_data()
        result = bootstrap_model(df, "y", ["x1", "x2"], fit_logit, n_boot=50, ci=95.0)
        assert "x1" in result.boot_means
        assert result.n_boot == 50

    def test_bootstrap_command(self):
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_bootstrap(session, "n=50 ci=95")
        assert "Bootstrap" in output

    def test_bootstrap_no_model(self):
        session = Session()
        session.df = _make_ols_data()
        output = cmd_bootstrap(session, "")
        assert "No model fitted" in output


# ===========================================================================
# FitResult summary additions
# ===========================================================================

# ===========================================================================
# Interaction terms with predict/residuals (regression safety)
# ===========================================================================

class TestInteractionWithPredictResiduals:
    def test_predict_with_interaction(self):
        from openstat.commands.stat_cmds import cmd_predict
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1*x2")
        output = cmd_predict(session, "yhat")
        assert "yhat" in output
        assert "yhat" in session.df.columns

    def test_residuals_with_interaction(self):
        from openstat.commands.stat_cmds import cmd_residuals
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2 + x1:x2")
        output = cmd_residuals(session, "resid")
        assert "resid" in output


# ===========================================================================
# Additional fixes: VIF, stepwise, tau, multi-way
# ===========================================================================

class TestVIFWithInteractions:
    def test_vif_with_interaction_term(self):
        from openstat.stats.models import compute_vif
        df = _make_ols_data()
        vifs = compute_vif(df, ["x1", "x2", "x1:x2"])
        names = [v[0] for v in vifs]
        assert "x1:x2" in names
        assert all(v[1] > 0 for v in vifs)

    def test_vif_command_with_interaction(self):
        from openstat.commands.stat_cmds import cmd_vif
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1*x2")
        output = cmd_vif(session, "")
        assert "x1:x2" in output


class TestStepwiseWithInteractions:
    def test_stepwise_with_interaction_candidates(self):
        from openstat.stats.models import stepwise_ols
        df = _make_ols_data()
        result = stepwise_ols(df, "y", ["x1", "x2", "x1:x2"])
        assert "x1:x2" in result.selected


class TestTauValidation:
    def test_tau_zero_rejected(self):
        df = _make_ols_data()
        with pytest.raises(ValueError, match="tau must be between"):
            fit_quantreg(df, "y", ["x1"], tau=0.0)

    def test_tau_one_rejected(self):
        df = _make_ols_data()
        with pytest.raises(ValueError, match="tau must be between"):
            fit_quantreg(df, "y", ["x1"], tau=1.0)

    def test_tau_negative_rejected(self):
        df = _make_ols_data()
        with pytest.raises(ValueError, match="tau must be between"):
            fit_quantreg(df, "y", ["x1"], tau=-0.5)


class TestMultiWayInteraction:
    def test_three_way_expansion(self):
        dep, indeps = parse_formula("y ~ x1*x2*x3")
        assert "x1" in indeps
        assert "x2" in indeps
        assert "x3" in indeps
        assert "x1:x2" in indeps
        assert "x1:x3" in indeps
        assert "x2:x3" in indeps
        assert "x1:x2:x3" in indeps
        assert len(indeps) == 7  # 3 main + 3 pairwise + 1 three-way

    def test_three_way_ols(self):
        rng = np.random.default_rng(42)
        n = 200
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        x3 = rng.normal(0, 1, n)
        y = 1 + x1 + x2 + x3 + rng.normal(0, 0.5, n)
        df = pl.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
        dep, indeps = parse_formula("y ~ x1*x2*x3")
        result, _ = fit_ols(df, dep, indeps)
        assert "x1:x2:x3" in result.params


class TestEstat:
    def test_estat_hettest(self):
        from openstat.commands.stat_cmds import cmd_estat
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_estat(session, "hettest")
        assert "Breusch-Pagan" in output

    def test_estat_ovtest(self):
        from openstat.commands.stat_cmds import cmd_estat
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_estat(session, "ovtest")
        assert "RESET" in output

    def test_estat_linktest(self):
        from openstat.commands.stat_cmds import cmd_estat
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_estat(session, "linktest")
        assert "_hat" in output

    def test_estat_ic(self):
        from openstat.commands.stat_cmds import cmd_estat
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_estat(session, "ic")
        assert "AIC" in output

    def test_estat_all(self):
        from openstat.commands.stat_cmds import cmd_estat
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_estat(session, "all")
        assert "Breusch-Pagan" in output
        assert "RESET" in output
        assert "AIC" in output

    def test_estat_no_model(self):
        from openstat.commands.stat_cmds import cmd_estat
        session = Session()
        session.df = _make_ols_data()
        output = cmd_estat(session, "hettest")
        assert "No model fitted" in output

    def test_estat_no_subcommand(self):
        from openstat.commands.stat_cmds import cmd_estat
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_estat(session, "")
        assert "Usage" in output


class TestEstimatesTable:
    def test_estimates_table(self):
        from openstat.commands.stat_cmds import cmd_estimates
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1")
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_estimates(session, "table")
        assert "Model Comparison" in output
        assert "x1" in output
        assert "x2" in output

    def test_estimates_needs_two_models(self):
        from openstat.commands.stat_cmds import cmd_estimates
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1")
        output = cmd_estimates(session, "table")
        assert "Need at least 2" in output

    def test_estimates_mixed_models(self):
        from openstat.commands.stat_cmds import cmd_estimates
        session = Session()
        session.df = _make_binary_data()
        cmd_logit(session, "y ~ x1")
        cmd_probit(session, "y ~ x1 + x2")
        output = cmd_estimates(session, "table")
        assert "Model Comparison" in output

    def test_estimates_shows_aic(self):
        from openstat.commands.stat_cmds import cmd_estimates
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1")
        cmd_ols(session, "y ~ x1 + x2")
        output = cmd_estimates(session, "table")
        assert "AIC" in output


class TestFitResultExtensions:
    def test_log_likelihood_in_summary(self):
        df = _make_count_data()
        result, _ = fit_poisson(df, "y", ["x1", "x2"])
        summary = result.summary_table()
        assert "LL" in summary

    def test_dispersion_in_summary(self):
        df = _make_overdispersed_count_data()
        result, _ = fit_negbin(df, "y", ["x1"])
        summary = result.summary_table()
        assert "alpha" in summary

    def test_log_likelihood_in_markdown(self):
        df = _make_count_data()
        result, _ = fit_poisson(df, "y", ["x1", "x2"])
        md = result.to_markdown()
        assert "Log-Likelihood" in md

    def test_dispersion_in_latex(self):
        df = _make_overdispersed_count_data()
        result, _ = fit_negbin(df, "y", ["x1"])
        latex = result.to_latex()
        assert "alpha" in latex


# ===========================================================================
# Bug-fix regression tests (critical review)
# ===========================================================================

class TestInteractionWhitespace:
    """parse_formula should handle spaces around ':' in interaction terms."""

    def test_colon_with_spaces(self):
        dep, indeps = parse_formula("y ~ x1 + x2 + x1 : x2")
        assert "x1:x2" in indeps  # spaces stripped

    def test_colon_with_spaces_only(self):
        dep, indeps = parse_formula("y ~ x1 : x2")
        assert indeps == ["x1:x2"]

    def test_star_with_spaces(self):
        dep, indeps = parse_formula("y ~ x1 * x2")
        assert "x1:x2" in indeps

    def test_interaction_spaces_in_ols(self):
        """Ensure OLS works with space-around-colon formula."""
        df = _make_ols_data()
        dep, indeps = parse_formula("y ~ x1 + x2 + x1 : x2")
        result, _ = fit_ols(df, dep, indeps)
        assert "x1:x2" in result.params


class TestPoissonExposureNulls:
    """Exposure column with nulls should not cause dimension mismatch."""

    def test_exposure_with_nulls(self):
        df = _make_count_data()
        # Introduce nulls in the exposure column
        exposure_vals = df["exposure"].to_list()
        for i in range(0, 20):
            exposure_vals[i] = None
        df = df.with_columns(pl.Series("exposure", exposure_vals))
        result, _ = fit_poisson(df, "y", ["x1", "x2"], exposure_col="exposure")
        assert result.n_obs > 0
        assert result.n_obs <= 500 - 20  # at least 20 rows should be dropped


class TestBootstrapPreservesKwargs:
    """Bootstrap should use the same model kwargs as the original fit."""

    def test_bootstrap_quantreg_tau(self):
        """Bootstrap after quantreg(tau=0.75) should use tau=0.75."""
        session = Session()
        session.df = _make_ols_data()
        cmd_quantreg(session, "y ~ x1 + x2 tau=0.75")
        assert session._last_fit_kwargs.get("tau") == 0.75
        output = cmd_bootstrap(session, "n=30 ci=95")
        assert "Bootstrap" in output

    def test_bootstrap_poisson_exposure(self):
        """Bootstrap after poisson --exposure should preserve exposure_col."""
        session = Session()
        session.df = _make_count_data()
        cmd_poisson(session, "y ~ x1 + x2 --exposure=exposure")
        assert session._last_fit_kwargs.get("exposure_col") == "exposure"
        output = cmd_bootstrap(session, "n=30 ci=95")
        assert "Bootstrap" in output

    def test_ols_kwargs_empty(self):
        """OLS should store empty fit_kwargs."""
        session = Session()
        session.df = _make_ols_data()
        cmd_ols(session, "y ~ x1 + x2")
        assert session._last_fit_kwargs == {}
