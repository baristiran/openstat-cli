"""Tests for causal inference models: DiD, PSM."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.causal_cmds import cmd_did, cmd_psmatch


@pytest.fixture
def causal_session(tmp_path):
    """Session with synthetic data for causal inference."""
    np.random.seed(42)
    n = 400

    # DiD data: 2 groups x 2 time periods
    treatment = np.array([0] * 200 + [1] * 200, dtype=float)
    post = np.tile([0.0] * 100 + [1.0] * 100, 2)
    # True treatment effect = 3.0
    y = (5.0
         + 2.0 * treatment
         + 1.5 * post
         + 3.0 * treatment * post
         + np.random.randn(n) * 0.5)

    # PSM data
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    # Treatment assignment depends on x1
    prob_treat = 1 / (1 + np.exp(-0.8 * x1))
    t_binary = (np.random.rand(n) < prob_treat).astype(float)
    # Outcome depends on treatment and covariates
    outcome = 1.0 + 2.0 * t_binary + 0.5 * x1 - 0.3 * x2 + np.random.randn(n) * 0.5

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "y": y,
        "treatment": treatment,
        "post": post,
        "outcome": outcome,
        "t_binary": t_binary,
        "x1": x1,
        "x2": x2,
    })
    return s


class TestDiD:
    def test_did_basic(self, causal_session):
        result = cmd_did(causal_session, "y ~ treatment post")
        assert "DiD" in result
        assert "Coef" in result

    def test_did_estimate(self, causal_session):
        result = cmd_did(causal_session, "y ~ treatment post")
        assert "DiD estimate" in result
        # The true effect is 3.0, check it's in reasonable range
        assert "treatment:post" in result

    def test_did_group_means(self, causal_session):
        result = cmd_did(causal_session, "y ~ treatment post")
        assert "Group means" in result

    def test_did_robust(self, causal_session):
        result = cmd_did(causal_session, "y ~ treatment post --robust")
        assert "DiD" in result

    def test_did_stores_result(self, causal_session):
        cmd_did(causal_session, "y ~ treatment post")
        assert len(causal_session.results) > 0

    def test_did_usage(self, causal_session):
        result = cmd_did(causal_session, "")
        assert "Usage" in result

    def test_did_missing_var(self, causal_session):
        result = cmd_did(causal_session, "y ~ treatment")
        assert "Usage" in result


class TestPSM:
    def test_psm_basic(self, causal_session):
        result = cmd_psmatch(causal_session, "outcome ~ x1 x2, treatment(t_binary)")
        assert "Propensity Score Matching" in result
        assert "ATT" in result

    def test_psm_att_positive(self, causal_session):
        result = cmd_psmatch(causal_session, "outcome ~ x1 x2, treatment(t_binary)")
        # True ATT is around 2.0
        assert "ATT" in result
        assert "SE" in result
        assert "p-value" in result

    def test_psm_matched_counts(self, causal_session):
        result = cmd_psmatch(causal_session, "outcome ~ x1 x2, treatment(t_binary)")
        assert "Matched" in result
        assert "N treated" in result

    def test_psm_balance_table(self, causal_session):
        result = cmd_psmatch(causal_session, "outcome ~ x1 x2, treatment(t_binary)")
        assert "Covariate Balance" in result

    def test_psm_with_caliper(self, causal_session):
        result = cmd_psmatch(causal_session, "outcome ~ x1 x2, treatment(t_binary) caliper(0.1)")
        assert "ATT" in result

    def test_psm_with_nn(self, causal_session):
        result = cmd_psmatch(causal_session, "outcome ~ x1 x2, treatment(t_binary) nn(3)")
        assert "ATT" in result

    def test_psm_usage(self, causal_session):
        result = cmd_psmatch(causal_session, "outcome ~ x1 x2")
        assert "Usage" in result

    def test_psm_missing_col(self, causal_session):
        result = cmd_psmatch(causal_session, "outcome ~ nonexist, treatment(t_binary)")
        assert "not found" in result
