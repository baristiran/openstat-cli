"""Tests for discrete/censored models: Tobit, MNLogit, Ordered Logit/Probit."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.discrete_cmds import cmd_tobit, cmd_mlogit, cmd_ologit, cmd_oprobit


@pytest.fixture
def disc_session(tmp_path):
    """Session with synthetic data for discrete models."""
    np.random.seed(42)
    n = 300
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)

    # Continuous outcome (for Tobit)
    y_latent = 2.0 + 1.0 * x1 - 0.5 * x2 + np.random.randn(n) * 0.8
    y_censored = np.maximum(y_latent, 0.0)  # left-censored at 0

    # Multinomial outcome (3 categories)
    probs = np.column_stack([
        np.ones(n),
        np.exp(0.5 * x1),
        np.exp(-0.3 * x2),
    ])
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_multi = np.array([np.random.choice([0, 1, 2], p=p) for p in probs]).astype(float)

    # Ordered outcome (4 categories)
    y_ordered_latent = 1.0 * x1 - 0.5 * x2 + np.random.randn(n)
    y_ordered = np.digitize(y_ordered_latent, bins=[-1, 0, 1]).astype(float)

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "y_cens": y_censored,
        "y_multi": y_multi,
        "y_ord": y_ordered,
        "x1": x1,
        "x2": x2,
    })
    return s


class TestTobit:
    def test_tobit_basic(self, disc_session):
        result = cmd_tobit(disc_session, "y_cens ~ x1 + x2, ll(0)")
        assert "Tobit" in result
        assert "Coef" in result

    def test_tobit_no_censoring(self, disc_session):
        result = cmd_tobit(disc_session, "y_cens ~ x1 + x2")
        assert "Tobit" in result
        assert "No censoring" in result

    def test_tobit_sigma_in_output(self, disc_session):
        result = cmd_tobit(disc_session, "y_cens ~ x1 + x2, ll(0)")
        assert "sigma" in result

    def test_tobit_censoring_info(self, disc_session):
        result = cmd_tobit(disc_session, "y_cens ~ x1 + x2, ll(0)")
        assert "Left-censored" in result
        assert "Uncensored" in result

    def test_tobit_stores_result(self, disc_session):
        cmd_tobit(disc_session, "y_cens ~ x1 + x2, ll(0)")
        assert len(disc_session.results) > 0

    def test_tobit_upper_limit(self, disc_session):
        result = cmd_tobit(disc_session, "y_cens ~ x1 + x2, ul(5)")
        assert "Tobit" in result

    def test_tobit_usage(self, disc_session):
        result = cmd_tobit(disc_session, "")
        assert "Usage" in result


class TestMNLogit:
    def test_mlogit_basic(self, disc_session):
        result = cmd_mlogit(disc_session, "y_multi ~ x1 + x2")
        assert "MNLogit" in result
        assert "Coef" in result

    def test_mlogit_categories(self, disc_session):
        result = cmd_mlogit(disc_session, "y_multi ~ x1 + x2")
        assert "y=" in result  # per-category coefficients

    def test_mlogit_base_category(self, disc_session):
        result = cmd_mlogit(disc_session, "y_multi ~ x1 + x2")
        assert "Base category" in result

    def test_mlogit_stores_result(self, disc_session):
        cmd_mlogit(disc_session, "y_multi ~ x1 + x2")
        assert len(disc_session.results) > 0

    def test_mlogit_usage(self, disc_session):
        result = cmd_mlogit(disc_session, "")
        assert "Usage" in result


class TestOrderedLogit:
    def test_ologit_basic(self, disc_session):
        result = cmd_ologit(disc_session, "y_ord ~ x1 + x2")
        assert "Logit" in result or "Coef" in result

    def test_ologit_thresholds(self, disc_session):
        result = cmd_ologit(disc_session, "y_ord ~ x1 + x2")
        assert "cut" in result

    def test_ologit_categories(self, disc_session):
        result = cmd_ologit(disc_session, "y_ord ~ x1 + x2")
        assert "Ordered categories" in result

    def test_ologit_stores_result(self, disc_session):
        cmd_ologit(disc_session, "y_ord ~ x1 + x2")
        assert len(disc_session.results) > 0

    def test_ologit_usage(self, disc_session):
        result = cmd_ologit(disc_session, "")
        assert "Usage" in result


class TestOrderedProbit:
    def test_oprobit_basic(self, disc_session):
        result = cmd_oprobit(disc_session, "y_ord ~ x1 + x2")
        assert "Probit" in result or "Coef" in result

    def test_oprobit_thresholds(self, disc_session):
        result = cmd_oprobit(disc_session, "y_ord ~ x1 + x2")
        assert "cut" in result

    def test_oprobit_stores_result(self, disc_session):
        cmd_oprobit(disc_session, "y_ord ~ x1 + x2")
        assert len(disc_session.results) > 0
