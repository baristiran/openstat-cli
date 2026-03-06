"""Tests for survival analysis (F3)."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.surv_cmds import cmd_stset, cmd_stcox, cmd_sts

try:
    import lifelines
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False


@pytest.fixture
def surv_session(tmp_path):
    np.random.seed(42)
    n = 200
    x1 = np.random.randn(n)
    x2 = (np.random.rand(n) > 0.5).astype(float)
    # Exponential survival with hazard = exp(0.5*x1 + 0.3*x2)
    lam = np.exp(0.5 * x1 + 0.3 * x2)
    time = np.random.exponential(1 / lam)
    # Censoring
    censor_time = np.random.exponential(2, n)
    observed_time = np.minimum(time, censor_time)
    event = (time <= censor_time).astype(float)
    group = np.where(x2 > 0.5, "A", "B")

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({
        "time": observed_time,
        "event": event,
        "x1": x1,
        "x2": x2,
        "group": group,
    })
    return s


class TestStset:
    def test_stset_basic(self, surv_session):
        result = cmd_stset(surv_session, "time, failure(event)")
        assert "Survival time: time" in result
        assert "Failure event: event" in result
        assert surv_session._surv_time_var == "time"
        assert surv_session._surv_event_var == "event"

    def test_stset_missing_col(self, surv_session):
        result = cmd_stset(surv_session, "nonexistent, failure(event)")
        assert "not found" in result

    def test_stset_usage(self, surv_session):
        result = cmd_stset(surv_session, "time")
        assert "Usage" in result


@pytest.mark.skipif(not HAS_LIFELINES, reason="lifelines not installed")
class TestStcox:
    def test_stcox_basic(self, surv_session):
        cmd_stset(surv_session, "time, failure(event)")
        result = cmd_stcox(surv_session, "x1 x2")
        assert "Cox PH" in result or "Coef" in result

    def test_stcox_no_stset(self, surv_session):
        result = cmd_stcox(surv_session, "x1 x2")
        assert "not set" in result

    def test_stcox_missing_col(self, surv_session):
        cmd_stset(surv_session, "time, failure(event)")
        result = cmd_stcox(surv_session, "nonexistent")
        assert "not found" in result

    def test_stcox_concordance(self, surv_session):
        cmd_stset(surv_session, "time, failure(event)")
        result = cmd_stcox(surv_session, "x1 x2")
        assert "Concordance" in result


@pytest.mark.skipif(not HAS_LIFELINES, reason="lifelines not installed")
class TestSts:
    def test_sts_graph(self, surv_session):
        cmd_stset(surv_session, "time, failure(event)")
        result = cmd_sts(surv_session, "graph")
        assert "Kaplan-Meier" in result

    def test_sts_test(self, surv_session):
        cmd_stset(surv_session, "time, failure(event)")
        result = cmd_sts(surv_session, "test group")
        assert "Log-Rank" in result

    def test_sts_no_stset(self, surv_session):
        result = cmd_sts(surv_session, "graph")
        assert "not set" in result
