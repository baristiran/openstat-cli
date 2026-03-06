"""Tests for time series analysis (F2)."""

import pytest
import numpy as np
import polars as pl

from openstat.session import Session
from openstat.commands.ts_cmds import cmd_tsset, cmd_dfuller, cmd_arima, cmd_var, cmd_forecast


@pytest.fixture
def ts_session(tmp_path):
    np.random.seed(42)
    n = 200
    # AR(1) process
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = 0.7 * y[i - 1] + np.random.randn()

    time = np.arange(n)
    x1 = np.random.randn(n)

    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({"t": time, "y": y, "x1": x1})
    return s


@pytest.fixture
def var_session(tmp_path):
    np.random.seed(42)
    n = 200
    y1 = np.cumsum(np.random.randn(n) * 0.5)
    y2 = np.cumsum(np.random.randn(n) * 0.5)
    t = np.arange(n)
    s = Session(output_dir=tmp_path / "out")
    s.df = pl.DataFrame({"t": t, "y1": y1, "y2": y2})
    return s


class TestTsset:
    def test_tsset_basic(self, ts_session):
        result = cmd_tsset(ts_session, "t")
        assert "Time variable: t" in result
        assert ts_session._time_var == "t"

    def test_tsset_missing_col(self, ts_session):
        result = cmd_tsset(ts_session, "nonexistent")
        assert "not found" in result


class TestDfuller:
    def test_dfuller_stationary(self, ts_session):
        result = cmd_dfuller(ts_session, "y")
        assert "ADF Statistic" in result

    def test_dfuller_missing_col(self, ts_session):
        result = cmd_dfuller(ts_session, "nonexistent")
        assert "not found" in result

    def test_dfuller_usage(self, ts_session):
        result = cmd_dfuller(ts_session, "")
        assert "Usage" in result


class TestArima:
    def test_arima_basic(self, ts_session):
        result = cmd_arima(ts_session, "y, order(1,0,0)")
        assert "ARIMA" in result
        assert "Coef" in result

    def test_arima_with_diff(self, ts_session):
        result = cmd_arima(ts_session, "y, order(1,1,0)")
        assert "ARIMA" in result

    def test_arima_usage(self, ts_session):
        result = cmd_arima(ts_session, "y")
        assert "Usage" in result

    def test_arima_store_result(self, ts_session):
        cmd_arima(ts_session, "y, order(1,0,0)")
        assert ts_session._last_model is not None
        assert len(ts_session.results) > 0


class TestVar:
    def test_var_basic(self, var_session):
        result = cmd_var(var_session, "y1 y2, lags(2)")
        assert "VAR" in result or "Summary" in result

    def test_var_missing_lags(self, var_session):
        result = cmd_var(var_session, "y1 y2")
        assert "Usage" in result

    def test_var_too_few_variables(self, var_session):
        result = cmd_var(var_session, "y1, lags(2)")
        assert "at least 2" in result


class TestForecast:
    def test_forecast_no_model(self, ts_session):
        result = cmd_forecast(ts_session, "12")
        assert "No model" in result

    def test_forecast_after_arima(self, ts_session):
        cmd_arima(ts_session, "y, order(1,0,0)")
        result = cmd_forecast(ts_session, "5")
        assert "Forecast" in result or "Step" in result
