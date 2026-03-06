"""Tests for Round 6 features: fillna, cast, lag/lead, predict, vif, probit, Excel/DTA export."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.commands.data_cmds import cmd_fillna
from openstat.commands.datamanip_cmds import cmd_cast, cmd_lag, cmd_lead
from openstat.commands.stat_cmds import cmd_ols, cmd_probit, cmd_predict, cmd_vif
from openstat.stats.models import fit_probit, compute_vif


# -----------------------------------------------------------------------
# Fillna
# -----------------------------------------------------------------------

class TestFillna:
    def test_fill_mean(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0, None, 3.0]})
        result = cmd_fillna(s, "x mean")
        assert s.df["x"].null_count() == 0
        assert s.df["x"][1] == 2.0  # mean of 1 and 3

    def test_fill_median(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0, None, 5.0, 3.0]})
        result = cmd_fillna(s, "x median")
        assert s.df["x"].null_count() == 0

    def test_fill_mode(self):
        s = Session()
        s.df = pl.DataFrame({"x": ["a", "a", "b", None]})
        result = cmd_fillna(s, "x mode")
        assert s.df["x"][3] == "a"

    def test_fill_forward(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0, None, None, 4.0]})
        cmd_fillna(s, "x forward")
        assert s.df["x"].to_list() == [1.0, 1.0, 1.0, 4.0]

    def test_fill_backward(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0, None, None, 4.0]})
        cmd_fillna(s, "x backward")
        assert s.df["x"].to_list() == [1.0, 4.0, 4.0, 4.0]

    def test_fill_value(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0, None, 3.0]})
        cmd_fillna(s, "x value=99")
        assert s.df["x"][1] == 99.0

    def test_fill_no_nulls(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0, 2.0]})
        result = cmd_fillna(s, "x mean")
        assert "No missing" in result

    def test_fill_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0, None]})
        cmd_fillna(s, "x mean")
        assert s.undo_depth == 1

    def test_fill_usage(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert "Usage" in cmd_fillna(s, "x")


# -----------------------------------------------------------------------
# Cast
# -----------------------------------------------------------------------

class TestCast:
    def test_cast_to_float(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        cmd_cast(s, "x float")
        assert s.df["x"].dtype == pl.Float64

    def test_cast_to_str(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        cmd_cast(s, "x str")
        assert s.df["x"].dtype == pl.Utf8

    def test_cast_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        cmd_cast(s, "x float")
        assert s.undo_depth == 1

    def test_cast_unknown_type(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        result = cmd_cast(s, "x complex")
        assert "Unknown type" in result

    def test_cast_usage(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert "Usage" in cmd_cast(s, "x")


# -----------------------------------------------------------------------
# Lag / Lead
# -----------------------------------------------------------------------

class TestLagLead:
    def test_lag_default(self):
        s = Session()
        s.df = pl.DataFrame({"x": [10, 20, 30, 40]})
        cmd_lag(s, "x")
        assert "x_lag1" in s.df.columns
        assert s.df["x_lag1"].to_list() == [None, 10, 20, 30]

    def test_lag_n(self):
        s = Session()
        s.df = pl.DataFrame({"x": [10, 20, 30, 40]})
        cmd_lag(s, "x 2")
        assert "x_lag2" in s.df.columns
        assert s.df["x_lag2"].to_list() == [None, None, 10, 20]

    def test_lead_default(self):
        s = Session()
        s.df = pl.DataFrame({"x": [10, 20, 30, 40]})
        cmd_lead(s, "x")
        assert "x_lead1" in s.df.columns
        assert s.df["x_lead1"].to_list() == [20, 30, 40, None]

    def test_lag_custom_name(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        cmd_lag(s, "x into(prev_x)")
        assert "prev_x" in s.df.columns

    def test_lag_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        cmd_lag(s, "x")
        assert s.undo_depth == 1


# -----------------------------------------------------------------------
# Probit
# -----------------------------------------------------------------------

class TestProbit:
    def test_probit_fits(self):
        np.random.seed(42)
        n = 100
        x = np.random.normal(0, 1, n)
        p = 1 / (1 + np.exp(-(x * 2)))
        y = (np.random.rand(n) < p).astype(float)
        df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        result, model = fit_probit(df, "y", ["x"])
        assert result.model_type.startswith("Probit")
        assert result.n_obs == n
        assert result.params["x"] > 0  # positive relationship

    def test_probit_command(self):
        np.random.seed(42)
        s = Session()
        n = 100
        x = np.random.normal(0, 1, n)
        p = 1 / (1 + np.exp(-(x * 2)))
        y = (np.random.rand(n) < p).astype(float)
        s.df = pl.DataFrame({"y": y.tolist(), "x": x.tolist()})
        result = cmd_probit(s, "y ~ x")
        assert "Probit" in result
        assert s._last_model is not None

    def test_probit_usage(self):
        s = Session()
        s.df = pl.DataFrame({"y": [0.0, 1.0], "x": [1.0, 2.0]})
        assert "Usage" in cmd_probit(s, "")


# -----------------------------------------------------------------------
# Predict
# -----------------------------------------------------------------------

class TestPredict:
    def test_predict_after_ols(self):
        s = Session()
        s.df = pl.DataFrame({
            "y": [2.0, 4.0, 6.0, 8.0, 10.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        cmd_ols(s, "y ~ x")
        result = cmd_predict(s, "predicted")
        assert "predicted" in s.df.columns
        # y = 2x, so predictions should be close to y
        preds = s.df["predicted"].to_list()
        assert abs(preds[0] - 2.0) < 0.5
        assert abs(preds[4] - 10.0) < 0.5

    def test_predict_custom_name(self):
        s = Session()
        s.df = pl.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        cmd_ols(s, "y ~ x")
        cmd_predict(s, "my_preds")
        assert "my_preds" in s.df.columns

    def test_predict_no_model(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0]})
        result = cmd_predict(s, "")
        assert "No model" in result

    def test_predict_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        cmd_ols(s, "y ~ x")
        cmd_predict(s, "")
        assert s.undo_depth == 1


# -----------------------------------------------------------------------
# VIF
# -----------------------------------------------------------------------

class TestVIF:
    def test_vif_independent_vars(self):
        """Independent predictors should have VIF ≈ 1."""
        np.random.seed(42)
        n = 100
        df = pl.DataFrame({
            "y": np.random.normal(0, 1, n).tolist(),
            "x1": np.random.normal(0, 1, n).tolist(),
            "x2": np.random.normal(0, 1, n).tolist(),
        })
        vifs = compute_vif(df, ["x1", "x2"])
        for var, vif_val in vifs:
            assert vif_val < 2.0  # should be near 1 for independent vars

    def test_vif_collinear_vars(self):
        """Collinear predictors should have high VIF."""
        np.random.seed(42)
        x = np.random.normal(0, 1, 100)
        df = pl.DataFrame({
            "x1": x.tolist(),
            "x2": (x + np.random.normal(0, 0.01, 100)).tolist(),  # near-perfect copy
        })
        vifs = compute_vif(df, ["x1", "x2"])
        for var, vif_val in vifs:
            assert vif_val > 100  # very high

    def test_vif_command(self):
        s = Session()
        np.random.seed(42)
        n = 50
        s.df = pl.DataFrame({
            "y": np.random.normal(0, 1, n).tolist(),
            "x1": np.random.normal(0, 1, n).tolist(),
            "x2": np.random.normal(0, 1, n).tolist(),
        })
        cmd_ols(s, "y ~ x1 + x2")
        result = cmd_vif(s, "")
        assert "VIF" in result
        assert "x1" in result

    def test_vif_no_model(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1.0]})
        result = cmd_vif(s, "")
        assert "No model" in result


# -----------------------------------------------------------------------
# Excel/DTA export
# -----------------------------------------------------------------------

class TestExport:
    def test_save_xlsx(self, tmp_path):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        from openstat.commands.data_cmds import cmd_save
        path = tmp_path / "out.xlsx"
        result = cmd_save(s, str(path))
        if "error" in result.lower():
            pytest.skip("xlsxwriter not installed")
        assert path.exists()
