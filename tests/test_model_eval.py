"""Tests for ROC, confmatrix, calibration, SHAP."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.stats.model_eval import roc_auc, confusion_matrix, calibration_curve, compute_shap_linear


@pytest.fixture()
def cls_df():
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 0.5 * x1 + 0.3 * x2
    prob = 1 / (1 + np.exp(-logit))
    y = (prob > 0.5).astype(int)
    return pl.DataFrame({"y": y, "prob": prob, "x1": x1, "x2": x2})


@pytest.fixture()
def reg_df():
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 2.0 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 0.5, n)
    return pl.DataFrame({"y": y, "x1": x1, "x2": x2})


class TestROC:
    def test_keys(self, cls_df):
        r = roc_auc(cls_df, "y", "prob")
        assert "auc" in r and "fpr" in r and "tpr" in r

    def test_auc_range(self, cls_df):
        r = roc_auc(cls_df, "y", "prob")
        assert 0.5 <= r["auc"] <= 1.0

    def test_good_model(self, cls_df):
        r = roc_auc(cls_df, "y", "prob")
        assert r["auc"] > 0.7

    def test_cmd(self, cls_df):
        from openstat.commands.model_eval_cmds import cmd_roc
        s = Session()
        s.df = cls_df
        out = cmd_roc(s, "y prob")
        assert "AUC" in out

    def test_cmd_missing_col(self, cls_df):
        from openstat.commands.model_eval_cmds import cmd_roc
        s = Session()
        s.df = cls_df
        out = cmd_roc(s, "y nonexistent")
        assert "not found" in out.lower()


class TestConfusionMatrix:
    def test_keys(self, cls_df):
        r = confusion_matrix(cls_df, "y", "prob")
        assert all(k in r for k in ["tp", "fp", "fn", "tn", "accuracy", "f1_score"])

    def test_accuracy_range(self, cls_df):
        r = confusion_matrix(cls_df, "y", "prob")
        assert 0 <= r["accuracy"] <= 1

    def test_good_accuracy(self, cls_df):
        r = confusion_matrix(cls_df, "y", "prob")
        assert r["accuracy"] > 0.6

    def test_cmd(self, cls_df):
        from openstat.commands.model_eval_cmds import cmd_confmatrix
        s = Session()
        s.df = cls_df
        out = cmd_confmatrix(s, "y prob threshold(0.5)")
        assert "Confusion" in out or "Accuracy" in out or "accuracy" in out.lower()


class TestCalibration:
    def test_keys(self, cls_df):
        r = calibration_curve(cls_df, "y", "prob")
        assert "brier_score" in r and "fraction_positive" in r

    def test_brier_range(self, cls_df):
        r = calibration_curve(cls_df, "y", "prob")
        assert 0 <= r["brier_score"] <= 1

    def test_cmd(self, cls_df):
        from openstat.commands.model_eval_cmds import cmd_calibration
        s = Session()
        s.df = cls_df
        out = cmd_calibration(s, "y prob bins(5)")
        assert "Calibration" in out or "Brier" in out


class TestSHAP:
    def test_keys(self, reg_df):
        r = compute_shap_linear(reg_df, "y", ["x1", "x2"])
        assert "mean_abs_shap" in r and "feature_ranking" in r

    def test_feature_ranking_order(self, reg_df):
        r = compute_shap_linear(reg_df, "y", ["x1", "x2"])
        # x1 has larger coefficient, should rank first
        assert r["feature_ranking"][0] == "x1"

    def test_shap_values_shape(self, reg_df):
        r = compute_shap_linear(reg_df, "y", ["x1", "x2"])
        assert len(r["shap_values"]) == len(reg_df)
        assert len(r["shap_values"][0]) == 2

    def test_cmd(self, reg_df):
        from openstat.commands.model_eval_cmds import cmd_shap
        s = Session()
        s.df = reg_df
        out = cmd_shap(s, "y x1 x2")
        assert "SHAP" in out
        assert "x1" in out
