"""Tests for bootstrap, permtest, jackknife."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.stats.resampling import bootstrap_ci, bootstrap_diff, permutation_test, jackknife_ci


@pytest.fixture()
def sample_df():
    rng = np.random.default_rng(42)
    n = 100
    vals = rng.normal(5.0, 1.0, n)
    group = ["A"] * 50 + ["B"] * 50
    # Group B has higher mean
    vals[50:] += 1.5
    return pl.DataFrame({"val": vals, "group": group})


class TestBootstrapCI:
    def test_keys(self, sample_df):
        r = bootstrap_ci(sample_df, "val")
        assert "ci_lo" in r and "ci_hi" in r and "observed" in r

    def test_ci_covers_mean(self, sample_df):
        r = bootstrap_ci(sample_df, "val", stat="mean", n_boot=1000)
        # True mean ~5 for first 50, ~6.5 for all combined
        assert r["ci_lo"] < r["observed"] < r["ci_hi"]

    def test_different_stats(self, sample_df):
        for stat in ["mean", "median", "std"]:
            r = bootstrap_ci(sample_df, "val", stat=stat, n_boot=500)
            assert "observed" in r

    def test_cmd(self, sample_df):
        from openstat.commands.resampling_cmds import cmd_bootstrap
        s = Session()
        s.df = sample_df
        out = cmd_bootstrap(s, "val stat(mean) n(500)")
        assert "Bootstrap" in out

    def test_invalid_stat(self, sample_df):
        with pytest.raises(ValueError):
            bootstrap_ci(sample_df, "val", stat="geomean")


class TestBootstrapDiff:
    def test_keys(self, sample_df):
        r = bootstrap_diff(sample_df, "val", "group", n_boot=500)
        assert "observed_diff" in r and "p_value" in r

    def test_detects_difference(self, sample_df):
        r = bootstrap_diff(sample_df, "val", "group", n_boot=1000)
        assert r["p_value"] < 0.05

    def test_cmd_by(self, sample_df):
        from openstat.commands.resampling_cmds import cmd_bootstrap
        s = Session()
        s.df = sample_df
        out = cmd_bootstrap(s, "val by(group) n(500)")
        assert "Bootstrap" in out


class TestPermutationTest:
    def test_keys(self, sample_df):
        r = permutation_test(sample_df, "val", "group", n_perm=500)
        assert "p_value" in r and "observed_diff" in r

    def test_significant(self, sample_df):
        r = permutation_test(sample_df, "val", "group", n_perm=1000)
        assert r["p_value"] < 0.05

    def test_cmd(self, sample_df):
        from openstat.commands.resampling_cmds import cmd_permtest
        s = Session()
        s.df = sample_df
        out = cmd_permtest(s, "val by(group) n(500)")
        assert "Permutation" in out

    def test_missing_by(self, sample_df):
        from openstat.commands.resampling_cmds import cmd_permtest
        s = Session()
        s.df = sample_df
        out = cmd_permtest(s, "val n(100)")
        assert "Specify" in out or "by" in out.lower()


class TestJackknife:
    def test_keys(self, sample_df):
        r = jackknife_ci(sample_df, "val")
        assert "bias" in r and "se_jackknife" in r

    def test_bias_small_for_mean(self, sample_df):
        r = jackknife_ci(sample_df, "val", stat="mean")
        assert abs(r["bias"]) < 0.1

    def test_cmd(self, sample_df):
        from openstat.commands.resampling_cmds import cmd_jackknife
        s = Session()
        s.df = sample_df
        out = cmd_jackknife(s, "val stat(mean)")
        assert "Jackknife" in out
