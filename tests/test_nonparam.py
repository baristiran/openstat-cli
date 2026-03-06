"""Tests for nonparametric tests."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.stats.nonparametric import (
    spearman_corr,
    ranksum_test,
    signrank_test,
    kruskal_wallis_test,
)
from openstat.session import Session


@pytest.fixture()
def two_group_df():
    rng = np.random.default_rng(42)
    return pl.DataFrame({
        "value": np.concatenate([rng.normal(0, 1, 50), rng.normal(1, 1, 50)]).tolist(),
        "group": (["A"] * 50 + ["B"] * 50),
    })


@pytest.fixture()
def multi_group_df():
    rng = np.random.default_rng(0)
    return pl.DataFrame({
        "score": np.concatenate([
            rng.normal(0, 1, 30), rng.normal(1, 1, 30), rng.normal(2, 1, 30)
        ]).tolist(),
        "group": (["X"] * 30 + ["Y"] * 30 + ["Z"] * 30),
    })


@pytest.fixture()
def numeric_df():
    rng = np.random.default_rng(7)
    n = 100
    x = rng.normal(0, 1, n)
    return pl.DataFrame({
        "x": x.tolist(),
        "y": (x * 0.8 + rng.normal(0, 0.5, n)).tolist(),
        "z": rng.normal(0, 1, n).tolist(),
    })


class TestSpearman:
    def test_keys(self, numeric_df):
        r = spearman_corr(numeric_df, ["x", "y", "z"])
        assert "rho" in r and "pvalues" in r and "cols" in r

    def test_diagonal_one(self, numeric_df):
        r = spearman_corr(numeric_df, ["x", "y"])
        assert abs(r["rho"][0][0] - 1.0) < 1e-9
        assert abs(r["rho"][1][1] - 1.0) < 1e-9

    def test_symmetric(self, numeric_df):
        r = spearman_corr(numeric_df, ["x", "y", "z"])
        rho = r["rho"]
        assert abs(rho[0][1] - rho[1][0]) < 1e-9

    def test_correlated(self, numeric_df):
        r = spearman_corr(numeric_df, ["x", "y"])
        assert r["rho"][0][1] > 0.5  # x and y are positively correlated

    def test_two_vars(self, numeric_df):
        r = spearman_corr(numeric_df, ["x", "z"])
        assert len(r["rho"]) == 2


class TestRanksum:
    def test_basic(self, two_group_df):
        r = ranksum_test(two_group_df, "value", "group")
        assert "U_statistic" in r
        assert r["p_value"] < 0.05  # groups differ significantly

    def test_keys(self, two_group_df):
        r = ranksum_test(two_group_df, "value", "group")
        assert {"test", "n1", "n2", "U_statistic", "z_statistic", "p_value"}.issubset(r)

    def test_sample_sizes(self, two_group_df):
        r = ranksum_test(two_group_df, "value", "group")
        assert r["n1"] == 50
        assert r["n2"] == 50

    def test_wrong_groups(self):
        df = pl.DataFrame({"x": [1, 2, 3], "g": ["A", "B", "C"]})
        with pytest.raises(ValueError):
            ranksum_test(df, "x", "g")

    def test_alternative_less(self, two_group_df):
        r = ranksum_test(two_group_df, "value", "group", alternative="less")
        assert r["alternative"] == "less"


class TestSignrank:
    def test_one_sample(self):
        df = pl.DataFrame({"x": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 0.5, 1.0]})
        r = signrank_test(df, "x", mu=2.0)
        assert "W_statistic" in r
        assert 0 <= r["p_value"] <= 1

    def test_paired(self):
        rng = np.random.default_rng(1)
        pre = rng.normal(5, 1, 40)
        post = pre + rng.normal(1, 0.5, 40)
        df = pl.DataFrame({"pre": pre.tolist(), "post": post.tolist()})
        r = signrank_test(df, "pre", "post")
        assert r["p_value"] < 0.05  # significantly different

    def test_keys(self):
        df = pl.DataFrame({"x": list(range(1, 21))})
        r = signrank_test(df, "x", mu=10)
        assert {"W_statistic", "p_value", "n"}.issubset(r)


class TestKruskalWallis:
    def test_significant(self, multi_group_df):
        r = kruskal_wallis_test(multi_group_df, "score", "group")
        assert r["p_value"] < 0.05

    def test_keys(self, multi_group_df):
        r = kruskal_wallis_test(multi_group_df, "score", "group")
        assert {"H_statistic", "df", "p_value", "k_groups"}.issubset(r)

    def test_k_groups(self, multi_group_df):
        r = kruskal_wallis_test(multi_group_df, "score", "group")
        assert r["k_groups"] == 3
        assert r["df"] == 2

    def test_null(self):
        rng = np.random.default_rng(99)
        df = pl.DataFrame({
            "x": rng.normal(0, 1, 90).tolist(),
            "g": (["A"] * 30 + ["B"] * 30 + ["C"] * 30),
        })
        r = kruskal_wallis_test(df, "x", "g")
        # Should not reject (same distribution)
        assert r["p_value"] > 0.01


class TestNonparamCommands:
    def _session(self, df):
        s = Session()
        s.df = df
        return s

    def test_ranksum_cmd(self, two_group_df):
        from openstat.commands.nonparam_cmds import cmd_ranksum
        s = self._session(two_group_df)
        out = cmd_ranksum(s, "value by(group)")
        assert "rank-sum" in out.lower() or "Mann" in out

    def test_ranksum_no_by(self, two_group_df):
        from openstat.commands.nonparam_cmds import cmd_ranksum
        s = self._session(two_group_df)
        out = cmd_ranksum(s, "value")
        assert "Specify" in out or "by" in out.lower()

    def test_signrank_cmd(self):
        df = pl.DataFrame({"x": list(range(1, 21))})
        from openstat.commands.nonparam_cmds import cmd_signrank
        s = self._session(df)
        out = cmd_signrank(s, "x mu(10)")
        assert "signed-rank" in out.lower() or "Wilcoxon" in out

    def test_kwallis_cmd(self, multi_group_df):
        from openstat.commands.nonparam_cmds import cmd_kwallis
        s = self._session(multi_group_df)
        out = cmd_kwallis(s, "score by(group)")
        assert "Kruskal" in out

    def test_spearman_cmd(self, numeric_df):
        from openstat.commands.nonparam_cmds import cmd_spearman
        s = self._session(numeric_df)
        out = cmd_spearman(s, "x y z")
        assert "Spearman" in out
        assert "x" in out

    def test_spearman_too_few(self, numeric_df):
        from openstat.commands.nonparam_cmds import cmd_spearman
        s = self._session(numeric_df)
        out = cmd_spearman(s, "x")
        assert "at least 2" in out.lower() or "require" in out.lower()
