"""Tests for Round 4 features: hypothesis tests, merge, reshape, sample, replace."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.commands.data_cmds import (
    cmd_merge, cmd_pivot, cmd_melt, cmd_sample, cmd_replace,
)
from openstat.commands.stat_cmds import cmd_ttest, cmd_chi2, cmd_anova
from openstat.stats.models import run_ttest, run_chi2, run_anova


# -----------------------------------------------------------------------
# Hypothesis Tests — numerical correctness
# -----------------------------------------------------------------------

class TestTTestCorrectness:
    """Verify t-test statistics against known values."""

    def test_one_sample_zero_mean(self):
        """Sample clearly above 0 → significant."""
        df = pl.DataFrame({"x": [10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.0, 11.0]})
        result = run_ttest(df, "x", mu=0.0)
        assert result.p_value < 0.001
        assert result.statistic > 0  # mean > 0
        assert result.df == 7  # n-1

    def test_one_sample_matches_mu(self):
        """When mu equals sample mean → not significant."""
        vals = [10.0, 12.0, 11.0, 13.0, 9.0]
        mu = float(np.mean(vals))
        df = pl.DataFrame({"x": vals})
        result = run_ttest(df, "x", mu=mu)
        assert abs(result.statistic) < 0.001  # t ≈ 0
        assert result.p_value > 0.99

    def test_two_sample_different_means(self):
        """Two clearly separated groups → significant."""
        np.random.seed(42)
        g1 = np.random.normal(10, 1, 50)
        g2 = np.random.normal(20, 1, 50)
        df = pl.DataFrame({
            "val": g1.tolist() + g2.tolist(),
            "group": ["A"] * 50 + ["B"] * 50,
        })
        result = run_ttest(df, "val", by="group")
        assert result.p_value < 0.001
        assert "Two-sample" in result.test_name

    def test_two_sample_same_means(self):
        """Same distribution → not significant."""
        np.random.seed(42)
        vals = np.random.normal(10, 1, 100)
        df = pl.DataFrame({
            "val": vals.tolist(),
            "group": ["A"] * 50 + ["B"] * 50,
        })
        result = run_ttest(df, "val", by="group")
        assert result.p_value > 0.05

    def test_paired_ttest(self):
        """Paired samples with known difference."""
        df = pl.DataFrame({
            "before": [10.0, 12.0, 14.0, 8.0, 11.0],
            "after": [15.0, 17.0, 19.0, 13.0, 16.0],
        })
        result = run_ttest(df, "before", paired_col="after")
        assert result.p_value < 0.001  # clear difference of 5
        assert "Paired" in result.test_name

    def test_two_sample_needs_binary_groups(self):
        df = pl.DataFrame({
            "val": [1.0, 2.0, 3.0],
            "group": ["A", "B", "C"],
        })
        with pytest.raises(ValueError, match="2 groups"):
            run_ttest(df, "val", by="group")


class TestChi2Correctness:
    """Verify chi-square test."""

    def test_independent_variables(self):
        """Truly independent → not significant."""
        np.random.seed(42)
        n = 200
        df = pl.DataFrame({
            "coin1": np.random.choice(["H", "T"], n).tolist(),
            "coin2": np.random.choice(["H", "T"], n).tolist(),
        })
        result = run_chi2(df, "coin1", "coin2")
        assert result.p_value > 0.05

    def test_dependent_variables(self):
        """Perfectly associated → significant."""
        df = pl.DataFrame({
            "x": ["A"] * 50 + ["B"] * 50,
            "y": ["Yes"] * 50 + ["No"] * 50,
        })
        result = run_chi2(df, "x", "y")
        assert result.p_value < 0.001
        assert result.statistic > 0

    def test_cramers_v_reported(self):
        df = pl.DataFrame({
            "x": ["A"] * 50 + ["B"] * 50,
            "y": ["Yes"] * 50 + ["No"] * 50,
        })
        result = run_chi2(df, "x", "y")
        assert "Cramér's V" in result.details
        assert result.details["Cramér's V"] > 0.9  # near perfect association


class TestAnovaCorrectness:
    """Verify one-way ANOVA."""

    def test_different_group_means(self):
        """Groups with clearly different means → significant."""
        np.random.seed(42)
        df = pl.DataFrame({
            "score": (
                np.random.normal(10, 1, 30).tolist() +
                np.random.normal(20, 1, 30).tolist() +
                np.random.normal(30, 1, 30).tolist()
            ),
            "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        })
        result = run_anova(df, "score", "group")
        assert result.p_value < 0.001
        assert result.statistic > 10  # large F

    def test_same_group_means(self):
        """Same distribution → not significant."""
        np.random.seed(42)
        df = pl.DataFrame({
            "score": np.random.normal(10, 2, 90).tolist(),
            "group": ["A"] * 30 + ["B"] * 30 + ["C"] * 30,
        })
        result = run_anova(df, "score", "group")
        assert result.p_value > 0.05

    def test_needs_two_groups(self):
        df = pl.DataFrame({"x": [1.0, 2.0], "g": ["A", "A"]})
        with pytest.raises(ValueError, match="at least 2"):
            run_anova(df, "x", "g")


# -----------------------------------------------------------------------
# Hypothesis test commands (via command interface)
# -----------------------------------------------------------------------

class TestHypothesisCommands:
    @pytest.fixture
    def session(self, tmp_path):
        s = Session(output_dir=tmp_path / "outputs")
        np.random.seed(42)
        s.df = pl.DataFrame({
            "score": np.random.normal(75, 10, 100).tolist(),
            "group": (["control"] * 50 + ["treatment"] * 50),
            "gender": np.random.choice(["M", "F"], 100).tolist(),
            "region": np.random.choice(["N", "S", "E"], 100).tolist(),
        })
        return s

    def test_ttest_usage(self, session):
        assert "Usage" in cmd_ttest(session, "")

    def test_ttest_one_sample(self, session):
        result = cmd_ttest(session, "score mu=75")
        assert "t-test" in result.lower() or "One-sample" in result

    def test_ttest_two_sample(self, session):
        result = cmd_ttest(session, "score by group")
        assert "Two-sample" in result or "p-value" in result

    def test_chi2_command(self, session):
        result = cmd_chi2(session, "group gender")
        assert "Chi-square" in result or "p-value" in result

    def test_chi2_usage(self, session):
        assert "Usage" in cmd_chi2(session, "single")

    def test_anova_command(self, session):
        result = cmd_anova(session, "score by region")
        assert "ANOVA" in result or "p-value" in result

    def test_anova_usage(self, session):
        assert "Usage" in cmd_anova(session, "score region")  # missing 'by'


# -----------------------------------------------------------------------
# Merge / Join
# -----------------------------------------------------------------------

class TestMerge:
    @pytest.fixture
    def session_and_file(self, tmp_path):
        s = Session(output_dir=tmp_path / "outputs")
        s.df = pl.DataFrame({
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Carol", "Dave"],
        })
        other = pl.DataFrame({
            "id": [2, 3, 4, 5],
            "score": [85.0, 90.0, 78.0, 92.0],
        })
        other_path = tmp_path / "scores.csv"
        other.write_csv(other_path)
        return s, str(other_path)

    def test_inner_join(self, session_and_file):
        s, path = session_and_file
        result = cmd_merge(s, f"{path} on id")
        assert "3" in result  # 3 matches (ids 2,3,4)
        assert s.df.height == 3

    def test_left_join(self, session_and_file):
        s, path = session_and_file
        result = cmd_merge(s, f"{path} on id how=left")
        assert s.df.height == 4  # keeps all left rows

    def test_outer_join(self, session_and_file):
        s, path = session_and_file
        result = cmd_merge(s, f"{path} on id how=outer")
        assert s.df.height == 5  # ids 1-5

    def test_merge_creates_snapshot(self, session_and_file):
        s, path = session_and_file
        cmd_merge(s, f"{path} on id")
        assert s.undo_depth == 1

    def test_merge_usage(self, session_and_file):
        s, _ = session_and_file
        assert "Usage" in cmd_merge(s, "somefile.csv key")  # missing 'on'

    def test_merge_missing_key(self, session_and_file):
        s, path = session_and_file
        result = cmd_merge(s, f"{path} on nonexistent")
        assert "not found" in result


# -----------------------------------------------------------------------
# Pivot / Melt
# -----------------------------------------------------------------------

class TestReshape:
    def test_pivot(self):
        s = Session()
        s.df = pl.DataFrame({
            "name": ["Alice", "Alice", "Bob", "Bob"],
            "subject": ["math", "eng", "math", "eng"],
            "score": [90, 85, 78, 92],
        })
        result = cmd_pivot(s, "score by subject over name")
        assert "wide" in result.lower()
        assert s.df.width > 2  # name + math + eng

    def test_melt(self):
        s = Session()
        s.df = pl.DataFrame({
            "name": ["Alice", "Bob"],
            "math": [90, 78],
            "eng": [85, 92],
        })
        result = cmd_melt(s, "name, math eng")
        assert "long" in result.lower()
        assert s.df.height == 4  # 2 people x 2 subjects

    def test_pivot_undo(self):
        s = Session()
        s.df = pl.DataFrame({
            "name": ["A", "A", "B", "B"],
            "sub": ["x", "y", "x", "y"],
            "val": [1, 2, 3, 4],
        })
        original_shape = (s.df.height, s.df.width)
        cmd_pivot(s, "val by sub over name")
        assert (s.df.height, s.df.width) != original_shape
        s.undo()
        assert (s.df.height, s.df.width) == original_shape

    def test_melt_usage(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert "Usage" in cmd_melt(s, "x y z")  # no comma

    def test_pivot_usage(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert "Usage" in cmd_pivot(s, "x y z")  # no 'by'


# -----------------------------------------------------------------------
# Sample / Replace
# -----------------------------------------------------------------------

class TestSample:
    @pytest.fixture
    def session(self):
        s = Session()
        s.df = pl.DataFrame({"x": list(range(100)), "y": list(range(100))})
        return s

    def test_sample_n(self, session):
        result = cmd_sample(session, "10")
        assert session.df.height == 10
        assert "10" in result

    def test_sample_pct(self, session):
        result = cmd_sample(session, "25%")
        assert session.df.height == 25

    def test_sample_creates_snapshot(self, session):
        cmd_sample(session, "50")
        assert session.undo_depth == 1

    def test_sample_usage(self, session):
        assert "Usage" in cmd_sample(session, "")


class TestReplace:
    def test_replace_string(self):
        s = Session()
        s.df = pl.DataFrame({"region": ["North", "South", "North"]})
        result = cmd_replace(s, "region North Norte")
        assert "2" in result  # 2 occurrences
        assert s.df["region"].to_list() == ["Norte", "South", "Norte"]

    def test_replace_numeric(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3, 2]})
        result = cmd_replace(s, "x 2 99")
        assert "2" in result  # 2 occurrences
        assert s.df["x"].to_list() == [1, 99, 3, 99]

    def test_replace_creates_snapshot(self):
        s = Session()
        s.df = pl.DataFrame({"x": ["a", "b"]})
        cmd_replace(s, "x a z")
        assert s.undo_depth == 1

    def test_replace_missing_col(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        result = cmd_replace(s, "nonexistent 1 2")
        assert "not found" in result

    def test_replace_usage(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1]})
        assert "Usage" in cmd_replace(s, "x 1")
