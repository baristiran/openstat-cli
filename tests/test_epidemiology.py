"""Tests for epidemiology commands."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from openstat.session import Session
from openstat.stats.epidemiology import cohort_study, case_control, incidence_rate


@pytest.fixture()
def epi_df():
    # outcome=1 more likely when exposure=1
    rng = np.random.default_rng(42)
    n = 200
    exposure = rng.integers(0, 2, n)
    outcome = (exposure * 0.4 + rng.uniform(0, 1, n) < 0.5).astype(int)
    person_time = rng.uniform(1, 5, n)
    return pl.DataFrame({"outcome": outcome, "exposure": exposure, "pt": person_time})


class TestCohortStudy:
    def test_keys(self, epi_df):
        r = cohort_study(epi_df, "outcome", "exposure")
        assert "risk_ratio" in r and "arr" in r and "nnt" in r

    def test_rr_positive(self, epi_df):
        r = cohort_study(epi_df, "outcome", "exposure")
        assert r["risk_ratio"] > 0

    def test_table_2x2(self, epi_df):
        r = cohort_study(epi_df, "outcome", "exposure")
        t = r["table_2x2"]
        assert t["a"] + t["b"] + t["c"] + t["d"] == len(epi_df)

    def test_cmd(self, epi_df):
        from openstat.commands.epi_cmds import cmd_cs
        s = Session()
        s.df = epi_df
        out = cmd_cs(s, "outcome exposure")
        assert "Cohort" in out or "Risk" in out


class TestCaseControl:
    def test_keys(self, epi_df):
        r = case_control(epi_df, "outcome", "exposure")
        assert "odds_ratio" in r

    def test_or_positive(self, epi_df):
        r = case_control(epi_df, "outcome", "exposure")
        assert r["odds_ratio"] > 0

    def test_cmd(self, epi_df):
        from openstat.commands.epi_cmds import cmd_cc
        s = Session()
        s.df = epi_df
        out = cmd_cc(s, "outcome exposure")
        assert "Odds" in out or "Case-Control" in out


class TestIncidenceRate:
    def test_keys(self, epi_df):
        r = incidence_rate(epi_df, "outcome", "pt")
        assert "incidence_rate" in r and "cases" in r

    def test_rate_positive(self, epi_df):
        r = incidence_rate(epi_df, "outcome", "pt")
        assert r["incidence_rate"] > 0

    def test_cmd(self, epi_df):
        from openstat.commands.epi_cmds import cmd_ir
        s = Session()
        s.df = epi_df
        out = cmd_ir(s, "outcome pt")
        assert "Incidence" in out or "Rate" in out
