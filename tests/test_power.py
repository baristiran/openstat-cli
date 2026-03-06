"""Tests for power analysis (stats/power.py and commands/power_cmds.py)."""

from __future__ import annotations

import math

import pytest
import polars as pl

from openstat.stats.power import (
    power_onemean,
    power_twomeans,
    power_oneproportion,
    power_twoproportions,
    power_ols,
    sampsi,
)
from openstat.session import Session


# ── Unit tests: stats/power.py ─────────────────────────────────────────────

class TestPowerOnemean:
    def test_compute_n(self):
        r = power_onemean(effect_size=0.5, alpha=0.05, power=0.80)
        assert r["n"] > 0
        assert r["power"] >= 0.79

    def test_compute_power(self):
        r = power_onemean(effect_size=0.5, alpha=0.05, n=34)
        assert 0.0 < r["power"] <= 1.0

    def test_compute_effect_size(self):
        r = power_onemean(alpha=0.05, n=34, power=0.80)
        assert r["effect_size"] > 0

    def test_delta_sd(self):
        r = power_onemean(alpha=0.05, n=34, power=None, sd=2.0, delta=1.0)
        assert r["effect_size"] == pytest.approx(0.5, abs=1e-4)

    def test_raises_without_two_params(self):
        with pytest.raises((ValueError, TypeError)):
            power_onemean(alpha=0.05)

    def test_onesided(self):
        r2 = power_onemean(effect_size=0.5, alpha=0.05, n=34, two_sided=True)
        r1 = power_onemean(effect_size=0.5, alpha=0.05, n=34, two_sided=False)
        assert r1["power"] > r2["power"]


class TestPowerTwomeans:
    def test_compute_n(self):
        r = power_twomeans(effect_size=0.5, alpha=0.05, power=0.80)
        assert r["n1"] > 0
        assert r["power"] >= 0.79

    def test_ratio(self):
        r = power_twomeans(effect_size=0.5, alpha=0.05, power=0.80, ratio=2.0)
        assert r["n2"] == math.ceil(r["n1"] * 2.0)

    def test_compute_power(self):
        r = power_twomeans(effect_size=0.5, alpha=0.05, n=64)
        assert 0.0 < r["power"] <= 1.0

    def test_keys(self):
        r = power_twomeans(effect_size=0.5, alpha=0.05, n=64)
        assert {"test", "effect_size", "alpha", "n1", "n2", "power"}.issubset(r)


class TestPowerProportions:
    def test_oneprop_compute_n(self):
        r = power_oneproportion(p0=0.5, pa=0.65, alpha=0.05, power=0.80)
        assert r["n"] > 0
        assert r["power"] >= 0.79

    def test_oneprop_compute_power(self):
        r = power_oneproportion(p0=0.5, pa=0.65, alpha=0.05, n=100)
        assert 0.0 < r["power"] <= 1.0

    def test_twoprop_compute_n(self):
        r = power_twoproportions(p1=0.3, p2=0.5, alpha=0.05, power=0.80)
        assert r["n"] > 0

    def test_twoprop_compute_power(self):
        r = power_twoproportions(p1=0.3, p2=0.5, alpha=0.05, n=80)
        assert 0.0 < r["power"] <= 1.0


class TestPowerOLS:
    def test_compute_n(self):
        r = power_ols(f2=0.15, alpha=0.05, power=0.80, k=3)
        assert r["n"] > r["k"]
        assert r["power"] >= 0.79

    def test_compute_power(self):
        r = power_ols(f2=0.15, alpha=0.05, n=100, k=3)
        assert 0.0 < r["power"] <= 1.0

    def test_keys(self):
        r = power_ols(f2=0.15, alpha=0.05, n=100, k=1)
        assert {"test", "f2", "alpha", "n", "k", "power"}.issubset(r)


class TestSampsi:
    def test_basic(self):
        r = sampsi(mu1=0, mu2=0.5, sd=1.0, alpha=0.05, power=0.80)
        assert r["n1"] > 0
        assert "Two-sample" in r["test"]

    def test_larger_effect_smaller_n(self):
        r_small = sampsi(mu1=0, mu2=0.5, sd=1.0)
        r_large = sampsi(mu1=0, mu2=1.0, sd=1.0)
        assert r_small["n1"] > r_large["n1"]


# ── Integration tests: commands ────────────────────────────────────────────

class TestPowerCommand:
    def _session(self):
        s = Session()
        s.df = pl.DataFrame({"x": [1, 2, 3]})
        return s

    def test_power_onemean_cmd(self):
        from openstat.commands.power_cmds import cmd_power
        s = self._session()
        out = cmd_power(s, "onemean, n(34) delta(0.5) sd(1)")
        assert "One-sample" in out
        assert "power" in out.lower()

    def test_power_twomeans_cmd(self):
        from openstat.commands.power_cmds import cmd_power
        s = self._session()
        out = cmd_power(s, "twomeans, n(64) delta(0.5) sd(1)")
        assert "Two-sample" in out

    def test_power_oneprop_cmd(self):
        from openstat.commands.power_cmds import cmd_power
        s = self._session()
        out = cmd_power(s, "oneprop, n(100) p0(0.5) pa(0.65)")
        assert "proportion" in out.lower()

    def test_power_twoprop_cmd(self):
        from openstat.commands.power_cmds import cmd_power
        s = self._session()
        out = cmd_power(s, "twoprop, p1(0.3) p2(0.5) power(0.80)")
        assert "proportion" in out.lower()

    def test_power_ols_cmd(self):
        from openstat.commands.power_cmds import cmd_power
        s = self._session()
        out = cmd_power(s, "ols, n(100) f2(0.15) k(3)")
        assert "OLS" in out

    def test_power_unknown_sub(self):
        from openstat.commands.power_cmds import cmd_power
        s = self._session()
        out = cmd_power(s, "unknown_test")
        assert "Unknown" in out

    def test_power_no_args(self):
        from openstat.commands.power_cmds import cmd_power
        s = self._session()
        out = cmd_power(s, "")
        assert "Usage" in out or "Subcommand" in out

    def test_sampsi_cmd(self):
        from openstat.commands.power_cmds import cmd_sampsi
        s = self._session()
        out = cmd_sampsi(s, "0 0.5 sd(1) alpha(0.05) power(0.80)")
        assert "Two-sample" in out

    def test_sampsi_missing_args(self):
        from openstat.commands.power_cmds import cmd_sampsi
        s = self._session()
        out = cmd_sampsi(s, "0")
        assert "Usage" in out
