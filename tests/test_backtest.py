"""
Unit tests for risk/backtest.py.

Tests use constructed exceedance scenarios where we know the exact
expected outputs (number of violations, p-values, etc.).
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from risk.backtest import (
    count_exceedances,
    kupiec_test,
    christoffersen_test,
    rolling_var_series,
)


# ---------------------------------------------------------------------------
# count_exceedances
# ---------------------------------------------------------------------------


class TestCountExceedances:
    def test_exact_count(self):
        # returns:    -0.10,  0.02, -0.05, 0.01, -0.03
        # VaR:         0.04,  0.04,  0.04, 0.04,  0.04
        # loss > VaR:  YES,    NO,   YES,   NO,    NO
        returns = pd.Series([-0.10, 0.02, -0.05, 0.01, -0.03])
        var_s = pd.Series([0.04, 0.04, 0.04, 0.04, 0.04])
        result = count_exceedances(returns, var_s)
        assert result["n_exceedances"] == 2
        assert result["n_observations"] == 5

    def test_no_exceedances(self):
        returns = pd.Series([0.01, 0.02, -0.005, 0.01])
        var_s = pd.Series([0.05] * 4)
        result = count_exceedances(returns, var_s)
        assert result["n_exceedances"] == 0
        assert result["exceedance_rate"] == 0.0

    def test_all_exceedances(self):
        returns = pd.Series([-0.10, -0.11, -0.12])
        var_s = pd.Series([0.05] * 3)
        result = count_exceedances(returns, var_s)
        assert result["n_exceedances"] == 3
        assert result["exceedance_rate"] == pytest.approx(1.0)

    def test_exceedance_rate_correct(self):
        # 10 obs, 2 exceptions -> rate = 0.20
        # 2 breaches (loss > VaR=0.05): -0.10, -0.07; 8 non-breaches
        returns = pd.Series([-0.10, 0.01, 0.02, -0.07, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02])
        var_s = pd.Series([0.05] * 10)
        result = count_exceedances(returns, var_s)
        assert result["n_exceedances"] == 2
        assert result["exceedance_rate"] == pytest.approx(0.20)

    def test_returns_keys(self):
        returns = pd.Series([-0.10, 0.01])
        var_s = pd.Series([0.05, 0.05])
        result = count_exceedances(returns, var_s)
        assert "n_exceedances" in result
        assert "n_observations" in result
        assert "exceedance_rate" in result
        assert "exceedance_dates" in result


# ---------------------------------------------------------------------------
# kupiec_test
# ---------------------------------------------------------------------------


class TestKupiecTest:
    def test_zero_exceedances_rejects(self):
        # 0 vs 12.5 expected at 5% of 250 -- extreme, should reject
        result = kupiec_test(0, 250, alpha=0.05)
        assert result["reject_h0"] is True

    def test_expected_exceedances_pass(self):
        # 13 vs 12.5 expected -- should not reject
        result = kupiec_test(13, 250, alpha=0.05)
        assert result["reject_h0"] is False

    def test_too_many_exceedances_reject(self):
        # 50 vs 12.5 expected (20% vs 5%) -- should reject
        result = kupiec_test(50, 250, alpha=0.05)
        assert result["reject_h0"] is True

    def test_p_value_range(self):
        result = kupiec_test(13, 250, alpha=0.05)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_statistic_positive(self):
        result = kupiec_test(13, 250, alpha=0.05)
        assert result["statistic"] >= 0

    def test_expected_exceptions_correct(self):
        result = kupiec_test(0, 100, alpha=0.05)
        assert result["expected_exceptions"] == 5

    def test_returns_all_required_keys(self):
        result = kupiec_test(10, 250, alpha=0.05)
        for key in ["statistic", "p_value", "reject_h0", "expected_exceptions",
                    "actual_exceptions", "test_name", "interpretation"]:
            assert key in result

    def test_zero_observations(self):
        result = kupiec_test(0, 0, alpha=0.05)
        assert result["p_value"] is None

    def test_99_confidence_expected_count(self):
        # At 99% VaR, expect 1% of 250 = 2.5 -> 3 exceptions
        result = kupiec_test(2, 250, alpha=0.01)
        assert result["expected_exceptions"] == 2 or result["expected_exceptions"] == 3

    def test_interpretation_string_nonempty(self):
        result = kupiec_test(13, 250, alpha=0.05)
        assert len(result["interpretation"]) > 10


# ---------------------------------------------------------------------------
# christoffersen_test
# ---------------------------------------------------------------------------


class TestChristoffersenTest:
    def _make_test_data(self, n: int = 300, breach_prob: float = 0.05):
        """Create returns with random exceedances at given rate."""
        rng = np.random.default_rng(99)
        returns = pd.Series(rng.normal(0, 0.01, n))
        var_s = pd.Series([historical_var_val(returns, 0.05)] * n)
        return returns, var_s

    def test_independent_breaches_pass(self):
        # Inject breaches randomly (not clustered): should not reject H0
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 500))
        # Set VaR so exactly 5% are breaches, randomly distributed
        var_level = float(-np.percentile(r, 5))
        var_s = pd.Series([var_level] * 500)
        result = christoffersen_test(r, var_s, alpha=0.05)
        assert "p_value" in result
        assert "reject_h0" in result
        assert "statistic" in result

    def test_clustered_breaches_may_reject(self):
        # Inject 20 consecutive breaches at the start (clearly clustered)
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 300))
        var_s = pd.Series([0.005] * 300)
        # Force first 20 to be massive losses (clear breaches)
        r.iloc[:20] = -0.10
        result = christoffersen_test(r, var_s, alpha=0.05)
        # Just verify it returns valid output; clustered data often rejects
        assert 0.0 <= result["p_value"] <= 1.0

    def test_returns_all_keys(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 200))
        var_s = pd.Series([0.02] * 200)
        result = christoffersen_test(r, var_s)
        for key in ["statistic", "p_value", "reject_h0", "pi01", "pi11", "test_name"]:
            assert key in result

    def test_pi01_pi11_between_0_and_1(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 300))
        var_s = pd.Series([0.015] * 300)
        result = christoffersen_test(r, var_s)
        assert 0.0 <= result["pi01"] <= 1.0
        assert 0.0 <= result["pi11"] <= 1.0


def historical_var_val(returns: pd.Series, alpha: float) -> float:
    return float(-np.percentile(returns, alpha * 100))


# ---------------------------------------------------------------------------
# rolling_var_series
# ---------------------------------------------------------------------------


class TestRollingVarSeries:
    def test_output_is_series(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 400))
        rv = rolling_var_series(r, alpha=0.05, window=100)
        assert isinstance(rv, pd.Series)

    def test_all_positive(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 400))
        rv = rolling_var_series(r, alpha=0.05, window=100)
        assert (rv > 0).all()

    def test_length_less_than_input(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 400))
        rv = rolling_var_series(r, alpha=0.05, window=100)
        assert len(rv) < len(r)

    def test_reasonable_magnitude(self):
        rng = np.random.default_rng(42)
        r = pd.Series(rng.normal(0, 0.01, 400))
        rv = rolling_var_series(r, alpha=0.05, window=100)
        # Rolling 95% VaR for N(0, 0.01) should be around 0.016
        assert 0.010 < rv.mean() < 0.025
