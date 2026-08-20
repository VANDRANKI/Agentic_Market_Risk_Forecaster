"""
Tests for risk/portfolio.py.

These require PyPortfolioOpt. If it is not installed, every function returns
None or logs a warning rather than raising, so the tests are skipped instead
of failing the whole suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

pypfopt = pytest.importorskip("pypfopt")

from risk.portfolio import run_all_optimizations  # noqa: E402


def _prices(n=300):
    rng = np.random.default_rng(2)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "AAA": (1 + rng.normal(0.0004, 0.01, n)).cumprod() * 100,
            "BBB": (1 + rng.normal(0.0003, 0.012, n)).cumprod() * 100,
        },
        index=idx,
    )


class TestCurrentPortfolioWeightHandling:
    def test_matches_manually_renormalized_weights(self):
        """Regression: EfficientFrontier.set_weights builds its vector as
        [input_weights[t] for t in self.tickers], so a weight key for a ticker
        not in prices.columns was silently dropped rather than renormalized.
        Normalizing before that drop left the vector pypfopt actually used
        summing to less than 1, which understated volatility by exactly the
        missing weight fraction with no error and no warning."""
        prices = _prices()

        with_ghost = run_all_optimizations(
            prices, {"AAA": 0.5, "BBB": 0.3, "GHOST": 0.2}
        )["current"]
        manually_renormalized = run_all_optimizations(
            prices, {"AAA": 0.625, "BBB": 0.375}
        )["current"]

        assert with_ghost["weights"] == manually_renormalized["weights"]
        assert with_ghost["volatility"] == pytest.approx(
            manually_renormalized["volatility"]
        )
        assert with_ghost["expected_return"] == pytest.approx(
            manually_renormalized["expected_return"]
        )

    def test_returned_weights_sum_to_one(self):
        prices = _prices()
        out = run_all_optimizations(prices, {"AAA": 0.5, "BBB": 0.3, "GHOST": 0.2})
        assert sum(out["current"]["weights"].values()) == pytest.approx(1.0)

    def test_all_tickers_present_is_unaffected(self):
        prices = _prices()
        out = run_all_optimizations(prices, {"AAA": 0.6, "BBB": 0.4})
        assert out["current"]["weights"] == {"AAA": 0.6, "BBB": 0.4}

    def test_zero_sum_weights_return_none_not_a_crash(self):
        prices = _prices()
        out = run_all_optimizations(prices, {"AAA": 1.0, "BBB": -1.0})
        assert out["current"] is None

    def test_no_matching_tickers_returns_none_not_a_crash(self):
        prices = _prices()
        out = run_all_optimizations(prices, {"GHOST_A": 0.5, "GHOST_B": 0.5})
        assert out["current"] is None


class TestRunAllOptimizationsShape:
    def test_all_four_strategies_present(self):
        prices = _prices()
        out = run_all_optimizations(prices, {"AAA": 0.6, "BBB": 0.4})
        assert set(out) == {"current", "max_sharpe", "min_vol", "equal_weight"}

    def test_equal_weight_always_present(self):
        """equal_weight_portfolio has no failure path that returns None; it
        falls back to weights-only if performance calc fails."""
        prices = _prices()
        out = run_all_optimizations(prices, {"AAA": 0.6, "BBB": 0.4})
        assert out["equal_weight"] is not None
        assert out["equal_weight"]["weights"] == {
            "AAA": pytest.approx(0.5), "BBB": pytest.approx(0.5)
        }
