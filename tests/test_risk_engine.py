"""
Unit tests for risk/engine.py.

Uses synthetic return data with known statistical properties so that
we can verify the numerical outputs against ground-truth values.

All VaR/ES values are in decimal form (0.01 = 1%).
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from risk.engine import (
    compute_returns,
    compute_portfolio_returns,
    historical_var,
    historical_es,
    variance_covariance_var,
    variance_covariance_es,
    monte_carlo_var,
    monte_carlo_es,
    scale_var_to_horizon,
    compute_drawdown,
    max_drawdown,
    rolling_volatility,
    regime_classifier,
    current_regime,
    detect_anomalies_zscore,
    return_stats,
)

# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N = 1000

# Normal daily returns: mean=0, std=0.01 (1% daily)
NORMAL_RETURNS = pd.Series(RNG.normal(0, 0.01, N), name="test")

# Prices derived from normal returns
PRICES = pd.Series((1 + NORMAL_RETURNS).cumprod() * 100)
PRICES.name = "price"


# ---------------------------------------------------------------------------
# compute_returns
# ---------------------------------------------------------------------------


class TestComputeReturns:
    def test_log_returns_length(self):
        r = compute_returns(PRICES, method="log")
        assert len(r) == len(PRICES) - 1

    def test_simple_returns_length(self):
        r = compute_returns(PRICES, method="simple")
        assert len(r) == len(PRICES) - 1

    def test_log_returns_mean_near_zero(self):
        r = compute_returns(PRICES, method="log")
        assert abs(r.mean()) < 0.005

    def test_log_returns_std_near_target(self):
        r = compute_returns(PRICES, method="log")
        # std should be close to 0.01
        assert 0.007 < r.std() < 0.013

    def test_dataframe_input(self):
        df_prices = pd.DataFrame({"A": PRICES, "B": PRICES * 1.1})
        r = compute_returns(df_prices, method="log")
        assert isinstance(r, pd.DataFrame)
        assert r.shape[0] == len(df_prices) - 1


# ---------------------------------------------------------------------------
# historical_var
# ---------------------------------------------------------------------------


class TestHistoricalVaR:
    def test_var_positive(self):
        var = historical_var(NORMAL_RETURNS, 0.05)
        assert var > 0

    def test_var_95_reasonable(self):
        # For N(0, 0.01), 95% VaR ~ z(0.05)*0.01 = 1.645*0.01 = 0.01645
        var = historical_var(NORMAL_RETURNS, 0.05)
        assert 0.010 < var < 0.025, f"Expected ~0.016, got {var:.4f}"

    def test_var_99_greater_than_var_95(self):
        v95 = historical_var(NORMAL_RETURNS, 0.05)
        v99 = historical_var(NORMAL_RETURNS, 0.01)
        assert v99 > v95, "99% VaR must exceed 95% VaR"

    def test_var_99_reasonable(self):
        # For N(0, 0.01), 99% VaR ~ 2.326*0.01 = 0.02326
        var = historical_var(NORMAL_RETURNS, 0.01)
        assert 0.015 < var < 0.035, f"Expected ~0.023, got {var:.4f}"


# ---------------------------------------------------------------------------
# historical_es
# ---------------------------------------------------------------------------


class TestHistoricalES:
    def test_es_greater_than_var(self):
        var = historical_var(NORMAL_RETURNS, 0.05)
        es = historical_es(NORMAL_RETURNS, 0.05)
        assert es >= var, f"ES {es:.4f} must be >= VaR {var:.4f}"

    def test_es_positive(self):
        es = historical_es(NORMAL_RETURNS, 0.05)
        assert es > 0

    def test_es_99_greater_than_es_95(self):
        es95 = historical_es(NORMAL_RETURNS, 0.05)
        es99 = historical_es(NORMAL_RETURNS, 0.01)
        assert es99 > es95


# ---------------------------------------------------------------------------
# variance_covariance_var
# ---------------------------------------------------------------------------


class TestParametricVaR:
    def test_close_to_historical_for_normal_data(self):
        # For truly normal data, parametric and historical should agree closely
        hist = historical_var(NORMAL_RETURNS, 0.05)
        param = variance_covariance_var(NORMAL_RETURNS, 0.05)
        assert abs(hist - param) < 0.005, (
            f"Parametric {param:.4f} should be close to historical {hist:.4f} "
            "for normal data."
        )

    def test_es_greater_than_var(self):
        var = variance_covariance_var(NORMAL_RETURNS, 0.05)
        es = variance_covariance_es(NORMAL_RETURNS, 0.05)
        assert es >= var

    def test_var_positive(self):
        var = variance_covariance_var(NORMAL_RETURNS, 0.05)
        assert var > 0


# ---------------------------------------------------------------------------
# monte_carlo_var
# ---------------------------------------------------------------------------


class TestMonteCarloVaR:
    def test_mc_var_positive(self):
        var = monte_carlo_var(NORMAL_RETURNS, 0.05, n_sims=5000)
        assert var > 0

    def test_mc_close_to_historical(self):
        hist = historical_var(NORMAL_RETURNS, 0.05)
        mc = monte_carlo_var(NORMAL_RETURNS, 0.05, n_sims=10000)
        assert abs(hist - mc) < 0.005

    def test_mc_es_greater_than_mc_var(self):
        var = monte_carlo_var(NORMAL_RETURNS, 0.05, n_sims=5000)
        es = monte_carlo_es(NORMAL_RETURNS, 0.05, n_sims=5000)
        assert es >= var

    def test_mc_10day_greater_than_1day(self):
        var_1d = monte_carlo_var(NORMAL_RETURNS, 0.05, n_sims=5000, horizon=1)
        var_10d = monte_carlo_var(NORMAL_RETURNS, 0.05, n_sims=5000, horizon=10)
        assert var_10d > var_1d


# ---------------------------------------------------------------------------
# scale_var_to_horizon
# ---------------------------------------------------------------------------


class TestScaling:
    def test_sqrt_of_time(self):
        var_1d = 0.02
        var_10d = scale_var_to_horizon(var_1d, 10)
        expected = var_1d * np.sqrt(10)
        assert abs(var_10d - expected) < 1e-10

    def test_1day_horizon_unchanged(self):
        var = 0.02
        assert scale_var_to_horizon(var, 1) == pytest.approx(var)


# ---------------------------------------------------------------------------
# compute_portfolio_returns
# ---------------------------------------------------------------------------


class TestPortfolioReturns:
    def test_single_asset_full_weight(self):
        r_df = pd.DataFrame({"SPY": NORMAL_RETURNS})
        port = compute_portfolio_returns(r_df, {"SPY": 1.0})
        pd.testing.assert_series_equal(port, NORMAL_RETURNS, check_names=False)

    def test_equal_weight_two_assets(self):
        double_returns = NORMAL_RETURNS * 2.0
        r_df = pd.DataFrame({"A": NORMAL_RETURNS, "B": double_returns})
        port = compute_portfolio_returns(r_df, {"A": 0.5, "B": 0.5})
        expected = (NORMAL_RETURNS + double_returns) / 2
        pd.testing.assert_series_equal(port, expected, check_names=False, rtol=1e-10)

    def test_weights_are_normalized(self):
        r_df = pd.DataFrame({"A": NORMAL_RETURNS, "B": NORMAL_RETURNS})
        # Weights sum to 2, should be normalized to 0.5 each
        port = compute_portfolio_returns(r_df, {"A": 1.0, "B": 1.0})
        pd.testing.assert_series_equal(port, NORMAL_RETURNS, check_names=False)

    def test_raises_on_no_overlap(self):
        r_df = pd.DataFrame({"SPY": NORMAL_RETURNS})
        with pytest.raises(ValueError):
            compute_portfolio_returns(r_df, {"AAPL": 1.0})


# ---------------------------------------------------------------------------
# Drawdown
# ---------------------------------------------------------------------------


class TestDrawdown:
    def test_drawdown_all_non_positive(self):
        dd = compute_drawdown(NORMAL_RETURNS)
        assert (dd <= 0).all(), "All drawdown values must be <= 0"

    def test_max_drawdown_negative(self):
        mdd = max_drawdown(NORMAL_RETURNS)
        assert mdd < 0

    def test_all_gains_no_drawdown(self):
        all_positive = pd.Series([0.01] * 50)
        dd = compute_drawdown(all_positive)
        assert dd.max() == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------


class TestRollingVol:
    def test_annualized_scale(self):
        rv = rolling_volatility(NORMAL_RETURNS, window=20)
        rv_clean = rv.dropna()
        # Annualized vol should be around 0.01 * sqrt(252) ~ 0.159
        mean_rv = rv_clean.mean()
        assert 0.10 < mean_rv < 0.25, f"Mean annualized vol {mean_rv:.3f} out of range"

    def test_length_with_dropna(self):
        rv = rolling_volatility(NORMAL_RETURNS, window=20)
        assert len(rv.dropna()) == len(NORMAL_RETURNS) - 19


# ---------------------------------------------------------------------------
# Regime classifier
# ---------------------------------------------------------------------------


class TestRegimeClassifier:
    def test_regime_values(self):
        regimes = regime_classifier(NORMAL_RETURNS, window=20)
        assert set(regimes.unique()).issubset({"low", "mid", "high"})

    def test_regime_roughly_balanced(self):
        regimes = regime_classifier(NORMAL_RETURNS, window=20)
        counts = regimes.value_counts(normalize=True)
        # Each regime should have roughly 1/3 of days (with some tolerance)
        for regime in ["low", "mid", "high"]:
            assert 0.20 < counts.get(regime, 0) < 0.50

    def test_current_regime_valid(self):
        regime = current_regime(NORMAL_RETURNS)
        assert regime in {"low", "mid", "high"}


# ---------------------------------------------------------------------------
# Return statistics
# ---------------------------------------------------------------------------


class TestReturnStats:
    def test_all_keys_present(self):
        stats = return_stats(NORMAL_RETURNS)
        required = [
            "mean_daily", "mean_annual", "std_daily", "std_annual",
            "skewness", "excess_kurtosis", "min_return", "max_return",
            "recent_vol_20d", "sharpe_ratio_annual", "n_observations",
        ]
        for key in required:
            assert key in stats, f"Missing key: {key}"

    def test_std_reasonable(self):
        stats = return_stats(NORMAL_RETURNS)
        # Daily std should be close to 0.01
        assert 0.007 < stats["std_daily"] < 0.013

    def test_n_observations(self):
        stats = return_stats(NORMAL_RETURNS)
        assert stats["n_observations"] == N


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


class TestAnomalyDetection:
    def test_zscore_flags_are_bool(self):
        flags = detect_anomalies_zscore(NORMAL_RETURNS, threshold=2.5, window=60)
        assert flags.dtype == bool

    def test_zscore_detects_outliers(self):
        # Inject obvious outliers
        r = NORMAL_RETURNS.copy()
        r.iloc[100] = -0.20  # massive negative return
        r.iloc[500] = 0.20
        flags = detect_anomalies_zscore(r, threshold=2.5, window=60)
        assert flags.iloc[100] or flags.iloc[101], "Should flag day after outlier"

    def test_zscore_flag_rate_reasonable(self):
        # Should flag roughly ~2-5% of normal data at threshold 2.5
        flags = detect_anomalies_zscore(NORMAL_RETURNS, threshold=2.5, window=60)
        rate = flags.mean()
        assert 0.00 < rate < 0.15, f"Anomaly rate {rate:.3f} seems off"
