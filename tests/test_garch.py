"""
Tests for risk/garch.py.

Returns are simulated from a known GARCH(1,1) process with a fixed seed, so
the fitted parameters and forecasts can be checked against the truth rather
than only for "does not crash".
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from risk.garch import (  # noqa: E402
    fit_garch,
    garch_conditional_vol_series,
    garch_var_forecast,
)

OMEGA, ALPHA, BETA = 2e-6, 0.08, 0.90


def _simulate(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = OMEGA / (1 - ALPHA - BETA)
    for t in range(n):
        if t > 0:
            h[t] = OMEGA + ALPHA * r[t - 1] ** 2 + BETA * h[t - 1]
        r[t] = np.sqrt(h[t]) * rng.standard_normal()
    return pd.Series(r, index=pd.bdate_range("2018-01-01", periods=n), name="ret")


@pytest.fixture(scope="module")
def returns():
    return _simulate()


@pytest.fixture(scope="module")
def fit(returns):
    return fit_garch(returns)


class TestFitGarch:
    def test_too_few_observations_returns_none(self):
        assert fit_garch(_simulate(n=99)) is None

    def test_nans_do_not_count_towards_the_minimum(self):
        r = _simulate(n=150)
        r.iloc[:60] = np.nan
        assert fit_garch(r) is None

    def test_result_has_the_documented_keys(self, fit):
        assert {"result", "aic", "bic", "persistence", "params", "annualized_long_run_vol"} <= set(fit)

    def test_persistence_is_near_the_true_value(self, fit):
        assert fit["persistence"] == pytest.approx(ALPHA + BETA, abs=0.05)
        assert fit["persistence"] == pytest.approx(
            fit["params"]["alpha[1]"] + fit["params"]["beta[1]"], abs=1e-5
        )

    def test_annualized_long_run_vol_is_in_a_plausible_range(self, fit):
        true_vol = np.sqrt(OMEGA / (1 - ALPHA - BETA) * 252)
        assert 0.5 * true_vol < fit["annualized_long_run_vol"] < 2.0 * true_vol

    def test_input_series_is_not_modified(self, returns):
        before = returns.copy()
        fit_garch(returns)
        pd.testing.assert_series_equal(returns, before)


class TestGarchVarForecast:
    def test_too_few_observations_returns_none(self):
        assert garch_var_forecast(_simulate(n=50)) is None

    def test_one_day_var_is_close_to_the_normal_quantile_of_the_daily_vol(self, returns):
        out = garch_var_forecast(returns, alpha=0.05, horizon=1)
        # 1.645 * daily sigma, with the mean near zero
        assert out["var"] == pytest.approx(1.645 * out["conditional_sigma"], rel=0.15)
        assert out["horizon"] == 1

    def test_es_exceeds_var_and_99_exceeds_95(self, returns):
        v95 = garch_var_forecast(returns, alpha=0.05)
        v99 = garch_var_forecast(returns, alpha=0.01)
        assert v95["es"] > v95["var"] > 0
        assert v99["es"] > v99["var"] > v95["var"]

    def test_annualized_sigma_is_sqrt_252_times_daily(self, returns):
        out = garch_var_forecast(returns)
        assert out["conditional_sigma_annual"] == pytest.approx(
            out["conditional_sigma"] * np.sqrt(252), rel=1e-3
        )

    @pytest.mark.parametrize("horizon", [5, 10, 20])
    def test_multi_day_var_grows_with_the_horizon(self, returns, horizon):
        """Regression: only the last day's variance was used, so a 10-day VaR
        came out almost identical to the 1-day VaR."""
        one = garch_var_forecast(returns, horizon=1)["var"]
        many = garch_var_forecast(returns, horizon=horizon)["var"]
        # roughly sqrt(horizon) times larger; allow for mean reversion of the variance
        assert many > one * np.sqrt(horizon) * 0.7
        assert many < one * np.sqrt(horizon) * 1.5

    def test_multi_day_es_also_grows(self, returns):
        one = garch_var_forecast(returns, horizon=1)["es"]
        ten = garch_var_forecast(returns, horizon=10)["es"]
        assert ten > 2.5 * one


class TestConditionalVolSeries:
    def test_too_few_observations_returns_none(self):
        assert garch_conditional_vol_series(_simulate(n=40)) is None

    def test_series_is_aligned_with_the_input_and_annualized(self, returns):
        out = garch_conditional_vol_series(returns)
        assert out.name == "garch_cond_vol_annual"
        assert out.index.equals(returns.index)
        assert (out > 0).all()
        true_vol = np.sqrt(OMEGA / (1 - ALPHA - BETA) * 252)
        assert 0.5 * true_vol < out.median() < 2.0 * true_vol

    def test_leading_nans_are_dropped_from_the_index(self):
        r = _simulate(n=300)
        r.iloc[:10] = np.nan
        out = garch_conditional_vol_series(r)
        assert out.index.equals(r.dropna().index)
