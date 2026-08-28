"""
Tests for the agent context builder in agents/crew.py.

The focus is what the prompt says when an optional model fails. GARCH is the
only estimator that can come back empty, and analysis.py signals that with None.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.crew import (  # noqa: E402
    _fmt_pct,
    _fmt_opt_pct,
    _fmt_opt_num,
    _fmt_portfolio,
    _fmt_risk,
    _fmt_backtest_num,
    _fmt_backtest_verdict,
    _MODEL_UNAVAILABLE,
    _OPTIMIZATION_UNAVAILABLE,
    _BACKTEST_UNAVAILABLE,
    compile_analysis_context,
)


def _frame(n=120):
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = pd.DataFrame(
        {t: (1 + rng.normal(0.0004, 0.01, n)).cumprod() * 100 for t in ("AAA", "BBB")},
        index=idx,
    )
    returns = np.log(prices / prices.shift(1)).dropna()
    port = returns.mean(axis=1)
    return prices, returns, port


BASE_METRICS = {
    "hist_var_95": 0.0164, "hist_es_95": 0.0207,
    "param_var_95": 0.0161, "param_es_95": 0.0202,
    "mc_var_95": 0.0163, "mc_es_95": 0.0205,
    "hist_var_99": 0.0233, "hist_es_99": 0.0271,
    "param_var_99": 0.0228, "mc_var_99": 0.0231,
    "var_10d_95": 0.0519, "max_drawdown": -0.12, "regime": "mid",
}


def _context(extra=None, portfolio_results=None):
    prices, returns, port = _frame()
    metrics = dict(BASE_METRICS)
    metrics.update(extra or {})
    return compile_analysis_context(
        tickers=["AAA", "BBB"],
        weights={"AAA": 0.5, "BBB": 0.5},
        prices=prices,
        returns=returns,
        portfolio_returns=port,
        risk_metrics=metrics,
        backtest_results={},
        portfolio_results=portfolio_results or {},
        anomalies={},
    )


_SUCCESSFUL_STRATEGY = {
    "weights": {"AAA": 0.5, "BBB": 0.5},
    "volatility": 0.15,
    "expected_return": 0.08,
    "sharpe_ratio": 0.4,
}


class TestFmtPct:
    def test_none_is_named_not_zeroed(self):
        assert _fmt_pct(None) == _MODEL_UNAVAILABLE
        assert "0.000" not in _fmt_pct(None)

    def test_value_is_formatted(self):
        assert _fmt_pct(1.6437) == "1.644%"
        assert _fmt_pct(1.6437, 2) == "1.64%"

    def test_zero_is_still_rendered_as_zero(self):
        """A genuine zero is a real measurement and must not be relabelled."""
        assert _fmt_pct(0.0) == "0.000%"


class TestFailedGarchIsNotReportedAsZeroRisk:
    def test_context_keeps_none(self):
        """Regression: (value or 0) * 100 turned a GARCH that did not converge
        into 0.000%, so the prompt told the LLM the model measured no tail risk.
        A failure must not render as the most reassuring number available."""
        ctx = _context({"garch_var_95": None, "garch_es_95": None, "garch_var_99": None,
                        "garch_long_run_vol": None, "garch_persistence": None})
        assert ctx["garch_var_95_pct"] is None
        assert ctx["garch_es_95_pct"] is None
        assert ctx["garch_var_99_pct"] is None
        assert ctx["garch_long_run_vol_pct"] is None

    def test_successful_garch_still_scales_to_percent(self):
        ctx = _context({"garch_var_95": 0.0181, "garch_es_95": 0.0229,
                        "garch_var_99": 0.0256, "garch_long_run_vol": 0.1712,
                        "garch_persistence": 0.964})
        assert ctx["garch_var_95_pct"] == pytest.approx(1.81, abs=1e-6)
        assert ctx["garch_es_95_pct"] == pytest.approx(2.29, abs=1e-6)
        assert ctx["garch_long_run_vol_pct"] == pytest.approx(17.12, abs=1e-6)

    def test_failed_garch_is_excluded_from_method_spread(self):
        """The spread across methods measures model disagreement, so a model
        that produced nothing must not count as a 0% estimate."""
        ctx = _context({"garch_var_95": None})
        assert ctx["var_95_method_spread_pct"] < 0.5
        assert ctx["garch_vs_hist_var_95_delta_pct"] is None

    def test_spread_uses_garch_when_it_converged(self):
        ctx = _context({"garch_var_95": 0.0400})
        assert ctx["var_95_method_spread_pct"] > 2.0
        assert ctx["garch_vs_hist_var_95_delta_pct"] is not None


class TestFmtOptHelpers:
    def test_none_is_named_not_zeroed(self):
        assert _fmt_opt_pct(None) == _OPTIMIZATION_UNAVAILABLE
        assert _fmt_opt_num(None) == _OPTIMIZATION_UNAVAILABLE

    def test_value_is_formatted(self):
        assert _fmt_opt_pct(18.0) == "18.0%"
        assert _fmt_opt_num(0.65) == "0.650"

    def test_genuine_zero_is_not_mistaken_for_missing(self):
        assert _fmt_opt_pct(0.0) == "0.0%"
        assert _fmt_opt_num(0.0) == "0.000"


class TestFailedOptimizationIsNotReportedAsZeroReturnPortfolio:
    """Regression: max_sharpe_portfolio / min_volatility_portfolio return None
    when pypfopt finds no feasible solution (e.g. no asset's expected return
    exceeds the risk-free rate, common for a short lookback or a down market).
    compile_analysis_context defaulted every one of nine metrics to 0 via
    (result or {}).get(key, 0), so a failed optimization read to the LLM as a
    portfolio with a genuine 0% volatility and 0% return: risk-free
    arbitrage, not "the optimizer produced nothing." """

    def test_failed_strategy_keeps_none_in_context(self):
        ctx = _context(portfolio_results={
            "current": _SUCCESSFUL_STRATEGY,
            "max_sharpe": None,
            "min_vol": None,
        })
        assert ctx["max_sharpe_vol_pct"] is None
        assert ctx["max_sharpe_return_pct"] is None
        assert ctx["max_sharpe_sharpe"] is None
        assert ctx["min_vol_vol_pct"] is None
        assert ctx["min_vol_sharpe"] is None

    def test_successful_strategy_still_scales_correctly(self):
        ctx = _context(portfolio_results={"current": _SUCCESSFUL_STRATEGY})
        assert ctx["current_vol_pct"] == pytest.approx(15.0)
        assert ctx["current_return_pct"] == pytest.approx(8.0)
        assert ctx["current_sharpe"] == pytest.approx(0.4)

    def test_prompt_names_the_failure_not_a_fake_zero(self):
        ctx = _context(portfolio_results={
            "current": _SUCCESSFUL_STRATEGY,
            "max_sharpe": None,
            "min_vol": None,
        })
        out = _fmt_portfolio(ctx)
        assert _OPTIMIZATION_UNAVAILABLE in out
        assert "Max Sharpe vol: 0.0%" not in out
        assert "0.000" not in out.split("Max Sharpe Portfolio")[1].split("===")[0]

    def test_all_strategies_succeeding_renders_real_numbers(self):
        ctx = _context(portfolio_results={
            "current": _SUCCESSFUL_STRATEGY,
            "max_sharpe": {"weights": {"AAA": 0.7, "BBB": 0.3}, "volatility": 0.18,
                          "expected_return": 0.12, "sharpe_ratio": 0.65},
            "min_vol": {"weights": {"AAA": 0.3, "BBB": 0.7}, "volatility": 0.10,
                       "expected_return": 0.05, "sharpe_ratio": 0.3},
        })
        out = _fmt_portfolio(ctx)
        assert _OPTIMIZATION_UNAVAILABLE not in out
        assert "18.0%" in out
        assert "0.650" in out


class TestFmtBacktestHelpers:
    def test_none_is_named_not_zeroed(self):
        assert _fmt_backtest_num(None) == _BACKTEST_UNAVAILABLE

    def test_value_is_formatted(self):
        assert _fmt_backtest_num(0.812, 4) == "0.8120"
        assert _fmt_backtest_num(1.25, suffix="%") == "1.250%"

    def test_genuine_zero_is_not_mistaken_for_missing(self):
        assert _fmt_backtest_num(0.0, 4) == "0.0000"

    def test_verdict_none_is_named_not_a_pass(self):
        """reject_h0=None means the test had zero observations to evaluate
        (see kupiec_test/christoffersen_test in risk/backtest.py), not that
        it ran and cleared the bar."""
        verdict = _fmt_backtest_verdict(None, "REJECT H0", "PASS")
        assert verdict != "PASS"
        assert _BACKTEST_UNAVAILABLE in verdict

    def test_verdict_true_and_false_still_map_correctly(self):
        assert _fmt_backtest_verdict(True, "REJECT H0", "PASS") == "REJECT H0"
        assert _fmt_backtest_verdict(False, "REJECT H0", "PASS") == "PASS"


class TestBacktestWithZeroObservationsIsNotReportedAsAPass:
    """Regression: kupiec_test and christoffersen_test return p_value=None,
    reject_h0=None when there were zero observations to evaluate (e.g. a
    short lookback leaves the rolling VaR backtest window empty). _fmt_risk
    used to read that with ctx.get(key, 'N/A'), which never applies because
    the key IS present with value None, printing the literal word "None".
    Worse, `'REJECT...' if ctx.get('kupiec_95_reject') else 'PASS...'`
    treated the None reject flag as falsy and told the LLM the model's
    calibration was a clean PASS, when the test never ran at all."""

    _INSUFFICIENT = {"p_value": None, "reject_h0": None,
                      "interpretation": "Insufficient data."}

    def test_prompt_does_not_leak_the_word_none(self):
        ctx = _context()
        ctx["kupiec_95_pval"] = None
        ctx["kupiec_95_reject"] = None
        ctx["christoffersen_95_pval"] = None
        ctx["christoffersen_95_reject"] = None
        ctx["christoffersen_95_pi01"] = None
        ctx["christoffersen_95_pi11"] = None
        out = _fmt_risk(ctx)
        assert "None" not in out

    def test_prompt_does_not_falsely_claim_a_pass(self):
        ctx = _context()
        ctx["kupiec_95_pval"] = None
        ctx["kupiec_95_reject"] = None
        ctx["christoffersen_95_pval"] = None
        ctx["christoffersen_95_reject"] = None
        out = _fmt_risk(ctx)
        assert "PASS (calibration acceptable)" not in out
        assert "PASS (breaches are independent)" not in out
        assert _BACKTEST_UNAVAILABLE in out

    def test_compile_analysis_context_propagates_none_from_backtest_results(self):
        """End-to-end: a kupiec_95/christoffersen_95 dict with zero
        observations (as risk/backtest.py returns) must reach _fmt_risk
        with None fields, and _fmt_risk must not render that as a pass."""
        ctx = _context()
        prices, returns, port = _frame()
        ctx = compile_analysis_context(
            tickers=["AAA", "BBB"],
            weights={"AAA": 0.5, "BBB": 0.5},
            prices=prices,
            returns=returns,
            portfolio_returns=port,
            risk_metrics=dict(BASE_METRICS),
            backtest_results={
                "kupiec_95": dict(self._INSUFFICIENT),
                "christoffersen_95": dict(self._INSUFFICIENT),
                "n_exceedances_95": 0,
                "expected_exceedances_95": 0,
            },
            portfolio_results={},
            anomalies={},
        )
        assert ctx["kupiec_95_reject"] is None
        assert ctx["christoffersen_95_reject"] is None
        out = _fmt_risk(ctx)
        assert "None" not in out
        assert "PASS (calibration acceptable)" not in out
        assert "PASS (breaches are independent)" not in out
