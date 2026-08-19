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

from agents.crew import _fmt_pct, _MODEL_UNAVAILABLE, compile_analysis_context  # noqa: E402


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


def _context(extra):
    prices, returns, port = _frame()
    metrics = dict(BASE_METRICS)
    metrics.update(extra)
    return compile_analysis_context(
        tickers=["AAA", "BBB"],
        weights={"AAA": 0.5, "BBB": 0.5},
        prices=prices,
        returns=returns,
        portfolio_returns=port,
        risk_metrics=metrics,
        backtest_results={},
        portfolio_results={},
        anomalies={},
    )


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
