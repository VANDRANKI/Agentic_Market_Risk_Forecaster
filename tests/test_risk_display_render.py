"""
Render tests for the section functions in app/components/risk_display.py.

Each function is run through Streamlit's AppTest harness with a hand-built
results dict, and the rendered metrics, tables and messages are checked.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest


def _run(function_name, *args):
    def script(function_name, *args):
        from app.components import risk_display

        getattr(risk_display, function_name)(*args)

    at = AppTest.from_function(script, args=(function_name, *args))
    at.run(timeout=30)
    assert not at.exception, f"{function_name} raised: {at.exception}"
    return at


def _markdown(at):
    return "\n".join(m.value for m in at.markdown)


RISK = {
    "hist_var_95": 0.0210, "hist_es_95": 0.0310,
    "hist_var_99": 0.0400, "hist_es_99": 0.0500,
    "param_var_95": 0.0190, "param_es_95": 0.0250,
    "param_var_99": 0.0270, "param_es_99": 0.0320,
    "mc_var_95": 0.0200, "mc_es_95": 0.0300,
    "mc_var_99": 0.0380, "mc_es_99": 0.0480,
    "garch_var_95": None, "garch_es_95": None,
    "garch_var_99": None, "garch_es_99": None,
    "var_10d_95": 0.0664,
    "max_drawdown": -0.1234,
    "regime": "high",
}


class TestSummaryCards:
    def test_95_percent_cards(self):
        text = _markdown(_run("render_summary_cards", RISK, 0.95))
        assert "95% VaR (1-day)" in text
        assert "2.10%" in text
        assert "3.10%" in text
        assert "-12.3%" in text
        assert "HIGH" in text

    def test_99_percent_cards_use_the_99_values(self):
        text = _markdown(_run("render_summary_cards", RISK, 0.99))
        assert "99% VaR (1-day)" in text
        assert "4.00%" in text
        assert "5.00%" in text
        assert "2.10%" not in text

    def test_unknown_regime_uses_the_neutral_colour(self):
        text = _markdown(_run("render_summary_cards", {**RISK, "regime": "unknown"}, 0.95))
        assert "#94a3b8" in text
        assert "UNKNOWN" in text


class TestVarTable:
    def test_rows_for_every_method_and_na_for_failed_garch(self):
        at = _run("render_var_table", RISK)
        df = at.dataframe[0].value
        assert list(df.index) == ["Historical", "Parametric", "Monte Carlo", "GARCH"]
        assert df.loc["Historical", "VaR 95%"] == "2.100%"
        assert df.loc["Monte Carlo", "ES 99%"] == "4.800%"
        assert (df.loc["GARCH"] == "N/A").all()

    def test_zero_var_is_a_number_not_na(self):
        at = _run("render_var_table", {**RISK, "param_var_95": 0.0})
        assert at.dataframe[0].value.loc["Parametric", "VaR 95%"] == "0.000%"

    def test_ten_day_callout(self):
        text = _markdown(_run("render_var_table", RISK))
        assert "6.640%" in text


BACKTEST = {
    "n_exceedances_95": 12,
    "expected_exceedances_95": 12.6,
    "exceedance_rate_95": 0.0476,
    "kupiec_95": {"p_value": 0.8123, "reject_h0": False, "interpretation": "Calibrated."},
    "christoffersen_95": {"p_value": 0.0123, "reject_h0": True, "interpretation": "Clustered."},
    "exceedance_dates_95": ["2024-03-01", "2024-03-04"],
}


class TestBacktestResults:
    def test_counts_rate_and_verdicts(self):
        at = _run("render_backtest_results", BACKTEST)
        metrics = {m.label: m.value for m in at.metric}
        assert metrics == {
            "Actual Exceedances": "12",
            "Expected Exceedances": "12.6",
            "Exceedance Rate": "4.76%",
        }
        text = _markdown(at)
        assert "PASS  (p = 0.812)" in text
        assert "FAIL  (p = 0.012)" in text
        assert "Calibrated." in text and "Clustered." in text

    def test_exceedance_dates_are_listed(self):
        text = _markdown(_run("render_backtest_results", BACKTEST))
        assert "- 2024-03-01" in text and "- 2024-03-04" in text

    def test_insufficient_data_is_reported_not_raised(self):
        empty = {
            "kupiec_95": {"p_value": None, "reject_h0": None},
            "christoffersen_95": {"p_value": None, "reject_h0": None},
        }
        text = _markdown(_run("render_backtest_results", empty))
        assert text.count("insufficient data") == 2


def _portfolio(vol, ret, sharpe, weights):
    return {"volatility": vol, "expected_return": ret, "sharpe_ratio": sharpe, "weights": weights}


class TestPortfolioResults:
    def test_failed_strategies_are_skipped(self):
        results = {
            "current": _portfolio(0.18, 0.09, 0.5, {"A": 0.6, "B": 0.4}),
            "max_sharpe": None,
            "min_vol": _portfolio(0.12, 0.06, 0.4, {"A": 1.0}),
        }
        at = _run("render_portfolio_results", results, {"A": 0.6, "B": 0.4})
        text = _markdown(at)
        assert "**Current Portfolio**" in text
        assert "**Min Volatility**" in text
        assert "Max Sharpe" not in text
        assert "Equal Weight" not in text

    def test_metrics_are_formatted(self):
        results = {"current": _portfolio(0.18, 0.09, 0.5, {"A": 1.0})}
        at = _run("render_portfolio_results", results, {"A": 1.0})
        metrics = {m.label: m.value for m in at.metric}
        assert metrics == {
            "Annual Volatility": "18.0%",
            "Expected Return": "9.0%",
            "Sharpe Ratio": "0.500",
        }

    def test_tiny_weights_are_hidden_and_large_first(self):
        results = {"current": _portfolio(0.18, 0.09, 0.5, {"SMALL": 0.004, "B": 0.3, "A": 0.696})}
        text = _markdown(_run("render_portfolio_results", results, {}))
        assert "SMALL" not in text
        assert text.index(">A<") < text.index(">B<")

    def test_nothing_to_show_warns(self):
        at = _run("render_portfolio_results", {}, {})
        assert [w.value for w in at.warning] == ["Portfolio optimization results unavailable."]


class TestGarchDetails:
    def test_unavailable_model_shows_info(self):
        at = _run("render_garch_details", {"garch_var_95": None})
        assert len(at.info) == 1
        assert "not available" in at.info[0].value
        assert len(at.metric) == 0

    @pytest.mark.parametrize(
        "persistence, kind",
        [(0.97, "warning"), (0.90, "info"), (0.50, "success")],
    )
    def test_persistence_message_level(self, persistence, kind):
        at = _run(
            "render_garch_details",
            {"garch_var_95": 0.02, "garch_persistence": persistence, "garch_long_run_vol": 0.2},
        )
        assert len(getattr(at, kind)) == 1
        values = {m.label: m.value for m in at.metric}
        assert values["Conditional VaR 95%"] == "2.000%"
        assert values["Persistence (alpha+beta)"] == f"{persistence:.4f}"
        assert values["Long-run Annual Vol"] == "20.0%"

    def test_missing_optional_stats_are_left_out(self):
        at = _run("render_garch_details", {"garch_var_95": 0.02})
        assert [m.label for m in at.metric] == ["Conditional VaR 95%"]
