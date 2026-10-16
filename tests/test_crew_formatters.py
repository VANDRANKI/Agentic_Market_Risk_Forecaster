"""
Tests for the market and anomaly prompt formatters and the Groq call helper
in agents/crew.py. These are pure string builders and a thin API wrapper, so
they run offline with a fake client.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.crew import _call_groq_agent, _fmt_anomaly, _fmt_market  # noqa: E402


def _market_ctx(**overrides):
    ctx = {
        "tickers": ["AAA", "BBB"],
        "period_start": "2024-01-02",
        "period_end": "2024-12-31",
        "n_trading_days": 252,
        "portfolio_mean_daily": 0.0004,
        "portfolio_mean_annual_pct": 10.1,
        "portfolio_std_daily": 0.01,
        "portfolio_std_annual_pct": 15.9,
        "portfolio_skewness": -0.321,
        "portfolio_excess_kurtosis": 1.5,
        "portfolio_min_return_pct": -4.25,
        "portfolio_max_return_pct": 3.1,
        "sharpe_ratio_annual": 0.64,
        "recent_vol_20d_annual_pct": 18.2,
        "current_regime": "normal",
        "weights": {"AAA": 0.6, "BBB": 0.4},
        "asset_stats": {
            "AAA": {"mean_annual_pct": 12.0, "std_annual_pct": 20.0, "skewness": -0.1},
            "BBB": {"mean_annual_pct": 5.0, "std_annual_pct": 10.0, "skewness": 0.2},
        },
        "correlation_pairs": {"AAA/BBB": 0.4567},
        "worst_5_days": ["2024-03-01 (-4.25%)"],
        "best_3_days": ["2024-05-02 (3.10%)"],
    }
    ctx.update(overrides)
    return ctx


class TestFmtMarket:
    def test_headline_numbers(self):
        out = _fmt_market(_market_ctx())
        assert "Portfolio: AAA, BBB" in out
        assert "Period: 2024-01-02 to 2024-12-31 (252 trading days)" in out
        assert "Mean return: 0.0004 daily / 10.1% annualized" in out
        assert "Volatility: 0.0100 daily / 15.9% annualized" in out
        assert "Skewness: -0.321" in out
        assert "Min single-day return: -4.25%" in out
        assert "Annual Sharpe ratio: 0.640" in out
        assert "Current volatility regime: NORMAL" in out

    def test_asset_lines_show_weight_as_percent(self):
        out = _fmt_market(_market_ctx())
        assert "AAA (weight 60.0%): return 12.0%, vol 20.0%, skew -0.10" in out
        assert "BBB (weight 40.0%): return 5.0%, vol 10.0%, skew 0.20" in out

    def test_asset_missing_from_weights_is_shown_with_zero_weight(self):
        out = _fmt_market(_market_ctx(weights={"AAA": 1.0}))
        assert "BBB (weight 0.0%)" in out

    def test_correlations_and_best_worst_days(self):
        out = _fmt_market(_market_ctx())
        assert "AAA/BBB: 0.457" in out
        worst = out.index("Worst 5 single-day losses:")
        best = out.index("Best 3 single-day gains:")
        assert worst < out.index("2024-03-01 (-4.25%)") < best < out.index("2024-05-02 (3.10%)")

    def test_empty_optional_sections_are_left_out(self):
        out = _fmt_market(_market_ctx(asset_stats={}, correlation_pairs={}))
        assert "Individual Asset Statistics" not in out
        assert "Pairwise Correlations" not in out

    def test_missing_day_lists_do_not_raise(self):
        ctx = _market_ctx()
        del ctx["worst_5_days"]
        del ctx["best_3_days"]
        out = _fmt_market(ctx)
        assert out.endswith("Best 3 single-day gains:")


def _anomaly_ctx(**overrides):
    ctx = {
        "current_regime": "high",
        "recent_vol_20d_annual_pct": 31.5,
        "n_anomalies_zscore": 2,
        "anomaly_dates_zscore_enriched": ["2024-03-01 (-4.25%)", "2024-03-04 (3.10%)"],
        "n_anomalies_isolation_forest": 1,
        "anomaly_dates_if_enriched": ["2024-03-01 (-4.25%)"],
        "n_agreed_anomalies": 1,
        "anomaly_dates_both_agreed": ["2024-03-01"],
        "worst_5_days": ["2024-03-01 (-4.25%)"],
    }
    ctx.update(overrides)
    return ctx


class TestFmtAnomaly:
    def test_counts_and_dates(self):
        out = _fmt_anomaly(_anomaly_ctx())
        assert "Current volatility regime: HIGH" in out
        assert "Recent 20d annualized vol: 31.5%" in out
        assert "Z-score anomalies (threshold: 2.5 rolling std): 2 days" in out
        assert "    2024-03-04 (3.10%)" in out
        assert "Isolation Forest anomalies (contamination 5%): 1 days" in out

    def test_agreed_dates_are_joined(self):
        out = _fmt_anomaly(_anomaly_ctx(
            n_agreed_anomalies=2, anomaly_dates_both_agreed=["2024-03-01", "2024-03-04"]
        ))
        assert "Days flagged by BOTH methods (2 total): 2024-03-01, 2024-03-04" in out

    def test_no_agreed_dates_reads_none(self):
        out = _fmt_anomaly(_anomaly_ctx(n_agreed_anomalies=0, anomaly_dates_both_agreed=[]))
        assert "Days flagged by BOTH methods (0 total): none" in out

    def test_missing_optional_lists_do_not_raise(self):
        ctx = _anomaly_ctx()
        for key in ("anomaly_dates_zscore_enriched", "anomaly_dates_if_enriched",
                    "anomaly_dates_both_agreed", "worst_5_days"):
            del ctx[key]
        out = _fmt_anomaly(ctx)
        assert "BOTH methods (1 total): none" in out


class _FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


class TestCallGroqAgent:
    def test_returns_stripped_text_and_sends_system_and_task(self):
        comp = _FakeCompletions("  The market looks calm.  \n")
        out = _call_groq_agent(_client(comp), "m1", "You are an analyst.", "Summarise.", [])
        assert out == "The market looks calm."
        assert comp.kwargs["model"] == "m1"
        assert comp.kwargs["messages"] == [
            {"role": "system", "content": "You are an analyst."},
            {"role": "user", "content": "Summarise."},
        ]

    def test_prior_outputs_are_prepended_and_empty_ones_skipped(self):
        comp = _FakeCompletions("ok")
        _call_groq_agent(_client(comp), "m1", "sys", "Task", ["first", "", "second"])
        user = comp.kwargs["messages"][1]["content"]
        assert user == "Previous analysis:\nfirst\n\nPrevious analysis:\nsecond\n\nTask"

    def test_typographic_dashes_are_replaced(self):
        comp = _FakeCompletions("a—b and 1–2")
        assert _call_groq_agent(_client(comp), "m1", "s", "t", []) == "a -- b and 1 - 2"

    def test_none_content_becomes_empty_string(self):
        comp = _FakeCompletions(None)
        assert _call_groq_agent(_client(comp), "m1", "s", "t", []) == ""

    def test_api_error_returns_empty_string(self):
        comp = _FakeCompletions(error=RuntimeError("rate limited"))
        assert _call_groq_agent(_client(comp), "m1", "s", "t", []) == ""
