"""
Tests for app/components/charts.py.

Plotly Figure objects are plain data structures, so their trace arrays can be
inspected directly without a browser or any rendering step.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.charts import (  # noqa: E402
    chart_cumulative_returns,
    chart_portfolio_weights_comparison,
    chart_returns_over_time,
    chart_rolling_volatility,
    chart_var_exceedance,
    chart_var_comparison,
    chart_correlation_heatmap,
    garch_value_to_pct,
)


def _sample(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(4)
    returns = pd.Series(rng.normal(0, 0.01, n), index=idx)
    var_series = pd.Series(0.02, index=idx)
    return returns, var_series


class TestChartVarExceedanceAlignment:
    """Regression: this chart aligned returns and VaR with a bare
    Series.align(join="inner"), independently of the same operation in
    risk/backtest.py. A line trace draws points in index order, not sorted-by
    -x order, so an out-of-order index drew a zigzag instead of a clean time
    series, and a duplicated date (prices are merged from two sources
    upstream) doubled a marker. Now shares align_returns_and_var with
    risk/backtest.py instead of a second, independently-buggy copy."""

    def test_var_line_is_chronologically_sorted_despite_shuffled_input(self):
        returns, var_series = _sample()
        shuffled = returns.sample(frac=1.0, random_state=1)

        fig = chart_var_exceedance(shuffled, var_series)

        var_trace = next(t for t in fig.data if "VaR" in t.name)
        xs = list(var_trace.x)
        assert xs == sorted(xs)

    def test_duplicate_date_does_not_double_a_marker(self):
        returns, var_series = _sample()
        dup = pd.Series(
            list(returns.values) + [returns.values[-1]],
            index=returns.index.tolist() + [returns.index[-1]],
        )

        fig = chart_var_exceedance(dup, var_series)

        scatter = next(t for t in fig.data if t.name == "Daily Return")
        assert len(scatter.x) == len(returns)

    def test_ordinary_input_is_unaffected(self):
        returns, var_series = _sample()
        fig = chart_var_exceedance(returns, var_series)
        scatter = next(t for t in fig.data if t.name == "Daily Return")
        assert len(scatter.x) == len(returns)


class TestGarchValueToPct:
    """Regression: app/main.py built the Method Comparison bar chart with
    (risk.get("garch_var_95") or 0) * 100, the same false-reassurance shape
    fixed earlier today in agents/crew.py. A GARCH fit that returns None
    (risk/garch.py's documented failure signal) became a real zero-height
    bar reading as "GARCH measured no tail risk," rather than the gap a
    missing estimate should leave."""

    def test_none_becomes_nan(self):
        assert np.isnan(garch_value_to_pct(None))

    def test_value_is_scaled_to_percent(self):
        assert garch_value_to_pct(0.0181) == pytest.approx(1.81)

    def test_genuine_zero_stays_zero_not_nan(self):
        assert garch_value_to_pct(0.0) == 0.0


class TestChartVarComparisonOmitsFailedGarch:
    def test_nan_garch_produces_no_bar_value(self):
        var_df = pd.DataFrame(
            {
                "Historical": [1.64, 2.33],
                "Parametric": [1.61, 2.28],
                "Monte Carlo": [1.63, 2.31],
                "GARCH": [garch_value_to_pct(None), garch_value_to_pct(None)],
            },
            index=["95% VaR", "99% VaR"],
        )
        fig = chart_var_comparison(var_df)
        garch_trace = next(t for t in fig.data if t.name == "GARCH")
        assert all(np.isnan(v) for v in garch_trace.y)

    def test_converged_garch_renders_normally(self):
        var_df = pd.DataFrame(
            {
                "Historical": [1.64, 2.33],
                "GARCH": [garch_value_to_pct(0.0181), garch_value_to_pct(0.0256)],
            },
            index=["95% VaR", "99% VaR"],
        )
        fig = chart_var_comparison(var_df)
        garch_trace = next(t for t in fig.data if t.name == "GARCH")
        assert list(garch_trace.y) == pytest.approx([1.81, 2.56])


class TestChartCorrelationHeatmapLayout:
    """Regression: the layout dict set layout["xaxis"]["side"] = "bottom"
    and then unconditionally called layout.pop("xaxis", None) and
    layout.pop("yaxis", None) two lines later, discarding that assignment
    -- and every other axis setting from _base_layout, including the
    dark-theme gridcolor -- before fig.update_layout() ever saw it. The
    resulting figure had no xaxis/yaxis layout at all, falling back to
    Plotly's own heatmap defaults instead of "side": "bottom"."""

    def _sample_returns(self, n=40):
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        rng = np.random.default_rng(7)
        return pd.DataFrame(
            rng.normal(0, 0.01, size=(n, 3)),
            index=idx,
            columns=["AAPL", "MSFT", "SPY"],
        )

    def test_xaxis_side_is_actually_applied(self):
        fig = chart_correlation_heatmap(self._sample_returns())
        assert fig.layout.xaxis.side == "bottom"

    def test_axes_keep_the_shared_dark_theme_styling(self):
        fig = chart_correlation_heatmap(self._sample_returns())
        assert fig.layout.xaxis.gridcolor == "#1e293b"
        assert fig.layout.yaxis.gridcolor == "#1e293b"


class TestChartCumulativeReturns:
    def test_log_returns_add_up_to_the_price_move(self):
        """Regression: the analysis pipeline holds log returns, which were
        compounded as (1 + r) and so understated the cumulative return."""
        idx = pd.date_range("2024-01-01", periods=4, freq="B")
        prices = pd.Series([100.0, 110.0, 99.0, 121.0], index=idx)
        log_r = np.log(prices / prices.shift(1)).dropna()

        fig = chart_cumulative_returns(log_r, log_returns=True)

        y = np.asarray(fig.data[0].y)
        np.testing.assert_allclose(y, (prices.iloc[1:] / prices.iloc[0] - 1).values * 100)

    def test_simple_returns_compound_by_default(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        simple_r = pd.Series([0.10, -0.10, 0.20], index=idx)

        fig = chart_cumulative_returns(simple_r)

        np.testing.assert_allclose(np.asarray(fig.data[0].y), [10.0, -1.0, 18.8])


class TestChartReturnsOverTime:
    def test_bars_are_percent_and_coloured_by_sign(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        r = pd.Series([0.01, -0.02, 0.0], index=idx)
        fig = chart_returns_over_time(r)
        bars = fig.data[0]
        np.testing.assert_allclose(bars.y, [1.0, -2.0, 0.0])
        # A zero return counts as a gain, a negative one as a loss.
        assert bars.marker.color[0] == bars.marker.color[2]
        assert bars.marker.color[1] != bars.marker.color[0]

    def test_nan_returns_are_dropped(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="B")
        fig = chart_returns_over_time(pd.Series([0.01, np.nan, 0.02], index=idx))
        assert len(fig.data[0].x) == 2

    def test_anomaly_markers_sit_on_the_flagged_days(self):
        returns, _ = _sample(20)
        flags = pd.Series(False, index=returns.index)
        flags.iloc[[3, 9]] = True
        fig = chart_returns_over_time(returns, flags)
        assert [t.name for t in fig.data] == ["Daily Return", "Anomaly"]
        assert list(fig.data[1].x) == [returns.index[3], returns.index[9]]
        np.testing.assert_allclose(fig.data[1].y, returns.iloc[[3, 9]].values * 100)

    def test_flags_covering_only_part_of_the_range(self):
        """Days the flag series does not cover count as not anomalous."""
        returns, _ = _sample(20)
        flags = pd.Series(True, index=returns.index[:2])
        fig = chart_returns_over_time(returns, flags)
        assert len(fig.data[1].x) == 2

    def test_no_flagged_days_adds_no_anomaly_trace(self):
        returns, _ = _sample(20)
        flags = pd.Series(False, index=returns.index)
        fig = chart_returns_over_time(returns, flags)
        assert [t.name for t in fig.data] == ["Daily Return"]


class TestChartRollingVolatility:
    def test_realised_vol_trace_and_optional_garch_trace(self):
        returns, _ = _sample(60)
        fig = chart_rolling_volatility(returns, window=10)
        assert len(fig.data) == 1
        assert fig.data[0].name == "Rolling 10d Vol (annualized)"

        garch = pd.Series(0.01, index=returns.index)
        fig = chart_rolling_volatility(returns, garch_vol=garch, window=10)
        assert [t.name for t in fig.data][1] == "GARCH Conditional Vol"
        np.testing.assert_allclose(fig.data[1].y, 1.0)

    def test_empty_garch_series_is_ignored(self):
        returns, _ = _sample(60)
        fig = chart_rolling_volatility(returns, garch_vol=pd.Series(dtype=float), window=10)
        assert len(fig.data) == 1


class TestChartPortfolioWeightsComparison:
    def test_one_trace_per_ticker_with_percent_values(self):
        fig = chart_portfolio_weights_comparison(
            {"Current": {"A": 0.5, "B": 0.5}, "MaxSharpe": {"A": 1.0}}
        )
        traces = {t.name: list(t.y) for t in fig.data}
        assert traces == {"A": [50.0, 100.0], "B": [50.0, 0]}
        assert list(fig.data[0].x) == ["Current", "MaxSharpe"]

    def test_stacked_on_a_zero_to_hundred_axis(self):
        fig = chart_portfolio_weights_comparison({"Current": {"A": 1.0}})
        assert fig.layout.barmode == "stack"
        assert tuple(fig.layout.yaxis.range) == (0, 100)

    def test_more_tickers_than_colours_reuses_the_palette(self):
        weights = {f"T{i}": 1 / 9 for i in range(9)}
        fig = chart_portfolio_weights_comparison({"Current": weights})
        assert len(fig.data) == 9
        assert fig.data[0].marker.color == fig.data[7].marker.color
