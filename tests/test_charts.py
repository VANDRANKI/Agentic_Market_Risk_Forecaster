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

from app.components.charts import chart_var_exceedance  # noqa: E402


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
