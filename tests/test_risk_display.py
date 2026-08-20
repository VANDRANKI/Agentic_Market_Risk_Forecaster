"""
Tests for app/components/risk_display.py's pure formatting helpers.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.risk_display import _backtest_color, _pval_label  # noqa: E402


class TestPvalLabelHandlesInsufficientData:
    """Regression: kupiec_test returns {"p_value": None, "reject_h0": None}
    when there are zero observations to test. The UI read that with
    kupiec.get("p_value", 0), but the key is present with value None, so the
    0 default never applies and f"{None:.3f}" raised TypeError. Reachable
    from the Streamlit sidebar with a short enough lookback window."""

    def test_none_p_value_does_not_raise(self):
        label = _pval_label(None, None)
        assert "insufficient data" in label.lower()

    def test_none_p_value_alone_is_handled(self):
        label = _pval_label(None, False)
        assert "insufficient data" in label.lower()

    def test_none_reject_alone_is_handled(self):
        label = _pval_label(0.03, None)
        assert "insufficient data" in label.lower()

    def test_normal_pass_case(self):
        assert _pval_label(0.42, False) == "PASS  (p = 0.420)"

    def test_normal_fail_case(self):
        assert _pval_label(0.01, True) == "FAIL  (p = 0.010)"

    def test_zero_p_value_is_not_mistaken_for_missing(self):
        """A genuine p-value of 0.0 must render normally, not as N/A."""
        label = _pval_label(0.0, True)
        assert "insufficient data" not in label.lower()
        assert "0.000" in label


class TestBacktestColorHandlesNone:
    def test_none_reject_is_neutral_not_pass_green(self):
        """None is falsy in Python, so a naive `"#ef4444" if reject else
        "#10b981"` would render insufficient-data as if the test had passed."""
        assert _backtest_color(None) == "#94a3b8"

    def test_true_is_fail_red(self):
        assert _backtest_color(True) == "#ef4444"

    def test_false_is_pass_green(self):
        assert _backtest_color(False) == "#10b981"
