"""
Tests for app/components/agent_panel.py.

st.markdown is patched so the HTML it would send to the browser can be
inspected directly, without needing a running Streamlit app.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.components.agent_panel import render_agent_card, render_all_agent_outputs  # noqa: E402


def _captured_markdown(fn, *args, **kwargs):
    captured = {}

    def fake_markdown(text, **kw):
        captured["text"] = text

    with patch("streamlit.markdown", side_effect=fake_markdown), \
         patch("streamlit.info"):
        fn(*args, **kwargs)
    return captured.get("text", "")


class TestAgentContentIsEscaped:
    """Regression: content is LLM-generated text embedded directly into an
    unsafe_allow_html=True block with no escaping. Anything that steered the
    model into emitting HTML or script tags, for example through prompt
    injection reachable via portfolio inputs, would render as live markup
    rather than the plain text the component's own docstring promises."""

    def test_script_tag_is_escaped_not_live(self):
        text = _captured_markdown(
            render_agent_card, "risk_forecaster", "<script>alert(1)</script>"
        )
        assert "<script>" not in text
        assert "&lt;script&gt;" in text

    def test_event_handler_payload_is_escaped(self):
        text = _captured_markdown(
            render_agent_card, "market_monitor", '<img src=x onerror="alert(1)">'
        )
        # The tag can no longer be parsed as an element and the quotes that
        # would close the attribute early are both neutralized; "onerror="
        # surviving as inert text is fine, only the markup structure matters.
        assert '<img src=x onerror="alert(1)">' not in text
        assert "&lt;img" in text
        assert "&quot;alert(1)&quot;" in text

    def test_ordinary_text_is_unaffected(self):
        text = _captured_markdown(
            render_agent_card, "anomaly_detector", "Volatility rose 12% this week."
        )
        assert "Volatility rose 12% this week." in text

    def test_em_dash_cleanup_still_applies(self):
        text = _captured_markdown(
            render_agent_card, "portfolio_optimizer", "Rebalance — reduce tech exposure"
        )
        assert "—" not in text
        assert "--" in text


class TestUnknownAgentKey:
    def test_unrecognized_key_gets_a_generic_title(self):
        text = _captured_markdown(render_agent_card, "some_new_agent", "output")
        assert "Some New Agent" in text


class TestRenderAllAgentOutputs:
    def test_empty_outputs_shows_info_not_cards(self):
        with patch("streamlit.info") as mock_info, patch("streamlit.markdown") as mock_md:
            render_all_agent_outputs({})
            mock_info.assert_called_once()
            mock_md.assert_not_called()

    def test_all_blank_values_shows_info(self):
        with patch("streamlit.info") as mock_info, patch("streamlit.markdown") as mock_md:
            render_all_agent_outputs({"market_monitor": "", "risk_forecaster": None})
            mock_info.assert_called_once()
            mock_md.assert_not_called()
