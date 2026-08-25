"""
Tests for app/main.py's render_sidebar(), specifically the
"Set Equal Weights" button.

Regression: the per-ticker weight inputs are keyed Streamlit widgets
(key=f"w_{ticker}"). Once a keyed widget has been instantiated once,
Streamlit binds its displayed value to st.session_state[key] and ignores
the `value=` argument passed to the widget on every later rerun -- the
only supported way to change what a keyed widget shows is to write
st.session_state[key] directly before that widget is instantiated again.

The "Set Equal Weights" button only wrote the new weights into
st.session_state["sidebar_weights"][ticker], a different key than the
widget's own "w_{ticker}" key, and then passed
value=st.session_state["sidebar_weights"][ticker] to number_input. Because
the widget key already existed after the first render, that `value=`
argument was silently ignored, so a manually-typed weight (e.g. 0.90)
stayed on screen forever, while the "Total weight" caption (computed from
the number_input's returned value, not from "sidebar_weights") also kept
showing the old value. Clicking "Set Equal Weights" appeared to do nothing.

Uses Streamlit's official AppTest harness (streamlit.testing.v1) to drive
app/main.py's real widget tree instead of unit-testing helper functions in
isolation, since the bug is specifically about Streamlit's key-vs-value
precedence rules for widgets across reruns.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

_SECRETS_PATH = _ROOT / ".streamlit" / "secrets.toml"


@pytest.fixture
def app_test():
    """Run app/main.py's real script through Streamlit's AppTest harness.

    app/main.py reads st.secrets at import time (to forward Streamlit Cloud
    secrets into os.environ); with no secrets.toml present, `key in
    st.secrets` raises StreamlitSecretNotFoundError before the sidebar is
    ever rendered. A dev machine may or may not already have a real
    secrets.toml (which can hold live API keys and must never be touched or
    committed), so this fixture only creates an empty placeholder when one
    is not already present, and only removes what it created.
    """
    created = False
    if not _SECRETS_PATH.exists():
        _SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SECRETS_PATH.write_text("", encoding="utf-8")
        created = True

    try:
        at = AppTest.from_file(str(_ROOT / "app" / "main.py"))
        at.run(timeout=60)
        assert not at.exception, f"Initial run raised: {at.exception}"
        yield at
    finally:
        if created:
            try:
                _SECRETS_PATH.unlink(missing_ok=True)
            except PermissionError:
                # On Windows, Streamlit's file-watcher can still hold the
                # secrets.toml handle open for a moment after the script run
                # finishes. The placeholder is empty and gitignored, so
                # leaving it behind is harmless; it will be cleaned up next
                # run once the watcher releases the handle.
                pass


def _weight_value(at, ticker: str) -> float:
    return at.sidebar.number_input(key=f"w_{ticker}").value


class TestSetEqualWeightsButton:
    def test_manual_edit_persists_across_an_unrelated_rerun(self, app_test):
        """Sanity check that the widget key really does bind across reruns
        (establishes the precondition the bug and its fix both depend on)."""
        at = app_test
        at.sidebar.number_input(key="w_AAPL").set_value(0.90).run(timeout=60)
        assert not at.exception
        assert _weight_value(at, "AAPL") == pytest.approx(0.90)

    def test_set_equal_weights_resets_a_manually_edited_input(self, app_test):
        at = app_test
        tickers = ["SPY", "QQQ", "IWM", "AAPL", "MSFT"]
        equal = round(1.0 / len(tickers), 4)

        # Manually push AAPL away from the equal weight.
        at.sidebar.number_input(key="w_AAPL").set_value(0.90).run(timeout=60)
        assert not at.exception
        assert _weight_value(at, "AAPL") == pytest.approx(0.90)

        # Click "Set Equal Weights".
        btn = [b for b in at.sidebar.button if b.label == "Set Equal Weights"][0]
        btn.click().run(timeout=60)
        assert not at.exception

        for ticker in tickers:
            assert _weight_value(at, ticker) == pytest.approx(equal), (
                f"{ticker} widget still shows {_weight_value(at, ticker)} "
                f"after 'Set Equal Weights' was clicked; expected {equal}"
            )
