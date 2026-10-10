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

@pytest.fixture
def app_test():
    """Run app/main.py's real script through Streamlit's AppTest harness.

    app/main.py forwards Streamlit Cloud secrets into os.environ at import
    time. It must tolerate a missing secrets.toml (the normal local setup,
    where keys come from .env), so no placeholder file is created here.
    """
    at = AppTest.from_file(str(_ROOT / "app" / "main.py"))
    at.run(timeout=60)
    assert not at.exception, f"Initial run raised: {at.exception}"
    return at


class TestStartsWithoutSecretsFile:
    """Regression: app/main.py read `key in st.secrets` unguarded, so with no
    secrets.toml the whole app died with StreamlitSecretNotFoundError before
    rendering anything, even though the README says to configure keys in .env."""

    @pytest.mark.skipif(
        (_ROOT / ".streamlit" / "secrets.toml").exists(),
        reason="a real secrets.toml exists on this machine",
    )
    def test_initial_run_succeeds_with_no_secrets_toml(self, app_test):
        assert not app_test.exception


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


class TestPresetPortfolioLoadsItsOwnWeights:
    """
    Regression: selecting a predefined portfolio from the "Load preset
    portfolio" selectbox silently ignored that preset's own weights.

    render_sidebar() computed `init_weights = preset["weights"]` right next
    to `init_tickers = preset["tickers"]`, but only init_tickers ever made it
    into a widget (the ticker text_area's `value=`). init_weights was dead:
    nothing wired it into st.session_state["sidebar_weights"]. Once that
    session-state key existed from a prior render, switching presets only
    ran the "fill in missing tickers at 1/n" sync loop, so every preset
    displayed equal weights (or, worse, a stale weight left over from a
    different preset's ticker of the same name) instead of the specific
    weights defined in configs/universe.yaml -- e.g. "US Indices" showing
    25/25/25/25 instead of its actual 30/30/20/20.

    Verified by loading "US Indices" (weights [0.30, 0.30, 0.20, 0.20], not
    uniform) via Streamlit's AppTest harness and reading back the real
    number_input widget values, since the bug is specifically about what
    session state a keyed widget displays across a rerun.
    """

    def test_selecting_a_preset_loads_its_configured_weights(self, app_test):
        at = app_test
        preset_sb = at.sidebar.selectbox[0]
        assert "US Indices" in preset_sb.options

        preset_sb.set_value("US Indices").run(timeout=60)
        assert not at.exception

        expected = {"SPY": 0.30, "QQQ": 0.30, "IWM": 0.20, "DIA": 0.20}
        for ticker, weight in expected.items():
            assert _weight_value(at, ticker) == pytest.approx(weight), (
                f"{ticker} shows {_weight_value(at, ticker)} after loading "
                f"the US Indices preset; expected its configured weight {weight}"
            )

    def test_switching_between_two_presets_does_not_leak_stale_weights(self, app_test):
        """AAPL appears in both 'Diversified' (weight 0.20) and 'Magnificent 7'
        (weight 0.15). Switching presets must load the new preset's own
        weight for a shared ticker rather than keeping the previous one."""
        at = app_test
        preset_sb = at.sidebar.selectbox[0]

        preset_sb.set_value("Diversified").run(timeout=60)
        assert not at.exception
        assert _weight_value(at, "AAPL") == pytest.approx(0.20)

        preset_sb = at.sidebar.selectbox[0]
        preset_sb.set_value("Magnificent 7").run(timeout=60)
        assert not at.exception
        assert _weight_value(at, "AAPL") == pytest.approx(0.15)
