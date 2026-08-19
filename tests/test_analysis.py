"""
Tests for app/analysis.py.

run_full_analysis reaches the network through DataProvider, so the provider is
replaced with a stub that serves synthetic prices. Everything downstream of the
fetch is the real pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.analysis import run_full_analysis  # noqa: E402


def _prices(tickers, n=400):
    rng = np.random.default_rng(11)
    idx = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n, freq="B")
    return pd.DataFrame(
        {t: (1 + rng.normal(0.0003, 0.011, n)).cumprod() * 100 for t in tickers},
        index=idx,
    )


@pytest.fixture
def stub_provider(monkeypatch):
    """Serve prices for AAPL and MSFT only. FAILED never downloads."""
    import data.provider as provider_mod

    class StubProvider:
        def __init__(self, *a, **k):
            pass

        def fetch_prices(self, tickers, start, end=None):
            served = [t for t in tickers if t in ("AAPL", "MSFT")]
            if not served:
                raise ValueError("no data")
            return _prices(served)

    monkeypatch.setattr(provider_mod, "DataProvider", StubProvider)
    return StubProvider


class TestWeightNormalization:
    def test_weights_sum_to_one_when_a_ticker_is_missing(self, stub_provider):
        """Regression: weights were normalized over the full request and only then
        filtered to the tickers that actually downloaded, so the surviving weights
        summed to less than 1. norm_weights is returned and shown in the UI and
        handed to the LLM context, so the book was reported as partly uninvested."""
        out = run_full_analysis(
            tickers=["AAPL", "MSFT", "FAILED"],
            weights={"AAPL": 0.5, "MSFT": 0.3, "FAILED": 0.2},
            lookback_days=250,
            run_agents=False,
            n_mc_sims=200,
        )
        nw = out["norm_weights"]
        assert set(nw) == {"AAPL", "MSFT"}
        assert sum(nw.values()) == pytest.approx(1.0)
        assert nw["AAPL"] == pytest.approx(0.625)
        assert nw["MSFT"] == pytest.approx(0.375)

    def test_all_weights_kept_when_nothing_is_missing(self, stub_provider):
        out = run_full_analysis(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 1.0, "MSFT": 1.0},
            lookback_days=250,
            run_agents=False,
            n_mc_sims=200,
        )
        nw = out["norm_weights"]
        assert sum(nw.values()) == pytest.approx(1.0)
        assert nw["AAPL"] == pytest.approx(0.5)

    def test_offsetting_weights_raise_instead_of_dividing_by_zero(self, stub_provider):
        """Regression: sum(weights.values()) was used as a divisor unguarded, so a
        long/short book that nets to zero raised ZeroDivisionError from a dict
        comprehension rather than a message naming the problem."""
        with pytest.raises(ValueError, match="sum to zero"):
            run_full_analysis(
                tickers=["AAPL", "MSFT"],
                weights={"AAPL": 1.0, "MSFT": -1.0},
                lookback_days=250,
                run_agents=False,
                n_mc_sims=200,
            )

    def test_no_available_tickers_raises(self, stub_provider):
        with pytest.raises((ValueError, RuntimeError)):
            run_full_analysis(
                tickers=["FAILED"],
                weights={"FAILED": 1.0},
                lookback_days=250,
                run_agents=False,
                n_mc_sims=200,
            )


class TestPipelineOutputs:
    def test_both_confidence_levels_are_always_returned(self, stub_provider):
        """confidence_level selects what the caller displays; the pipeline always
        computes both pairs, so both keys must be present regardless."""
        out = run_full_analysis(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.6, "MSFT": 0.4},
            lookback_days=250,
            confidence_level=0.95,
            run_agents=False,
            n_mc_sims=200,
        )
        rm = out["risk_metrics"]
        for key in ("hist_var_95", "hist_es_95", "hist_var_99", "hist_es_99"):
            assert key in rm, f"missing {key}"
        assert rm["hist_var_99"] >= rm["hist_var_95"]

    def test_confidence_level_does_not_change_the_numbers(self, stub_provider):
        kwargs = dict(
            tickers=["AAPL", "MSFT"],
            weights={"AAPL": 0.6, "MSFT": 0.4},
            lookback_days=250,
            run_agents=False,
            n_mc_sims=200,
        )
        a = run_full_analysis(confidence_level=0.95, **kwargs)["risk_metrics"]
        b = run_full_analysis(confidence_level=0.99, **kwargs)["risk_metrics"]
        assert a["hist_var_95"] == b["hist_var_95"]
        assert a["hist_var_99"] == b["hist_var_99"]
