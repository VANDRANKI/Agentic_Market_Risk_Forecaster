"""
Unit tests for data/provider.py.

These tests use only the local cache logic and do not make network calls.
For CI/CD: tests that would hit real APIs are skipped unless an env var is set.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.provider import DataProvider


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_fake_series(ticker: str, n: int = 252) -> pd.Series:
    """Return a fake price series for testing cache logic."""
    rng = np.random.default_rng(42)
    prices = (1 + rng.normal(0.0003, 0.01, n)).cumprod() * 100
    end = datetime.today()
    dates = pd.date_range(end=end, periods=n, freq="B")
    return pd.Series(prices, index=dates, name=ticker)


# ---------------------------------------------------------------------------
# Cache logic tests
# ---------------------------------------------------------------------------


class TestCacheLogic:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.provider = DataProvider(cache_dir=self.tmpdir, cache_max_age_hours=24)

    def test_save_and_load_cache(self):
        ticker = "TEST"
        series = make_fake_series(ticker)
        self.provider._save_cache(ticker, series)

        start = str(series.index[0].date())
        end = str(series.index[-1].date())
        loaded = self.provider._load_cache(ticker, start, end)

        assert loaded is not None
        assert len(loaded) > 0

    def test_cache_path_format(self):
        path = self.provider._cache_path("SPY")
        assert str(path).endswith("SPY.parquet")

    def test_cache_path_special_chars(self):
        # Tickers with "/" should not break path
        path = self.provider._cache_path("BRK/B")
        assert "BRK_B.parquet" in str(path)

    def test_fresh_cache_returns_true(self):
        ticker = "FRESH"
        series = make_fake_series(ticker)
        self.provider._save_cache(ticker, series)
        path = self.provider._cache_path(ticker)
        assert self.provider._cache_is_fresh(path) is True

    def test_missing_cache_not_fresh(self):
        path = self.provider._cache_path("NONEXISTENT_TICKER_XYZ")
        assert self.provider._cache_is_fresh(path) is False

    def test_old_cache_not_fresh(self):
        import os
        import time
        ticker = "OLD"
        series = make_fake_series(ticker)
        self.provider._save_cache(ticker, series)
        path = self.provider._cache_path(ticker)

        # Artificially age the file by 25 hours
        old_time = time.time() - 25 * 3600
        os.utime(str(path), (old_time, old_time))

        assert self.provider._cache_is_fresh(path) is False

    def test_load_cache_date_slice(self):
        ticker = "SLICE"
        series = make_fake_series(ticker, n=252)
        self.provider._save_cache(ticker, series)

        # Request a subset
        mid_start = str(series.index[50].date())
        mid_end = str(series.index[100].date())
        loaded = self.provider._load_cache(ticker, mid_start, mid_end)

        assert loaded is not None
        assert len(loaded) <= 51  # max 51 days in range

    def test_load_cache_start_before_data_returns_none(self):
        ticker = "BEFORE"
        series = make_fake_series(ticker, n=100)
        self.provider._save_cache(ticker, series)

        # Request data 10 years before cache start
        very_old_start = "2000-01-01"
        end = str(series.index[-1].date())
        loaded = self.provider._load_cache(ticker, very_old_start, end)

        # Should return None because cache doesn't cover the start date
        assert loaded is None


# ---------------------------------------------------------------------------
# DataProvider init
# ---------------------------------------------------------------------------


class TestDataProviderInit:
    def test_creates_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "new_cache"
            assert not new_dir.exists()
            provider = DataProvider(cache_dir=str(new_dir))
            assert new_dir.exists()

    def test_reads_av_key_from_env(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_val = os.environ.get("ALPHA_VANTAGE_API_KEY")
            os.environ["ALPHA_VANTAGE_API_KEY"] = "test_key_123"
            provider = DataProvider(cache_dir=tmpdir)
            assert provider.av_key == "test_key_123"
            if old_val is None:
                del os.environ["ALPHA_VANTAGE_API_KEY"]
            else:
                os.environ["ALPHA_VANTAGE_API_KEY"] = old_val

    def test_default_cache_age(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = DataProvider(cache_dir=tmpdir)
            assert provider.cache_max_age_hours == 24


# ---------------------------------------------------------------------------
# Live API tests (skipped unless ENABLE_LIVE_TESTS=1)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("ENABLE_LIVE_TESTS") != "1",
    reason="Live API tests disabled. Set ENABLE_LIVE_TESTS=1 to run.",
)
class TestLiveDataFetch:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.provider = DataProvider(cache_dir=self.tmpdir)

    def test_fetch_spy_prices(self):
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        prices = self.provider.fetch_prices(["SPY"], start, end)

        assert isinstance(prices, pd.DataFrame)
        assert "SPY" in prices.columns
        assert len(prices) > 200
        assert prices.isna().sum().sum() == 0

    def test_fetch_multiple_tickers(self):
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=180)).strftime("%Y-%m-%d")
        prices = self.provider.fetch_prices(["SPY", "QQQ"], start, end)

        assert "SPY" in prices.columns
        assert "QQQ" in prices.columns
        assert not prices.empty

    def test_invalid_ticker_drops_gracefully(self):
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
        # Only one real ticker; NOTAREALTICKER should be dropped
        prices = self.provider.fetch_prices(["SPY", "NOTAREALTICKER"], start, end)
        assert "SPY" in prices.columns
        assert "NOTAREALTICKER" not in prices.columns

    def test_all_invalid_tickers_raises(self):
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
        with pytest.raises(ValueError):
            self.provider.fetch_prices(["FAKETICKERXYZ999"], start, end)
