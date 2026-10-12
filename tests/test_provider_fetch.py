"""
Offline tests for DataProvider.fetch_prices and its two sources.

yfinance and the Alpha Vantage HTTP call are replaced with fakes, and the
cache goes to a temp directory, so nothing here touches the network.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.provider import DataProvider  # noqa: E402

START = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
END = datetime.today().strftime("%Y-%m-%d")


def _series(ticker, n=10, base=100.0, end=None):
    end = end or pd.Timestamp(datetime.today().date())
    idx = pd.bdate_range(end=end, periods=n)
    return pd.Series(base + np.arange(n, dtype=float), index=idx, name=ticker)


def _window_series(ticker, start, end):
    """Business-day prices from exactly `start` to `end`, like a real download."""
    idx = pd.bdate_range(start, end)
    return pd.Series(100.0 + np.arange(len(idx), dtype=float), index=idx, name=ticker)


@pytest.fixture
def provider(tmp_path, monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    return DataProvider(cache_dir=str(tmp_path))


class TestFetchPrices:
    def test_tickers_are_uppercased_stripped_and_blanks_dropped(self, provider, monkeypatch):
        seen = {}

        def fake_batch(tickers, start, end):
            seen["tickers"] = tickers
            return {t: _series(t) for t in tickers}

        monkeypatch.setattr(provider, "_fetch_yfinance_batch", fake_batch)
        df = provider.fetch_prices([" spy ", "", "aapl", "  "], START, END)
        assert seen["tickers"] == ["SPY", "AAPL"]
        assert list(df.columns) == ["SPY", "AAPL"]

    def test_result_is_the_inner_join_on_dates(self, provider, monkeypatch):
        a = _series("A", n=10)
        b = _series("B", n=6)  # starts later
        monkeypatch.setattr(provider, "_fetch_yfinance_batch", lambda t, s, e: {"A": a, "B": b})
        df = provider.fetch_prices(["A", "B"], START, END)
        assert len(df) == 6
        assert df.index.equals(b.index)
        assert df.notna().all().all()

    def test_index_is_sorted_and_date_only(self, provider, monkeypatch):
        s = _series("A")
        shuffled = s.sample(frac=1, random_state=0)
        shuffled.index = shuffled.index + pd.Timedelta(hours=16)  # intraday timestamps
        monkeypatch.setattr(provider, "_fetch_yfinance_batch", lambda t, s_, e: {"A": shuffled})
        df = provider.fetch_prices(["A"], START, END)
        assert df.index.is_monotonic_increasing
        assert (df.index == df.index.normalize()).all()

    def test_duplicate_dates_keep_the_first_value(self, provider, monkeypatch):
        s = _series("A", n=3)
        dup = pd.concat([s, s.iloc[[-1]] + 1000])
        monkeypatch.setattr(provider, "_fetch_yfinance_batch", lambda t, s_, e: {"A": dup})
        df = provider.fetch_prices(["A"], START, END)
        assert len(df) == 3
        assert df["A"].iloc[-1] == s.iloc[-1]

    def test_fetched_series_are_written_to_the_cache(self, provider, monkeypatch):
        monkeypatch.setattr(provider, "_fetch_yfinance_batch", lambda t, s, e: {"A": _series("A")})
        provider.fetch_prices(["A"], START, END)
        assert provider._cache_path("A").exists()

    def test_second_call_is_served_from_the_cache(self, provider, monkeypatch):
        calls = []

        def fake_batch(tickers, start, end):
            calls.append(list(tickers))
            return {t: _window_series(t, start, end) for t in tickers}

        monkeypatch.setattr(provider, "_fetch_yfinance_batch", fake_batch)
        first = provider.fetch_prices(["A"], START, END)
        second = provider.fetch_prices(["A"], START, END)
        assert calls == [["A"]]
        pd.testing.assert_frame_equal(first, second, check_freq=False)

    def test_only_uncached_tickers_are_downloaded(self, provider, monkeypatch):
        provider._save_cache("A", _window_series("A", START, END))
        calls = []

        def fake_batch(tickers, start, end):
            calls.append(list(tickers))
            return {t: _window_series(t, start, end) for t in tickers}

        monkeypatch.setattr(provider, "_fetch_yfinance_batch", fake_batch)
        df = provider.fetch_prices(["A", "B"], START, END)
        assert calls == [["B"]]
        assert list(df.columns) == ["A", "B"]

    def test_unknown_ticker_is_dropped_when_others_succeed(self, provider, monkeypatch):
        monkeypatch.setattr(provider, "_fetch_yfinance_batch", lambda t, s, e: {"A": _series("A")})
        df = provider.fetch_prices(["A", "ZZZZ"], START, END)
        assert list(df.columns) == ["A"]

    def test_all_sources_failing_raises_value_error(self, provider, monkeypatch):
        monkeypatch.setattr(provider, "_fetch_yfinance_batch", lambda t, s, e: {})
        with pytest.raises(ValueError, match="any of the requested tickers"):
            provider.fetch_prices(["A", "B"], START, END)

    def test_no_overlapping_dates_raises_value_error(self, provider, monkeypatch):
        a = _series("A", n=5, end=pd.Timestamp("2024-03-01"))
        b = _series("B", n=5, end=pd.Timestamp("2024-06-03"))
        monkeypatch.setattr(provider, "_fetch_yfinance_batch", lambda t, s, e: {"A": a, "B": b})
        with pytest.raises(ValueError, match="No overlapping trading days"):
            provider.fetch_prices(["A", "B"], "2024-01-01", "2024-07-01")

    def test_missing_ticker_falls_back_to_alpha_vantage(self, provider, monkeypatch):
        av_calls = []

        def fake_av(ticker, start, end):
            av_calls.append(ticker)
            return _series(ticker, base=50.0)

        monkeypatch.setattr(provider, "_fetch_yfinance_batch", lambda t, s, e: {"A": _series("A")})
        monkeypatch.setattr(provider, "_fetch_alpha_vantage", fake_av)
        df = provider.fetch_prices(["A", "B"], START, END)
        assert av_calls == ["B"]
        assert list(df.columns) == ["A", "B"]
        assert df["B"].iloc[0] == 50.0


class TestYfinanceMultiIndex:
    @staticmethod
    def _raw(tickers, n=5, metrics=("Close", "Volume")):
        idx = pd.date_range("2024-01-01", periods=n) + pd.Timedelta(hours=5)
        cols = pd.MultiIndex.from_product([metrics, tickers])
        data = np.arange(n * len(cols), dtype=float).reshape(n, len(cols))
        return pd.DataFrame(data, index=idx, columns=cols)

    def test_close_level_is_split_per_ticker(self, provider, monkeypatch):
        raw = self._raw(["SPY", "AAPL"])
        monkeypatch.setattr("data.provider.yf.download", lambda *a, **k: raw)
        out = provider._fetch_yfinance_batch(["SPY", "AAPL"], "2024-01-01", "2024-01-06")
        assert set(out) == {"SPY", "AAPL"}
        assert out["SPY"].name == "SPY"
        assert (out["SPY"].index == out["SPY"].index.normalize()).all()
        assert not np.array_equal(out["SPY"].values, out["AAPL"].values)

    def test_ticker_missing_from_the_response_is_skipped(self, provider, monkeypatch):
        raw = self._raw(["SPY"])
        monkeypatch.setattr("data.provider.yf.download", lambda *a, **k: raw)
        out = provider._fetch_yfinance_batch(["SPY", "NOPE"], "2024-01-01", "2024-01-06")
        assert list(out) == ["SPY"]

    def test_all_nan_ticker_is_skipped(self, provider, monkeypatch):
        raw = self._raw(["SPY", "DEAD"])
        raw[("Close", "DEAD")] = np.nan
        monkeypatch.setattr("data.provider.yf.download", lambda *a, **k: raw)
        out = provider._fetch_yfinance_batch(["SPY", "DEAD"], "2024-01-01", "2024-01-06")
        assert list(out) == ["SPY"]

    def test_no_close_level_returns_nothing(self, provider, monkeypatch):
        raw = self._raw(["SPY"], metrics=("Open",))
        monkeypatch.setattr("data.provider.yf.download", lambda *a, **k: raw)
        assert provider._fetch_yfinance_batch(["SPY"], "2024-01-01", "2024-01-06") == {}

    def test_empty_or_failed_download_returns_nothing(self, provider, monkeypatch):
        monkeypatch.setattr("data.provider.yf.download", lambda *a, **k: pd.DataFrame())
        assert provider._fetch_yfinance_batch(["SPY"], "2024-01-01", "2024-01-06") == {}

        def boom(*a, **k):
            raise RuntimeError("rate limited")

        monkeypatch.setattr("data.provider.yf.download", boom)
        assert provider._fetch_yfinance_batch(["SPY"], "2024-01-01", "2024-01-06") == {}


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self._payload


def _av_payload(days):
    return {
        "Time Series (Daily)": {
            d: {"5. adjusted close": str(100.0 + i)} for i, d in enumerate(days)
        }
    }


class TestAlphaVantage:
    def test_without_an_api_key_nothing_is_requested(self, provider, monkeypatch):
        def fail(*a, **k):
            raise AssertionError("network call without an API key")

        monkeypatch.setattr("data.provider.requests.get", fail)
        provider.av_key = None
        assert provider._fetch_alpha_vantage("SPY", "2024-01-01", "2024-02-01") is None

    def test_prices_are_parsed_sorted_and_clipped_to_the_window(self, provider, monkeypatch):
        provider.av_key = "k"
        payload = _av_payload(["2024-03-01", "2024-02-01", "2024-01-15", "2023-12-01"])
        monkeypatch.setattr("data.provider.requests.get", lambda *a, **k: _FakeResponse(payload))
        out = provider._fetch_alpha_vantage("SPY", "2024-01-01", "2024-02-15")
        assert out.index.tolist() == [pd.Timestamp("2024-01-15"), pd.Timestamp("2024-02-01")]
        assert out.name == "SPY"
        assert out.is_monotonic_increasing or out.index.is_monotonic_increasing

    def test_throttle_note_returns_none(self, provider, monkeypatch):
        provider.av_key = "k"
        payload = {"Note": "Thank you for using Alpha Vantage! API call frequency is 25 per day."}
        monkeypatch.setattr("data.provider.requests.get", lambda *a, **k: _FakeResponse(payload))
        assert provider._fetch_alpha_vantage("SPY", "2024-01-01", "2024-02-01") is None

    def test_window_with_no_rows_returns_none(self, provider, monkeypatch):
        provider.av_key = "k"
        payload = _av_payload(["2020-01-02"])
        monkeypatch.setattr("data.provider.requests.get", lambda *a, **k: _FakeResponse(payload))
        assert provider._fetch_alpha_vantage("SPY", "2024-01-01", "2024-02-01") is None

    def test_http_error_returns_none(self, provider, monkeypatch):
        provider.av_key = "k"
        monkeypatch.setattr(
            "data.provider.requests.get", lambda *a, **k: _FakeResponse({}, status=500)
        )
        assert provider._fetch_alpha_vantage("SPY", "2024-01-01", "2024-02-01") is None


class TestCacheStartSlack:
    """The first cached row is the first trading day on or after the requested
    start. A start on a weekend or holiday must still be served from the cache
    (analysis.py derives its start as "today minus N days", which is a weekend
    about two days in seven)."""

    @staticmethod
    def _cache(provider, first_trading_day):
        end = pd.Timestamp(datetime.today().date())
        idx = pd.bdate_range(first_trading_day, end)
        provider._save_cache("SPY", pd.Series(np.arange(len(idx), dtype=float), index=idx))

    def test_saturday_start_hits_the_cache(self, provider):
        self._cache(provider, "2026-03-02")  # a Monday
        assert provider._load_cache("SPY", "2026-02-28", END) is not None  # Saturday

    def test_holiday_start_hits_the_cache(self, provider):
        self._cache(provider, "2026-01-02")  # first trading day after New Year's Day
        assert provider._load_cache("SPY", "2026-01-01", END) is not None

    def test_cache_starting_weeks_late_is_still_rejected(self, provider):
        self._cache(provider, "2026-03-02")
        assert provider._load_cache("SPY", "2026-02-02", END) is None
