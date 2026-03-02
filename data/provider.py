"""
Market data provider with yfinance as primary and Alpha Vantage as fallback.
Caches data locally in Parquet format to avoid repeated API calls and
to stay resilient when upstream APIs throttle or go down.

yfinance 1.x compatibility notes:
  - All downloads (single and multi-ticker) return MultiIndex columns:
    ('Close', 'SPY'), ('High', 'SPY'), etc.
  - We extract the 'Close' level and normalize the DatetimeIndex to date-only.
  - Multi-ticker fetches are done in one batch call for aligned indices.
"""

import os
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf

logger = logging.getLogger(__name__)


class DataProvider:
    """
    Fetches and caches adjusted close prices for a list of tickers.

    Fetch strategy:
      1. Check local Parquet cache (fresh within cache_max_age_hours).
      2. Download all tickers at once via yfinance (single aligned call).
      3. For any ticker still missing, try Alpha Vantage individually.

    Cache lives in cache_dir, one Parquet file per ticker.
    """

    def __init__(
        self,
        cache_dir: str = "data/raw",
        cache_max_age_hours: int = 24,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_hours = cache_max_age_hours
        self.av_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_prices(
        self,
        tickers: list[str],
        start: str,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return a DataFrame of daily adjusted close prices.

        Rows: DatetimeIndex (dates only, no time component).
        Columns: ticker symbols.
        Only rows where ALL tickers have data are returned (inner join).

        Parameters
        ----------
        tickers : list of ticker symbols, e.g. ["SPY", "AAPL"]
        start   : start date string "YYYY-MM-DD"
        end     : end date string (defaults to today)

        Raises
        ------
        ValueError if no data could be fetched for any ticker.
        """
        if end is None:
            end = datetime.today().strftime("%Y-%m-%d")

        tickers = [t.upper().strip() for t in tickers if t.strip()]

        # Step 1: Check which tickers need fetching (not in fresh cache)
        cached_data: dict[str, pd.Series] = {}
        need_fetch: list[str] = []

        for ticker in tickers:
            cached = self._load_cache(ticker, start, end)
            if cached is not None:
                cached_data[ticker] = cached
            else:
                need_fetch.append(ticker)

        # Step 2: Batch-fetch missing tickers via yfinance
        if need_fetch:
            yf_data = self._fetch_yfinance_batch(need_fetch, start, end)
            for ticker, series in yf_data.items():
                self._save_cache(ticker, series)
                cached_data[ticker] = series

        # Step 3: Alpha Vantage fallback for still-missing tickers
        still_missing = [t for t in tickers if t not in cached_data or cached_data[t] is None]
        for ticker in still_missing:
            logger.info("Trying Alpha Vantage fallback for %s", ticker)
            series = self._fetch_alpha_vantage(ticker, start, end)
            if series is not None and not series.empty:
                self._save_cache(ticker, series)
                cached_data[ticker] = series
            else:
                logger.warning("Could not fetch %s from any source. Dropping.", ticker)

        # Filter to only tickers we actually got
        valid: dict[str, pd.Series] = {
            k: v for k, v in cached_data.items()
            if v is not None and not v.empty
        }

        if not valid:
            raise ValueError(
                f"Could not fetch data for any of the requested tickers: {tickers}"
            )

        # Build aligned DataFrame - normalize index before building
        normalized: dict[str, pd.Series] = {}
        for ticker, series in valid.items():
            s = series.copy()
            s.index = pd.to_datetime(s.index).normalize()
            s = s[~s.index.duplicated(keep='first')]
            normalized[ticker] = s

        df = pd.DataFrame(normalized)
        df.index = pd.to_datetime(df.index).normalize()
        df = df.sort_index()
        df = df.dropna()  # keep only rows where all tickers have data

        if df.empty:
            raise ValueError(
                "No overlapping trading days with complete data for all tickers. "
                f"Tickers fetched: {list(valid.keys())}"
            )

        return df

    # ------------------------------------------------------------------
    # yfinance (batch)
    # ------------------------------------------------------------------

    def _fetch_yfinance_batch(
        self, tickers: list[str], start: str, end: str
    ) -> dict[str, pd.Series]:
        """
        Download all tickers in one yfinance call.
        Returns dict of ticker -> adjusted close Series (date-only index).

        yfinance 1.x always returns MultiIndex columns: ('metric', 'ticker').
        """
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                actions=False,
            )
        except Exception as exc:
            logger.warning("yfinance batch download failed: %s", exc)
            return {}

        if raw is None or raw.empty:
            return {}

        result: dict[str, pd.Series] = {}

        # Normalize index to dates
        raw.index = pd.to_datetime(raw.index).normalize()

        if isinstance(raw.columns, pd.MultiIndex):
            # Multi or single ticker with MultiIndex columns: ('Close', 'SPY')
            try:
                close_df = raw["Close"]
            except KeyError:
                logger.warning("yfinance output has no 'Close' level.")
                return {}

            if isinstance(close_df, pd.Series):
                # Single ticker returned as Series
                close_df = close_df.to_frame(name=tickers[0])

            for ticker in tickers:
                if ticker in close_df.columns:
                    series = close_df[ticker].dropna()
                    if not series.empty:
                        series.name = ticker
                        result[ticker] = series
                else:
                    logger.warning("Ticker %s not found in yfinance response.", ticker)
        else:
            # Flat columns (should not happen in yfinance 1.x but handle anyway)
            if "Close" in raw.columns:
                for ticker in tickers:
                    series = raw["Close"].dropna()
                    series.name = ticker
                    if not series.empty:
                        result[ticker] = series

        return result

    # ------------------------------------------------------------------
    # Alpha Vantage fallback
    # ------------------------------------------------------------------

    def _fetch_alpha_vantage(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.Series]:
        if not self.av_key:
            logger.warning("No ALPHA_VANTAGE_API_KEY set. Skipping fallback.")
            return None

        try:
            url = (
                "https://www.alphavantage.co/query"
                f"?function=TIME_SERIES_DAILY_ADJUSTED"
                f"&symbol={ticker}"
                f"&outputsize=full"
                f"&apikey={self.av_key}"
            )
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            payload = resp.json()

            ts = payload.get("Time Series (Daily)", {})
            if not ts:
                msg = (
                    payload.get("Note")
                    or payload.get("Information")
                    or "empty response"
                )
                logger.warning("Alpha Vantage no data for %s: %s", ticker, msg)
                return None

            closes = {k: float(v["5. adjusted close"]) for k, v in ts.items()}
            series = pd.Series(closes, name=ticker)
            series.index = pd.to_datetime(series.index).normalize()
            series = series.sort_index()

            start_dt = pd.to_datetime(start).normalize()
            end_dt = pd.to_datetime(end).normalize()
            series = series.loc[start_dt:end_dt]

            return series if not series.empty else None

        except Exception as exc:
            logger.warning("Alpha Vantage failed for %s: %s", ticker, exc)
            return None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, ticker: str) -> Path:
        safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.cache_dir / f"{safe}.parquet"

    def _cache_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
        return age_hours < self.cache_max_age_hours

    def _load_cache(
        self, ticker: str, start: str, end: str
    ) -> Optional[pd.Series]:
        path = self._cache_path(ticker)
        if not self._cache_is_fresh(path):
            return None

        try:
            df = pd.read_parquet(path)
            series = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
            series.index = pd.to_datetime(series.index).normalize()

            start_dt = pd.to_datetime(start).normalize()
            end_dt = pd.to_datetime(end).normalize()

            # Only use cache if it covers the requested start date
            if series.index.min() > start_dt:
                return None

            sliced = series.loc[start_dt:end_dt]
            return sliced if not sliced.empty else None

        except Exception as exc:
            logger.warning("Cache read failed for %s: %s", ticker, exc)
            return None

    def _save_cache(self, ticker: str, series: pd.Series) -> None:
        path = self._cache_path(ticker)
        try:
            s = series.copy()
            s.index = pd.to_datetime(s.index).normalize()
            s.to_frame(name="close").to_parquet(path)
        except Exception as exc:
            logger.warning("Cache write failed for %s: %s", ticker, exc)
