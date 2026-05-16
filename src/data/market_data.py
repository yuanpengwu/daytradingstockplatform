"""OHLCV market-data feed.

Providers:
  - alpaca   : Alpaca historical bars (recommended — uses your ALPACA_* keys,
               reliable, no rate-limiting). Free IEX feed.
  - yfinance : free Yahoo data; frequently rate-limited / returns empty.
  - polygon  : optional, requires POLYGON_API_KEY.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import pandas as pd

from ..utils.logger import get_logger

log = get_logger(__name__)


class MarketData:
    """Fetches and caches OHLCV bars per ticker."""

    def __init__(self, provider: str = "alpaca", interval: str = "5m", lookback_days: int = 30):
        self.provider = provider.lower()
        self.interval = interval
        self.lookback_days = lookback_days
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ts: Dict[str, datetime] = {}
        self._alpaca_client = None  # lazily created

    def get_bars(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        """Return a DataFrame with columns [Open, High, Low, Close, Volume]."""
        now = datetime.utcnow()
        cached_at = self._cache_ts.get(symbol)
        max_age = self._max_cache_age()
        if (
            not force_refresh
            and cached_at
            and (now - cached_at) < max_age
            and symbol in self._cache
        ):
            return self._cache[symbol]

        if self.provider == "alpaca":
            df = self._fetch_alpaca(symbol)
        elif self.provider == "polygon":
            df = self._fetch_polygon(symbol)
        else:
            df = self._fetch_yfinance(symbol)

        if df is not None and not df.empty:
            self._cache[symbol] = df
            self._cache_ts[symbol] = now
        return df if df is not None else pd.DataFrame()

    def get_latest_price(self, symbol: str) -> float:
        df = self.get_bars(symbol)
        if df.empty:
            return 0.0
        return float(df["Close"].iloc[-1])

    # ---------- Alpaca ----------
    def _get_alpaca_client(self):
        if self._alpaca_client is not None:
            return self._alpaca_client
        from alpaca.data.historical import StockHistoricalDataClient

        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not (key and secret):
            raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in env.")
        self._alpaca_client = StockHistoricalDataClient(key, secret)
        return self._alpaca_client

    def _alpaca_timeframe(self):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        unit = self.interval[-1]
        amt = int(self.interval[:-1]) if self.interval[:-1].isdigit() else 1
        if unit == "m":
            return TimeFrame(amt, TimeFrameUnit.Minute)
        if unit == "h":
            return TimeFrame(amt, TimeFrameUnit.Hour)
        if unit == "d":
            return TimeFrame(amt, TimeFrameUnit.Day)
        return TimeFrame(5, TimeFrameUnit.Minute)

    def _fetch_alpaca(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch recent bars from Alpaca (IEX feed — works on free/paper accounts).

        Note: the free IEX feed cannot serve roughly the last 15 minutes of
        data, so the most recent bar may lag the true market by a few minutes.
        That is acceptable for paper trading and far more reliable than the
        yfinance free endpoint.
        """
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.enums import DataFeed
        except ImportError:
            log.warning("alpaca-py not installed; falling back to yfinance.")
            return self._fetch_yfinance(symbol)

        try:
            client = self._get_alpaca_client()
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=self.lookback_days)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self._alpaca_timeframe(),
                start=start,
                end=end,
                feed=DataFeed.IEX,
            )
            resp = client.get_stock_bars(req)
            df = resp.df
            if df is None or df.empty:
                log.warning("Alpaca returned no bars for %s.", symbol)
                return None
            # resp.df is MultiIndex (symbol, timestamp) for a single symbol too.
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol]
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            return df[["Open", "High", "Low", "Close", "Volume"]]
        except Exception as e:
            log.warning("Alpaca data fetch failed for %s: %s", symbol, e)
            return None

    # ---------- yfinance ----------
    def _fetch_yfinance(self, symbol: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch bars from yfinance with a few retries.

        yfinance is rate-limited and intermittently returns empty data even
        for valid tickers, so we retry a couple of times with a short backoff
        before giving up. Failures are handled gracefully by the caller.
        """
        import time as _time

        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed.")
            return None

        period_days = min(self.lookback_days, 59 if self.interval.endswith("m") else 365)
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                df = yf.download(
                    symbol,
                    period=f"{period_days}d",
                    interval=self.interval,
                    progress=False,
                    auto_adjust=False,
                    prepost=False,
                )
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df = df.rename(columns=str.title)
                    return df[["Open", "High", "Low", "Close", "Volume"]]
                last_err = "empty response"
            except Exception as e:
                last_err = str(e)
            if attempt < retries:
                _time.sleep(1.5 * attempt)

        log.warning(
            "yfinance returned no data for %s after %d attempts (%s). "
            "Consider data.provider: alpaca in config.yaml.",
            symbol, retries, last_err,
        )
        return None

    # ---------- Polygon ----------
    def _fetch_polygon(self, symbol: str) -> Optional[pd.DataFrame]:
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            log.warning("POLYGON_API_KEY not set; falling back to yfinance.")
            return self._fetch_yfinance(symbol)
        try:
            from polygon import RESTClient  # type: ignore
        except ImportError:
            log.warning("polygon-api-client not installed; falling back to yfinance.")
            return self._fetch_yfinance(symbol)

        try:
            client = RESTClient(key)
            end = datetime.utcnow()
            start = end - timedelta(days=self.lookback_days)
            mult, span = self._parse_interval()
            aggs = client.get_aggs(symbol, mult, span, start, end, limit=50_000)
            df = pd.DataFrame(
                [
                    {
                        "Open": a.open,
                        "High": a.high,
                        "Low": a.low,
                        "Close": a.close,
                        "Volume": a.volume,
                        "ts": pd.Timestamp(a.timestamp, unit="ms", tz="UTC"),
                    }
                    for a in aggs
                ]
            )
            if df.empty:
                return None
            return df.set_index("ts").sort_index()
        except Exception as e:
            log.warning("Polygon fetch failed for %s: %s", symbol, e)
            return None

    # ---------- helpers ----------
    def _parse_interval(self):
        unit = self.interval[-1]
        amt = int(self.interval[:-1]) if self.interval[:-1].isdigit() else 1
        span = {"m": "minute", "h": "hour", "d": "day"}[unit]
        return amt, span

    def _max_cache_age(self) -> timedelta:
        unit = self.interval[-1]
        amt = int(self.interval[:-1]) if self.interval[:-1].isdigit() else 1
        if unit == "m":
            return timedelta(seconds=max(10, amt * 30))
        if unit == "h":
            return timedelta(minutes=max(5, amt * 30))
        return timedelta(hours=1)
