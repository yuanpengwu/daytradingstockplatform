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


def is_crypto_symbol(symbol: str) -> bool:
    """Return True for crypto tickers like BTC/USD, ETH/USD, BTCUSD."""
    return "/" in symbol or symbol.upper().endswith("USD") and len(symbol) >= 6


class MarketData:
    """Fetches and caches OHLCV bars per ticker (stocks + crypto)."""

    def __init__(self, provider: str = "alpaca", interval: str = "5m", lookback_days: int = 30,
                 feed: str = "iex"):
        self.provider = provider.lower()
        self.interval = interval
        self.lookback_days = lookback_days
        self.feed = feed.lower()
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_ts: Dict[str, datetime] = {}
        self._alpaca_client = None        # StockHistoricalDataClient (lazily created)
        self._alpaca_crypto_client = None # CryptoHistoricalDataClient (lazily created)

    def get_bars(
        self,
        symbol: str,
        force_refresh: bool = False,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Return a DataFrame with columns [Open, High, Low, Close, Volume].

        *start_dt* / *end_dt* — if provided, fetch exactly that date range
        instead of using ``lookback_days`` from now.  Both must be timezone-aware
        (UTC) or naive UTC datetimes.  Bypasses the cache so every call fetches
        fresh data for that range.
        """
        # Fixed-range fetch — skip cache entirely (each range is unique)
        if start_dt is not None or end_dt is not None:
            if self.provider == "alpaca" and is_crypto_symbol(symbol):
                result = self._fetch_alpaca_crypto(symbol, start_dt=start_dt, end_dt=end_dt)
            elif self.provider == "alpaca":
                result = self._fetch_alpaca(symbol, start_dt=start_dt, end_dt=end_dt)
            elif self.provider == "polygon":
                result = self._fetch_polygon(symbol)
            else:
                result = self._fetch_yfinance(symbol, start_dt=start_dt, end_dt=end_dt)
            return result if result is not None else pd.DataFrame()

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

        if self.provider == "alpaca" and is_crypto_symbol(symbol):
            df = self._fetch_alpaca_crypto(symbol)
        elif self.provider == "alpaca":
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

    def _get_alpaca_crypto_client(self):
        if self._alpaca_crypto_client is not None:
            return self._alpaca_crypto_client
        from alpaca.data.historical.crypto import CryptoHistoricalDataClient
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not (key and secret):
            raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in env.")
        self._alpaca_crypto_client = CryptoHistoricalDataClient(key, secret)
        return self._alpaca_crypto_client

    def _fetch_alpaca_crypto(
        self,
        symbol: str,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV bars for a crypto pair (e.g. BTC/USD) from Alpaca.

        Crypto markets are 24/7 so no feed restriction applies.
        Returns the same [Open, High, Low, Close, Volume] schema as stocks.
        """
        try:
            from alpaca.data.requests import CryptoBarsRequest
        except ImportError:
            log.warning("alpaca-py crypto module not available.")
            return None

        try:
            client = self._get_alpaca_crypto_client()
            if end_dt is not None:
                end = end_dt if end_dt.tzinfo else end_dt.replace(tzinfo=timezone.utc)
            else:
                end = datetime.now(timezone.utc)
            if start_dt is not None:
                start = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
            else:
                start = end - timedelta(days=self.lookback_days)

            req = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self._alpaca_timeframe(),
                start=start,
                end=end,
            )
            resp = client.get_crypto_bars(req)
            df = resp.df
            if df is None or df.empty:
                log.warning("Alpaca crypto returned no bars for %s.", symbol)
                return None
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol] if symbol in df.index.get_level_values(0) else df.droplevel(0)
            df = df.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "volume": "Volume",
            })
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            return df[cols]
        except Exception as e:
            log.warning("Alpaca crypto fetch failed for %s: %s", symbol, e)
            return None

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

    def _fetch_alpaca(
        self,
        symbol: str,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch bars from Alpaca.

        Feed selection:
          iex — free/paper accounts. Cannot serve roughly the last 15 minutes
                of data; most recent bar may lag true market by a few minutes.
                Acceptable for paper trading.
          sip — full consolidated tape (NYSE + NASDAQ + all exchanges).
                Requires Algo Trader Plus subscription ($99/mo). Use this for
                live accounts to get accurate VWAP, prices, and stop levels.

        *start_dt* / *end_dt* override the default lookback-from-now window.
        """
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.enums import DataFeed
        except ImportError:
            log.warning("alpaca-py not installed; falling back to yfinance.")
            return self._fetch_yfinance(symbol)

        feed_enum = DataFeed.SIP if self.feed == "sip" else DataFeed.IEX
        log.debug("Alpaca data feed: %s", feed_enum)

        try:
            client = self._get_alpaca_client()
            if end_dt is not None:
                end = end_dt if end_dt.tzinfo else end_dt.replace(tzinfo=timezone.utc)
            else:
                end = datetime.now(timezone.utc)
            if start_dt is not None:
                start = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
            else:
                start = end - timedelta(days=self.lookback_days)
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=self._alpaca_timeframe(),
                start=start,
                end=end,
                feed=feed_enum,
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
    def _fetch_yfinance(
        self,
        symbol: str,
        retries: int = 3,
        start_dt: Optional[datetime] = None,
        end_dt: Optional[datetime] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch bars from yfinance with a few retries.

        yfinance is rate-limited and intermittently returns empty data even
        for valid tickers, so we retry a couple of times with a short backoff
        before giving up. Failures are handled gracefully by the caller.

        *start_dt* / *end_dt* override the default lookback-from-now window.
        Note: yfinance only provides intraday bars for the last 60 days.
        """
        import time as _time

        try:
            import yfinance as yf
        except ImportError:
            log.warning("yfinance not installed.")
            return None

        last_err = None
        for attempt in range(1, retries + 1):
            try:
                if start_dt is not None or end_dt is not None:
                    _end = (end_dt or datetime.utcnow()).strftime("%Y-%m-%d")
                    _start = (start_dt or (datetime.utcnow() - timedelta(days=self.lookback_days))).strftime("%Y-%m-%d")
                    df = yf.download(
                        symbol,
                        start=_start,
                        end=_end,
                        interval=self.interval,
                        progress=False,
                        auto_adjust=False,
                        prepost=False,
                    )
                else:
                    period_days = min(self.lookback_days, 59 if self.interval.endswith("m") else 365)
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
