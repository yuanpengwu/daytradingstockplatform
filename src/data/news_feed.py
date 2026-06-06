"""Multi-source news headline fetcher."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import requests

from ..utils.logger import get_logger

log = get_logger(__name__)

# ── Circuit-breaker constants ─────────────────────────────────────────────────
_CB_FAIL_THRESHOLD = 3      # consecutive failures before tripping
_CB_COOLDOWN_SECS  = 300    # seconds to stay offline after tripping (5 min)

# ── StockTwits cache TTL ──────────────────────────────────────────────────────
# Free tier: ~200 req/hour unauthenticated. 10 symbols × 60 cycles/hr = 600 req
# without caching → hits rate limit. Cache for 5 min to stay well within limits.
_ST_CACHE_TTL_SECS = 300


@dataclass
class Headline:
    symbol: str
    title: str
    source: str
    url: str
    published_at: datetime
    summary: str = ""

    @property
    def age_minutes(self) -> float:
        return (datetime.now(timezone.utc) - self.published_at).total_seconds() / 60.0


class NewsFeed:
    """Fetches recent news per ticker from configured sources."""

    def __init__(self, sources: Iterable[str] = ("alpaca", "polygon", "stocktwits"), max_age_minutes: int = 120):
        self.sources = [s.lower() for s in sources]
        self.max_age_minutes = max_age_minutes
        # Circuit-breaker state per source: {src: {"fails": int, "tripped_at": datetime|None}}
        self._cb: Dict[str, dict] = {s: {"fails": 0, "tripped_at": None} for s in self.sources}
        # StockTwits per-symbol cache: {symbol: (headlines, fetched_at)}
        self._st_cache: Dict[str, Tuple[List[Headline], datetime]] = {}

    def _cb_ok(self, src: str) -> bool:
        """Return True if the source is healthy enough to try."""
        cb = self._cb.setdefault(src, {"fails": 0, "tripped_at": None})
        tripped = cb["tripped_at"]
        if tripped is None:
            return True
        elapsed = (datetime.now(timezone.utc) - tripped).total_seconds()
        if elapsed >= _CB_COOLDOWN_SECS:
            # Cooldown expired — reset and retry
            cb["fails"] = 0
            cb["tripped_at"] = None
            log.info("news_feed | %s back online after cooldown", src)
            return True
        return False  # still in cooldown; skip silently

    def _cb_success(self, src: str) -> None:
        cb = self._cb.setdefault(src, {"fails": 0, "tripped_at": None})
        cb["fails"] = 0
        cb["tripped_at"] = None

    def _cb_failure(self, src: str, symbol: str, exc: Exception) -> None:
        cb = self._cb.setdefault(src, {"fails": 0, "tripped_at": None})
        cb["fails"] += 1
        if cb["fails"] >= _CB_FAIL_THRESHOLD and cb["tripped_at"] is None:
            cb["tripped_at"] = datetime.now(timezone.utc)
            log.warning(
                "news_feed | %s tripped after %d failures — pausing %ds. Last error: %s",
                src, cb["fails"], _CB_COOLDOWN_SECS, exc,
            )
        elif cb["tripped_at"] is None:
            # First or second failure — still worth logging once
            log.warning("news_feed | News fetch failed (%s/%s): %s", src, symbol, exc)

    def get_headlines(self, symbol: str) -> List[Headline]:
        out: List[Headline] = []
        for src in self.sources:
            if not self._cb_ok(src):
                continue  # circuit open — skip silently
            try:
                if src == "alpaca":
                    result = self._fetch_alpaca(symbol)
                elif src == "polygon":
                    result = self._fetch_polygon(symbol)
                elif src == "stocktwits":
                    result = self._fetch_stocktwits(symbol)
                elif src == "newsapi":
                    result = self._fetch_newsapi(symbol)
                elif src == "finnhub":
                    result = self._fetch_finnhub(symbol)
                elif src == "yfinance":
                    result = self._fetch_yfinance(symbol)
                else:
                    continue
                out.extend(result)
                self._cb_success(src)
            except Exception as e:
                self._cb_failure(src, symbol, e)
        # Filter by age
        cutoff = self.max_age_minutes
        return [h for h in out if h.age_minutes <= cutoff]

    # ---------- providers ----------

    def _fetch_alpaca(self, symbol: str) -> List[Headline]:
        """Alpaca market-data news — real-time, no extra cost (uses existing API key)."""
        key    = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not (key and secret):
            return []
        url = "https://data.alpaca.markets/v1beta1/news"
        params = {
            "symbols": symbol,
            "limit":   50,
            "sort":    "desc",
            "start":   (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        headers = {
            "APCA-API-KEY-ID":     key,
            "APCA-API-SECRET-KEY": secret,
        }
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            log.debug("Alpaca news HTTP %s for %s", r.status_code, symbol)
            return []
        out: List[Headline] = []
        for a in r.json().get("news", []):
            try:
                pub = datetime.fromisoformat(a["created_at"].replace("Z", "+00:00"))
            except Exception:
                continue
            out.append(Headline(
                symbol=symbol,
                title=a.get("headline", "") or "",
                source=a.get("source", "alpaca"),
                url=a.get("url", ""),
                published_at=pub,
                summary=a.get("summary", "") or "",
            ))
        return out

    def _fetch_polygon(self, symbol: str) -> List[Headline]:
        """Polygon reference/news — high-quality, ticker-filtered (uses existing API key)."""
        key = os.getenv("POLYGON_API_KEY")
        if not key:
            return []
        url = "https://api.polygon.io/v2/reference/news"
        params = {
            "ticker":              symbol,
            "published_utc.gte":   (datetime.now(timezone.utc) - timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "order":               "desc",
            "limit":               50,
            "apiKey":              key,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            log.debug("Polygon news HTTP %s for %s", r.status_code, symbol)
            return []
        out: List[Headline] = []
        for a in r.json().get("results", []):
            try:
                pub = datetime.fromisoformat(a["published_utc"].replace("Z", "+00:00"))
            except Exception:
                continue
            out.append(Headline(
                symbol=symbol,
                title=a.get("title", "") or "",
                source=a.get("publisher", {}).get("name", "polygon"),
                url=a.get("article_url", ""),
                published_at=pub,
                summary=a.get("description", "") or "",
            ))
        return out

    # ── StockTwits helpers ────────────────────────────────────────────────────

    @staticmethod
    def _st_symbol(symbol: str) -> str:
        """Map internal symbol to StockTwits format.

        Stocks:  AAPL        → AAPL   (unchanged)
        Crypto:  BTC/USD     → BTC.X
        """
        if "/" in symbol:
            return symbol.split("/")[0] + ".X"
        return symbol

    def _fetch_stocktwits(self, symbol: str) -> List[Headline]:
        """Fetch recent StockTwits messages for *symbol*.

        No API key required for the free tier (200 req/hour unauthenticated).
        Results are cached per-symbol for _ST_CACHE_TTL_SECS (5 min) to stay
        well within the rate limit when the engine polls every 60 seconds.

        Each message tagged Bullish / Bearish is prefixed so VADER reliably
        scores it in the correct direction.  An additional aggregate headline
        summarising the bullish-ratio is appended — this gives the sentiment
        signal a clean directional anchor even when individual messages are
        emoji-heavy or too short for VADER to parse.
        """
        # ── Cache check ───────────────────────────────────────────────────────
        now = datetime.now(timezone.utc)
        cached = self._st_cache.get(symbol)
        if cached:
            headlines, fetched_at = cached
            if (now - fetched_at).total_seconds() < _ST_CACHE_TTL_SECS:
                return headlines   # serve from cache — no network call

        st_sym = self._st_symbol(symbol)
        url    = f"https://api.stocktwits.com/api/2/streams/symbol/{st_sym}.json"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 429:
                log.warning("news_feed | StockTwits rate-limited — cached result reused if available.")
                return cached[0] if cached else []
            if r.status_code != 200:
                log.debug("StockTwits HTTP %s for %s", r.status_code, symbol)
                return []
            data = r.json()
        except Exception as e:
            raise   # let _cb_failure in the caller handle it

        messages = data.get("messages", [])
        if not messages:
            self._st_cache[symbol] = ([], now)
            return []

        out: List[Headline] = []
        bullish_count = 0
        bearish_count = 0

        for m in messages:
            body = (m.get("body") or "").strip()
            if len(body) < 10:
                continue   # skip emoji-only / blank messages

            # Parse timestamp
            try:
                pub = datetime.fromisoformat(
                    m["created_at"].replace("Z", "+00:00")
                )
            except Exception:
                pub = now

            # Explicit Bullish / Bearish label from StockTwits
            label = (
                (m.get("entities") or {})
                .get("sentiment", {})
                .get("basic", "")
                .lower()
            )
            if label == "bullish":
                bullish_count += 1
                # Prefix helps VADER score this message correctly
                title = f"Bullish: {body}"
            elif label == "bearish":
                bearish_count += 1
                title = f"Bearish: {body}"
            else:
                title = body   # unlabelled — let VADER decide

            out.append(Headline(
                symbol=symbol,
                title=title,
                source="stocktwits",
                url=f"https://stocktwits.com/message/{m.get('id', '')}",
                published_at=pub,
            ))

        # ── Aggregate sentiment headline ──────────────────────────────────────
        # Converts the bullish/bearish ratio into a single VADER-friendly
        # sentence that anchors the overall crowd sentiment score.
        total_labeled = bullish_count + bearish_count
        if total_labeled >= 3:
            pct = int(bullish_count / total_labeled * 100)
            if pct >= 65:
                summary = (
                    f"Strong bullish crowd sentiment: {pct}% of {total_labeled} "
                    f"StockTwits messages are positive and optimistic."
                )
            elif pct <= 35:
                summary = (
                    f"Strong bearish crowd sentiment: {100-pct}% of {total_labeled} "
                    f"StockTwits messages are negative and pessimistic."
                )
            else:
                summary = (
                    f"Mixed crowd sentiment: {pct}% bullish vs {100-pct}% bearish "
                    f"across {total_labeled} StockTwits messages."
                )
            out.append(Headline(
                symbol=symbol,
                title=summary,
                source="stocktwits",
                url=f"https://stocktwits.com/symbol/{st_sym}",
                published_at=now,
            ))
            log.debug(
                "StockTwits %s: %d bullish / %d bearish / %d unlabelled",
                symbol, bullish_count, bearish_count, len(messages) - total_labeled,
            )

        self._st_cache[symbol] = (out, now)
        return out

    def _fetch_newsapi(self, symbol: str) -> List[Headline]:
        key = os.getenv("NEWSAPI_KEY")
        if not key:
            return []
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "from": (datetime.utcnow() - timedelta(hours=4)).isoformat(),
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 25,
            "apiKey": key,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        out: List[Headline] = []
        for a in r.json().get("articles", []):
            try:
                pub = datetime.fromisoformat(a["publishedAt"].replace("Z", "+00:00"))
            except Exception:
                continue
            out.append(
                Headline(
                    symbol=symbol,
                    title=a.get("title", "") or "",
                    source=a.get("source", {}).get("name", "newsapi"),
                    url=a.get("url", ""),
                    published_at=pub,
                    summary=a.get("description", "") or "",
                )
            )
        return out

    def _fetch_finnhub(self, symbol: str) -> List[Headline]:
        key = os.getenv("FINNHUB_API_KEY")
        if not key:
            return []
        url = "https://finnhub.io/api/v1/company-news"
        today = datetime.utcnow().date()
        params = {
            "symbol": symbol,
            "from": (today - timedelta(days=1)).isoformat(),
            "to": today.isoformat(),
            "token": key,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        out: List[Headline] = []
        for a in r.json():
            try:
                pub = datetime.fromtimestamp(a.get("datetime", 0), tz=timezone.utc)
            except Exception:
                continue
            out.append(
                Headline(
                    symbol=symbol,
                    title=a.get("headline", "") or "",
                    source=a.get("source", "finnhub"),
                    url=a.get("url", ""),
                    published_at=pub,
                    summary=a.get("summary", "") or "",
                )
            )
        return out

    def _fetch_yfinance(self, symbol: str) -> List[Headline]:
        """Fallback — yfinance also exposes some news (limited)."""
        try:
            import yfinance as yf

            t = yf.Ticker(symbol)
            items = t.news or []
        except Exception:
            return []
        out: List[Headline] = []
        for a in items:
            try:
                pub = datetime.fromtimestamp(a.get("providerPublishTime", 0), tz=timezone.utc)
            except Exception:
                continue
            out.append(
                Headline(
                    symbol=symbol,
                    title=a.get("title", "") or "",
                    source=a.get("publisher", "yfinance"),
                    url=a.get("link", ""),
                    published_at=pub,
                )
            )
        return out
