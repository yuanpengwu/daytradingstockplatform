"""Multi-source news headline fetcher."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional

import requests

from ..utils.logger import get_logger

log = get_logger(__name__)


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

    def __init__(self, sources: Iterable[str] = ("newsapi", "finnhub"), max_age_minutes: int = 120):
        self.sources = [s.lower() for s in sources]
        self.max_age_minutes = max_age_minutes

    def get_headlines(self, symbol: str) -> List[Headline]:
        out: List[Headline] = []
        for src in self.sources:
            try:
                if src == "newsapi":
                    out.extend(self._fetch_newsapi(symbol))
                elif src == "finnhub":
                    out.extend(self._fetch_finnhub(symbol))
                elif src == "yfinance":
                    out.extend(self._fetch_yfinance(symbol))
            except Exception as e:
                log.warning("News fetch failed (%s/%s): %s", src, symbol, e)
        # Filter by age
        cutoff = self.max_age_minutes
        return [h for h in out if h.age_minutes <= cutoff]

    # ---------- providers ----------
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
