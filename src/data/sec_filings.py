"""SEC EDGAR filing monitor — flags recent material events."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional

import requests

from ..utils.logger import get_logger

log = get_logger(__name__)

EDGAR_HEADERS = {
    "User-Agent": "DayTradingBot research-bot 6052home@gmail.com",
    "Accept": "application/json",
}

# ── Circuit-breaker constants ─────────────────────────────────────────────────
_CB_FAIL_THRESHOLD = 3      # consecutive failures before tripping
_CB_COOLDOWN_SECS  = 300    # seconds to stay offline after tripping (5 min)


@dataclass
class Filing:
    symbol: str
    form_type: str
    title: str
    url: str
    filed_at: datetime

    @property
    def age_hours(self) -> float:
        return (datetime.now(timezone.utc) - self.filed_at).total_seconds() / 3600.0


class SECFilings:
    """Fetch recent EDGAR filings for tickers using their CIK lookup."""

    CIK_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

    def __init__(self, form_types: Iterable[str] = ("8-K", "10-Q", "10-K", "4")):
        self.form_types = set(t.upper() for t in form_types)
        # In-memory CIK cache so we don't re-resolve every cycle.
        self._cik_cache: Dict[str, str] = {}
        # Circuit-breaker state for SEC.gov connectivity (shared by all methods)
        self._cb_fails: int = 0
        self._cb_tripped_at: Optional[datetime] = None

    # ── Circuit-breaker helpers ───────────────────────────────────────────────

    def _cb_ok(self) -> bool:
        """Return True if SEC.gov is healthy enough to try."""
        if self._cb_tripped_at is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._cb_tripped_at).total_seconds()
        if elapsed >= _CB_COOLDOWN_SECS:
            self._cb_fails = 0
            self._cb_tripped_at = None
            log.info("sec_filings | EDGAR back online after cooldown")
            return True
        return False  # still in cooldown — skip silently

    def _cb_success(self) -> None:
        self._cb_fails = 0
        self._cb_tripped_at = None

    def _cb_failure(self, symbol: str, exc: Exception) -> None:
        self._cb_fails += 1
        if self._cb_fails >= _CB_FAIL_THRESHOLD and self._cb_tripped_at is None:
            self._cb_tripped_at = datetime.now(timezone.utc)
            log.warning(
                "sec_filings | EDGAR tripped after %d failures — pausing %ds. "
                "Last error: %s",
                self._cb_fails, _CB_COOLDOWN_SECS, exc,
            )
        elif self._cb_tripped_at is None:
            # First or second failure — still worth logging once
            log.warning("sec_filings | EDGAR fetch failed for %s: %s", symbol, exc)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_recent_filings(self, symbol: str, lookback_hours: int = 48) -> List[Filing]:
        if not self._cb_ok():
            return []  # circuit open — skip silently

        cik = self._resolve_cik(symbol)
        if not cik:
            return []

        params = {
            "action": "getcompany",
            "CIK": cik,
            "type": "",
            "dateb": "",
            "owner": "include",
            "count": 40,
            "output": "atom",
        }
        try:
            import feedparser  # lazy: optional dependency
            r = requests.get(self.CIK_URL, params=params, headers=EDGAR_HEADERS, timeout=10)
            r.raise_for_status()
            feed = feedparser.parse(r.text)
            self._cb_success()
        except Exception as e:
            self._cb_failure(symbol, e)
            return []

        out: List[Filing] = []
        for entry in feed.entries:
            title = entry.get("title", "")
            form_type = self._extract_form_type(title)
            if self.form_types and form_type not in self.form_types:
                continue
            try:
                filed_at = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
            except Exception:
                continue
            if (datetime.now(timezone.utc) - filed_at) > timedelta(hours=lookback_hours):
                continue
            out.append(
                Filing(
                    symbol=symbol,
                    form_type=form_type,
                    title=title,
                    url=entry.get("link", ""),
                    filed_at=filed_at,
                )
            )
        return out

    # ---------- helpers ----------
    def _extract_form_type(self, title: str) -> str:
        # Title format: "8-K - Current report"  /  "10-Q - Quarterly report"
        if " - " in title:
            return title.split(" - ", 1)[0].strip().upper()
        return title.strip().upper()

    def _resolve_cik(self, symbol: str) -> Optional[str]:
        symbol = symbol.upper()
        if symbol in self._cik_cache:
            return self._cik_cache[symbol]  # cached — no network call needed

        if not self._cb_ok():
            return None  # circuit open — skip silently

        try:
            r = requests.get(
                "https://www.sec.gov/files/company_tickers.json",
                headers=EDGAR_HEADERS,
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            for _, row in data.items():
                if row.get("ticker", "").upper() == symbol:
                    cik = str(row["cik_str"]).zfill(10)
                    self._cik_cache[symbol] = cik
                    self._cb_success()
                    return cik
            # Symbol not found in EDGAR — cache a sentinel so we don't retry every cycle
            self._cik_cache[symbol] = ""
            self._cb_success()
        except Exception as e:
            self._cb_failure(symbol, e)
        return None
