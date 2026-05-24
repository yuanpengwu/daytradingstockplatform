"""Persistent closed-trade log with win-rate analytics.

Two files are maintained:

  transactions.json  — every filled order (entries AND exits), append-only.
                       Use this for your own local analysis / spreadsheet import.

  trades.json        — one record per closed round-trip (entry→exit), with P&L
                       and won/lost flag.  Used by the dashboard win-rate panel.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .logger import get_logger

log = get_logger(__name__)

_WINDOWS = {
    "1d":  1,
    "1w":  7,
    "1m":  30,
    "3m":  90,
    "6m":  180,
    "1y":  365,
}


# ---------------------------------------------------------------------------
# Transaction log  (every filled order)
# ---------------------------------------------------------------------------

@dataclass
class Transaction:
    """One filled order — either an entry or an exit."""
    type:        str    # "entry" | "exit"
    symbol:      str
    side:        str    # "buy" | "sell"
    qty:         float
    price:       float  # actual fill price
    score:       float  # aggregated signal score at the time
    confidence:  float
    reason:      str    # "signal" | "stop_loss" | "take_profit" | "trailing_stop" | "eod" | …
    timestamp:   str    # ISO-8601 local time
    # Exit-only fields (None for entries)
    entry_price: Optional[float] = None
    pnl:         Optional[float] = None
    pnl_pct:     Optional[float] = None
    won:         Optional[bool]  = None


# ---------------------------------------------------------------------------
# Closed round-trip record  (one per trade lifecycle, for win-rate stats)
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    symbol:      str
    side:        str        # "buy" | "sell"
    qty:         float
    entry_price: float
    exit_price:  float
    pnl:         float      # realized dollar P&L
    pnl_pct:     float      # as a fraction, e.g. 0.02 = +2%
    closed_at:   str        # ISO-8601
    reason:      str        # stop_loss | take_profit | eod | signal | manual
    won:         bool


# ---------------------------------------------------------------------------
# TradeHistory
# ---------------------------------------------------------------------------

class TradeHistory:
    def __init__(self, path: str = "trades.json"):
        self._path       = Path(path)
        self._tx_path    = self._path.with_name("transactions.json")
        self._trades: List[TradeRecord] = []
        self._load()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def log_entry(
        self,
        symbol:     str,
        side:       str,
        qty:        float,
        price:      float,
        score:      float,
        confidence: float,
    ) -> None:
        """Record a filled entry order to transactions.json."""
        tx = Transaction(
            type="entry",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            score=round(score, 4),
            confidence=round(confidence, 4),
            reason="signal",
            timestamp=datetime.now().isoformat(),
        )
        self._append_transaction(tx)
        log.info(
            "[ENTRY LOGGED] %s %s x%.0f @ %.2f  score=%+.3f  conf=%.2f",
            side.upper(), symbol, qty, price, score, confidence,
        )

    def log_exit(
        self,
        symbol:      str,
        side:        str,
        qty:         float,
        price:       float,
        entry_price: float,
        pnl:         float,
        pnl_pct:     float,
        score:       float,
        confidence:  float,
        reason:      str,
    ) -> None:
        """Record a filled exit order to transactions.json."""
        tx = Transaction(
            type="exit",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            score=round(score, 4),
            confidence=round(confidence, 4),
            reason=reason,
            timestamp=datetime.now().isoformat(),
            entry_price=entry_price,
            pnl=round(pnl, 4),
            pnl_pct=round(pnl_pct, 6),
            won=pnl > 0,
        )
        self._append_transaction(tx)
        log.info(
            "[EXIT  LOGGED] %s %s x%.0f @ %.2f  entry=%.2f  pnl=%+.2f (%+.2f%%)  reason=%s",
            side.upper(), symbol, qty, price, entry_price,
            pnl, pnl_pct * 100, reason,
        )

    def record(
        self,
        symbol:      str,
        side:        str,
        qty:         float,
        entry_price: float,
        exit_price:  float,
        pnl:         float,
        pnl_pct:     float,
        reason:      str,
    ) -> None:
        """Append one closed round-trip to trades.json (used for win-rate stats)."""
        trade = TradeRecord(
            symbol=symbol,
            side=side,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            closed_at=datetime.now().isoformat(),
            reason=reason,
            won=pnl > 0,
        )
        self._trades.append(trade)
        self._save()
        log.info(
            "[TRADE CLOSED] %s %s x%.0f  entry=%.2f exit=%.2f  pnl=%+.2f (%+.2f%%)  won=%s  reason=%s",
            side.upper(), symbol, qty, entry_price, exit_price,
            pnl, pnl_pct * 100, trade.won, reason,
        )

    def win_rate_summary(self) -> dict:
        """Return win-rate stats for every time window."""
        return {label: self._stats_for_days(days) for label, days in _WINDOWS.items()}

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _stats_for_days(self, days: int) -> dict:
        cutoff = datetime.now() - timedelta(days=days)
        trades = [
            t for t in self._trades
            if datetime.fromisoformat(t.closed_at) >= cutoff
        ]
        if not trades:
            return {"trades": 0, "wins": 0, "losses": 0, "win_rate": None, "total_pnl": 0.0}
        wins = sum(1 for t in trades if t.won)
        return {
            "trades":    len(trades),
            "wins":      wins,
            "losses":    len(trades) - wins,
            "win_rate":  round(wins / len(trades), 4),
            "total_pnl": round(sum(t.pnl for t in trades), 2),
        }

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                rows = json.load(f)
            self._trades = [TradeRecord(**r) for r in rows]
            log.info("Loaded %d trade records from %s.", len(self._trades), self._path)
        except Exception as e:
            log.warning("Could not load trade history from %s: %s", self._path, e)

    def _save(self) -> None:
        try:
            rows = [asdict(t) for t in self._trades]
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
        except Exception as e:
            log.warning("Could not save trade history: %s", e)

    def _append_transaction(self, tx: Transaction) -> None:
        """Append a single transaction to transactions.json (never rewrites full file)."""
        try:
            # Append-only: read existing list, add new entry, write back.
            if self._tx_path.exists():
                with open(self._tx_path, "r", encoding="utf-8") as f:
                    rows = json.load(f)
            else:
                rows = []
            rows.append(asdict(tx))
            with open(self._tx_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
        except Exception as e:
            log.warning("Could not append transaction to %s: %s", self._tx_path, e)
