"""Crypto position lifecycle management.

Handles entry placement, stop-loss / take-profit / trailing-stop checks,
and position exit for crypto pairs.

Crypto-specific rules enforced here (not shared with the stock Trader):
  • GTC time-in-force  — DAY orders are invalid on 24/7 crypto markets
  • Notional sizing    — orders sized in $ not shares
  • Fractional qty     — rounded to 8 dp (Alpaca's max crypto precision)
  • No PDT rules       — crypto has no pattern-day-trader restriction
  • Wider stops/TP     — 5% SL, 10% TP, 4% trailing (vs 2.5%/6%/3% for stocks)

The stock Trader is never imported here and vice versa.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..brokers.base import BrokerBase, Order, OrderSide, OrderType, Position  # Order kept for place_entry
from ..signals.aggregator import AggregatedDecision
from ..utils.logger import get_logger
from ..utils.notifications import notify_order_entry, notify_order_exit

log = get_logger(__name__)


class CryptoTrader:
    """Owns all per-position state and order submission for crypto pairs.

    CryptoEngine calls this for every position lifecycle event — entries,
    stops, TPs, and reconciliation after restarts.  CryptoEngine itself
    only handles signal evaluation and the main 24/7 loop.
    """

    def __init__(self, broker: BrokerBase, config: dict):
        self.broker = broker
        ccfg = config.get("crypto", {})

        self._stop_pct:     float = float(ccfg.get("per_trade_stop_loss_pct", 0.05))
        self._tp_pct:       float = float(ccfg.get("take_profit_pct", 0.10))
        self._trail_pct:    float = float(ccfg.get("trailing_stop_pct", 0.04))
        self._max_notional: float = float(ccfg.get("max_position_notional", 500))

        notif_cfg = config.get("notifications", {})
        self._channels: List[str] = list(notif_cfg.get("channels", ["console"]))

        # Per-position tracking (keyed by symbol, e.g. "BTC/USD")
        self._trail_high:  Dict[str, float]               = {}
        self._stops:       Dict[str, Tuple[float, float]] = {}
        self._entry_time:  Dict[str, datetime]            = {}
        self._entry_price: Dict[str, float]               = {}

    # ── Public read-only state ─────────────────────────────────────────────────

    @property
    def tracked_symbols(self) -> List[str]:
        return list(self._trail_high.keys())

    def is_tracking(self, sym: str) -> bool:
        return sym in self._trail_high

    # ── Reconciliation (restart recovery) ─────────────────────────────────────

    def reconcile_positions(self, positions: Dict[str, Position]) -> None:
        """Bootstrap tracking for positions that exist at the broker but not
        in our in-memory dicts (e.g. after a bot restart).

        ``positions`` is already crypto-only — callers pass the result of
        ``broker.get_crypto_positions()``, so no filtering is needed here.
        """
        for sym, pos in positions.items():
            if sym in self._trail_high:
                continue  # already tracked

            entry = pos.avg_entry_price or pos.current_price or 0.0
            if entry <= 0:
                continue

            current    = pos.current_price or entry
            stop_price = entry * (1 - self._stop_pct)
            tp_price   = entry * (1 + self._tp_pct)

            # Trail high starts at the higher of entry or current so we don't
            # immediately fire a trailing stop on a position that has run up.
            self._trail_high[sym]  = max(entry, current)
            self._stops[sym]       = (stop_price, tp_price)
            self._entry_price[sym] = entry
            self._entry_time[sym]  = datetime.now()   # actual time unknown after restart

            log.info(
                "CryptoTrader reconciled %s | entry=%.4f current=%.4f "
                "SL=%.4f TP=%.4f",
                sym, entry, current, stop_price, tp_price,
            )

    # ── Position management ────────────────────────────────────────────────────

    def manage_position(self, pos: Position) -> None:
        """Update trailing high; fire stop-loss, take-profit, or trailing stop."""
        sym   = pos.symbol
        price = pos.current_price or self.broker.get_last_price(sym)
        if price <= 0:
            return

        self._trail_high[sym] = max(self._trail_high.get(sym, price), price)
        peak = self._trail_high[sym]
        stop_price, tp_price = self._stops.get(sym, (None, None))

        reason: Optional[str] = None
        if tp_price and price >= tp_price:
            reason = "take_profit"
        elif stop_price and price <= stop_price:
            reason = "stop_loss"
        elif price <= peak * (1 - self._trail_pct):
            reason = "trailing_stop"

        if reason:
            self.close_position(pos, reason)

    # ── Order submission ───────────────────────────────────────────────────────

    def place_entry(
        self,
        sym: str,
        price: float,
        notional: float,
        dec: AggregatedDecision,
    ) -> bool:
        """Submit a notional BUY order and start tracking the position.

        Returns True if the order was filled.
        """
        stop_price = price * (1 - self._stop_pct)
        tp_price   = price * (1 + self._tp_pct)

        order = Order(
            symbol=sym,
            side=OrderSide.BUY,
            qty=0.0,          # crypto uses notional sizing
            notional=notional,
            type=OrderType.MARKET,
        )
        result = self.broker.submit_order(order)
        if result.status.value != "filled":
            return False

        fill_price = result.filled_avg_price or price
        self._trail_high[sym]  = fill_price
        self._stops[sym]       = (stop_price, tp_price)
        self._entry_time[sym]  = datetime.now()
        self._entry_price[sym] = fill_price

        log.info(
            "CRYPTO ENTRY %s | notional=$%.2f fill=$%.4f SL=$%.4f TP=$%.4f "
            "score=%+.3f conf=%.2f",
            sym, notional, fill_price, stop_price, tp_price,
            dec.score, dec.confidence,
        )
        notify_order_entry(
            symbol=sym, side="buy",
            qty=notional / fill_price,
            price=fill_price,
            agg_score=dec.score, confidence=dec.confidence,
            min_confidence=dec.min_confidence, agreement_ok=True,
            components=dec.components,
            raw_scores=dec.raw_scores,
            stop_loss=round(stop_price, 4),
            take_profit=round(tp_price, 4),
            enter_threshold=dec.enter_long,
            channels=self._channels,
        )
        return True

    def close_position(self, pos: Position, reason: str) -> None:
        """Close the full position using Alpaca's native close_position endpoint.

        Using broker.close_position() instead of a qty-based sell order avoids
        floating-point mismatches (e.g. bot sends 15.020985 but Alpaca holds
        15.020984997, causing "available" rejection errors).  Alpaca calculates
        the exact qty to close server-side.
        """
        sym = pos.symbol

        # Cancel any pending orders first to avoid "insufficient holdings" rejection
        self.broker.cancel_orders_for_symbol(sym)

        # Ghost position (qty rounds to zero) — still call close_position so
        # Alpaca removes it on their side, then clean up locally.
        if round(abs(pos.qty), 8) <= 0:
            log.warning(
                "CryptoTrader: near-zero qty (%.2e) for %s — "
                "closing orphan via broker.close_position().",
                abs(pos.qty), sym,
            )
            self.broker.close_position(sym)
            self.cleanup(sym)
            return

        # Use broker.close_position() — avoids all qty precision issues
        result = self.broker.close_position(sym)

        if result.status.value not in ("filled", "pending", "cancelled"):
            log.warning(
                "CryptoTrader: close_position(%s) status=%s reason=%s "
                "— will retry next cycle.",
                sym, result.status.value, reason,
            )
            return   # position stays tracked; next cycle retries

        if result.status.value == "cancelled":
            # Position was already gone on the broker side (404 → cleaned)
            log.info("CryptoTrader: %s already closed on broker — cleaning up.", sym)
            self.cleanup(sym)
            return

        # Use fill price from result, or fall back to current market price
        exit_price  = result.filled_avg_price or pos.current_price or pos.avg_entry_price
        filled_qty  = result.filled_qty if result.filled_qty > 0 else abs(pos.qty)
        entry_price = self._entry_price.get(sym, pos.avg_entry_price)
        pnl         = (exit_price - entry_price) * filled_qty
        pnl_pct     = (exit_price - entry_price) / entry_price if entry_price else 0.0

        log.info(
            "CRYPTO EXIT %s | reason=%s exit=$%.4f qty=%.6f pnl=%+.2f (%+.2f%%)",
            sym, reason, exit_price, filled_qty, pnl, pnl_pct * 100,
        )
        notify_order_exit(
            symbol=sym, side="sell",
            qty=filled_qty, pnl=pnl, pnl_pct=pnl_pct,
            reason=reason,
            entry_price=entry_price, exit_price=exit_price,
            held_since=self._entry_time.get(sym),
            channels=self._channels,
        )
        self.cleanup(sym)

    def cleanup(self, sym: str) -> None:
        """Remove a symbol from all tracking dicts."""
        self._trail_high.pop(sym, None)
        self._stops.pop(sym, None)
        self._entry_time.pop(sym, None)
        self._entry_price.pop(sym, None)
