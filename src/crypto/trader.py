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

from ..brokers.base import BrokerBase, Order, OrderSide, OrderType, Position
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
        """Cancel any pending orders, then submit a SELL to close the position.

        If the sell is not immediately filled, the position stays in tracking
        and the next cycle will retry.
        """
        sym = pos.symbol

        # Cancel pending orders first — a PENDING sell from a prior cycle
        # causes Alpaca to reject the new sell ("insufficient holdings").
        self.broker.cancel_orders_for_symbol(sym)

        # Round to 8 dp (Alpaca's max crypto precision).
        qty = round(abs(pos.qty), 8)
        if qty <= 0:
            # Ghost position: qty is too small to sell via qty-based order
            # (e.g. 4e-09 LINK remnant).  Use Alpaca's native close_position
            # endpoint so the position is also removed on the broker side,
            # not just from our in-memory tracking.
            log.warning(
                "CryptoTrader: near-zero qty (%.2e) for %s — "
                "closing orphan on broker via close_position().",
                abs(pos.qty), sym,
            )
            self.broker.close_position(sym)
            self.cleanup(sym)
            return

        order = Order(
            symbol=sym,
            side=OrderSide.SELL,
            qty=qty,
            type=OrderType.MARKET,
        )
        result = self.broker.submit_order(order)

        if result.status.value != "filled":
            log.warning(
                "CryptoTrader: SELL %s NOT filled | status=%s reason=%s qty=%.8f "
                "— will retry next cycle.",
                sym, result.status.value, reason, qty,
            )
            return  # position stays tracked; next cycle retries

        exit_price  = result.filled_avg_price or pos.current_price
        entry_price = self._entry_price.get(sym, pos.avg_entry_price)
        pnl         = (exit_price - entry_price) * abs(pos.qty)
        pnl_pct     = (exit_price - entry_price) / entry_price if entry_price else 0.0

        log.info(
            "CRYPTO EXIT %s | reason=%s exit=$%.4f pnl=%+.2f (%+.2f%%)",
            sym, reason, exit_price, pnl, pnl_pct * 100,
        )
        notify_order_exit(
            symbol=sym, side="sell",
            qty=abs(pos.qty), pnl=pnl, pnl_pct=pnl_pct,
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
