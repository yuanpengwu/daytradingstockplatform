"""Crypto position lifecycle management.

Handles entry placement, stop-loss / take-profit / trailing-stop checks,
and position exit for crypto pairs — both LONG and SHORT.

Crypto-specific rules enforced here (not shared with the stock Trader):
  • GTC time-in-force  — DAY orders are invalid on 24/7 crypto markets
  • Notional sizing    — orders sized in $ not shares
  • Fractional qty     — Alpaca calculates exact qty via close_position()
  • No PDT rules       — crypto has no pattern-day-trader restriction
  • Wider stops/TP     — 5% SL, 10% TP, 4% trailing (vs 2.5%/6%/3% for stocks)
  • Short selling      — enabled via config crypto.shorting_enabled
                         requires Alpaca margin account; paper trading supports it

Long vs short mechanics
──────────────────────
  Long  (qty > 0): trail_watermark = max price seen
    stop  fires when  price <=  stop_price  (entry × 0.95)
    TP    fires when  price >=  tp_price    (entry × 1.10)
    trail fires when  price <=  watermark × (1 − trail_pct)

  Short (qty < 0): trail_watermark = min price seen
    stop  fires when  price >=  stop_price  (entry × 1.05)
    TP    fires when  price <=  tp_price    (entry × 0.90)
    trail fires when  price >=  watermark × (1 + trail_pct)

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

        self._stop_pct:         float = float(ccfg.get("per_trade_stop_loss_pct", 0.05))
        self._tp_pct:           float = float(ccfg.get("take_profit_pct",         0.10))
        self._trail_pct:        float = float(ccfg.get("trailing_stop_pct",       0.04))
        self._max_notional:     float = float(ccfg.get("max_position_notional",   500))
        self._short_notional:   float = float(ccfg.get("short_max_notional",      300))
        self.shorting_enabled:  bool  = bool(ccfg.get("shorting_enabled",         False))

        notif_cfg = config.get("notifications", {})
        self._channels: List[str] = list(notif_cfg.get("channels", ["console"]))

        # Per-position tracking (keyed by symbol, e.g. "BTC/USD")
        # _side: "long" | "short"
        # _trail_watermark: max price for longs, min price for shorts
        self._side:             Dict[str, str]               = {}
        self._trail_watermark:  Dict[str, float]             = {}
        self._stops:            Dict[str, Tuple[float, float]] = {}   # (stop_price, tp_price)
        self._entry_time:       Dict[str, datetime]          = {}
        self._entry_price:      Dict[str, float]             = {}

    # ── Public read-only state ─────────────────────────────────────────────────

    @property
    def tracked_symbols(self) -> List[str]:
        return list(self._trail_watermark.keys())

    def is_tracking(self, sym: str) -> bool:
        return sym in self._trail_watermark

    # ── Reconciliation (restart recovery) ─────────────────────────────────────

    def reconcile_positions(self, positions: Dict[str, Position]) -> None:
        """Bootstrap tracking for positions that exist at the broker but not
        in our in-memory dicts (e.g. after a bot restart).

        Handles both long (qty > 0) and short (qty < 0) positions.
        """
        for sym, pos in positions.items():
            if sym in self._trail_watermark:
                continue  # already tracked

            entry = pos.avg_entry_price or pos.current_price or 0.0
            if entry <= 0:
                continue

            current = pos.current_price or entry
            is_short = pos.qty < 0

            if is_short:
                stop_price = entry * (1 + self._stop_pct)   # above entry for shorts
                tp_price   = entry * (1 - self._tp_pct)     # below entry for shorts
                watermark  = min(entry, current)             # lowest price seen
                side       = "short"
            else:
                stop_price = entry * (1 - self._stop_pct)
                tp_price   = entry * (1 + self._tp_pct)
                watermark  = max(entry, current)
                side       = "long"

            self._side[sym]            = side
            self._trail_watermark[sym] = watermark
            self._stops[sym]           = (stop_price, tp_price)
            self._entry_price[sym]     = entry
            self._entry_time[sym]      = datetime.now()

            log.info(
                "CryptoTrader reconciled %s [%s] | entry=%.4f current=%.4f "
                "SL=%.4f TP=%.4f",
                sym, side, entry, current, stop_price, tp_price,
            )

    # ── Position management ────────────────────────────────────────────────────

    def manage_position(self, pos: Position) -> None:
        """Update trailing watermark; fire stop-loss, take-profit, or trailing stop."""
        sym   = pos.symbol
        price = pos.current_price or self.broker.get_last_price(sym)
        if price <= 0:
            return

        side = self._side.get(sym, "long" if pos.qty >= 0 else "short")
        wm   = self._trail_watermark.get(sym, price)
        stop_price, tp_price = self._stops.get(sym, (None, None))

        if side == "long":
            self._trail_watermark[sym] = max(wm, price)
            wm = self._trail_watermark[sym]
            reason: Optional[str] = None
            if tp_price   and price >= tp_price:
                reason = "take_profit"
            elif stop_price and price <= stop_price:
                reason = "stop_loss"
            elif price <= wm * (1 - self._trail_pct):
                reason = "trailing_stop"
        else:  # short
            self._trail_watermark[sym] = min(wm, price)
            wm = self._trail_watermark[sym]
            reason = None
            if tp_price   and price <= tp_price:
                reason = "take_profit"
            elif stop_price and price >= stop_price:
                reason = "stop_loss"
            elif price >= wm * (1 + self._trail_pct):
                reason = "trailing_stop"

        if reason:
            self.close_position(pos, reason)

    # ── Long entry ─────────────────────────────────────────────────────────────

    def place_entry(
        self,
        sym: str,
        price: float,
        notional: float,
        dec: AggregatedDecision,
    ) -> bool:
        """Submit a notional BUY order and start tracking the long position."""
        stop_price = price * (1 - self._stop_pct)
        tp_price   = price * (1 + self._tp_pct)

        order = Order(
            symbol=sym, side=OrderSide.BUY,
            qty=0.0, notional=notional, type=OrderType.MARKET,
        )
        result = self.broker.submit_order(order)
        if result.status.value != "filled":
            return False

        fill_price = result.filled_avg_price or price
        self._side[sym]            = "long"
        self._trail_watermark[sym] = fill_price
        self._stops[sym]           = (stop_price, tp_price)
        self._entry_time[sym]      = datetime.now()
        self._entry_price[sym]     = fill_price

        log.info(
            "CRYPTO LONG %s | notional=$%.2f fill=$%.4f SL=$%.4f TP=$%.4f "
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

    # ── Short entry ────────────────────────────────────────────────────────────

    def place_short_entry(
        self,
        sym: str,
        price: float,
        notional: float,
        dec: AggregatedDecision,
    ) -> bool:
        """Submit a notional SELL order to open a short position.

        Requires Alpaca margin account. For paper trading this works out of the box.
        Stop and TP are inverted vs longs:
          stop  = entry × (1 + stop_pct)   ← price rises against us
          TP    = entry × (1 − tp_pct)     ← price falls in our favour
        """
        stop_price = price * (1 + self._stop_pct)
        tp_price   = price * (1 - self._tp_pct)

        order = Order(
            symbol=sym, side=OrderSide.SELL,
            qty=0.0, notional=notional, type=OrderType.MARKET,
        )
        result = self.broker.submit_order(order)
        if result.status.value != "filled":
            log.warning(
                "CryptoTrader: SHORT entry %s NOT filled (status=%s) — skipping.",
                sym, result.status.value,
            )
            return False

        fill_price = result.filled_avg_price or price
        self._side[sym]            = "short"
        self._trail_watermark[sym] = fill_price   # trail_low starts at entry
        self._stops[sym]           = (stop_price, tp_price)
        self._entry_time[sym]      = datetime.now()
        self._entry_price[sym]     = fill_price

        log.info(
            "CRYPTO SHORT %s | notional=$%.2f fill=$%.4f SL=$%.4f TP=$%.4f "
            "score=%+.3f conf=%.2f",
            sym, notional, fill_price, stop_price, tp_price,
            dec.score, dec.confidence,
        )
        notify_order_entry(
            symbol=sym, side="sell",
            qty=notional / fill_price,
            price=fill_price,
            agg_score=dec.score, confidence=dec.confidence,
            min_confidence=dec.min_confidence, agreement_ok=True,
            components=dec.components,
            raw_scores=dec.raw_scores,
            stop_loss=round(stop_price, 4),
            take_profit=round(tp_price, 4),
            enter_threshold=dec.enter_short,
            channels=self._channels,
        )
        return True

    # ── Exit (works for both long and short) ───────────────────────────────────

    def close_position(self, pos: Position, reason: str) -> None:
        """Close the full position via Alpaca's native close_position endpoint.

        Works for both long (buys in) and short (buys back) positions.
        Using broker.close_position() avoids floating-point qty mismatches.
        """
        sym  = pos.symbol
        side = self._side.get(sym, "long" if pos.qty >= 0 else "short")

        self.broker.cancel_orders_for_symbol(sym)

        # Ghost position
        if round(abs(pos.qty), 8) <= 0:
            log.warning(
                "CryptoTrader: near-zero qty (%.2e) for %s [%s] — "
                "closing orphan via broker.",
                abs(pos.qty), sym, side,
            )
            self.broker.close_position(sym)
            self.cleanup(sym)
            return

        result = self.broker.close_position(sym)

        if result.status.value not in ("filled", "pending", "cancelled"):
            log.warning(
                "CryptoTrader: close_position(%s) [%s] status=%s reason=%s "
                "— will retry next cycle.",
                sym, side, result.status.value, reason,
            )
            return

        if result.status.value == "cancelled":
            log.info("CryptoTrader: %s [%s] already closed on broker.", sym, side)
            self.cleanup(sym)
            return

        exit_price  = result.filled_avg_price or pos.current_price or pos.avg_entry_price
        filled_qty  = result.filled_qty if result.filled_qty > 0 else abs(pos.qty)
        entry_price = self._entry_price.get(sym, pos.avg_entry_price)

        # P&L: long = (exit − entry) × qty | short = (entry − exit) × qty
        if side == "short":
            pnl     = (entry_price - exit_price) * filled_qty
            pnl_pct = (entry_price - exit_price) / entry_price if entry_price else 0.0
            exit_side = "buy"   # covering the short
        else:
            pnl     = (exit_price - entry_price) * filled_qty
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
            exit_side = "sell"

        log.info(
            "CRYPTO EXIT %s [%s] | reason=%s exit=$%.4f qty=%.6f pnl=%+.2f (%+.2f%%)",
            sym, side, reason, exit_price, filled_qty, pnl, pnl_pct * 100,
        )
        notify_order_exit(
            symbol=sym, side=exit_side,
            qty=filled_qty, pnl=pnl, pnl_pct=pnl_pct,
            reason=reason,
            entry_price=entry_price, exit_price=exit_price,
            held_since=self._entry_time.get(sym),
            channels=self._channels,
        )
        self.cleanup(sym)

    def cleanup(self, sym: str) -> None:
        """Remove a symbol from all tracking dicts."""
        self._side.pop(sym, None)
        self._trail_watermark.pop(sym, None)
        self._stops.pop(sym, None)
        self._entry_time.pop(sym, None)
        self._entry_price.pop(sym, None)
