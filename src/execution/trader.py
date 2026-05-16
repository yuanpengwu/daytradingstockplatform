"""Order placement + open-position lifecycle management."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

from ..brokers.base import BrokerBase, Order, OrderSide, OrderType, Position
from ..risk.risk_manager import RiskManager
from ..signals.aggregator import AggregatedDecision
from ..utils.logger import get_logger
from ..utils.notifications import notify

log = get_logger(__name__)


class Trader:
    """Glue between aggregated signals, risk checks, and broker calls."""

    def __init__(self, broker: BrokerBase, risk: RiskManager, notify_channels=("console",)):
        self.broker = broker
        self.risk = risk
        self.notify_channels = list(notify_channels)
        # Track peak price since entry per symbol (for trailing stops).
        self._trail_high: Dict[str, float] = {}
        # Track absolute stop / take-profit levels set at entry per symbol.
        self._stops: Dict[str, Tuple[Optional[float], Optional[float]]] = {}

    # ---------- entries ----------
    def handle_decision(self, dec: AggregatedDecision, atr: Optional[float]) -> None:
        side = "buy" if dec.score > 0 else "sell"
        price = self.broker.get_last_price(dec.symbol)
        if price <= 0:
            log.warning("Skipping %s — no price.", dec.symbol)
            return

        existing = self.broker.get_position(dec.symbol)
        if existing:
            # We already hold this. Update trail and check for exit.
            self._update_trailing_high(existing)
            stop_price, tp_price = self._stops.get(dec.symbol, (None, None))
            should_exit, reason = self.risk.check_exit(
                existing,
                score=dec.score,
                trail_high=self._trail_high.get(dec.symbol),
                stop_price=stop_price,
                tp_price=tp_price,
            )
            if should_exit:
                self._close_position(existing, reason)
            return

        # No position — consider entry.
        rd = self.risk.check_entry(
            symbol=dec.symbol,
            side=side,
            score=dec.score,
            confidence=dec.confidence,
            price=price,
            atr=atr,
            broker=self.broker,
        )
        if not rd.approved:
            log.debug("Entry blocked for %s: %s", dec.symbol, rd.reason)
            return

        order = Order(
            symbol=dec.symbol,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            qty=rd.qty,
            type=OrderType.MARKET,
            stop_loss=rd.stop_loss,
            take_profit=rd.take_profit,
        )
        result = self.broker.submit_order(order)
        if result.status.value == "filled":
            self.risk.record_day_trade()
            self._trail_high[dec.symbol] = price
            self._stops[dec.symbol] = (rd.stop_loss, rd.take_profit)
            notify(
                f"[ENTRY] {side.upper()} {rd.qty} {dec.symbol} @ ~${price:.2f} "
                f"score={dec.score:+.2f} conf={dec.confidence:.2f} "
                f"SL=${rd.stop_loss} TP=${rd.take_profit}",
                self.notify_channels,
            )

    # ---------- exits ----------
    def manage_open_positions(self, aggregated: Dict[str, AggregatedDecision]) -> None:
        """Sweep current positions; close any whose decision has flipped or
        whose stop/take-profit has triggered."""
        positions = self.broker.get_positions()
        for sym, pos in positions.items():
            self._update_trailing_high(pos)
            score = aggregated.get(sym).score if sym in aggregated else 0.0
            stop_price, tp_price = self._stops.get(sym, (None, None))
            should_exit, reason = self.risk.check_exit(
                pos,
                score=score,
                trail_high=self._trail_high.get(sym),
                stop_price=stop_price,
                tp_price=tp_price,
            )
            if should_exit:
                self._close_position(pos, reason)

    def flatten_all(self, reason: str = "end-of-day flatten") -> None:
        for sym, pos in self.broker.get_positions().items():
            self._close_position(pos, reason)

    # ---------- internals ----------
    def _close_position(self, position: Position, reason: str) -> None:
        side = OrderSide.SELL if position.qty > 0 else OrderSide.BUY
        order = Order(
            symbol=position.symbol,
            side=side,
            qty=abs(position.qty),
            type=OrderType.MARKET,
        )
        result = self.broker.submit_order(order)
        if result.status.value == "filled":
            self.risk.record_day_trade()
            self._trail_high.pop(position.symbol, None)
            self._stops.pop(position.symbol, None)
            notify(
                f"[EXIT] {side.value.upper()} {abs(position.qty)} {position.symbol} "
                f"PnL={position.unrealized_pnl:+.2f} ({position.unrealized_pnl_pct*100:+.2f}%) "
                f"reason={reason}",
                self.notify_channels,
            )

    def _update_trailing_high(self, pos: Position) -> None:
        cur = pos.current_price or self.broker.get_last_price(pos.symbol)
        if cur <= 0:
            return
        prev = self._trail_high.get(pos.symbol, pos.avg_entry_price)
        if cur > prev:
            self._trail_high[pos.symbol] = cur
