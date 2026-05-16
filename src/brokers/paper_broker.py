"""In-memory paper-trading simulator.

Use this for development, backtesting-with-live-data, and validation
before risking real money. It models slippage, tracks positions, and
exposes the same BrokerBase interface as the live adapters.
"""
from __future__ import annotations

import uuid
from datetime import datetime, time as dtime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import yfinance as yf

from ..utils.logger import get_logger
from .base import (
    BrokerBase,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
)

log = get_logger(__name__)
NY = ZoneInfo("America/New_York")


class PaperBroker(BrokerBase):
    def __init__(self, starting_cash: float = 10_000.0, slippage_bps: float = 5.0):
        self._cash = float(starting_cash)
        self._slippage = float(slippage_bps) / 10_000.0
        self._positions: Dict[str, Position] = {}
        self._orders: List[Order] = []
        self._price_cache: Dict[str, float] = {}
        self._realized_pnl = 0.0
        log.info(
            "PaperBroker initialized | cash=$%.2f | slippage=%.2fbps",
            self._cash,
            slippage_bps,
        )

    # ---------- account ----------
    def get_cash(self) -> float:
        return self._cash

    def get_equity(self) -> float:
        market_value = sum(p.market_value for p in self._positions.values())
        return self._cash + market_value

    def get_buying_power(self) -> float:
        # Cash account semantics — no margin.
        return self._cash

    # ---------- positions ----------
    def get_positions(self) -> Dict[str, Position]:
        # Refresh marks before returning.
        for sym, pos in self._positions.items():
            pos.current_price = self.get_last_price(sym)
        return dict(self._positions)

    # ---------- orders ----------
    def submit_order(self, order: Order) -> Order:
        order.id = order.id or str(uuid.uuid4())
        order.submitted_at = datetime.utcnow()
        px = self.get_last_price(order.symbol)
        if px <= 0:
            order.status = OrderStatus.REJECTED
            log.warning("Order rejected — no price for %s", order.symbol)
            self._orders.append(order)
            return order

        # Apply slippage (buys pay up, sells get hit).
        fill = px * (1 + self._slippage) if order.side == OrderSide.BUY else px * (1 - self._slippage)

        if order.side == OrderSide.BUY:
            cost = fill * order.qty
            if cost > self._cash:
                order.status = OrderStatus.REJECTED
                log.warning("Order rejected — insufficient cash for %s", order.symbol)
                self._orders.append(order)
                return order
            self._cash -= cost
            existing = self._positions.get(order.symbol)
            if existing:
                total_qty = existing.qty + order.qty
                existing.avg_entry_price = (
                    existing.avg_entry_price * existing.qty + fill * order.qty
                ) / total_qty
                existing.qty = total_qty
            else:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol, qty=order.qty, avg_entry_price=fill
                )
        else:  # SELL
            pos = self._positions.get(order.symbol)
            if not pos or pos.qty < order.qty:
                order.status = OrderStatus.REJECTED
                log.warning("Order rejected — no position to sell in %s", order.symbol)
                self._orders.append(order)
                return order
            proceeds = fill * order.qty
            self._realized_pnl += (fill - pos.avg_entry_price) * order.qty
            self._cash += proceeds
            pos.qty -= order.qty
            if pos.qty <= 1e-9:
                del self._positions[order.symbol]

        order.status = OrderStatus.FILLED
        order.filled_qty = order.qty
        order.filled_avg_price = fill
        self._orders.append(order)
        log.info(
            "FILL %s %s %.4f @ %.4f | cash=$%.2f | equity=$%.2f",
            order.side.value.upper(),
            order.symbol,
            order.qty,
            fill,
            self._cash,
            self.get_equity(),
        )
        return order

    def cancel_order(self, order_id: str) -> None:
        # Market orders fill instantly in the sim — nothing to cancel.
        return None

    def get_open_orders(self) -> List[Order]:
        return [o for o in self._orders if o.status == OrderStatus.PENDING]

    # ---------- market info ----------
    def is_market_open(self) -> bool:
        now = datetime.now(NY)
        if now.weekday() >= 5:
            return False
        return dtime(9, 30) <= now.time() <= dtime(16, 0)

    def get_last_price(self, symbol: str) -> float:
        try:
            t = yf.Ticker(symbol)
            data = t.history(period="1d", interval="1m")
            if not data.empty:
                px = float(data["Close"].iloc[-1])
                self._price_cache[symbol] = px
                return px
        except Exception as e:
            log.warning("yfinance price fetch failed for %s: %s", symbol, e)
        return self._price_cache.get(symbol, 0.0)

    # ---------- helpers for backtest / reporting ----------
    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl
