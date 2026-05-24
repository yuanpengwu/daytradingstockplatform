"""Alpaca live & paper broker adapter.

Alpaca is the recommended live broker:
  * Real, supported public API.
  * Free paper-trading endpoint (https://paper-api.alpaca.markets).
  * Commission-free US equities + crypto.

Requires:  ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_BASE_URL  in .env

Network resilience: a long-running bot will hit transient network blips
(dropped connections, timeouts, brief API hiccups). Read calls here retry
with backoff. If they still fail, they raise a clean error — the engine's
main loop catches it, logs it, and simply tries again next cycle. We never
act on stale or missing data, and we never auto-retry order submission
(that could double-fill).
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, TypeVar

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

T = TypeVar("T")

# Exceptions worth retrying — transient network/transport failures.
_TRANSIENT = (ConnectionError, TimeoutError, OSError)


def _retry(fn: Callable[[], T], what: str, attempts: int = 3) -> T:
    """Call fn(), retrying transient network errors with linear backoff.

    Raises the last error if every attempt fails.
    """
    last = None
    for i in range(1, attempts + 1):
        try:
            return fn()
        except _TRANSIENT as e:
            last = e
            log.warning("Alpaca %s failed (attempt %d/%d): %s", what, i, attempts, e)
        except Exception as e:
            # Non-transient (auth, bad request, etc.) — don't retry.
            log.error("Alpaca %s error: %s", what, e)
            raise
        if i < attempts:
            time.sleep(1.0 * i)
    raise ConnectionError(f"Alpaca {what} failed after {attempts} attempts: {last}")


def _map_alpaca_status(status: str) -> OrderStatus:
    s = status.lower()
    if s == "filled":
        return OrderStatus.FILLED
    if s == "partially_filled":
        return OrderStatus.PARTIAL
    if s in ("canceled", "cancelled", "expired", "replaced", "done_for_day"):
        return OrderStatus.CANCELLED
    if s in ("rejected", "suspended"):
        return OrderStatus.REJECTED
    return OrderStatus.PENDING


class AlpacaBroker(BrokerBase):
    def __init__(self):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as e:
            raise RuntimeError(
                "alpaca-py is not installed. Run: pip install alpaca-py"
            ) from e

        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        base = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not (key and secret):
            raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set in env.")

        paper = "paper" in base
        self._client = TradingClient(key, secret, paper=paper)
        self._data = StockHistoricalDataClient(key, secret)
        log.info("AlpacaBroker connected | paper=%s", paper)

    # ---------- account ----------
    def _account(self):
        return _retry(self._client.get_account, "get_account")

    def get_equity(self) -> float:
        return float(self._account().equity)

    def get_cash(self) -> float:
        return float(self._account().cash)

    def get_buying_power(self) -> float:
        return float(self._account().buying_power)

    # ---------- positions ----------
    def get_positions(self) -> Dict[str, Position]:
        # Retried; if it ultimately fails it raises, and the engine loop
        # safely skips this cycle rather than acting on missing positions.
        raw = _retry(self._client.get_all_positions, "get_positions")
        out: Dict[str, Position] = {}
        for p in raw:
            out[p.symbol] = Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry_price=float(p.avg_entry_price),
                current_price=float(p.current_price or 0),
            )
        return out

    # ---------- orders ----------
    def submit_order(self, order: Order) -> Order:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide as AOS, TimeInForce

        side = AOS.BUY if order.side == OrderSide.BUY else AOS.SELL
        if order.type == OrderType.LIMIT and order.limit_price:
            req = LimitOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
            )
        else:
            req = MarketOrderRequest(
                symbol=order.symbol,
                qty=order.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        # NOTE: order submission is intentionally NOT retried — a retry after a
        # dropped connection could place the same order twice. On any failure
        # we mark it rejected; the engine will reconsider it next cycle.
        try:
            resp = self._client.submit_order(req)
        except Exception as e:
            log.error("Alpaca order failed (%s %s %s): %s",
                      order.side.value, order.qty, order.symbol, e)
            order.status = OrderStatus.REJECTED
            return order

        order.id = str(resp.id)
        order.status = _map_alpaca_status(resp.status.value) if resp.status else OrderStatus.PENDING
        order.filled_qty = float(resp.filled_qty or 0)
        order.filled_avg_price = float(resp.filled_avg_price) if resp.filled_avg_price else None

        # Alpaca market orders fill asynchronously — the initial response is
        # "new" or "accepted", not "filled". Poll once after a short delay so
        # the caller sees the true fill status and fill price, which is required
        # for notifications and trade history to fire correctly.
        if order.type == OrderType.MARKET and order.status == OrderStatus.PENDING:
            time.sleep(2)
            try:
                updated = self._client.get_order_by_id(order.id)
                order.status = _map_alpaca_status(updated.status.value) if updated.status else order.status
                order.filled_qty = float(updated.filled_qty or order.filled_qty)
                if updated.filled_avg_price:
                    order.filled_avg_price = float(updated.filled_avg_price)
            except Exception as e:
                log.warning("Could not refresh fill status for order %s (%s): %s",
                            order.id, order.symbol, e)

        log.info("Alpaca %s %s %s -> %s (filled_qty=%s avg_px=%s)",
                 order.side.value, order.qty, order.symbol, order.status.value,
                 order.filled_qty, order.filled_avg_price)
        return order

    def cancel_order(self, order_id: str) -> None:
        try:
            self._client.cancel_order_by_id(order_id)
        except Exception as e:
            log.warning("Cancel failed for %s: %s", order_id, e)

    def get_open_orders(self) -> List[Order]:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus

        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        try:
            raw = _retry(lambda: self._client.get_orders(filter=req), "get_open_orders")
        except Exception as e:
            log.warning("Could not fetch open orders: %s", e)
            return []
        out: List[Order] = []
        for o in raw:
            out.append(
                Order(
                    id=str(o.id),
                    symbol=o.symbol,
                    side=OrderSide(o.side.value),
                    qty=float(o.qty),
                    type=OrderType.MARKET if o.order_type.value == "market" else OrderType.LIMIT,
                    status=OrderStatus.PENDING,
                )
            )
        return out

    # ---------- market info ----------
    def is_market_open(self) -> bool:
        try:
            clock = _retry(self._client.get_clock, "get_clock")
            return bool(clock.is_open)
        except Exception:
            # On persistent failure, assume closed — the safe default.
            return False

    def get_last_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestTradeRequest

        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        try:
            trades = _retry(
                lambda: self._data.get_stock_latest_trade(req),
                f"get_last_price({symbol})",
            )
            return float(trades[symbol].price)
        except Exception as e:
            log.warning("Alpaca price fetch failed for %s: %s", symbol, e)
            return 0.0
