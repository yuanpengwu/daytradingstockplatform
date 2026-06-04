"""Broker interface + shared data classes.

Every concrete broker (Robinhood, Alpaca, Paper) implements BrokerBase so
the rest of the system is broker-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


def is_crypto_symbol(symbol: str) -> bool:
    """Return True for crypto tickers like BTC/USD, ETH/USD."""
    return "/" in symbol


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    symbol: str
    side: OrderSide
    qty: float
    type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    notional: Optional[float] = None       # crypto: order by dollar amount instead of qty
    id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    submitted_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_entry_price) * self.qty

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price


class BrokerBase(ABC):
    """Minimal interface all brokers must implement."""

    # ---------- account ----------
    @abstractmethod
    def get_equity(self) -> float: ...

    @abstractmethod
    def get_cash(self) -> float: ...

    @abstractmethod
    def get_buying_power(self) -> float: ...

    # ---------- positions ----------
    @abstractmethod
    def get_positions(self) -> Dict[str, Position]: ...

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.get_positions().get(symbol)

    # ---------- orders ----------
    @abstractmethod
    def submit_order(self, order: Order) -> Order: ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...

    @abstractmethod
    def get_open_orders(self) -> List[Order]: ...

    # ---------- market info ----------
    @abstractmethod
    def is_market_open(self) -> bool: ...

    @abstractmethod
    def get_last_price(self, symbol: str) -> float: ...

    # ---------- lifecycle ----------
    def connect(self) -> None:
        """Override if the broker needs login / handshake."""
        return None

    def shutdown(self) -> None:
        return None
