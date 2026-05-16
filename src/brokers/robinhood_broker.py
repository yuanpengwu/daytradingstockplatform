"""Robinhood broker adapter (UNOFFICIAL — robin_stocks).

WARNING: Robinhood does NOT provide an official public trading API. This
adapter relies on the `robin_stocks` community library, which scrapes
Robinhood's private endpoints. Risks:
  - May violate Robinhood's Terms of Service for automated trading.
  - Endpoints can change without notice and silently break.
  - 2FA / device-verification flows are fragile in headless environments.

If you can use Alpaca instead, do — its API is purpose-built for bots.

Requires:  ROBINHOOD_USERNAME, ROBINHOOD_PASSWORD  in .env
"""
from __future__ import annotations

import os
from datetime import datetime, time as dtime
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

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


class RobinhoodBroker(BrokerBase):
    def __init__(self):
        try:
            import robin_stocks.robinhood as rh
        except ImportError as e:
            raise RuntimeError(
                "robin_stocks is not installed. Run: pip install robin-stocks"
            ) from e

        self._rh = rh
        self._logged_in = False
        self.connect()

    # ---------- connect ----------
    def connect(self) -> None:
        if self._logged_in:
            return
        user = os.getenv("ROBINHOOD_USERNAME")
        pw = os.getenv("ROBINHOOD_PASSWORD")
        mfa = os.getenv("ROBINHOOD_MFA_CODE") or None
        if not (user and pw):
            raise RuntimeError("ROBINHOOD_USERNAME / ROBINHOOD_PASSWORD not set.")
        try:
            self._rh.login(username=user, password=pw, mfa_code=mfa, store_session=True)
            self._logged_in = True
            log.warning(
                "RobinhoodBroker connected via UNOFFICIAL API — consider Alpaca."
            )
        except Exception as e:
            raise RuntimeError(f"Robinhood login failed: {e}") from e

    def shutdown(self) -> None:
        if self._logged_in:
            try:
                self._rh.logout()
            except Exception:
                pass

    # ---------- account ----------
    def get_equity(self) -> float:
        profile = self._rh.profiles.load_account_profile()
        return float(profile.get("portfolio_cash", 0)) + self._market_value()

    def get_cash(self) -> float:
        profile = self._rh.profiles.load_account_profile()
        return float(profile.get("cash", 0))

    def get_buying_power(self) -> float:
        profile = self._rh.profiles.load_account_profile()
        return float(profile.get("buying_power", 0))

    def _market_value(self) -> float:
        return sum(p.market_value for p in self.get_positions().values())

    # ---------- positions ----------
    def get_positions(self) -> Dict[str, Position]:
        out: Dict[str, Position] = {}
        try:
            holdings = self._rh.account.build_holdings()
        except Exception as e:
            log.warning("Robinhood holdings fetch failed: %s", e)
            return out
        for sym, info in holdings.items():
            try:
                qty = float(info.get("quantity", 0))
                if qty <= 0:
                    continue
                out[sym] = Position(
                    symbol=sym,
                    qty=qty,
                    avg_entry_price=float(info.get("average_buy_price", 0)),
                    current_price=float(info.get("price", 0)),
                )
            except (TypeError, ValueError):
                continue
        return out

    # ---------- orders ----------
    def submit_order(self, order: Order) -> Order:
        try:
            if order.side == OrderSide.BUY:
                if order.type == OrderType.LIMIT and order.limit_price:
                    resp = self._rh.orders.order_buy_limit(
                        order.symbol, int(order.qty), order.limit_price
                    )
                else:
                    resp = self._rh.orders.order_buy_market(order.symbol, int(order.qty))
            else:
                if order.type == OrderType.LIMIT and order.limit_price:
                    resp = self._rh.orders.order_sell_limit(
                        order.symbol, int(order.qty), order.limit_price
                    )
                else:
                    resp = self._rh.orders.order_sell_market(order.symbol, int(order.qty))
        except Exception as e:
            log.error("Robinhood order failed: %s", e)
            order.status = OrderStatus.REJECTED
            return order

        if not resp or "id" not in resp:
            order.status = OrderStatus.REJECTED
            log.error("Robinhood order rejected: %s", resp)
            return order
        order.id = resp["id"]
        order.status = OrderStatus.PENDING
        log.info("Robinhood submitted %s %s %s", order.side.value, order.qty, order.symbol)
        return order

    def cancel_order(self, order_id: str) -> None:
        try:
            self._rh.orders.cancel_stock_order(order_id)
        except Exception as e:
            log.warning("Robinhood cancel failed for %s: %s", order_id, e)

    def get_open_orders(self) -> List[Order]:
        out: List[Order] = []
        try:
            for o in self._rh.orders.get_all_open_stock_orders():
                instr = self._rh.stocks.get_instrument_by_url(o["instrument"])
                out.append(
                    Order(
                        id=o["id"],
                        symbol=instr.get("symbol", ""),
                        side=OrderSide(o["side"]),
                        qty=float(o["quantity"]),
                        type=OrderType.LIMIT if o.get("price") else OrderType.MARKET,
                        status=OrderStatus.PENDING,
                    )
                )
        except Exception as e:
            log.warning("Robinhood open orders fetch failed: %s", e)
        return out

    # ---------- market info ----------
    def is_market_open(self) -> bool:
        now = datetime.now(NY)
        if now.weekday() >= 5:
            return False
        return dtime(9, 30) <= now.time() <= dtime(16, 0)

    def get_last_price(self, symbol: str) -> float:
        try:
            quotes = self._rh.stocks.get_latest_price(symbol)
            return float(quotes[0]) if quotes else 0.0
        except Exception as e:
            log.warning("Robinhood price fetch failed for %s: %s", symbol, e)
            return 0.0
