"""Position sizing, stops, and kill switches.

Every order placed by the bot goes through RiskManager.check() first.
If `decision.approved` is False, the trade is blocked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

from ..brokers.base import BrokerBase, Position
from ..utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reason: str = ""
    qty: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class RiskManager:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.max_position_pct = float(cfg.get("max_position_pct", 0.10))
        self.max_total_exposure_pct = float(cfg.get("max_total_exposure_pct", 0.80))
        self.max_concurrent = int(cfg.get("max_concurrent_positions", 5))
        self.daily_loss_limit_pct = float(cfg.get("daily_loss_limit_pct", 0.03))
        self.stop_pct = float(cfg.get("per_trade_stop_loss_pct", 0.02))
        self.tp_pct = float(cfg.get("take_profit_pct", 0.04))
        self.trail_pct = float(cfg.get("trailing_stop_pct", 0.015))
        self.use_atr = bool(cfg.get("use_atr_stops", True))
        self.shorting = bool(cfg.get("shorting_enabled", False))
        self.pdt_protect = bool(cfg.get("pdt_protection", True))
        self.kelly = float(cfg.get("kelly_fraction", 0.25))

        # State across the trading day.
        self._equity_at_open: Optional[float] = None
        self._open_date: Optional[date] = None
        self._day_trade_dates: List[date] = []
        self._kill_switch = False

    # ---------- daily lifecycle ----------
    def begin_day(self, equity: float) -> None:
        self._equity_at_open = equity
        self._open_date = date.today()
        self._kill_switch = False
        log.info("RiskManager day start | equity=$%.2f", equity)

    def record_day_trade(self) -> None:
        self._day_trade_dates.append(date.today())
        # Trim history to last 5 business days worth.
        self._day_trade_dates = self._day_trade_dates[-20:]

    def kill_switch_engaged(self) -> bool:
        return self._kill_switch

    def check_daily_loss(self, equity: float) -> bool:
        if self._equity_at_open is None:
            return False
        loss_pct = (self._equity_at_open - equity) / self._equity_at_open
        if loss_pct >= self.daily_loss_limit_pct:
            if not self._kill_switch:
                log.error(
                    "KILL SWITCH ENGAGED — daily loss %.2f%% >= limit %.2f%%",
                    loss_pct * 100, self.daily_loss_limit_pct * 100,
                )
            self._kill_switch = True
            return True
        return False

    # ---------- pre-trade check ----------
    def check_entry(
        self,
        symbol: str,
        side: str,                           # "buy" or "sell"
        score: float,                        # aggregated signal score
        confidence: float,
        price: float,
        atr: Optional[float],
        broker: BrokerBase,
    ) -> RiskDecision:
        if self._kill_switch:
            return RiskDecision(False, "kill switch engaged (daily loss limit hit)")
        if side == "sell" and not self.shorting:
            return RiskDecision(False, "shorting disabled in config")
        equity = broker.get_equity()
        if equity <= 0:
            return RiskDecision(False, "equity is zero")

        self.check_daily_loss(equity)
        if self._kill_switch:
            return RiskDecision(False, "kill switch engaged")

        positions = broker.get_positions()
        if symbol in positions and side == "buy":
            return RiskDecision(False, "already long this symbol")
        if len(positions) >= self.max_concurrent and symbol not in positions:
            return RiskDecision(False, f"max concurrent positions ({self.max_concurrent}) reached")

        # PDT rule (under $25k cash account) — block new entry if it could trigger 4th day trade.
        if self.pdt_protect and equity < 25_000:
            recent = [d for d in self._day_trade_dates if (date.today() - d).days < 5]
            if len(recent) >= 3:
                return RiskDecision(False, "PDT-rule protection: 3 day trades in last 5 days on sub-$25k account")

        # Position size = fractional-Kelly  *  capped by max_position_pct
        kelly_pct = max(0.0, min(1.0, abs(score) * confidence)) * self.kelly
        target_pct = min(kelly_pct, self.max_position_pct)
        # Respect total-exposure cap
        cur_exposure = sum(p.market_value for p in positions.values())
        max_new = max(0.0, equity * self.max_total_exposure_pct - cur_exposure)
        notional = min(equity * target_pct, max_new, broker.get_buying_power())
        if notional <= 0 or price <= 0:
            return RiskDecision(False, "no buying power or invalid price")

        qty = round(notional / price, 4)
        # For brokers that only support whole shares (Robinhood market orders), floor.
        if qty < 1:
            return RiskDecision(False, f"sized position too small (qty={qty:.4f}) — increase equity or weights")
        qty = float(int(qty))

        # Stops
        if self.use_atr and atr and atr > 0:
            stop = price - 2 * atr if side == "buy" else price + 2 * atr
            tp = price + 3 * atr if side == "buy" else price - 3 * atr
        else:
            stop = price * (1 - self.stop_pct) if side == "buy" else price * (1 + self.stop_pct)
            tp = price * (1 + self.tp_pct) if side == "buy" else price * (1 - self.tp_pct)

        return RiskDecision(
            approved=True,
            reason="ok",
            qty=qty,
            stop_loss=round(stop, 2),
            take_profit=round(tp, 2),
        )

    # ---------- position management ----------
    def check_exit(
        self,
        position: Position,
        score: float,                        # current aggregated score
        trail_high: Optional[float] = None,
        stop_price: Optional[float] = None,  # absolute stop set at entry (e.g. ATR-based)
        tp_price: Optional[float] = None,    # absolute take-profit set at entry
    ) -> Tuple[bool, str]:
        """Return (should_exit, reason).

        Exit precedence: hard stop/take-profit price levels set at entry are
        checked FIRST (these honor `use_atr_stops`), then percentage fallbacks,
        then signal reversal, then the trailing stop.
        """
        if position.qty <= 0:
            return False, ""
        px = position.current_price
        is_long = position.qty > 0

        # 1. Absolute stop / take-profit levels set at entry (honors ATR stops).
        if px > 0 and stop_price is not None:
            if (is_long and px <= stop_price) or (not is_long and px >= stop_price):
                return True, f"stop price hit (px={px:.2f} stop={stop_price:.2f})"
        if px > 0 and tp_price is not None:
            if (is_long and px >= tp_price) or (not is_long and px <= tp_price):
                return True, f"take-profit price hit (px={px:.2f} tp={tp_price:.2f})"

        # 2. Percentage fallbacks (used when no absolute level was supplied).
        pnl_pct = position.unrealized_pnl_pct
        if pnl_pct <= -self.stop_pct:
            return True, f"stop loss hit ({pnl_pct*100:.2f}%)"
        if pnl_pct >= self.tp_pct:
            return True, f"take profit hit ({pnl_pct*100:.2f}%)"

        # 3. Score reversal.
        if is_long and score < -self.cfg.get("exit_threshold", 0.10):
            return True, "signal reversed"

        # 4. Trailing stop.
        if trail_high is not None and trail_high > position.avg_entry_price:
            drawdown = (trail_high - position.current_price) / trail_high
            if drawdown >= self.trail_pct:
                return True, f"trailing stop ({drawdown*100:.2f}% off peak)"
        return False, ""
