"""Position sizing, stops, and kill switches.

Every order placed by the bot goes through RiskManager.check() first.
If `decision.approved` is False, the trade is blocked.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..brokers.base import BrokerBase, Position
from ..data.sectors import get_sector, is_market_etf
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
        self.atr_stop_mult = float(cfg.get("atr_stop_multiplier", 2.5))   # was hard-coded 2×
        self.atr_tp_mult   = float(cfg.get("atr_tp_multiplier",   3.0))
        self.shorting = bool(cfg.get("shorting_enabled", False))
        self.pdt_protect = bool(cfg.get("pdt_protection", True))
        self.kelly = float(cfg.get("kelly_fraction", 0.25))
        self.min_hold_minutes = int(cfg.get("min_hold_minutes", 25))
        self.stale_exit_minutes = int(cfg.get("stale_exit_minutes", 20))
        self.stale_exit_threshold_pct = float(cfg.get("stale_exit_threshold_pct", 0.003))
        self.max_positions_per_sector = int(cfg.get("max_positions_per_sector", 1))

        # Regime-based position-size multiplier (set by engine each cycle).
        # 1.0 = full sizing; 0.5 = half sizing in bear / high-vol regimes.
        self.regime_size_mult: float = 1.0

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

        # Sector concentration limit — skip if we already hold a position in
        # the same sector (prevents correlated losses on sector-wide moves).
        if self.max_positions_per_sector > 0 and not is_market_etf(symbol):
            sym_sector = get_sector(symbol)
            if sym_sector != "Unknown":
                sector_count = sum(
                    1 for held in positions
                    if held != symbol and get_sector(held) == sym_sector and not is_market_etf(held)
                )
                if sector_count >= self.max_positions_per_sector:
                    held_in_sector = [h for h in positions if get_sector(h) == sym_sector and not is_market_etf(h)]
                    return RiskDecision(
                        False,
                        f"sector limit ({sym_sector}): already holding {held_in_sector}",
                    )

        # PDT rule (under $25k cash account) — block new entry if it could trigger 4th day trade.
        if self.pdt_protect and equity < 25_000:
            recent = [d for d in self._day_trade_dates if (date.today() - d).days < 5]
            if len(recent) >= 3:
                return RiskDecision(False, "PDT-rule protection: 3 day trades in last 5 days on sub-$25k account")

        # Position size = fractional-Kelly × regime multiplier, capped by max_position_pct.
        # regime_size_mult < 1.0 reduces exposure in choppy / bear / high-vol markets.
        kelly_pct = max(0.0, min(1.0, abs(score) * confidence)) * self.kelly
        target_pct = min(kelly_pct, self.max_position_pct) * self.regime_size_mult
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

        # Stops — always respect the configured % floors so ATR can only tighten
        # stops (never widen them) and can never shrink the profit target below
        # the configured take_profit_pct (currently 4 %).  This ensures partial
        # exits at +1 % / +2.5 % always get a chance to fire before hard TP.
        pct_stop = price * (1 - self.stop_pct) if side == "buy" else price * (1 + self.stop_pct)
        pct_tp   = price * (1 + self.tp_pct)   if side == "buy" else price * (1 - self.tp_pct)

        if self.use_atr and atr and atr > 0:
            atr_stop = price - self.atr_stop_mult * atr if side == "buy" else price + self.atr_stop_mult * atr
            atr_tp   = price + self.atr_tp_mult   * atr if side == "buy" else price - self.atr_tp_mult   * atr
            if side == "buy":
                # Tighter stop = higher price (closer to entry); larger TP = higher price
                stop = max(atr_stop, pct_stop)
                tp   = max(atr_tp,   pct_tp)
            else:
                # Tighter stop = lower price; larger TP = lower price
                stop = min(atr_stop, pct_stop)
                tp   = min(atr_tp,   pct_tp)
        else:
            stop = pct_stop
            tp   = pct_tp

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
        held_since: Optional[datetime] = None,  # when the position was entered
    ) -> Tuple[bool, str]:
        """Return (should_exit, reason).

        Exit precedence:
          1. Hard stop / take-profit price levels (ATR-based) — fires immediately
          1b. Breakeven stop — once position gains breakeven_trigger_pct, the
              effective floor is raised to entry price so a winner can never
              become a full-sized loser
          2. Percentage fallbacks — fires immediately
          3. Signal reversal — suppressed when position is in any profit;
             fires after min_hold_minutes only when at breakeven or a loss
          4. Trailing stop — fires immediately once in profit
        """
        if position.qty <= 0:
            return False, ""
        px = position.current_price
        is_long = position.qty > 0
        pnl_pct = position.unrealized_pnl_pct
        breakeven_trigger = float(self.cfg.get("breakeven_trigger_pct", 0.004))  # 0.4%

        # 1. Absolute stop / take-profit levels set at entry (honors ATR stops).
        #    Breakeven adjustment: once up breakeven_trigger_pct, floor the stop
        #    at entry price so a winning position can never become a full loss.
        effective_stop = stop_price
        if (
            stop_price is not None
            and position.avg_entry_price > 0
            and pnl_pct >= breakeven_trigger
        ):
            breakeven_price = position.avg_entry_price
            if is_long:
                effective_stop = max(stop_price, breakeven_price)
            else:
                effective_stop = min(stop_price, breakeven_price)
            if effective_stop != stop_price:
                log.debug(
                    "Breakeven stop active for %s: stop raised %.2f → %.2f (entry)",
                    position.symbol, stop_price, effective_stop,
                )

        if px > 0 and effective_stop is not None:
            if (is_long and px <= effective_stop) or (not is_long and px >= effective_stop):
                label = "breakeven stop" if effective_stop != stop_price else "stop price"
                return True, f"{label} hit (px={px:.2f} stop={effective_stop:.2f})"
        if px > 0 and tp_price is not None:
            if (is_long and px >= tp_price) or (not is_long and px <= tp_price):
                return True, f"take-profit price hit (px={px:.2f} tp={tp_price:.2f})"

        # 2. Percentage fallbacks (used when no absolute level was supplied).
        #    Same breakeven logic: once up breakeven_trigger_pct, don't allow
        #    the % fallback stop to trigger below entry.
        effective_stop_pct = self.stop_pct
        if pnl_pct >= breakeven_trigger:
            effective_stop_pct = 0.0   # floor at breakeven — don't stop below entry
        if pnl_pct <= -effective_stop_pct and effective_stop_pct > 0:
            return True, f"stop loss hit ({pnl_pct*100:.2f}%)"
        if pnl_pct >= self.tp_pct:
            return True, f"take profit hit ({pnl_pct*100:.2f}%)"

        # 3. Score reversal — gated by minimum hold time AND profitability.
        #
        # Key rule: if the position is already in meaningful profit (>= 0.5%),
        # do NOT exit on signal reversal.  Let the trailing stop protect the
        # gains instead — it will exit once the price pulls back trail_pct%
        # from the peak, capturing most of the move.
        #
        # Exiting a profitable position on a fleeting signal reversal is the
        # "cut winners short" anti-pattern: it produces tiny wins while losses
        # (which hit the hard stop before the signal reverses) remain full-sized.
        if is_long and score < -self.cfg.get("exit_threshold", 0.10):
            if pnl_pct > 0:
                # Any unrealized profit → suppress signal reversal entirely.
                # Let the trailing stop protect gains instead of cutting at pennies.
                log.debug(
                    "Signal reversed for %s but position is in profit (%.2f%%) "
                    "— suppressing reversal exit, trailing stop will handle it.",
                    position.symbol, pnl_pct * 100,
                )
            elif held_since is None:
                return True, "signal reversed"
            else:
                held_minutes = (datetime.now() - held_since).total_seconds() / 60
                if held_minutes >= self.min_hold_minutes:
                    return True, f"signal reversed (held {held_minutes:.0f}m)"
                else:
                    log.debug(
                        "Signal reversed for %s but min hold not met "
                        "(%.0fm < %dm) — keeping position.",
                        position.symbol, held_minutes, self.min_hold_minutes,
                    )

        # 4. Trailing stop.
        if trail_high is not None and trail_high > position.avg_entry_price:
            drawdown = (trail_high - position.current_price) / trail_high
            if drawdown >= self.trail_pct:
                return True, f"trailing stop ({drawdown*100:.2f}% off peak)"

        # 5. Stale position exit — no meaningful movement after N minutes.
        #    Catches positions that just sit at entry going nowhere.  Hard
        #    stops / trailing-stop protect positions that are already moving;
        #    this exits the "dead money" cases to free capital for better setups.
        if held_since is not None and position.avg_entry_price > 0 and px > 0:
            held_minutes = (datetime.now() - held_since).total_seconds() / 60
            if held_minutes >= self.stale_exit_minutes:
                move_pct = abs(pnl_pct)
                if move_pct < self.stale_exit_threshold_pct:
                    return True, (
                        f"stale position ({held_minutes:.0f}m held, "
                        f"only {move_pct*100:.2f}% move)"
                    )

        return False, ""
