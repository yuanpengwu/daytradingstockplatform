"""Position sizing, stops, and kill switches.

Every order placed by the bot goes through RiskManager.check() first.
If `decision.approved` is False, the trade is blocked.

Dead-signal position-size scaling
-----------------------------------
When signal sources are offline (e.g. Gemini → ML dead, EDGAR → fundamental
dead), the aggregated ``confidence`` metric is structurally suppressed —
those sources contribute nothing to the numerator, so even genuinely strong
active signals produce low confidence.  That causes Kelly-sized positions to
shrink even when the available information is good.

To compensate, the engine calls ``update_dead_signal_ratio()`` each cycle
with the active-weight ratio from SignalAggregator (same ratio used for
threshold scaling).  RiskManager scales up the *effective* kelly fraction
and max_position_pct inversely:

    eff_kelly   = min(kelly   / ratio, KELLY_SCALE_CAP)    e.g. 0.25/0.65 = 0.385
    eff_max_pos = min(max_pos / ratio, MAX_POS_SCALE_CAP)  e.g. 0.10/0.65 = 0.154

This restores expected dollar exposure as if all signals were working.
A safety ceiling prevents over-leveraging even when most sources are down:
    KELLY_SCALE_CAP   = 0.75   (3× the base 0.25)
    MAX_POS_SCALE_CAP = 0.20   (2× the base 10%)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..brokers.base import BrokerBase, Position
from ..data.sectors import get_sector, is_market_etf
from ..utils.logger import get_logger

log = get_logger(__name__)

# Safety ceilings for dead-signal position-size upscaling.
# Even if all signals are offline (ratio floored at 0.50 by the aggregator),
# the effective parameters never exceed these values.
_KELLY_SCALE_CAP: float   = 0.75   # base 0.25 × 3× max scaling
_MAX_POS_SCALE_CAP: float = 0.20   # base 10% × 2× max scaling


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
        self.max_hold_minutes = int(cfg.get("max_hold_minutes", 180))
        self.stale_exit_minutes = int(cfg.get("stale_exit_minutes", 20))
        self.stale_exit_threshold_pct = float(cfg.get("stale_exit_threshold_pct", 0.003))
        self.max_positions_per_sector = int(cfg.get("max_positions_per_sector", 1))

        # Time-decaying take-profit: TP target drops linearly after
        # time_decay_tp_start_minutes, reaching time_decay_tp_min_pct
        # at time_decay_tp_full_minutes.  Only fires if the position is
        # profitable and below the original TP.
        self._decay_tp_enabled   = bool(cfg.get("time_decay_tp_enabled", True))
        self._decay_tp_start_min = float(cfg.get("time_decay_tp_start_minutes", 45))
        self._decay_tp_full_min  = float(cfg.get("time_decay_tp_full_minutes", 120))
        self._decay_tp_min_pct   = float(cfg.get("time_decay_tp_min_pct", 0.015))

        # Regime-based position-size multiplier (set by engine each cycle).
        # 1.0 = full sizing; 0.5 = half sizing in bear / high-vol regimes.
        self.regime_size_mult: float = 1.0

        # ── Dead-signal position-size scaling ──────────────────────────────────
        # Engine calls update_dead_signal_ratio() each cycle after aggregation.
        # 1.0 = all sources live (no scaling).
        # 0.65 = ML + sentiment + fundamental dead (35 % weight offline).
        # 0.50 = aggregator floor (maximum upscaling: 2×, still within caps).
        self._signal_ratio: float      = 1.0   # current active-weight ratio
        self._last_signal_ratio: float = 1.0   # previous value — for change-detection log

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

    # ---------- dead-signal position-size scaling ----------
    def update_dead_signal_ratio(self, ratio: float) -> None:
        """Receive the active-weight ratio from SignalAggregator and log any change.

        Called by the engine each cycle immediately after ``agg.aggregate()``.

        The ratio equals ``active_weight / total_weight`` (already floored at
        0.50 by the aggregator).  ``check_entry`` reads ``self._signal_ratio``
        when it computes effective kelly / max_pos — no further work is needed
        here beyond storing the value and emitting a one-time log.

        Example (ML + sentiment + fundamental dead, ratio = 0.65):
            eff_kelly   = min(0.25 / 0.65, 0.75) = 0.385   (was 0.250)
            eff_max_pos = min(0.10 / 0.65, 0.20) = 15.4%   (was 10.0%)

        Args:
            ratio: active_weight / total_weight in [0.50, 1.0].
        """
        self._signal_ratio = max(1e-6, min(1.0, float(ratio)))

        # Only log when the ratio changes meaningfully — avoids every-cycle spam.
        if abs(self._signal_ratio - self._last_signal_ratio) > 1e-4:
            eff_kelly   = min(self.kelly            / self._signal_ratio, _KELLY_SCALE_CAP)
            eff_max_pos = min(self.max_position_pct / self._signal_ratio, _MAX_POS_SCALE_CAP)

            if self._signal_ratio < 1.0 - 1e-6:
                log.warning(
                    "Position scaling active: kelly=%.3f (base %.3f), "
                    "max_pos=%.1f%% (base %.1f%%) — active weight %.0f%%",
                    eff_kelly,   self.kelly,
                    eff_max_pos * 100, self.max_position_pct * 100,
                    self._signal_ratio * 100,
                )
            else:
                log.info(
                    "Position scaling lifted — all signal sources active. "
                    "Restored: kelly=%.3f, max_pos=%.1f%%",
                    self.kelly, self.max_position_pct * 100,
                )

            self._last_signal_ratio = self._signal_ratio

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
        #
        # Dead-signal adjustment: when _signal_ratio < 1.0, scale the effective
        # kelly and max_pos UP by 1/ratio so that structurally-suppressed
        # confidence (caused by offline ML/sentiment/fundamental) does not
        # over-shrink dollar exposure.  Hard ceilings prevent runaway scaling.
        eff_kelly   = min(self.kelly            / self._signal_ratio, _KELLY_SCALE_CAP)
        eff_max_pos = min(self.max_position_pct / self._signal_ratio, _MAX_POS_SCALE_CAP)
        kelly_pct   = max(0.0, min(1.0, abs(score) * confidence)) * eff_kelly
        target_pct  = min(kelly_pct, eff_max_pos) * self.regime_size_mult
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

    # ---------- helpers ----------
    def _time_decayed_tp(self, held_minutes: float) -> float:
        """Return the current effective take-profit threshold for a position
        held for *held_minutes*.

        Before decay starts  → original tp_pct (e.g. 6%)
        Between start and full decay window → linearly interpolated
        After full decay     → min_pct (e.g. 1.5%)

        Example with defaults (start=45, full=120, min=1.5%, base=6%):
            45 min  → 6.0%   (full target, no decay)
            70 min  → 4.2%   (40% decayed)
            90 min  → 3.0%   (60% decayed)
           120 min  → 1.5%   (fully decayed to minimum)
        """
        if held_minutes <= self._decay_tp_start_min:
            return self.tp_pct
        decay_window = self._decay_tp_full_min - self._decay_tp_start_min
        if decay_window <= 0:
            return self._decay_tp_min_pct
        t = min((held_minutes - self._decay_tp_start_min) / decay_window, 1.0)
        return self.tp_pct + t * (self._decay_tp_min_pct - self.tp_pct)

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

        # 1c. Time-decaying take-profit — fires ONLY on profitable positions and
        #     only when the decayed target is below the original TP (i.e. position
        #     has been open long enough that we should accept a lower gain).
        #     We never exit below the minimum or at a loss.
        if self._decay_tp_enabled and held_since is not None and pnl_pct > 0:
            held_minutes = (datetime.now() - held_since).total_seconds() / 60
            decayed_tp = self._time_decayed_tp(held_minutes)
            if decayed_tp < self.tp_pct and pnl_pct >= decayed_tp:
                return True, (
                    f"time-decayed TP hit ({pnl_pct*100:.2f}% >= "
                    f"{decayed_tp*100:.1f}% target at {held_minutes:.0f}m)"
                )

        # 1d. Max hold time — hard ceiling regardless of profit/loss.
        #     Prevents tying up capital all day in a position going nowhere and
        #     ensures all intraday positions close well before market close.
        if held_since is not None:
            held_minutes = (datetime.now() - held_since).total_seconds() / 60
            if held_minutes >= self.max_hold_minutes:
                return True, (
                    f"max hold time ({self.max_hold_minutes}m) reached "
                    f"(held {held_minutes:.0f}m, pnl={pnl_pct*100:.2f}%)"
                )

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
