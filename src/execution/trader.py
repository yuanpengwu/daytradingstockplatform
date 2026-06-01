"""Order placement + open-position lifecycle management."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ..brokers.base import BrokerBase, Order, OrderSide, OrderType, Position
from ..risk.performance_tracker import SymbolPerformanceTracker
from ..risk.risk_manager import RiskManager
from ..signals.aggregator import AggregatedDecision
from ..utils.logger import get_logger
from ..utils.notifications import notify, notify_order_entry, notify_order_exit
from ..utils.trade_history import TradeHistory

log = get_logger(__name__)


class Trader:
    """Glue between aggregated signals, risk checks, and broker calls."""

    def __init__(
        self,
        broker: BrokerBase,
        risk: RiskManager,
        notify_channels=("console",),
        trade_history: Optional[TradeHistory] = None,
        persistence_bars: int = 2,
    ):
        self.broker = broker
        self.risk = risk
        self.notify_channels = list(notify_channels)
        self.trade_history = trade_history

        # ── Position tracking (carry across cycles) ────────────────────────
        # Peak price since entry per symbol (for trailing stops).
        self._trail_high: Dict[str, float] = {}
        # Absolute stop / take-profit levels set at entry per symbol.
        self._stops: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
        # When each position was entered (for minimum hold time + stale exit).
        self._entry_time: Dict[str, datetime] = {}

        # ── Strategy improvement state ─────────────────────────────────────
        rcfg = risk.cfg

        # 1. Signal persistence filter
        self._persistence_bars: int = persistence_bars
        self._signal_history: Dict[str, List[float]] = {}   # last N scores per sym

        # 2. Daily trend filter (only enter long if price > session open)
        self._use_trend_filter: bool = bool(rcfg.get("use_daily_trend_filter", True))
        self._day_open: Dict[str, float] = {}

        # 3. Partial profit taking
        self._partial_1_pct: float = float(rcfg.get("partial_profit_1_pct", 0.010))
        self._partial_2_pct: float = float(rcfg.get("partial_profit_2_pct", 0.025))
        self._partial_exits: Dict[str, int] = {}   # 0 / 1 / 2 partials taken
        self._entry_qty: Dict[str, float] = {}     # original qty at entry for sizing

        # Per-position regime params (set at entry by the engine's regime detector).
        # Keys: pp1_pct, pp2_pct, eod_flatten (bool).
        # TRENDING positions: pp1/pp2 = 9999 (never fire), eod_flatten = False.
        # CHOPPY/NEUTRAL:     pp1/pp2 from config,         eod_flatten = True.
        self._regime_params: Dict[str, dict] = {}

        # 4. Per-symbol daily loss streak filter
        self._max_daily_losses: int = int(rcfg.get("max_symbol_daily_losses", 2))
        self._symbol_daily_losses: Dict[str, int] = {}

        # 4b. Dynamic exclusion: symbols with <33% win rate for 3 consecutive
        #     trading days are removed from the universe until they recover.
        self._perf_tracker = SymbolPerformanceTracker(
            min_win_rate=float(rcfg.get("dynamic_exclusion_win_rate", 0.33)),
            streak_limit=int(rcfg.get("dynamic_exclusion_streak_days", 3)),
        )

        # 5. ADX entry filters — three-tier treatment by trend strength:
        #      ADX < min_entry_adx              → hard block (no trend at all)
        #      min_entry_adx ≤ ADX < weak_max   → weak-trend zone: higher score
        #                                          required + tighter daily loss cap
        #      ADX ≥ weak_max                   → full trend, normal rules apply
        self._min_entry_adx: float = float(rcfg.get("min_entry_adx", 0.0))
        self._weak_trend_adx_max: float = float(rcfg.get("weak_trend_adx_max", 25.0))
        self._weak_trend_threshold: float = float(rcfg.get("weak_trend_entry_threshold", 0.55))
        self._weak_trend_max_losses: int = int(rcfg.get("weak_trend_max_daily_losses", 1))

        # 6. Next-day cooloff — symbols that stopped out this cycle
        #    Engine reads this after manage_open_positions and registers bans.
        self.recent_stop_losses: set = set()

    # ---------- daily reset ----------
    def begin_day(self, day_str: Optional[str] = None) -> None:
        """Reset per-day state. Must be called by the engine at day start.

        *day_str* ('YYYY-MM-DD') is the date of the day that just finished —
        used to trigger performance-tracker end-of-day evaluation before the
        new day's signals are processed.
        """
        if day_str:
            self._perf_tracker.end_of_day(day_str)
            st = self._perf_tracker.status()
            if st["excluded"] or st["bad_streak"]:
                log.warning(
                    "PerfTracker EOD %s | excluded=%s | bad_streak=%s",
                    day_str, st["excluded"], st["bad_streak"],
                )
        self._day_open.clear()
        self._signal_history.clear()
        self._symbol_daily_losses.clear()
        log.info("Trader: daily state reset (trend filter, signal history, loss streaks).")

    def update_day_open(self, symbol: str, open_price: float) -> None:
        """Record today's session-open price for a symbol (daily trend filter).

        Call once per day, as early as possible (first bar of today's session).
        Subsequent calls for the same symbol are ignored so the open is stable.
        """
        if symbol not in self._day_open and open_price > 0:
            self._day_open[symbol] = open_price
            log.debug("Day-open set %s = %.2f", symbol, open_price)

    # ---------- entries ----------
    def handle_decision(
        self,
        dec: AggregatedDecision,
        atr: Optional[float],
        regime_params: Optional[dict] = None,
    ) -> None:
        side = "buy" if dec.score > 0 else "sell"
        price = self.broker.get_last_price(dec.symbol)
        if price <= 0:
            log.warning("Skipping %s — no price.", dec.symbol)
            return

        existing = self.broker.get_position(dec.symbol)
        if existing:
            # Already hold this — update trail and check for exit.
            self._update_trailing_high(existing)
            stop_price, tp_price = self._stops.get(dec.symbol, (None, None))
            held_since = self._entry_time.get(dec.symbol)
            should_exit, reason = self.risk.check_exit(
                existing,
                score=dec.score,
                trail_high=self._trail_high.get(dec.symbol),
                stop_price=stop_price,
                tp_price=tp_price,
                held_since=held_since,
            )
            if should_exit:
                self._close_position(existing, reason)
            return

        # ── Filter 1: dynamic exclusion + per-symbol daily loss cap ──────────
        if self._perf_tracker.is_excluded(dec.symbol):
            log.info("SKIP %s — dynamically excluded (poor multi-day win rate).", dec.symbol)
            return
        eff_cap = self._perf_tracker.daily_loss_cap(dec.symbol, self._max_daily_losses)
        streak = self._symbol_daily_losses.get(dec.symbol, 0)
        if streak >= eff_cap:
            log.info(
                "SKIP %s — loss streak %d >= effective cap %d (today).",
                dec.symbol, streak, eff_cap,
            )
            return

        # ── Filter 2: signal persistence ──────────────────────────────────
        hist = self._signal_history.setdefault(dec.symbol, [])
        hist.append(dec.score)
        if len(hist) > self._persistence_bars:
            hist.pop(0)
        # Need N bars all on the same side AND above a minimum magnitude threshold.
        # Checking only sign lets a decaying signal (e.g. +0.50 → +0.12) pass
        # the filter and enter right before it reverses — the most common loss pattern.
        # Require each bar's score to be at least 70 % of the entry threshold so
        # the signal has been consistently strong, not merely positive.
        if len(hist) < self._persistence_bars:
            log.info(
                "SKIP %s — persistence not met yet (%d/%d bars; need %d consecutive).",
                dec.symbol, len(hist), self._persistence_bars, self._persistence_bars,
            )
            return
        same_side = all((s > 0) == (dec.score > 0) for s in hist)
        if not same_side:
            log.info(
                "SKIP %s — signal flipped direction in last %d bars (not persistent).",
                dec.symbol, self._persistence_bars,
            )
            return
        # Use the correct threshold for the direction of the trade:
        # longs compare against enter_long, shorts against abs(enter_short).
        ref_thresh = dec.enter_long if dec.score > 0 else abs(dec.enter_short)
        min_mag = ref_thresh * 0.70   # e.g. 0.45 × 0.70 = 0.315 for longs
        strong_enough = all(abs(s) >= min_mag for s in hist)
        if not strong_enough:
            weakest = min(abs(s) for s in hist)
            log.info(
                "SKIP %s — score decaying (min=%.3f < %.3f threshold); "
                "signal may be reversing.",
                dec.symbol, weakest, min_mag,
            )
            return

        # ── Filter 3: daily trend filter (longs only) ─────────────────────
        if self._use_trend_filter and side == "buy":
            day_open = self._day_open.get(dec.symbol)
            if day_open is not None and price < day_open:
                log.info(
                    "SKIP %s long — price %.2f below session open %.2f (daily trend filter).",
                    dec.symbol, price, day_open,
                )
                return

        # ── Filter 4: ADX-tiered entry gate ──────────────────────────────
        # Three zones based on the symbol's current ADX:
        #   ADX < min_entry_adx            → hard block (directionless noise)
        #   min_entry_adx ≤ ADX < weak_max → weak-trend zone: require higher
        #                                    score AND cap at 1 loss/day
        #   ADX ≥ weak_max                 → full trend, normal rules apply
        if self._min_entry_adx > 0 and regime_params is not None:
            adx_val = regime_params.get("adx")
            if adx_val is not None and not (adx_val != adx_val):  # not NaN
                if adx_val < self._min_entry_adx:
                    log.info(
                        "SKIP %s — ADX %.1f below min_entry_adx %.1f (no trend).",
                        dec.symbol, adx_val, self._min_entry_adx,
                    )
                    return
                if adx_val < self._weak_trend_adx_max:
                    # Weak-trend zone — require higher conviction score
                    if abs(dec.score) < self._weak_trend_threshold:
                        log.info(
                            "SKIP %s — weak trend (ADX=%.1f), score %.3f < %.3f required.",
                            dec.symbol, adx_val, abs(dec.score), self._weak_trend_threshold,
                        )
                        return
                    # Weak-trend zone — tighter daily loss cap (B+C combined)
                    weak_losses = self._symbol_daily_losses.get(dec.symbol, 0)
                    if weak_losses >= self._weak_trend_max_losses:
                        log.info(
                            "SKIP %s — weak trend (ADX=%.1f), daily loss cap %d reached.",
                            dec.symbol, adx_val, self._weak_trend_max_losses,
                        )
                        return

        # ── No position — consider entry ───────────────────────────────────
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
            log.info("SKIP %s — risk manager blocked entry: %s", dec.symbol, rd.reason)
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
            fill_price = result.filled_avg_price or price
            filled_qty = result.filled_qty or rd.qty
            self.risk.record_day_trade()
            self._trail_high[dec.symbol] = fill_price
            self._stops[dec.symbol] = (rd.stop_loss, rd.take_profit)
            self._entry_time[dec.symbol] = datetime.now()
            self._entry_qty[dec.symbol] = filled_qty
            self._partial_exits[dec.symbol] = 0
            # Store per-position regime params supplied by the engine.
            if regime_params:
                self._regime_params[dec.symbol] = regime_params
                log.info(
                    "ENTRY %s | regime=%s pp1=%.4f eod_flatten=%s",
                    dec.symbol,
                    regime_params.get("regime", "?"),
                    regime_params.get("pp1_pct", self._partial_1_pct),
                    regime_params.get("eod_flatten", True),
                )
            else:
                self._regime_params.pop(dec.symbol, None)
            if self.trade_history is not None:
                self.trade_history.log_entry(
                    symbol=dec.symbol,
                    side=side,
                    qty=filled_qty,
                    price=fill_price,
                    score=dec.score,
                    confidence=dec.confidence,
                )
            notify_order_entry(
                symbol=dec.symbol,
                side=side,
                qty=filled_qty,
                price=fill_price,
                agg_score=dec.score,
                confidence=dec.confidence,
                components=dec.components,
                stop_loss=rd.stop_loss,
                take_profit=rd.take_profit,
                channels=self.notify_channels,
            )

    # ---------- exits ----------
    def manage_open_positions(self, aggregated: Dict[str, AggregatedDecision]) -> None:
        """Sweep current positions; fire partial exits, then full exits."""
        self.recent_stop_losses.clear()   # reset each cycle; engine reads after this call
        positions = self.broker.get_positions()
        for sym, pos in positions.items():
            self._update_trailing_high(pos)

            # Partial profit exits — if one fires, skip full-exit this cycle
            # (next cycle will work with the updated reduced position).
            if self._check_partial_exits(sym, pos):
                continue

            score = aggregated.get(sym).score if sym in aggregated else 0.0
            stop_price, tp_price = self._stops.get(sym, (None, None))
            held_since = self._entry_time.get(sym)
            should_exit, reason = self.risk.check_exit(
                pos,
                score=score,
                trail_high=self._trail_high.get(sym),
                stop_price=stop_price,
                tp_price=tp_price,
                held_since=held_since,
            )
            if should_exit:
                self._close_position(pos, reason)

    def flatten_all(self, reason: str = "end-of-day flatten") -> None:
        for sym, pos in self.broker.get_positions().items():
            self._close_position(pos, reason)

    def flatten_eod_eligible(self, reason: str = "eod_flatten") -> None:
        """Close only positions that should be flattened at EOD.

        TRENDING positions (eod_flatten=False) are skipped — they are
        allowed to run overnight and exit via stop/TP/signal on a future cycle.
        CHOPPY and NEUTRAL positions (eod_flatten=True, the default) are closed.
        """
        positions = self.broker.get_positions()
        for sym, pos in positions.items():
            rp = self._regime_params.get(sym, {})
            if not rp.get("eod_flatten", True):
                log.info(
                    "EOD skip %s — TRENDING position, letting it run overnight.",
                    sym,
                )
                continue
            self._close_position(pos, reason)

    # ---------- partial exits ----------
    def _check_partial_exits(self, sym: str, pos: Position) -> bool:
        """Check and execute partial profit targets.

        Returns True if a partial exit was executed (caller should skip full
        exit for this cycle so the broker position is fresh next cycle).
        """
        if pos.avg_entry_price <= 0 or pos.qty <= 0:
            return False
        pnl_pct = (pos.current_price - pos.avg_entry_price) / pos.avg_entry_price
        partial_level = self._partial_exits.get(sym, 0)

        if partial_level >= 2:
            return False  # both partials already taken

        # Use per-position thresholds if set (regime routing), else global defaults.
        rp = self._regime_params.get(sym, {})
        pp1_pct = rp.get("pp1_pct", self._partial_1_pct)
        pp2_pct = rp.get("pp2_pct", self._partial_2_pct)

        orig_qty = self._entry_qty.get(sym, pos.qty)
        sell_qty = max(1.0, round(orig_qty * 0.33))

        if partial_level == 0 and pnl_pct >= pp1_pct:
            # Don't sell more than we actually hold.
            sell_qty = min(sell_qty, int(pos.qty))
            if sell_qty >= 1:
                self._execute_partial_exit(sym, sell_qty, "partial_profit_1", pos.avg_entry_price)
                self._partial_exits[sym] = 1
                return True

        elif partial_level == 1 and pnl_pct >= pp2_pct:
            sell_qty = min(sell_qty, int(pos.qty))
            if sell_qty >= 1:
                self._execute_partial_exit(sym, sell_qty, "partial_profit_2", pos.avg_entry_price)
                self._partial_exits[sym] = 2
                return True

        return False

    def _execute_partial_exit(
        self,
        symbol: str,
        qty: float,
        reason: str,
        entry_price: float,
    ) -> None:
        """Submit a partial sell order and notify."""
        order = Order(
            symbol=symbol,
            side=OrderSide.SELL,
            qty=qty,
            type=OrderType.MARKET,
        )
        result = self.broker.submit_order(order)
        if result.status.value == "filled":
            exit_price = result.filled_avg_price or entry_price
            realized_pnl = (exit_price - entry_price) * qty
            pnl_pct = (exit_price - entry_price) / entry_price if entry_price else 0.0
            log.info(
                "Partial exit %s | qty=%g reason=%s pnl=%.2f%%",
                symbol, qty, reason, pnl_pct * 100,
            )
            notify_order_exit(
                symbol=symbol,
                side="sell",
                qty=qty,
                pnl=realized_pnl,
                pnl_pct=pnl_pct,
                reason=reason,
                channels=self.notify_channels,
            )
            if self.trade_history is not None:
                self.trade_history.log_exit(
                    symbol=symbol,
                    side="sell",
                    qty=qty,
                    price=exit_price,
                    entry_price=entry_price,
                    pnl=realized_pnl,
                    pnl_pct=pnl_pct,
                    score=0.0,
                    confidence=0.0,
                    reason=reason,
                )
        else:
            log.warning("Partial exit %s qty=%g did not fill (status=%s).", symbol, qty, result.status)

    # ---------- full close ----------
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
            exit_price = result.filled_avg_price or position.current_price
            realized_pnl = (exit_price - position.avg_entry_price) * abs(position.qty)
            if position.qty < 0:  # short
                realized_pnl = -realized_pnl
            realized_pnl_pct = (
                (exit_price - position.avg_entry_price) / position.avg_entry_price
                if position.avg_entry_price else 0.0
            )
            self.risk.record_day_trade()

            # ── Per-symbol loss streak tracking + performance recorder ────────
            today_str = datetime.now().strftime("%Y-%m-%d")
            is_win    = realized_pnl_pct >= 0
            self._perf_tracker.record_trade(position.symbol, today_str, is_win)

            if realized_pnl_pct < 0:
                self._symbol_daily_losses[position.symbol] = (
                    self._symbol_daily_losses.get(position.symbol, 0) + 1
                )
                # Flag stop-loss exits for next-day cooloff (engine reads this).
                if "stop_loss" in reason or "breakeven_stop" in reason:
                    self.recent_stop_losses.add(position.symbol)
                log.debug(
                    "Loss streak for %s: %d consecutive.",
                    position.symbol, self._symbol_daily_losses[position.symbol],
                )
            else:
                # Any profitable exit (including partial-assisted rides) resets streak.
                self._symbol_daily_losses[position.symbol] = 0

            # ── Clean up per-symbol state ──────────────────────────────────
            self._trail_high.pop(position.symbol, None)
            self._stops.pop(position.symbol, None)
            self._entry_time.pop(position.symbol, None)
            self._partial_exits.pop(position.symbol, None)
            self._entry_qty.pop(position.symbol, None)
            self._regime_params.pop(position.symbol, None)

            notify_order_exit(
                symbol=position.symbol,
                side=side.value,
                qty=abs(position.qty),
                pnl=realized_pnl,
                pnl_pct=realized_pnl_pct,
                reason=reason,
                channels=self.notify_channels,
            )
            if self.trade_history is not None:
                self.trade_history.log_exit(
                    symbol=position.symbol,
                    side=side.value,
                    qty=abs(position.qty),
                    price=exit_price,
                    entry_price=position.avg_entry_price,
                    pnl=realized_pnl,
                    pnl_pct=realized_pnl_pct,
                    score=0.0,
                    confidence=0.0,
                    reason=reason,
                )
                self.trade_history.record(
                    symbol=position.symbol,
                    side="buy" if position.qty > 0 else "sell",
                    qty=abs(position.qty),
                    entry_price=position.avg_entry_price,
                    exit_price=exit_price,
                    pnl=realized_pnl,
                    pnl_pct=realized_pnl_pct,
                    reason=reason,
                )

    def _update_trailing_high(self, pos: Position) -> None:
        cur = pos.current_price or self.broker.get_last_price(pos.symbol)
        if cur <= 0:
            return
        prev = self._trail_high.get(pos.symbol, pos.avg_entry_price)
        if cur > prev:
            self._trail_high[pos.symbol] = cur
