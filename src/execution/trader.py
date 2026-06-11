"""Order placement + open-position lifecycle management."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..brokers.base import BrokerBase, Order, OrderSide, OrderType, Position
from ..risk.performance_tracker import SymbolPerformanceTracker
from ..risk.risk_manager import RiskManager
from ..signals.aggregator import AggregatedDecision
from ..utils.logger import get_logger
from ..utils.notifications import notify, notify_order_entry, notify_order_exit
from ..utils.trade_history import TradeHistory

log = get_logger(__name__)

# Per-position state (stops, partial-exit progress, regime params) persisted
# across restarts so ATR stops aren't silently replaced by the wider % fallbacks.
_STATE_PATH = Path(__file__).resolve().parents[2] / "trader_state.json"


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

        # Restore entry times from transactions.json so min_hold_minutes is
        # correctly applied to positions that were open before a restart.
        if trade_history is not None:
            recovered = trade_history.get_latest_entry_times()
            if recovered:
                self._entry_time.update(recovered)
                log.info(
                    "Trader: restored entry times for %d symbol(s) from transactions.json: %s",
                    len(recovered),
                    {s: t.strftime("%Y-%m-%d %H:%M") for s, t in recovered.items()},
                )

        # Restore stops / partial-exit progress / regime params from the last run.
        self._load_state()

    # ---------- state persistence ----------
    def _save_state(self) -> None:
        """Persist per-position stops, partial progress, and regime params."""
        try:
            state = {}
            for sym, (stop, tp) in self._stops.items():
                state[sym] = {
                    "stop": stop,
                    "tp": tp,
                    "entry_qty": self._entry_qty.get(sym),
                    "partial_exits": self._partial_exits.get(sym, 0),
                    "regime_params": self._regime_params.get(sym),
                    "entry_time": (
                        self._entry_time[sym].isoformat()
                        if sym in self._entry_time else None
                    ),
                }
            _STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("Could not save trader state: %s", e)

    def _load_state(self) -> None:
        """Restore per-position state saved by a previous run.

        Stale entries (positions closed while the bot was down) are harmless:
        stops are only consulted while the broker reports the position, and a
        fresh entry overwrites every per-symbol dict.
        """
        if not _STATE_PATH.exists():
            return
        try:
            state = json.loads(_STATE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("Could not load trader state: %s", e)
            return
        for sym, s in state.items():
            if s.get("stop") is not None or s.get("tp") is not None:
                self._stops[sym] = (s.get("stop"), s.get("tp"))
            if s.get("entry_qty"):
                self._entry_qty[sym] = float(s["entry_qty"])
            self._partial_exits[sym] = int(s.get("partial_exits") or 0)
            if s.get("regime_params"):
                self._regime_params[sym] = s["regime_params"]
            if sym not in self._entry_time and s.get("entry_time"):
                try:
                    self._entry_time[sym] = datetime.fromisoformat(s["entry_time"])
                except ValueError:
                    pass
        if state:
            log.info(
                "Trader: restored stops/partial state for %d symbol(s): %s",
                len(state), list(state),
            )

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
        side   = "buy" if dec.score > 0 else "sell"
        is_buy = side == "buy"
        price  = self.broker.get_last_price(dec.symbol)
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
                self._close_position(existing, reason, exit_scores=dec.raw_scores)
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
                    # Weak-trend zone (min_entry_adx ≤ ADX < weak_max):
                    # require higher conviction score AND cap at 1 loss/day.
                    if abs(dec.score) < self._weak_trend_threshold:
                        log.info(
                            "SKIP %s — weak trend (ADX=%.1f), score %.3f < %.3f required.",
                            dec.symbol, adx_val, abs(dec.score), self._weak_trend_threshold,
                        )
                        return
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
            self._save_state()
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
                min_confidence=dec.min_confidence,
                agreement_ok=dec.agreement_ok,
                components=dec.components,
                raw_scores=dec.raw_scores,
                stop_loss=rd.stop_loss,
                take_profit=rd.take_profit,
                regime_params=regime_params,
                enter_threshold=dec.enter_long if is_buy else dec.enter_short,
                position_size_info=rd.position_info,
                channels=self.notify_channels,
            )

    # ---------- exits ----------
    def manage_open_positions(self, aggregated: Dict[str, AggregatedDecision]) -> None:
        """Sweep current positions; fire partial exits, then full exits."""
        self.recent_stop_losses.clear()   # reset each cycle; engine reads after this call
        positions = self.broker.get_stock_positions()   # crypto excluded at broker level
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
                d = aggregated.get(sym)
                self._close_position(pos, reason, exit_scores=d.raw_scores if d else None)

    def flatten_all(self, reason: str = "end-of-day flatten") -> None:
        for pos in self.broker.get_stock_positions().values():
            self._close_position(pos, reason)

    def flatten_eod_eligible(self, reason: str = "eod_flatten") -> None:
        """Close only stock positions that should be flattened at EOD.

        TRENDING positions (eod_flatten=False) are skipped — they are
        allowed to run overnight and exit via stop/TP/signal on a future cycle.
        CHOPPY and NEUTRAL positions (eod_flatten=True, the default) are closed.
        Crypto positions never appear here — get_stock_positions() excludes them.
        """
        positions = self.broker.get_stock_positions()
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
                self._save_state()
                return True

        elif partial_level == 1 and pnl_pct >= pp2_pct:
            sell_qty = min(sell_qty, int(pos.qty))
            if sell_qty >= 1:
                self._execute_partial_exit(sym, sell_qty, "partial_profit_2", pos.avg_entry_price)
                self._partial_exits[sym] = 2
                self._save_state()
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
            partial_num = self._partial_exits.get(symbol, 0) + 1
            notify_order_exit(
                symbol=symbol,
                side="sell",
                qty=qty,
                pnl=realized_pnl,
                pnl_pct=pnl_pct,
                reason=reason,
                entry_price=entry_price,
                exit_price=exit_price,
                held_since=self._entry_time.get(symbol),
                partial_num=partial_num,
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
    def _close_position(self, position: Position, reason: str, exit_scores: Optional[dict] = None) -> None:
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
            if position.qty < 0:  # short: profit when exit is below entry
                realized_pnl_pct = -realized_pnl_pct
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
                # Reasons come from RiskManager.check_exit as free text:
                # "stop price hit (…)", "stop loss hit (…)", "breakeven stop hit (…)".
                if ("stop price hit" in reason or "stop loss hit" in reason
                        or "breakeven stop" in reason):
                    self.recent_stop_losses.add(position.symbol)
                log.debug(
                    "Loss streak for %s: %d consecutive.",
                    position.symbol, self._symbol_daily_losses[position.symbol],
                )
            else:
                # Any profitable exit (including partial-assisted rides) resets streak.
                self._symbol_daily_losses[position.symbol] = 0

            # ── Capture hold duration before cleanup ───────────────────────
            held_since = self._entry_time.get(position.symbol)

            # ── Clean up per-symbol state ──────────────────────────────────
            self._trail_high.pop(position.symbol, None)
            self._stops.pop(position.symbol, None)
            self._entry_time.pop(position.symbol, None)
            self._partial_exits.pop(position.symbol, None)
            self._entry_qty.pop(position.symbol, None)
            self._regime_params.pop(position.symbol, None)
            self._save_state()

            notify_order_exit(
                symbol=position.symbol,
                side=side.value,
                qty=abs(position.qty),
                pnl=realized_pnl,
                pnl_pct=realized_pnl_pct,
                reason=reason,
                entry_price=position.avg_entry_price,
                exit_price=exit_price,
                held_since=held_since,
                exit_scores=exit_scores,
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
        """Update the trailing watermark for stop calculation.

        Long  (qty > 0): track the MAXIMUM price — trail fires on a drop.
        Short (qty < 0): track the MINIMUM price — trail fires on a rally.
        """
        cur = pos.current_price or self.broker.get_last_price(pos.symbol)
        if cur <= 0:
            return
        prev = self._trail_high.get(pos.symbol, pos.avg_entry_price)
        if pos.qty >= 0:
            # Long: keep the highest price seen
            if cur > prev:
                self._trail_high[pos.symbol] = cur
        else:
            # Short: keep the lowest price seen (trough)
            if cur < prev:
                self._trail_high[pos.symbol] = cur
