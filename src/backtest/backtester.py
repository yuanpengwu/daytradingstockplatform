"""Vectorized backtester for the technical + ML signal stack.

Supports both long and short positions with partial profit taking and
a trailing stop that activates after the first partial exit.

Short-selling mechanics
───────────────────────
Entry  : margin equal to the full notional is reserved from cash
         (cash -= entry_fill × qty).
Cover  : margin is returned plus/minus the realised P&L
         (cash += entry_price × qty_remaining + short_pnl).
MtM    : short position equity contribution =
         (2 × entry_price − current_price) × qty_remaining
         This ensures  equity = cash + pos_value  correctly tracks
         initial_cash + unrealised_pnl  for both sides.

Macro multiplier for shorts
───────────────────────────
A bearish macro (multiplier < 1) should AMPLIFY short signals (divide
the raw score) rather than dampen them.  The aggregator handles this
sign-aware inversion when market_multiplier != 1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..signals.aggregator import SignalAggregator
from ..signals.finrl_signal import FinRLSignal
from ..signals.ml_model import MLSignal
from ..signals.macro import MacroSignal
from ..signals.technical import TechnicalSignal
from ..risk.performance_tracker import SymbolPerformanceTracker
from ..utils.logger import get_logger

log = get_logger(__name__)


def _is_eod(ts, close_buffer_minutes: int = 30) -> bool:
    """True if *ts* falls within *close_buffer_minutes* of 16:00 ET."""
    try:
        import pytz
        ny = pytz.timezone("America/New_York")
        if ts.tzinfo is None:
            ts_ny = ts.replace(tzinfo=pytz.utc).astimezone(ny)
        else:
            ts_ny = ts.astimezone(ny)
        bar_mins   = ts_ny.hour * 60 + ts_ny.minute
        close_mins = 16 * 60
        return bar_mins >= (close_mins - close_buffer_minutes)
    except Exception:
        return False


def _bars_per_year(interval: str) -> float:
    days = 252
    table = {
        "1m": 390 * days, "2m": 195 * days, "5m": 78 * days,
        "15m": 26 * days, "30m": 13 * days,
        "1h": 6.5 * days, "60m": 6.5 * days, "1d": days,
    }
    return table.get(interval, 78 * days)


@dataclass
class Trade:
    symbol:      str
    entry_time:  datetime
    entry_price: float
    side:        str = "long"          # "long" | "short"
    exit_time:   Optional[datetime] = None
    exit_price:  Optional[float]    = None
    qty:         float = 0.0
    reason:      str   = ""

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        if self.side == "short":
            return (self.entry_price - self.exit_price) * self.qty
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        if self.side == "short":
            return (self.entry_price - self.exit_price) / self.entry_price
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestResult:
    starting_cash:    float
    ending_cash:      float
    trades:           List[Trade] = field(default_factory=list)
    equity_curve:     List[float] = field(default_factory=list)
    periods_per_year: float = 78 * 252

    @property
    def total_return(self) -> float:
        return (self.ending_cash - self.starting_cash) / self.starting_cash

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / self.num_trades if self.num_trades else 0.0

    @property
    def sharpe(self) -> float:
        if len(self.equity_curve) < 2:
            return 0.0
        rets = pd.Series(self.equity_curve).pct_change().dropna()
        if rets.std() == 0:
            return 0.0
        return float(rets.mean() / rets.std() * np.sqrt(self.periods_per_year))

    def summary(self) -> str:
        return (
            f"Trades: {self.num_trades} | Win-rate: {self.win_rate*100:.1f}% | "
            f"Return: {self.total_return*100:+.2f}% | Sharpe: {self.sharpe:.2f}"
        )


class Backtester:
    def __init__(
        self,
        config: dict,
        starting_cash: float = 10_000.0,
        slippage_bps:  float = 5.0,
        interval:      str   = "5m",
    ):
        self.cfg             = config
        self.starting_cash   = starting_cash
        self.slippage        = slippage_bps / 10_000.0
        self.interval        = interval
        self.periods_per_year = _bars_per_year(interval)

        self.tech  = TechnicalSignal(config["signals"]["technical"])
        self.ml    = MLSignal(config["signals"]["ml"])
        self.finrl = FinRLSignal(config["signals"].get("finrl", {}))
        self.macro = MacroSignal()
        self.agg   = SignalAggregator(
            weights       = config["signals"]["weights"],
            enter_long    = config["signals"].get("enter_long_threshold",  0.35),
            enter_short   = config["signals"].get("enter_short_threshold", -0.35),
            exit_thresh   = config["signals"].get("exit_threshold",         0.10),
            require_ml_finrl_agreement = config["signals"].get(
                "require_ml_finrl_agreement", False),
        )

        r  = config["risk"]
        sc = config["schedule"]

        self.stop_pct          = r["per_trade_stop_loss_pct"]
        self.tp_pct            = r["take_profit_pct"]
        self.kelly             = r.get("kelly_fraction",      0.25)
        self.max_position_pct  = r.get("max_position_pct",   0.10)
        self.shorting_enabled  = bool(r.get("shorting_enabled", False))

        # Churn controls
        self.min_hold_minutes  = float(r.get("min_hold_minutes", 30))
        self.max_hold_minutes  = float(r.get("max_hold_minutes", 180))

        # Daily loss cap
        self.max_symbol_daily_losses = int(r.get("max_symbol_daily_losses", 2))

        # Partial profit + trailing stop
        self.partial_profit_1_pct  = float(r.get("partial_profit_1_pct",  0.01))
        self.partial_profit_2_pct  = float(r.get("partial_profit_2_pct",  0.025))
        self.trailing_stop_pct     = float(r.get("trailing_stop_pct",     0.03))
        self.breakeven_trigger_pct = float(r.get("breakeven_trigger_pct", 0.022))

        # Dynamic exclusion
        self._perf_tracker = SymbolPerformanceTracker(
            min_win_rate       = float(r.get("dynamic_exclusion_win_rate",    0.33)),
            streak_limit       = int(  r.get("dynamic_exclusion_streak_days", 3)),
            min_trades_per_day = int(  r.get("dynamic_exclusion_min_trades",  3)),
        )

        self.no_entry_before_close_min = int(
            sc.get("no_entry_before_close_minutes", 30)
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _pos_value(pos: dict, current_price: float) -> float:
        """Mark-to-market equity contribution of one open position.

        Long  :  qty_remaining × current_price
        Short :  (2 × entry_price − current_price) × qty_remaining

        The short formula gives the correct equity value because cash was
        already reduced by (entry_price × qty) as margin at entry:
          equity = cash_after_entry + pos_value
                 = (init_cash − entry × qty)
                   + (2×entry − current) × qty
                 = init_cash + (entry − current) × qty   ✓
        """
        rem = pos.get("qty_remaining", pos["qty"])
        if pos.get("side", "long") == "short":
            return (2.0 * pos["entry_price"] - current_price) * rem
        return current_price * rem

    @staticmethod
    def _price_at(df: pd.DataFrame, ts) -> float:
        try:
            return float(df.loc[:ts, "Close"].iloc[-1])
        except (KeyError, IndexError):
            return 0.0

    # ── Main simulation ────────────────────────────────────────────────────────

    def run(self, bars_by_symbol: Dict[str, pd.DataFrame]) -> BacktestResult:
        all_idx = sorted(set().union(*[df.index for df in bars_by_symbol.values()]))
        cash        = self.starting_cash
        open_pos:   Dict[str, dict] = {}
        trades:     List[Trade]     = []
        equity_curve: List[float]   = []
        daily_losses: Dict[str, int] = {}
        _current_day: Optional[str]  = None

        warmup    = 50
        exit_thresh = self.cfg["signals"].get("exit_threshold", 0.10)

        for i, ts in enumerate(all_idx):
            if i < warmup:
                continue

            bar_day = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            if bar_day != _current_day:
                if _current_day is not None:
                    self._perf_tracker.end_of_day(_current_day)
                    st = self._perf_tracker.status()
                    if st["excluded"] or st["bad_streak"]:
                        log.info(
                            "PerfTracker day=%s | excluded=%s | bad_streak=%s",
                            _current_day, st["excluded"], st["bad_streak"],
                        )
                _current_day = bar_day
                daily_losses.clear()

            # Mark-to-market equity (uses qty_remaining for accuracy after partials)
            mtm = cash + sum(
                self._pos_value(p, self._price_at(bars_by_symbol[s], ts))
                for s, p in open_pos.items()
            )
            equity_curve.append(mtm)

            # Macro signal
            spy_window = bars_by_symbol.get("SPY", pd.DataFrame())
            if not spy_window.empty and ts in spy_window.index:
                market_multiplier, _ = self.macro.evaluate(spy_window.loc[:ts])
            else:
                market_multiplier = 1.0

            # ── Per-symbol loop ────────────────────────────────────────────────
            for sym, df in bars_by_symbol.items():
                if ts not in df.index:
                    continue
                window = df.loc[:ts]
                if len(window) < warmup:
                    continue

                t  = self.tech.evaluate(sym, window)
                m  = self.ml.evaluate(sym, window, tech_signal=t)
                rl = self.finrl.evaluate(sym, window, tech_signal=t)
                sigs = [s for s in (t, m, rl) if s is not None]
                if not sigs:
                    continue
                dec = self.agg.aggregate(
                    sigs, market_multiplier=market_multiplier
                ).get(sym)
                if dec is None:
                    continue
                price = float(df.at[ts, "Close"])

                # ── Manage open position ───────────────────────────────────────
                if sym in open_pos:
                    pos  = open_pos[sym]
                    side = pos["side"]
                    rem  = pos["qty_remaining"]

                    # Directional P&L %
                    if side == "long":
                        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
                    else:
                        pnl_pct = (pos["entry_price"] - price) / pos["entry_price"]

                    hold_min = (ts - pos["entry_time"]).total_seconds() / 60.0

                    # ── Update watermark & trailing stop ───────────────────────
                    if side == "long":
                        pos["high_watermark"] = max(pos["high_watermark"], price)
                        if pos["trailing_stop"] is not None:
                            new_trail = pos["high_watermark"] * (1 - self.trailing_stop_pct)
                            pos["trailing_stop"] = max(pos["trailing_stop"], new_trail)
                    else:
                        pos["low_watermark"] = min(pos["low_watermark"], price)
                        if pos["trailing_stop"] is not None:
                            new_trail = pos["low_watermark"] * (1 + self.trailing_stop_pct)
                            pos["trailing_stop"] = min(pos["trailing_stop"], new_trail)

                    # Breakeven: move hard stop to entry once sufficiently in profit
                    if pnl_pct >= self.breakeven_trigger_pct:
                        if side == "long":
                            pos["stop"] = max(pos["stop"], pos["entry_price"])
                        else:
                            pos["stop"] = min(pos["stop"], pos["entry_price"])

                    # Effective stop (tightest of hard stop and trailing stop)
                    pos_min_hold = pos.get("min_hold_min", self.min_hold_minutes)
                    if side == "long":
                        eff_stop = pos["stop"]
                        if pos["trailing_stop"] is not None:
                            eff_stop = max(eff_stop, pos["trailing_stop"])
                        stop_hit    = price <= eff_stop
                        tp_hit      = price >= pos["tp"]
                        sig_reverse = (
                            dec.score < -exit_thresh
                            and hold_min >= pos_min_hold
                            and pnl_pct <= 0.0
                        )
                    else:
                        eff_stop = pos["stop"]
                        if pos["trailing_stop"] is not None:
                            eff_stop = min(eff_stop, pos["trailing_stop"])
                        stop_hit    = price >= eff_stop
                        tp_hit      = price <= pos["tp"]
                        sig_reverse = (
                            dec.score > exit_thresh
                            and hold_min >= pos_min_hold
                            and pnl_pct <= 0.0
                        )

                    # ── Partial exits ──────────────────────────────────────────
                    # Per-position overrides allow regime-adaptive strategies to
                    # store TRENDING params (pp1_pct=9999) vs CHOPPY params (0.01)
                    # at entry time; use them here rather than the global defaults.
                    pos_pp1_pct = pos.get("pp1_pct", self.partial_profit_1_pct)
                    pos_pp2_pct = pos.get("pp2_pct", self.partial_profit_2_pct)

                    if not pos["pp1_fired"] and pnl_pct >= pos_pp1_pct:
                        if rem >= 2:
                            pp_qty = max(1, rem // 2)
                            if side == "long":
                                pp_fill = price * (1 - self.slippage)
                                cash   += pp_fill * pp_qty
                                pp_pnl  = (pp_fill - pos["entry_price"]) * pp_qty
                            else:
                                pp_fill = price * (1 + self.slippage)
                                pp_pnl  = (pos["entry_price"] - pp_fill) * pp_qty
                                cash   += pos["entry_price"] * pp_qty + pp_pnl

                            pt = Trade(
                                symbol=sym, entry_time=pos["entry_time"],
                                entry_price=pos["entry_price"], side=side,
                                exit_time=ts, exit_price=pp_fill,
                                qty=pp_qty, reason="partial_profit_1",
                            )
                            trades.append(pt)
                            pos["realized_pnl"]  += pp_pnl
                            pos["qty_remaining"] -= pp_qty

                            if side == "long":
                                pos["trailing_stop"] = (
                                    pos["high_watermark"] * (1 - self.trailing_stop_pct)
                                )
                                pos["stop"] = max(pos["stop"], pos["entry_price"])
                            else:
                                pos["trailing_stop"] = (
                                    pos["low_watermark"] * (1 + self.trailing_stop_pct)
                                )
                                pos["stop"] = min(pos["stop"], pos["entry_price"])
                        pos["pp1_fired"] = True

                    elif pos["pp1_fired"] and not pos["pp2_fired"] \
                            and pnl_pct >= pos_pp2_pct:
                        rem = pos["qty_remaining"]
                        if rem >= 2:
                            pp_qty = max(1, rem // 2)
                            if side == "long":
                                pp_fill = price * (1 - self.slippage)
                                cash   += pp_fill * pp_qty
                                pp_pnl  = (pp_fill - pos["entry_price"]) * pp_qty
                            else:
                                pp_fill = price * (1 + self.slippage)
                                pp_pnl  = (pos["entry_price"] - pp_fill) * pp_qty
                                cash   += pos["entry_price"] * pp_qty + pp_pnl

                            pt = Trade(
                                symbol=sym, entry_time=pos["entry_time"],
                                entry_price=pos["entry_price"], side=side,
                                exit_time=ts, exit_price=pp_fill,
                                qty=pp_qty, reason="partial_profit_2",
                            )
                            trades.append(pt)
                            pos["realized_pnl"]  += pp_pnl
                            pos["qty_remaining"] -= pp_qty
                        pos["pp2_fired"] = True

                    # ── Full exit of remaining qty ─────────────────────────────
                    pos_max_hold = pos.get("max_hold_min", self.max_hold_minutes)
                    # Per-position EOD buffer: TRENDING positions use 0 (let run);
                    # CHOPPY/NEUTRAL positions use the configured buffer (flush early).
                    pos_eod_buf  = pos.get("eod_buffer_min", self.no_entry_before_close_min)
                    rem = pos["qty_remaining"]
                    if rem <= 0:
                        del open_pos[sym]
                    else:
                        full_exit = None
                        if stop_hit:
                            full_exit = "trailing_stop" if pos["pp1_fired"] else "stop"
                        elif tp_hit:
                            full_exit = "take_profit"
                        elif hold_min >= pos_max_hold:
                            full_exit = "max_hold"
                        elif _is_eod(ts, pos_eod_buf):
                            full_exit = "eod_flatten"
                        elif sig_reverse:
                            full_exit = "signal_reversed"

                        if full_exit:
                            if side == "long":
                                fill      = price * (1 - self.slippage)
                                cash     += fill * rem
                                final_pnl = (fill - pos["entry_price"]) * rem
                            else:
                                fill      = price * (1 + self.slippage)
                                final_pnl = (pos["entry_price"] - fill) * rem
                                cash     += pos["entry_price"] * rem + final_pnl

                            trade = Trade(
                                symbol=sym, entry_time=pos["entry_time"],
                                entry_price=pos["entry_price"], side=side,
                                exit_time=ts, exit_price=fill,
                                qty=rem, reason=full_exit,
                            )
                            trades.append(trade)

                            total_pnl = pos["realized_pnl"] + final_pnl
                            self._perf_tracker.record_trade(sym, bar_day, total_pnl > 0)
                            if total_pnl < 0:
                                daily_losses[sym] = daily_losses.get(sym, 0) + 1
                            del open_pos[sym]

                # ── Entry ──────────────────────────────────────────────────────
                eff_cap          = self._perf_tracker.daily_loss_cap(
                    sym, self.max_symbol_daily_losses
                )
                sym_losses_today = daily_losses.get(sym, 0)
                can_enter = (
                    sym not in open_pos
                    and not self._perf_tracker.is_excluded(sym)
                    and sym_losses_today < eff_cap
                    and not _is_eod(ts, self.no_entry_before_close_min)
                )

                if can_enter and dec.action == "BUY":
                    target_pct  = min(
                        abs(dec.score) * dec.confidence * self.kelly,
                        self.max_position_pct,
                    )
                    notional    = mtm * target_pct
                    qty         = int(notional / price)
                    entry_fill  = price * (1 + self.slippage)
                    cost        = qty * entry_fill
                    if qty >= 1 and cost <= cash:
                        cash -= cost
                        open_pos[sym] = {
                            "side":           "long",
                            "qty":            qty,
                            "qty_remaining":  qty,
                            "entry_time":     ts,
                            "entry_price":    entry_fill,
                            "stop":           price * (1 - self.stop_pct),
                            "tp":             price * (1 + self.tp_pct),
                            "high_watermark": price,
                            "pp1_fired":      False,
                            "pp2_fired":      False,
                            "trailing_stop":  None,
                            "realized_pnl":   0.0,
                        }

                elif can_enter and dec.action == "SELL" and self.shorting_enabled:
                    target_pct  = min(
                        abs(dec.score) * dec.confidence * self.kelly,
                        self.max_position_pct,
                    )
                    notional    = mtm * target_pct
                    qty         = int(notional / price)
                    entry_fill  = price * (1 - self.slippage)   # sell at slight discount
                    margin      = qty * entry_fill               # full collateral required
                    if qty >= 1 and margin <= cash:
                        cash -= margin                           # reserve collateral
                        open_pos[sym] = {
                            "side":          "short",
                            "qty":           qty,
                            "qty_remaining": qty,
                            "entry_time":    ts,
                            "entry_price":   entry_fill,
                            "stop":          price * (1 + self.stop_pct),  # stop ABOVE entry
                            "tp":            price * (1 - self.tp_pct),    # TP BELOW entry
                            "low_watermark": price,                         # for trailing stop
                            "pp1_fired":     False,
                            "pp2_fired":     False,
                            "trailing_stop": None,
                            "realized_pnl":  0.0,
                        }

        # ── End-of-run: flatten all remaining positions ────────────────────────
        last_ts = all_idx[-1] if all_idx else None
        for sym, pos in list(open_pos.items()):
            rem  = pos.get("qty_remaining", pos["qty"])
            if rem <= 0:
                continue
            side = pos.get("side", "long")
            df   = bars_by_symbol[sym]
            price = float(df["Close"].iloc[-1])
            if side == "long":
                fill  = price * (1 - self.slippage)
                cash += fill * rem
            else:
                fill      = price * (1 + self.slippage)
                final_pnl = (pos["entry_price"] - fill) * rem
                cash     += pos["entry_price"] * rem + final_pnl
            trades.append(Trade(
                symbol=sym, entry_time=pos["entry_time"],
                entry_price=pos["entry_price"], side=side,
                exit_time=last_ts, exit_price=fill,
                qty=rem, reason="eod_flatten",
            ))

        return BacktestResult(
            starting_cash    = self.starting_cash,
            ending_cash      = cash,
            trades           = trades,
            equity_curve     = equity_curve,
            periods_per_year = self.periods_per_year,
        )
