"""Regime-adaptive 6-month A/B/C backtest.

Compares three strategies on the original tuning universe:

  ADAPTIVE  — detects each day's regime from SPY ADX + vol-ratio, then:
                TRENDING  day  → OLD exit params (let winners run, no PP)
                CHOPPY    day  → BEST exit params (partial profit + trailing stop)
                NEUTRAL   day  → BEST exit params (conservative default)

  BEST      — always uses BEST exit params (partial profit + trailing stop)

  OLD       — always uses OLD exit params (baseline, no partial profit)

The regime detector uses:
  ADX(14d) > 25 → +2 trending votes
  ADX(14d) < 20 → +2 choppy  votes
  vol-ratio (5d σ / 20d σ) > 1.20 → +1 trending
  vol-ratio < 0.80 → +1 choppy
  votes_trending ≥ 2 → TRENDING  |  votes_choppy ≥ 2 → CHOPPY  |  else NEUTRAL

Per-position routing (stored at entry, respected until exit):
  TRENDING  → pp1_pct=9999, pp2_pct=9999, max_hold_min=9999, min_hold_min=30
  CHOPPY /
  NEUTRAL   → pp1_pct=0.01, pp2_pct=0.025, max_hold_min=480, min_hold_min=45
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.backtest.backtester import Backtester, BacktestResult, Trade, _is_eod
from src.data.market_data import MarketData
from src.signals.finrl_signal import FinRLSignal
from src.signals.ml_model import MLSignal
from src.signals.regime import MarketRegime, MarketRegimeDetector

# ── CLI arguments ─────────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime:
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Unrecognised date '{s}'. Use YYYY/MM/DD or YYYY-MM-DD.")

_ap = argparse.ArgumentParser(description="Regime-adaptive A/B/C backtest")
_ap.add_argument("--start", type=_parse_date, default=None,
                 metavar="YYYY/MM/DD",
                 help="Custom start date for fetched data (default: today − lookback)")
_ap.add_argument("--end",   type=_parse_date, default=None,
                 metavar="YYYY/MM/DD",
                 help="Custom end date for fetched data (default: today)")
_ap.add_argument("--interval", default=None, metavar="Nm",
                 help="Bar interval override, e.g. 1m, 5m, 1d (default: 5m)")
_ARGS = _ap.parse_args()

# ── Parameters ────────────────────────────────────────────────────────────────

LOOKBACK_DAYS = 180
INTERVAL      = _ARGS.interval or "5m"
FINRL_STEPS   = int(BASE_CFG.get("signals", {}).get("finrl", {}).get("total_timesteps", 150_000))
SLIPPAGE_BPS  = 5

TICKERS = [
    "CRM", "ADBE", "MSFT", "AVGO", "AMD",
    "VMC", "NUE", "AAPL", "NVDA", "TSLA",
    "META", "GOOGL", "AMZN", "SPY", "QQQ",
]

with open(ROOT / "config.yaml") as f:
    BASE_CFG = yaml.safe_load(f)

STARTING_CASH = float(BASE_CFG["broker"].get("starting_cash", 10_000))

# ── Regime detector thresholds ────────────────────────────────────────────────
_REGIME_CFG       = BASE_CFG.get("regime", {})
ADX_TREND_THRESH  = float(_REGIME_CFG.get("adx_trend_thresh",  25.0))
ADX_CHOPPY_THRESH = float(_REGIME_CFG.get("adx_choppy_thresh", 20.0))

# ── Regime params  ────────────────────────────────────────────────────────────

# Per-position params stored at entry time for TRENDING days (OLD-style).
# eod_buffer_min=0 means _is_eod() only fires at exactly 4 pm — with 5-min
# bars the last bar is 3:55 pm so the position is NOT force-closed intraday;
# it can run through signal-reversal or take-profit over multiple days.
_TREND_POS_PARAMS = dict(
    pp1_pct       = 9999.0,   # partial profit never fires — let winners run
    pp2_pct       = 9999.0,
    max_hold_min  = 9999.0,   # no hold cap
    min_hold_min  = 30.0,
    eod_buffer_min= 0,        # OLD-style: no forced intraday close
)

# Per-position params stored at entry time for CHOPPY / NEUTRAL days (BEST-style).
# eod_buffer_min=5 forces a flush at 3:55 pm — the position is always closed
# before EOD so gains are locked in rather than given back overnight.
_CHOPPY_POS_PARAMS = dict(
    pp1_pct       = 0.01,     # lock in 50 % at +1 %
    pp2_pct       = 0.025,    # lock in another 50 % of remainder at +2.5 %
    max_hold_min  = 480.0,
    min_hold_min  = 45.0,
    eod_buffer_min= 5,        # BEST-style: flush at 3:55 pm
)

# ── Config builders ───────────────────────────────────────────────────────────

def _best_cfg(base: dict) -> dict:
    cfg = copy.deepcopy(base)
    s  = cfg["signals"]
    r  = cfg["risk"]
    sc = cfg["schedule"]
    s["require_ml_finrl_agreement"]     = False
    s["ml"]["label_method"]             = "fixed"
    s["enter_short_threshold"]          = -0.45
    r["min_hold_minutes"]               = 45
    r["max_hold_minutes"]               = 480
    r["max_symbol_daily_losses"]        = 2
    r["min_entry_adx"]                  = 18        # Fix 1: block low-ADX entries
    r["weak_trend_adx_max"]             = 25        # B+C: weak-trend zone ceiling
    r["weak_trend_entry_threshold"]     = 0.55      # C: higher score req in weak zone
    r["weak_trend_max_daily_losses"]    = 1         # B: 1 loss/day cap in weak zone
    r["dynamic_exclusion_win_rate"]     = 0.30      # Fix 3: tighter bad-symbol gate
    r["dynamic_exclusion_streak_days"]  = 3         # Fix 3: react after 3 bad days
    r["partial_profit_1_pct"]           = 0.01
    r["partial_profit_2_pct"]           = 0.025
    r["trailing_stop_pct"]              = 0.03
    r["breakeven_trigger_pct"]          = 0.022
    r["shorting_enabled"]               = False
    sc["no_entry_before_close_minutes"] = 5
    return cfg


def _old_cfg(base: dict) -> dict:
    cfg = copy.deepcopy(base)
    s  = cfg["signals"]
    r  = cfg["risk"]
    sc = cfg["schedule"]
    s["require_ml_finrl_agreement"]     = False
    s["ml"]["label_method"]             = "fixed"
    r["min_hold_minutes"]               = 30
    r["max_hold_minutes"]               = 9999
    r["max_symbol_daily_losses"]        = 2
    r["dynamic_exclusion_win_rate"]     = 0.33
    r["dynamic_exclusion_streak_days"]  = 9999
    r["partial_profit_1_pct"]           = 9999.0
    r["partial_profit_2_pct"]           = 9999.0
    r["breakeven_trigger_pct"]          = 9999.0
    r["shorting_enabled"]               = False
    sc["no_entry_before_close_minutes"] = 0
    return cfg


CFG_BEST     = _best_cfg(BASE_CFG)
CFG_OLD      = _old_cfg(BASE_CFG)
CFG_ADAPTIVE = _best_cfg(BASE_CFG)   # base config for Adaptive (entry gate identical to BEST)


# ── Adaptive backtester ───────────────────────────────────────────────────────

class AdaptiveBacktester(Backtester):
    """Like Backtester but routes each position's exit params by per-symbol regime.

    At entry time the regime is detected from the *entering symbol's own bars*
    (not SPY), using ADX(14) + vol-ratio, looking only at prior-day bars to
    avoid lookahead bias.  Each symbol gets its own ``MarketRegimeDetector``
    instance so regime transitions are logged independently.

      TRENDING symbol → OLD-style exit params (no partial profit, no EOD close)
      CHOPPY / NEUTRAL → BEST-style exit params (partial profit + 3:55 pm close)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One detector per symbol — keeps per-symbol state (transition logging)
        self._regime_detectors: Dict[str, MarketRegimeDetector] = {}
        # (sym, entry_time, regime) — one record per entry
        self._regime_log: list = []
        # Full bars (train + test) — set via bt.full_bars before calling run().
        # Used ONLY for regime detection so ADX has enough history from day 1.
        # No lookahead risk: all training-period bars precede any test-period entry.
        self.full_bars: Dict[str, pd.DataFrame] = {}

    def _sym_regime(
        self,
        sym: str,
        sym_bars_past: pd.DataFrame,
    ) -> MarketRegime:
        """Return this symbol's current regime (no lookahead — past bars only)."""
        if sym not in self._regime_detectors:
            self._regime_detectors[sym] = MarketRegimeDetector(adx_trend_thresh=ADX_TREND_THRESH, adx_choppy_thresh=ADX_CHOPPY_THRESH)
        return self._regime_detectors[sym].detect(sym_bars_past)

    @staticmethod
    def _resolve_pos_params(regime: MarketRegime) -> dict:
        """Map a MarketRegime to the per-position exit param dict."""
        if regime == MarketRegime.TRENDING:
            return _TREND_POS_PARAMS.copy()
        return _CHOPPY_POS_PARAMS.copy()

    def run(self, bars_by_symbol: Dict[str, pd.DataFrame]) -> BacktestResult:
        all_idx = sorted(set().union(*[df.index for df in bars_by_symbol.values()]))
        cash          = self.starting_cash
        open_pos:     Dict[str, dict] = {}
        trades        = []
        equity_curve  = []
        daily_losses: Dict[str, int]  = {}
        _current_day: Optional[str]   = None
        _day_ts:      Optional[pd.Timestamp] = None   # tz-aware day boundary

        warmup      = 50
        exit_thresh = self.cfg["signals"].get("exit_threshold", 0.10)

        spy_df = bars_by_symbol.get("SPY", pd.DataFrame())

        for i, ts in enumerate(all_idx):
            if i < warmup:
                continue

            bar_day = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            if bar_day != _current_day:
                if _current_day is not None:
                    self._perf_tracker.end_of_day(_current_day)

                _current_day = bar_day
                daily_losses.clear()

                # Pre-compute a tz-aware day boundary once per day.
                # Used at entry to filter "strictly before today" bars per symbol.
                _day_ts = pd.Timestamp(bar_day)

            # Mark-to-market
            mtm = cash + sum(
                self._pos_value(p, self._price_at(bars_by_symbol[s], ts))
                for s, p in open_pos.items()
            )
            equity_curve.append(mtm)

            # Macro signal
            if not spy_df.empty and ts in spy_df.index:
                market_multiplier, _ = self.macro.evaluate(spy_df.loc[:ts])
            else:
                market_multiplier = 1.0

            # ── Per-symbol loop ───────────────────────────────────────────────
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

                # ── Manage open position ──────────────────────────────────────
                if sym in open_pos:
                    pos  = open_pos[sym]
                    side = pos["side"]
                    rem  = pos["qty_remaining"]

                    if side == "long":
                        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"]
                    else:
                        pnl_pct = (pos["entry_price"] - price) / pos["entry_price"]

                    hold_min = (ts - pos["entry_time"]).total_seconds() / 60.0

                    # Update watermarks
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

                    if pnl_pct >= self.breakeven_trigger_pct:
                        if side == "long":
                            pos["stop"] = max(pos["stop"], pos["entry_price"])
                        else:
                            pos["stop"] = min(pos["stop"], pos["entry_price"])

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

                    # Partial exits (use per-position thresholds stored at entry)
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
                            trades.append(Trade(
                                symbol=sym, entry_time=pos["entry_time"],
                                entry_price=pos["entry_price"], side=side,
                                exit_time=ts, exit_price=pp_fill,
                                qty=pp_qty, reason="partial_profit_1",
                            ))
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
                        rem2 = pos["qty_remaining"]
                        if rem2 >= 2:
                            pp_qty = max(1, rem2 // 2)
                            if side == "long":
                                pp_fill = price * (1 - self.slippage)
                                cash   += pp_fill * pp_qty
                                pp_pnl  = (pp_fill - pos["entry_price"]) * pp_qty
                            else:
                                pp_fill = price * (1 + self.slippage)
                                pp_pnl  = (pos["entry_price"] - pp_fill) * pp_qty
                                cash   += pos["entry_price"] * pp_qty + pp_pnl
                            trades.append(Trade(
                                symbol=sym, entry_time=pos["entry_time"],
                                entry_price=pos["entry_price"], side=side,
                                exit_time=ts, exit_price=pp_fill,
                                qty=pp_qty, reason="partial_profit_2",
                            ))
                            pos["realized_pnl"]  += pp_pnl
                            pos["qty_remaining"] -= pp_qty
                        pos["pp2_fired"] = True

                    # Full exit
                    pos_max_hold = pos.get("max_hold_min", self.max_hold_minutes)
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
                            trades.append(Trade(
                                symbol=sym, entry_time=pos["entry_time"],
                                entry_price=pos["entry_price"], side=side,
                                exit_time=ts, exit_price=fill,
                                qty=rem, reason=full_exit,
                            ))
                            total_pnl = pos["realized_pnl"] + final_pnl
                            self._perf_tracker.record_trade(sym, bar_day, total_pnl > 0)
                            if total_pnl < 0:
                                daily_losses[sym] = daily_losses.get(sym, 0) + 1
                            del open_pos[sym]

                # ── Entry ─────────────────────────────────────────────────────
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
                    # ── Per-symbol regime detection ───────────────────────────
                    # Use the FULL bars for this symbol (train + test) so
                    # ADX has adequate history from the very first test-period
                    # entry.  Filter to strictly prior-day bars to avoid lookahead.
                    hist_df = self.full_bars.get(sym, df)
                    sym_day_ts = _day_ts
                    if hist_df.index.tz is not None and sym_day_ts.tzinfo is None:
                        sym_day_ts = sym_day_ts.tz_localize(hist_df.index.tz)
                    sym_past = hist_df[hist_df.index.normalize() < sym_day_ts]
                    sym_regime = self._sym_regime(sym, sym_past)
                    pos_params = self._resolve_pos_params(sym_regime)

                    # ── ADX-tiered entry gate ─────────────────────────────────
                    # Zone 1 (ADX < min):          hard block — no trend at all
                    # Zone 2 (min ≤ ADX < weak_max): weak trend — need higher score
                    #                               + max 1 loss/day (B+C combined)
                    # Zone 3 (ADX ≥ weak_max):     full trend — normal rules apply
                    rcfg_bt   = self.cfg.get("risk", {})
                    min_adx   = rcfg_bt.get("min_entry_adx", 0.0)
                    weak_max  = rcfg_bt.get("weak_trend_adx_max", 25.0)
                    weak_thr  = rcfg_bt.get("weak_trend_entry_threshold", 0.55)
                    weak_cap  = int(rcfg_bt.get("weak_trend_max_daily_losses", 1))
                    det     = self._regime_detectors.get(sym)
                    adx_now = det._last_adx if det is not None else None
                    if min_adx > 0 and adx_now is not None and adx_now == adx_now:
                        if adx_now < min_adx:
                            continue  # zone 1 — directionless, skip
                        if adx_now < weak_max:
                            # zone 2 — weak trend: require higher conviction score
                            if abs(dec.score) < weak_thr:
                                continue
                            # zone 2 — tighter daily loss cap (B+C combined)
                            if daily_losses.get(sym, 0) >= weak_cap:
                                continue

                    # Only log regime for entries that actually pass all gates.
                    self._regime_log.append((sym, str(ts), sym_regime.value))

                    target_pct  = min(
                        abs(dec.score) * dec.confidence * self.kelly,
                        self.max_position_pct,
                    )
                    notional   = mtm * target_pct
                    qty        = int(notional / price)
                    entry_fill = price * (1 + self.slippage)
                    cost       = qty * entry_fill
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
                            **pos_params,     # inject regime-specific exit params
                        }

        # Flatten remaining positions
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

    def print_regime_summary(self) -> None:
        """Print per-symbol regime distribution across all entries."""
        if not self._regime_log:
            print("  No entries recorded.")
            return

        # Overall counts
        overall: dict = {}
        by_sym:  dict = {}   # sym → {regime: count}
        for sym, _ts, regime in self._regime_log:
            overall[regime] = overall.get(regime, 0) + 1
            by_sym.setdefault(sym, {})
            by_sym[sym][regime] = by_sym[sym].get(regime, 0) + 1

        total = len(self._regime_log)
        print(f"\n  Regime at entry — {total} entries total:")
        for r, n in sorted(overall.items(), key=lambda x: -x[1]):
            pct = n / total * 100
            print(f"    {r:<10s}  {n:3d}  ({pct:.0f}%)")

        print(f"\n  Per-symbol regime breakdown:")
        for sym in sorted(by_sym):
            parts = "  ".join(
                f"{r}={n}" for r, n in sorted(by_sym[sym].items())
            )
            det_status = self._regime_detectors.get(sym, MarketRegimeDetector(adx_trend_thresh=ADX_TREND_THRESH, adx_choppy_thresh=ADX_CHOPPY_THRESH)).status
            adx_str = f"ADX={det_status['adx']:.1f}" if det_status["adx"] else "ADX=n/a"
            print(f"    {sym:<6s}  {parts}  ({adx_str})")


# ── Summary helpers ───────────────────────────────────────────────────────────

def _summary(result: BacktestResult, trades) -> dict:
    groups: dict = defaultdict(list)
    for t in trades:
        groups[(t.symbol, t.entry_time)].append(t)

    wins, losses = [], []
    for group in groups.values():
        if sum(t.pnl for t in group) > 0:
            wins.append(group)
        else:
            losses.append(group)

    pf = (
        abs(sum(sum(t.pnl for t in g) for g in wins))
        / abs(sum(sum(t.pnl for t in g) for g in losses))
        if losses else 999.0
    )
    avg_w = float(np.mean([sum(t.pnl_pct for t in g) * 100 for g in wins]))   if wins   else 0.0
    avg_l = float(np.mean([sum(t.pnl_pct for t in g) * 100 for g in losses])) if losses else 0.0
    n_entries = len(groups)

    return {
        "net_pnl":    result.ending_cash - result.starting_cash,
        "return_pct": result.total_return * 100,
        "n_trades":   n_entries,
        "n_fills":    len(trades),
        "win_rate":   len(wins) / n_entries * 100 if n_entries else 0.0,
        "avg_win":    avg_w,
        "avg_loss":   avg_l,
        "pf":         pf,
        "sharpe":     result.sharpe,
        "exits":      Counter(t.reason for t in trades),
        "by_sym":     _by_sym(groups),
    }


def _by_sym(groups) -> dict:
    d: dict = {}
    for (sym, _), fills in groups.items():
        pnl = sum(t.pnl for t in fills)
        d.setdefault(sym, []).append(pnl)
    return d


def _print_result(label: str, s: dict, width: int = 60) -> None:
    bar = "=" * width
    print(f"\n{bar}")
    print(f"  {label}")
    print(bar)
    print(f"  Net P&L      : ${s['net_pnl']:>+9,.2f}  ({s['return_pct']:+.2f}%)")
    print(f"  Entries      : {s['n_trades']}  ({s['n_fills']} fills incl. partials)")
    print(f"  Win rate     : {s['win_rate']:.1f}%")
    print(f"  Avg win      : {s['avg_win']:+.2f}%")
    print(f"  Avg loss     : {s['avg_loss']:+.2f}%")
    print(f"  Profit factor: {s['pf']:.2f}x")
    print(f"  Sharpe ratio : {s['sharpe']:.2f}")
    print(f"  Exit reasons :")
    for r, n in s["exits"].most_common():
        pct = n / s["n_fills"] * 100 if s["n_fills"] else 0
        print(f"    {r:<22s} {n:3d} ({pct:.0f}%)")
    print(f"  Per-symbol   :")
    for sym, pnls in sorted(s["by_sym"].items(), key=lambda x: -sum(x[1])):
        w = sum(1 for p in pnls if p > 0)
        print(f"    {sym:<6s} {len(pnls):2d} entries  {w}/{len(pnls)} wins  ${sum(pnls):+.2f}")
    print(bar)


def _print_comparison(adaptive: dict, best: dict, old: dict) -> None:
    width = 80
    bar   = "=" * width
    dash  = "-" * width
    print(f"\n\n{bar}")
    print(f"  A/B/C COMPARISON  —  ADAPTIVE vs BEST vs OLD")
    print(f"  Universe: {', '.join(TICKERS)}")
    print(bar)
    print(f"  {'Metric':<28} {'ADAPTIVE':>14} {'BEST':>14} {'OLD':>14}")
    print(dash)

    def row(label, av, bv, ov, fmt="{:.2f}", higher_is_better=True):
        af = fmt.format(av) if av is not None else "—"
        bf = fmt.format(bv) if bv is not None else "—"
        of = fmt.format(ov) if ov is not None else "—"
        best_val = max(av, bv, ov) if higher_is_better else min(av, bv, ov)
        marker_a = " ★" if av == best_val else ""
        marker_b = " ★" if bv == best_val else ""
        marker_o = " ★" if ov == best_val else ""
        print(f"  {label:<28} {af+marker_a:>16} {bf+marker_b:>16} {of+marker_o:>16}")

    row("Net P&L ($)",    adaptive["net_pnl"],    best["net_pnl"],    old["net_pnl"],    fmt="${:+,.2f}")
    row("Return (%)",     adaptive["return_pct"], best["return_pct"], old["return_pct"], fmt="{:+.2f}%")
    row("Entries",        adaptive["n_trades"],   best["n_trades"],   old["n_trades"],   fmt="{:d}")
    row("Win rate (%)",   adaptive["win_rate"],   best["win_rate"],   old["win_rate"],   fmt="{:.1f}%")
    row("Avg win (%)",    adaptive["avg_win"],    best["avg_win"],    old["avg_win"],    fmt="{:+.2f}%")
    row("Avg loss (%)",   adaptive["avg_loss"],   best["avg_loss"],   old["avg_loss"],   fmt="{:+.2f}%",
        higher_is_better=False)
    row("Profit factor",  adaptive["pf"],         best["pf"],         old["pf"],         fmt="{:.2f}x")
    row("Sharpe ratio",   adaptive["sharpe"],     best["sharpe"],     old["sharpe"],     fmt="{:.2f}")
    print(bar)
    print("  ★ = best value for that metric\n")


# ── Main ──────────────────────────────────────────────────────────────────────

_date_range_str = (
    f"{_ARGS.start.strftime('%Y-%m-%d')} → {_ARGS.end.strftime('%Y-%m-%d')}"
    if _ARGS.start or _ARGS.end
    else f"last {LOOKBACK_DAYS} calendar days"
)

print(f"\n{'='*60}")
print(f"  DayTradingBot — Regime-Adaptive A/B/C backtest")
print(f"  Interval: {INTERVAL}  |  Range: {_date_range_str}")
print(f"  Cash: ${STARTING_CASH:,.0f}   Slippage: {SLIPPAGE_BPS} bps")
print(f"  Tickers ({len(TICKERS)}): {', '.join(TICKERS)}")
print(f"{'='*60}\n")

# Step 1 — Fetch data
print(f"Fetching {INTERVAL} bars ({_date_range_str}) …")
md = MarketData(
    provider=BASE_CFG["data"].get("provider", "alpaca"),
    interval=INTERVAL,
    lookback_days=LOOKBACK_DAYS,
    feed=BASE_CFG["data"].get("feed", "iex"),
)
bars_by_sym: dict = {}
for sym in TICKERS:
    df = md.get_bars(sym, start_dt=_ARGS.start, end_dt=_ARGS.end)
    if df is not None and not df.empty:
        bars_by_sym[sym] = df
        print(f"  {sym:<6s}  {len(df):6d} bars  "
              f"{df.index[0].strftime('%m/%d/%y')} → {df.index[-1].strftime('%m/%d/%y')}")
    else:
        print(f"  {sym:<6s}  NO DATA — skipped")

# Step 2 — Walk-forward split
print(f"\nWalk-forward split: train on first 50%, test on last 50%")
train_bars, test_bars = {}, {}
for sym, df in bars_by_sym.items():
    n = len(df)
    s = n // 2
    if s >= 200:
        train_bars[sym] = df.iloc[:s]
    if n - s >= 100:
        test_bars[sym]  = df.iloc[s:]

print(f"  Train symbols: {len(train_bars)}   Test symbols: {len(test_bars)}")
if train_bars:
    sample = next(iter(train_bars.values()))
    print(f"  Train period : {sample.index[0].strftime('%m/%d/%y')} → {sample.index[-1].strftime('%m/%d/%y')}")
if test_bars:
    sample = next(iter(test_bars.values()))
    print(f"  Test period  : {sample.index[0].strftime('%m/%d/%y')} → {sample.index[-1].strftime('%m/%d/%y')}")

# Step 3 — Train shared ML model
print(f"\nTraining ML model (LightGBM + fixed-horizon direction labels) …")
ml_shared = MLSignal(CFG_BEST["signals"]["ml"])
ml_shared.train(train_bars)

# Step 4 — Train shared FinRL model
print(f"Training FinRL PPO agent (shared, {FINRL_STEPS:,} steps) …")
finrl_cfg = dict(BASE_CFG["signals"].get("finrl", {}))
finrl_cfg["total_timesteps"] = FINRL_STEPS
finrl = FinRLSignal(finrl_cfg)
finrl.train(train_bars)

# Step 4b — Pre-compute all FinRL scores for the test period in one GPU batch.
# This eliminates the O(N²) per-bar feature recomputation and ~142k individual
# GPU dispatches, cutting backtest time from ~15 min → ~3-4 min.
print(f"Pre-computing FinRL scores for {len(test_bars)} symbols …")
finrl.precompute_backtest_scores(test_bars)

# Step 5 — Run ADAPTIVE strategy
print(f"\nRunning backtest — ADAPTIVE (regime-routing: TRENDING→OLD exits, CHOPPY→BEST exits) …")
bt_adaptive = AdaptiveBacktester(
    config=CFG_ADAPTIVE, starting_cash=STARTING_CASH,
    slippage_bps=SLIPPAGE_BPS, interval=INTERVAL,
)
bt_adaptive.ml    = ml_shared
bt_adaptive.finrl = finrl
# Supply the full bars (train + test) so the per-symbol ADX has adequate
# history from the very first test-period bar.  The regime detector filters
# to prior-day bars internally, so there is zero lookahead risk.
bt_adaptive.full_bars = bars_by_sym
result_adaptive = bt_adaptive.run(test_bars)
sum_adaptive = _summary(result_adaptive, result_adaptive.trades)

# Step 6 — Run BEST strategy
print(f"Running backtest — BEST (always partial profit + trailing stop) …")
bt_best = Backtester(config=CFG_BEST, starting_cash=STARTING_CASH,
                     slippage_bps=SLIPPAGE_BPS, interval=INTERVAL)
bt_best.ml = ml_shared
bt_best.finrl = finrl
result_best = bt_best.run(test_bars)
sum_best = _summary(result_best, result_best.trades)

# Step 7 — Run OLD strategy
print(f"Running backtest — OLD (plain baseline, no partial profit) …")
bt_old = Backtester(config=CFG_OLD, starting_cash=STARTING_CASH,
                    slippage_bps=SLIPPAGE_BPS, interval=INTERVAL)
bt_old.ml = ml_shared
bt_old.finrl = finrl
result_old = bt_old.run(test_bars)
sum_old = _summary(result_old, result_old.trades)

# Step 8 — Print results
_print_result("ADAPTIVE  (regime-routing: TRENDING→no-PP, CHOPPY→partial profit)", sum_adaptive)
bt_adaptive.print_regime_summary()
_print_result("BEST      (always partial profit + trailing stop)", sum_best)
_print_result("OLD       (no partial profit, no hold cap, plain baseline)", sum_old)
_print_comparison(sum_adaptive, sum_best, sum_old)
