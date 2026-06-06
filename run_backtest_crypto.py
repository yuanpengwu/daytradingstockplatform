"""Crypto backtest runner — walk-forward split, crypto-specific risk params.

Usage
-----
    python run_backtest_crypto.py
    python run_backtest_crypto.py --start 2025/08/01 --end 2025/10/31
    python run_backtest_crypto.py --start 2025/08/01 --end 2025/10/31 --interval 1m

Improvements over the stock backtester
----------------------------------------
  1. LightGBM ML model   — crypto-specific model (crypto_lgbm.pkl)
  2. Fractional qty      — notional / price (float, not int)
  3. No EOD flatten      — 24/7 market; positions never force-closed at 16:00 ET
  4. No SPY macro        — market_multiplier = 1.0 always
  5. Signal persistence  — requires N consecutive bars in same direction before
                           entry to avoid whipsawing on noisy 5m crypto signals
  6. Crypto-tuned params — tighter partials (2 % / 4 %), breakeven at 2 %,
                           60-min min-hold, wider stop (5 %) / TP (10 %)
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.backtest.backtester import Backtester, BacktestResult, Trade
from src.data.market_data import MarketData
from src.signals.ml_model import MLSignal

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime:
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise argparse.ArgumentTypeError(f"Unrecognised date '{s}'. Use YYYY/MM/DD.")

_ap = argparse.ArgumentParser(description="Crypto backtest")
_ap.add_argument("--start",    type=_parse_date, default=datetime(2025, 8,  1, tzinfo=timezone.utc))
_ap.add_argument("--end",      type=_parse_date, default=datetime(2025, 10, 31, tzinfo=timezone.utc))
_ap.add_argument("--interval", default="5m",  help="Bar interval (default: 5m)")
_ap.add_argument("--cash",     type=float,    default=None, help="Starting cash override")
_ap.add_argument("--persist",  type=int,      default=2,    help="Signal persistence bars (default: 2)")
_ARGS = _ap.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────

with open(ROOT / "config.yaml") as f:
    BASE_CFG = yaml.safe_load(f)

STARTING_CASH = _ARGS.cash or float(BASE_CFG["broker"].get("starting_cash", 10_000))
INTERVAL      = _ARGS.interval
SLIPPAGE_BPS  = float(BASE_CFG["broker"].get("slippage_bps", 5))
PERSIST_BARS  = _ARGS.persist   # consecutive bars required before entry

CRYPTO_TICKERS = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD"]

# ── Build crypto config ───────────────────────────────────────────────────────

def _build_crypto_cfg(base: dict) -> dict:
    cfg  = copy.deepcopy(base)
    ccfg = cfg.get("crypto", {})

    # Signals: technical + crypto ML only (50 / 50)
    cfg["signals"]["weights"] = {"technical": 0.50, "ml": 0.50}

    # ML: dedicated crypto model
    ml = dict(ccfg.get("ml", cfg["signals"].get("ml", {})))
    ml.setdefault("mode",                       "local")
    ml.setdefault("model_path",                 "models/crypto_lgbm.pkl")
    ml.setdefault("retrain_days",               1)
    ml.setdefault("prediction_horizon_minutes", 30)
    ml.setdefault("label_method",               "fixed")
    ml.setdefault("min_confidence",             0.45)
    ml.setdefault("tech_threshold",             0.05)
    cfg["signals"]["ml"] = ml

    # Thresholds
    cfg["signals"]["enter_long_threshold"]  =  float(ccfg.get("enter_long_threshold", 0.35))
    cfg["signals"]["enter_short_threshold"] = -float(ccfg.get("enter_long_threshold", 0.35))
    cfg["signals"]["min_confidence"]        =  float(ccfg.get("min_confidence", 0.45))

    # Risk — tuned for 5-min crypto bars
    r = cfg["risk"]
    r["per_trade_stop_loss_pct"]   = float(ccfg.get("per_trade_stop_loss_pct", 0.05))   # 5 %
    r["take_profit_pct"]           = float(ccfg.get("take_profit_pct",         0.10))   # 10 %
    r["trailing_stop_pct"]         = float(ccfg.get("trailing_stop_pct",       0.04))   # 4 %
    r["max_position_pct"]          = 0.20    # up to 20 % per crypto trade
    r["kelly_fraction"]            = 0.25
    r["shorting_enabled"]          = False
    r["min_hold_minutes"]          = 60      # 1 h min — avoid whipsawing on 5-min noise
    r["max_hold_minutes"]          = 99999   # no cap — crypto trends can last days
    r["partial_profit_1_pct"]      = 0.02    # first partial at +2 % (tighter than stocks)
    r["partial_profit_2_pct"]      = 0.04    # second partial at +4 %
    r["breakeven_trigger_pct"]     = 0.02    # move stop to breakeven at +2 %
    r["max_symbol_daily_losses"]   = 2
    r["dynamic_exclusion_win_rate"]    = 0.30
    r["dynamic_exclusion_streak_days"] = 3

    # Disable EOD entry cutoff (24/7 market)
    cfg["schedule"]["no_entry_before_close_minutes"] = -9999
    return cfg


CRYPTO_CFG = _build_crypto_cfg(BASE_CFG)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _bars_per_year(interval: str) -> float:
    """24/7 crypto — use 365 calendar days."""
    table = {
        "1m":  60 * 24 * 365,
        "5m":  12 * 24 * 365,
        "15m":  4 * 24 * 365,
        "1h":      24 * 365,
    }
    return table.get(interval, 12 * 24 * 365)

# ── CryptoBacktester ──────────────────────────────────────────────────────────

class CryptoBacktester(Backtester):
    """Backtester adapted for 24/7 crypto markets.

    Key overrides vs stock Backtester
    -----------------------------------
    * Fractional qty     — qty = notional / price (float, min $10 notional)
    * No EOD flatten     — 24/7 market; _is_eod never fires
    * No SPY macro       — market_multiplier = 1.0
    * Signal persistence — requires *persist_bars* consecutive bullish bars,
                           all above 70 % of the entry threshold, before entry.
                           Mirrors the live engine's entry_persistence_bars filter.
    """

    def __init__(self, *args, persist_bars: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.persist_bars = persist_bars

    def run(self, bars_by_symbol: Dict[str, pd.DataFrame]) -> BacktestResult:  # noqa: C901
        all_idx = sorted(set().union(*[df.index for df in bars_by_symbol.values()]))
        cash          = self.starting_cash
        open_pos:     Dict[str, dict] = {}
        trades:       List[Trade]     = []
        equity_curve: List[float]     = []
        daily_losses: Dict[str, int]  = {}
        _current_day: Optional[str]   = None

        # Signal persistence buffer  {sym: [score, score, ...]}
        sig_hist: Dict[str, List[float]] = {}

        warmup      = 50
        exit_thresh = self.cfg["signals"].get("exit_threshold", 0.10)

        for i, ts in enumerate(all_idx):
            if i < warmup:
                continue

            bar_day = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
            if bar_day != _current_day:
                if _current_day is not None:
                    self._perf_tracker.end_of_day(_current_day)
                _current_day = bar_day
                daily_losses.clear()

            # Mark-to-market
            mtm = cash + sum(
                self._pos_value(p, self._price_at(bars_by_symbol[s], ts))
                for s, p in open_pos.items()
            )
            equity_curve.append(mtm)

            for sym, df in bars_by_symbol.items():
                if ts not in df.index:
                    continue
                window = df.loc[:ts]
                if len(window) < warmup:
                    continue

                price = float(df.at[ts, "Close"])
                if price <= 0:
                    continue

                # Signals — technical + crypto ML only, no SPY macro
                t    = self.tech.evaluate(sym, window)
                m    = self.ml.evaluate(sym, window, tech_signal=t)
                sigs = [s for s in (t, m) if s is not None]
                if not sigs:
                    continue

                dec = self.agg.aggregate(sigs, market_multiplier=1.0).get(sym)
                if dec is None:
                    continue

                # ── Update signal persistence buffer ──────────────────────────
                hist = sig_hist.setdefault(sym, [])
                hist.append(dec.score)
                if len(hist) > self.persist_bars:
                    hist.pop(0)

                # ── Manage open position ──────────────────────────────────────
                if sym in open_pos:
                    pos      = open_pos[sym]
                    rem      = pos["qty_remaining"]
                    pnl_pct  = (price - pos["entry_price"]) / pos["entry_price"]
                    hold_min = (ts - pos["entry_time"]).total_seconds() / 60.0

                    # Trailing high + trailing stop
                    pos["high_watermark"] = max(pos["high_watermark"], price)
                    if pos["trailing_stop"] is not None:
                        pos["trailing_stop"] = max(
                            pos["trailing_stop"],
                            pos["high_watermark"] * (1 - self.trailing_stop_pct),
                        )

                    # Breakeven: move stop to entry once sufficiently in profit
                    if pnl_pct >= self.breakeven_trigger_pct:
                        pos["stop"] = max(pos["stop"], pos["entry_price"])

                    eff_stop = pos["stop"]
                    if pos["trailing_stop"] is not None:
                        eff_stop = max(eff_stop, pos["trailing_stop"])

                    stop_hit    = price <= eff_stop
                    tp_hit      = price >= pos["tp"]
                    sig_reverse = (
                        dec.score < -exit_thresh
                        and hold_min >= self.min_hold_minutes
                        and pnl_pct <= 0.0
                    )

                    # Partial exits (fractional qty — split by 50 %)
                    pos_pp1 = pos.get("pp1_pct", self.partial_profit_1_pct)
                    pos_pp2 = pos.get("pp2_pct", self.partial_profit_2_pct)

                    if not pos["pp1_fired"] and pnl_pct >= pos_pp1 and rem > 0:
                        pp_qty  = rem * 0.50
                        pp_fill = price * (1 - self.slippage)
                        cash   += pp_fill * pp_qty
                        pp_pnl  = (pp_fill - pos["entry_price"]) * pp_qty
                        trades.append(Trade(
                            symbol=sym, entry_time=pos["entry_time"],
                            entry_price=pos["entry_price"], side="long",
                            exit_time=ts, exit_price=pp_fill,
                            qty=pp_qty, reason="partial_profit_1",
                        ))
                        pos["realized_pnl"]  += pp_pnl
                        pos["qty_remaining"] -= pp_qty
                        pos["trailing_stop"]  = pos["high_watermark"] * (1 - self.trailing_stop_pct)
                        pos["stop"]           = max(pos["stop"], pos["entry_price"])
                        pos["pp1_fired"]      = True

                    elif pos["pp1_fired"] and not pos["pp2_fired"] and pnl_pct >= pos_pp2:
                        rem    = pos["qty_remaining"]
                        pp_qty = rem * 0.50
                        pp_fill = price * (1 - self.slippage)
                        cash   += pp_fill * pp_qty
                        pp_pnl  = (pp_fill - pos["entry_price"]) * pp_qty
                        trades.append(Trade(
                            symbol=sym, entry_time=pos["entry_time"],
                            entry_price=pos["entry_price"], side="long",
                            exit_time=ts, exit_price=pp_fill,
                            qty=pp_qty, reason="partial_profit_2",
                        ))
                        pos["realized_pnl"]  += pp_pnl
                        pos["qty_remaining"] -= pp_qty
                        pos["pp2_fired"]      = True

                    # Full exit of remaining qty
                    rem = pos["qty_remaining"]
                    if rem <= 0:
                        del open_pos[sym]
                        sig_hist.pop(sym, None)   # reset persistence on exit
                    else:
                        full_exit = None
                        if stop_hit:
                            full_exit = "trailing_stop" if pos["pp1_fired"] else "stop"
                        elif tp_hit:
                            full_exit = "take_profit"
                        elif hold_min >= self.max_hold_minutes:
                            full_exit = "max_hold"
                        elif sig_reverse:
                            full_exit = "signal_reversed"

                        if full_exit:
                            fill      = price * (1 - self.slippage)
                            cash     += fill * rem
                            final_pnl = (fill - pos["entry_price"]) * rem
                            trades.append(Trade(
                                symbol=sym, entry_time=pos["entry_time"],
                                entry_price=pos["entry_price"], side="long",
                                exit_time=ts, exit_price=fill,
                                qty=rem, reason=full_exit,
                            ))
                            total_pnl = pos["realized_pnl"] + final_pnl
                            self._perf_tracker.record_trade(sym, bar_day, total_pnl > 0)
                            if total_pnl < 0:
                                daily_losses[sym] = daily_losses.get(sym, 0) + 1
                            del open_pos[sym]
                            sig_hist.pop(sym, None)   # reset persistence on exit

                # ── Entry — long only, fractional qty, with persistence gate ──
                eff_cap = self._perf_tracker.daily_loss_cap(
                    sym, self.max_symbol_daily_losses
                )
                can_enter = (
                    sym not in open_pos
                    and not self._perf_tracker.is_excluded(sym)
                    and daily_losses.get(sym, 0) < eff_cap
                )

                if can_enter and dec.action == "BUY":
                    # ── Signal persistence check ──────────────────────────────
                    # Require *persist_bars* consecutive bars where:
                    #   (a) all scores are positive (same direction)
                    #   (b) all scores ≥ 70 % of the entry threshold (decaying
                    #       signals that are about to reverse are filtered out)
                    persistent = False
                    if len(hist) >= self.persist_bars:
                        all_positive  = all(s > 0 for s in hist)
                        min_mag       = dec.enter_long * 0.70
                        strong_enough = all(abs(s) >= min_mag for s in hist)
                        persistent    = all_positive and strong_enough

                    if not persistent:
                        continue   # wait for consistent signal before entering

                    # ── Size by notional (fractional crypto qty) ──────────────
                    target_pct = min(
                        abs(dec.score) * dec.confidence * self.kelly,
                        self.max_position_pct,
                    )
                    notional   = mtm * target_pct
                    qty        = notional / price      # fractional
                    entry_fill = price * (1 + self.slippage)
                    cost       = qty * entry_fill

                    if notional >= 10.0 and cost <= cash:   # $10 minimum notional
                        cash -= cost
                        open_pos[sym] = {
                            "side":          "long",
                            "qty":           qty,
                            "qty_remaining": qty,
                            "entry_time":    ts,
                            "entry_price":   entry_fill,
                            "stop":          price * (1 - self.stop_pct),
                            "tp":            price * (1 + self.tp_pct),
                            "high_watermark": price,
                            "pp1_fired":     False,
                            "pp2_fired":     False,
                            "trailing_stop": None,
                            "realized_pnl":  0.0,
                        }

        # Close any positions still open at end of period
        for sym, pos in list(open_pos.items()):
            df   = bars_by_symbol[sym]
            fill = float(df["Close"].iloc[-1]) * (1 - self.slippage)
            rem  = pos["qty_remaining"]
            cash += fill * rem
            final_pnl = (fill - pos["entry_price"]) * rem
            trades.append(Trade(
                symbol=sym, entry_time=pos["entry_time"],
                entry_price=pos["entry_price"], side="long",
                exit_time=df.index[-1], exit_price=fill,
                qty=rem, reason="end_of_period",
            ))

        return BacktestResult(
            starting_cash    = self.starting_cash,
            ending_cash      = cash,
            trades           = trades,
            equity_curve     = equity_curve,
            periods_per_year = _bars_per_year(INTERVAL),
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    start_dt = _ARGS.start
    end_dt   = _ARGS.end

    print(f"\n{'='*65}")
    print(f"  Crypto Backtest  {start_dt.strftime('%Y/%m/%d')} → {end_dt.strftime('%Y/%m/%d')}")
    print(f"  Tickers  : {', '.join(CRYPTO_TICKERS)}")
    print(f"  Interval : {INTERVAL}   Cash: ${STARTING_CASH:,.0f}   "
          f"Slippage: {SLIPPAGE_BPS} bps")
    print(f"  Persistence filter : {PERSIST_BARS} bar(s)")
    print(f"  Risk     : stop={CRYPTO_CFG['risk']['per_trade_stop_loss_pct']*100:.0f}%  "
          f"TP={CRYPTO_CFG['risk']['take_profit_pct']*100:.0f}%  "
          f"trail={CRYPTO_CFG['risk']['trailing_stop_pct']*100:.0f}%  "
          f"min_hold={CRYPTO_CFG['risk']['min_hold_minutes']:.0f}min")
    print(f"  Partials : PP1={CRYPTO_CFG['risk']['partial_profit_1_pct']*100:.0f}%  "
          f"PP2={CRYPTO_CFG['risk']['partial_profit_2_pct']*100:.0f}%  "
          f"breakeven={CRYPTO_CFG['risk']['breakeven_trigger_pct']*100:.0f}%")
    print(f"{'='*65}\n")

    # ── Fetch bars ────────────────────────────────────────────────────────────
    print("Fetching crypto bars …")
    md = MarketData(
        provider = BASE_CFG["data"].get("provider", "alpaca"),
        interval = INTERVAL,
        lookback_days = 1,
        feed     = BASE_CFG["data"].get("feed", "iex"),
    )

    bars_by_sym: Dict[str, pd.DataFrame] = {}
    for sym in CRYPTO_TICKERS:
        df = md.get_bars(sym, start_dt=start_dt, end_dt=end_dt)
        if df is not None and not df.empty:
            bars_by_sym[sym] = df
            span = (f"{df.index[0].strftime('%m/%d/%Y')} → "
                    f"{df.index[-1].strftime('%m/%d/%Y')}")
            print(f"  {sym:<10s}  {len(df):6,d} bars  {span}")
        else:
            print(f"  {sym:<10s}  NO DATA — skipping")

    if not bars_by_sym:
        print("\nNo bars fetched. Check ALPACA_API_KEY and date range.")
        return

    # ── Walk-forward split ────────────────────────────────────────────────────
    print(f"\nWalk-forward split : first 50% → train ML,  last 50% → test")
    train_bars: Dict[str, pd.DataFrame] = {}
    test_bars:  Dict[str, pd.DataFrame] = {}
    for sym, df in bars_by_sym.items():
        n = len(df); s = n // 2
        if s >= 100:
            train_bars[sym] = df.iloc[:s]
            print(f"  {sym:<10s}  train={s:,}  test={n-s:,}  "
                  f"split @ {df.index[s].strftime('%m/%d/%Y %H:%M')}")
        if n - s >= 50:
            test_bars[sym] = df.iloc[s:]

    if not test_bars:
        print("Not enough bars for split. Try a wider date range.")
        return

    # ── Train crypto ML (LightGBM) on crypto bars only ───────────────────────
    print(f"\nTraining crypto LightGBM on {len(train_bars)} pairs …")
    ml  = MLSignal(CRYPTO_CFG["signals"]["ml"])
    ok  = ml.train(train_bars)
    if ok:
        print("  Crypto ML trained successfully (LightGBM).")
    else:
        print("  WARNING: ML training failed — inference returns score=0.")

    # ── Run backtest ──────────────────────────────────────────────────────────
    total_test_bars = sum(len(df) for df in test_bars.values())
    print(f"\nRunning on {len(test_bars)} pairs,  "
          f"{total_test_bars:,} bars …\n")

    bt       = CryptoBacktester(
        config       = CRYPTO_CFG,
        starting_cash= STARTING_CASH,
        slippage_bps = SLIPPAGE_BPS,
        interval     = INTERVAL,
        persist_bars = PERSIST_BARS,
    )
    bt.ml    = ml
    result   = bt.run(test_bars)

    # ── Print results ─────────────────────────────────────────────────────────
    wins   = [t for t in result.trades if t.pnl > 0]
    losses = [t for t in result.trades if t.pnl <= 0]
    pf     = (
        abs(sum(t.pnl for t in wins)) / abs(sum(t.pnl for t in losses))
        if losses and any(t.pnl < 0 for t in losses) else 999
    )
    avg_w  = np.mean([t.pnl_pct * 100 for t in wins])   if wins   else 0.0
    avg_l  = np.mean([t.pnl_pct * 100 for t in losses]) if losses else 0.0
    net    = result.ending_cash - result.starting_cash

    print(f"{'='*65}")
    print(f"  CRYPTO BACKTEST RESULTS")
    print(f"{'='*65}")
    print(f"  Period         : {start_dt.strftime('%Y/%m/%d')} – {end_dt.strftime('%Y/%m/%d')}")
    print(f"  Starting cash  : ${result.starting_cash:>12,.2f}")
    print(f"  Ending cash    : ${result.ending_cash:>12,.2f}")
    pnl_sign = "+" if net >= 0 else ""
    print(f"  Net P&L        : ${net:>+12,.2f}  ({result.total_return*100:+.2f}%)")
    print(f"{'─'*65}")
    print(f"  Total trades   : {result.num_trades}")
    if result.num_trades:
        print(f"  Win rate       : {len(wins)}/{result.num_trades}  ({result.win_rate*100:.1f}%)")
        print(f"  Avg win        : {avg_w:+.2f}%")
        print(f"  Avg loss       : {avg_l:+.2f}%")
        print(f"  Profit factor  : {pf:.2f}x")
        print(f"  Sharpe ratio   : {result.sharpe:.2f}")
        print(f"{'─'*65}")
        print(f"  Exit breakdown :")
        for reason, n in Counter(t.reason for t in result.trades).most_common():
            pct = n / result.num_trades * 100
            print(f"    {reason:<28s}  {n:3d}  ({pct:.0f}%)")
        print(f"{'─'*65}")
        print(f"  Per-symbol     :")
        sym_data: Dict[str, list] = {}
        for t in result.trades:
            sym_data.setdefault(t.symbol, []).append(t)
        for sym, ts_list in sorted(sym_data.items(),
                                   key=lambda x: -sum(t.pnl for t in x[1])):
            total_pnl = sum(t.pnl for t in ts_list)
            w_list = [t for t in ts_list if t.pnl > 0]
            l_list = [t for t in ts_list if t.pnl <= 0]
            aw = np.mean([t.pnl_pct * 100 for t in w_list]) if w_list else 0.0
            al = np.mean([t.pnl_pct * 100 for t in l_list]) if l_list else 0.0
            print(f"    {sym:<10s}  {len(ts_list):3d} trades  "
                  f"{len(w_list)}/{len(ts_list)} wins  "
                  f"P&L: ${total_pnl:>+8.2f}  "
                  f"avg_w: {aw:>+5.2f}%  avg_l: {al:>+5.2f}%")
    else:
        print("  No trades — check signal thresholds or date range.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
