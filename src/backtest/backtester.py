"""Vectorized backtester for the technical + ML signal stack.

This is intentionally simple — it walks bars in order, computes the
technical signal on each one, optionally consults the trained ML model,
and simulates entries/exits with the configured stops.

News and SEC signals are excluded from the backtest because reliable
intraday news archives aren't free; if you have them, plug them in via
`extra_signals_fn`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..signals.aggregator import SignalAggregator
from ..signals.ml_model import MLSignal
from ..signals.macro import MacroSignal
from ..signals.technical import TechnicalSignal
from ..utils.logger import get_logger

log = get_logger(__name__)


def _bars_per_year(interval: str) -> float:
    """Approximate number of bars in a US-equity trading year for a given interval."""
    days = 252
    table = {
        "1m": 390 * days,
        "2m": 195 * days,
        "5m": 78 * days,
        "15m": 26 * days,
        "30m": 13 * days,
        "1h": 6.5 * days,
        "60m": 6.5 * days,
        "1d": days,
    }
    return table.get(interval, 78 * days)


@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    qty: float = 0.0
    reason: str = ""

    @property
    def pnl(self) -> float:
        if self.exit_price is None:
            return 0.0
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def pnl_pct(self) -> float:
        if self.exit_price is None or self.entry_price == 0:
            return 0.0
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestResult:
    starting_cash: float
    ending_cash: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    # Number of bars in a trading year, used to annualize the Sharpe ratio.
    # 5m -> 78*252, 1d -> 252, etc. Set by Backtester from the bar interval.
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
        # Approximate per-bar Sharpe
        if len(self.equity_curve) < 2:
            return 0.0
        rets = pd.Series(self.equity_curve).pct_change().dropna()
        if rets.std() == 0:
            return 0.0
        # Annualize using the bar interval (set by the Backtester).
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
        slippage_bps: float = 5.0,
        interval: str = "5m",
    ):
        self.cfg = config
        self.starting_cash = starting_cash
        self.slippage = slippage_bps / 10_000.0
        self.interval = interval
        self.periods_per_year = _bars_per_year(interval)
        self.tech = TechnicalSignal(config["signals"]["technical"])
        self.ml = MLSignal(config["signals"]["ml"])
        self.macro = MacroSignal()
        self.agg = SignalAggregator(
            weights=config["signals"]["weights"],
            enter_long=config["signals"].get("enter_long_threshold", 0.35),
            enter_short=config["signals"].get("enter_short_threshold", -0.35),
            exit_thresh=config["signals"].get("exit_threshold", 0.10),
        )
        self.stop_pct = config["risk"]["per_trade_stop_loss_pct"]
        self.tp_pct = config["risk"]["take_profit_pct"]
        self.kelly = config["risk"].get("kelly_fraction", 0.25)
        self.max_position_pct = config["risk"].get("max_position_pct", 0.10)

    def run(self, bars_by_symbol: Dict[str, pd.DataFrame]) -> BacktestResult:
        # Align timeline across symbols using the union of indices.
        all_idx = sorted(set().union(*[df.index for df in bars_by_symbol.values()]))
        cash = self.starting_cash
        open_pos: Dict[str, dict] = {}      # symbol -> {entry_price, qty, stop, tp}
        trades: List[Trade] = []
        equity_curve = []

        warmup = 50  # need enough bars for indicators
        for i, ts in enumerate(all_idx):
            if i < warmup:
                continue
            # Mark equity
            mtm = cash + sum(
                p["qty"] * self._price_at(bars_by_symbol[s], ts) for s, p in open_pos.items()
            )
            equity_curve.append(mtm)

            # Evaluate Macro Signal
            spy_window = bars_by_symbol.get("SPY", pd.DataFrame())
            if not spy_window.empty and ts in spy_window.index:
                market_multiplier, _ = self.macro.evaluate(spy_window.loc[:ts])
            else:
                market_multiplier = 1.0

            # Per-symbol updates
            for sym, df in bars_by_symbol.items():
                if ts not in df.index:
                    continue
                window = df.loc[:ts]
                if len(window) < warmup:
                    continue
                t = self.tech.evaluate(sym, window)
                m = self.ml.evaluate(sym, window, tech_signal=t)
                sigs = [s for s in (t, m) if s is not None]
                if not sigs:
                    continue
                dec = self.agg.aggregate(sigs, market_multiplier=market_multiplier).get(sym)
                if dec is None:
                    continue
                price = float(df.at[ts, "Close"])

                # ----- manage open position -----
                if sym in open_pos:
                    pos = open_pos[sym]
                    exit_reason = None
                    if price <= pos["stop"]:
                        exit_reason = "stop"
                    elif price >= pos["tp"]:
                        exit_reason = "take_profit"
                    elif dec.score < -self.cfg["signals"].get("exit_threshold", 0.10):
                        exit_reason = "signal_reversed"
                    if exit_reason:
                        fill = price * (1 - self.slippage)
                        cash += fill * pos["qty"]
                        trades.append(
                            Trade(
                                symbol=sym,
                                entry_time=pos["entry_time"],
                                entry_price=pos["entry_price"],
                                exit_time=ts,
                                exit_price=fill,
                                qty=pos["qty"],
                                reason=exit_reason,
                            )
                        )
                        del open_pos[sym]

                # ----- consider entry -----
                if sym not in open_pos and dec.action == "BUY":
                    target_pct = min(abs(dec.score) * dec.confidence * self.kelly, self.max_position_pct)
                    notional = mtm * target_pct
                    qty = int(notional / price)
                    cost = qty * price * (1 + self.slippage)
                    if qty >= 1 and cost <= cash:
                        cash -= cost
                        open_pos[sym] = {
                            "qty": qty,
                            "entry_time": ts,
                            "entry_price": price * (1 + self.slippage),
                            "stop": price * (1 - self.stop_pct),
                            "tp": price * (1 + self.tp_pct),
                        }

        # Flatten anything still open at the end.
        last_ts = all_idx[-1] if all_idx else None
        for sym, pos in list(open_pos.items()):
            df = bars_by_symbol[sym]
            price = float(df["Close"].iloc[-1])
            fill = price * (1 - self.slippage)
            cash += fill * pos["qty"]
            trades.append(
                Trade(
                    symbol=sym,
                    entry_time=pos["entry_time"],
                    entry_price=pos["entry_price"],
                    exit_time=last_ts,
                    exit_price=fill,
                    qty=pos["qty"],
                    reason="eod_flatten",
                )
            )
        return BacktestResult(
            starting_cash=self.starting_cash,
            ending_cash=cash,
            trades=trades,
            equity_curve=equity_curve,
            periods_per_year=self.periods_per_year,
        )

    @staticmethod
    def _price_at(df: pd.DataFrame, ts) -> float:
        try:
            return float(df.loc[:ts, "Close"].iloc[-1])
        except (KeyError, IndexError):
            return 0.0
