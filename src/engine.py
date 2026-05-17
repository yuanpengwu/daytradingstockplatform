"""Main orchestration loop.

Each cycle (every `schedule.poll_seconds`):
  1. Pull fresh OHLCV bars for the universe.
  2. Run every signal engine.
  3. Aggregate signals into a per-symbol decision.
  4. Hand each decision to the Trader (risk checks -> orders).
  5. Manage open positions (stops, trailing, end-of-day flatten).
"""
from __future__ import annotations

import time
from datetime import datetime, time as dtime, timedelta
from typing import Dict, List
from zoneinfo import ZoneInfo

from .brokers import get_broker
from .data.market_data import MarketData
from .data.universe import DynamicUniverse
from .execution.trader import Trader
from .risk.risk_manager import RiskManager
from .signals.aggregator import SignalAggregator
from .signals.base import Signal
from .signals.fundamental import FundamentalSignal
from .signals.macro import MacroSignal
from .signals.ml_model import MLSignal
from .signals.sentiment import SentimentSignal
from .signals.technical import TechnicalSignal
from .utils.logger import get_logger
from .utils.notifications import notify
from .utils.status_page import write_status_page

log = get_logger(__name__)
NY = ZoneInfo("America/New_York")


class TradingEngine:
    def __init__(self, config: dict):
        self.config = config

        # ----- data -----
        self.universe = DynamicUniverse(config.get("universe", {}))
        self.tickers: List[str] = self.universe.select_tickers()
        d = config.get("data", {})
        self.market = MarketData(
            provider=d.get("provider", "yfinance"),
            interval=d.get("bar_interval", "5m"),
            lookback_days=d.get("lookback_days", 30),
        )
        # ----- broker -----
        bcfg = config.get("broker", {})
        broker_name = bcfg.get("name", "paper")
        self.broker = get_broker(broker_name, bcfg)

        # ----- signal engines -----
        scfg = config.get("signals", {})
        self.tech = TechnicalSignal(scfg.get("technical", {}))
        self.sent = SentimentSignal(scfg.get("sentiment", {}))
        self.fund = FundamentalSignal(scfg.get("fundamental", {}))
        self.ml = MLSignal(scfg.get("ml", {}))
        self.macro = MacroSignal()
        self.agg = SignalAggregator(
            weights=scfg.get("weights", {}),
            enter_long=scfg.get("enter_long_threshold", 0.35),
            enter_short=scfg.get("enter_short_threshold", -0.35),
            exit_thresh=scfg.get("exit_threshold", 0.10),
        )

        # ----- risk + trader -----
        self.risk = RiskManager(config.get("risk", {}))
        ncfg = config.get("notifications", {})
        self.trader = Trader(
            broker=self.broker,
            risk=self.risk,
            notify_channels=ncfg.get("channels", ["console"]),
        )

        # ----- schedule -----
        sched = config.get("schedule", {})
        self.poll_seconds = int(sched.get("poll_seconds", 60))
        self.open_buffer_min = int(sched.get("market_open_buffer_minutes", 5))
        self.close_buffer_min = int(sched.get("market_close_buffer_minutes", 10))
        self.market_hours_only = bool(sched.get("trade_only_market_hours", True))
        self.status_path = sched.get("status_page", "status.html")

        self._last_day_started = None
        self._cycle_count = 0
        self._started_at = datetime.now()

    # ---------- main loop ----------
    def run_forever(self) -> None:
        log.info(
            "Engine starting | broker=%s | tickers=%s",
            self.config["broker"]["name"],
            ",".join(self.tickers),
        )
        try:
            while True:
                cycle_start = time.time()
                try:
                    self._cycle()
                except Exception as e:
                    log.exception("Cycle error: %s", e)
                # Sleep the remainder of the polling interval
                elapsed = time.time() - cycle_start
                time.sleep(max(1.0, self.poll_seconds - elapsed))
        except KeyboardInterrupt:
            log.info("Engine shutdown requested by user.")
            self.broker.shutdown()

    def run_once(self) -> Dict[str, "AggregatedDecision"]:
        """Run a single cycle. Useful for testing & cron-style scheduling."""
        return self._cycle()

    # ---------- one cycle ----------
    def _cycle(self):
        now_ny = datetime.now(NY)
        if self.market_hours_only and not self._within_trading_window(now_ny):
            log.debug("Outside trading window (%s NY); skipping.", now_ny.time())
            self._cycle_count += 1
            write_status_page(
                self.status_path,
                broker=self.broker,
                decisions={},
                cycle_count=self._cycle_count,
                started_at=self._started_at,
                refresh_seconds=max(10, self.poll_seconds),
                kill_switch=self.risk.kill_switch_engaged(),
            )
            return {}

        # Daily kick-off
        today = now_ny.date()
        if self._last_day_started != today:
            self.risk.begin_day(self.broker.get_equity())
            
            dynamic_tickers = self.universe.select_tickers()
            # Ensure we keep monitoring any open positions so we can exit them
            open_positions = self.broker.get_positions()
            current_holdings = list(open_positions.keys())
            
            # Combine and remove duplicates while keeping order
            combined = dynamic_tickers + [t for t in current_holdings if t not in dynamic_tickers]
            self.tickers = combined
            
            log.info(f"Target Universe updated for today: {self.tickers}")
            self._last_day_started = today

        # End-of-day flatten
        if self._near_close(now_ny):
            log.info("Approaching market close - flattening all positions.")
            self.trader.flatten_all("EOD flatten")
            return {}

        # 1. Collect signals
        spy_bars = self.market.get_bars("SPY")
        market_multiplier, macro_reason = self.macro.evaluate(spy_bars)
        
        signals: List[Signal] = []
        bars_by_sym = {}
        for sym in self.tickers:
            bars = self.market.get_bars(sym)
            bars_by_sym[sym] = bars
            t = self.tech.evaluate(sym, bars)
            if t: signals.append(t)
            s = self.sent.evaluate(sym)
            if s: signals.append(s)
            f = self.fund.evaluate(sym)
            if f: signals.append(f)
            m = self.ml.evaluate(sym, bars, tech_signal=t)
            if m: signals.append(m)

        # 2. Aggregate
        decisions = self.agg.aggregate(signals, market_multiplier=market_multiplier)
        for sym, d in decisions.items():
            log.info(
                "DECISION %s | score=%+.3f conf=%.2f action=%s comp=%s",
                sym, d.score, d.confidence, d.action,
                {k: round(v, 3) for k, v in d.components.items()},
            )

        # 3. Manage open positions (uses latest decisions)
        self.trader.manage_open_positions(decisions)

        # 4. Consider new entries
        for sym, d in decisions.items():
            if d.action == "HOLD":
                continue
            atr = None
            tech_sig = next((s for s in signals if s.symbol == sym and s.source.value == "technical"), None)
            if tech_sig:
                atr = tech_sig.metadata.get("atr")
            self.trader.handle_decision(d, atr=atr)

        # 5. Write the live status page (never let this break the loop).
        self._cycle_count += 1
        write_status_page(
            self.status_path,
            broker=self.broker,
            decisions=decisions,
            cycle_count=self._cycle_count,
            started_at=self._started_at,
            refresh_seconds=max(10, self.poll_seconds),
            kill_switch=self.risk.kill_switch_engaged(),
        )

        return decisions

    # ---------- windowing helpers ----------
    def _within_trading_window(self, now_ny: datetime) -> bool:
        if now_ny.weekday() >= 5:
            return False
        open_t = dtime(9, 30)
        close_t = dtime(16, 0)
        start = (datetime.combine(now_ny.date(), open_t) + timedelta(minutes=self.open_buffer_min)).time()
        end = (datetime.combine(now_ny.date(), close_t) - timedelta(minutes=self.close_buffer_min)).time()
        return start <= now_ny.time() <= end

    def _near_close(self, now_ny: datetime) -> bool:
        close_t = dtime(16, 0)
        end = (datetime.combine(now_ny.date(), close_t) - timedelta(minutes=self.close_buffer_min)).time()
        return end <= now_ny.time() <= close_t
