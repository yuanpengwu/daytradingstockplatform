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
from pathlib import Path
from typing import Dict, List
from zoneinfo import ZoneInfo

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
from .signals.orb import ORBSignal
from .signals.regime import RegimeDetector, REGIME_ADJUSTMENTS
from .signals.sentiment import SentimentSignal
from .signals.technical import TechnicalSignal
from .signals.vwap_bounce import VWAPBounceSignal
from .utils.logger import get_logger
from .utils.notifications import notify
from .utils.status_page import write_status_page
from .utils.trade_history import TradeHistory

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
        self.orb = ORBSignal(scfg.get("orb", {}))
        self.vwap_bounce = VWAPBounceSignal(scfg.get("vwap_bounce", {}))
        self.macro = MacroSignal()
        self.regime = RegimeDetector()
        self.agg = SignalAggregator(
            weights=scfg.get("weights", {}),
            enter_long=scfg.get("enter_long_threshold", 0.35),
            enter_short=scfg.get("enter_short_threshold", -0.35),
            exit_thresh=scfg.get("exit_threshold", 0.10),
            min_confidence=scfg.get("min_confidence", 0.25),
        )

        # ----- risk + trader -----
        self.risk = RiskManager(config.get("risk", {}))
        ncfg = config.get("notifications", {})
        self.trade_history = TradeHistory(path=str(_PROJECT_ROOT / "trades.json"))
        self.trader = Trader(
            broker=self.broker,
            risk=self.risk,
            notify_channels=ncfg.get("channels", ["console"]),
            trade_history=self.trade_history,
            persistence_bars=int(scfg.get("entry_persistence_bars", 2)),
        )

        # ----- schedule -----
        sched = config.get("schedule", {})
        self.poll_seconds = int(sched.get("poll_seconds", 60))
        self.open_buffer_min = int(sched.get("market_open_buffer_minutes", 5))
        self.close_buffer_min = int(sched.get("market_close_buffer_minutes", 10))
        self.market_hours_only = bool(sched.get("trade_only_market_hours", True))
        self.status_path = sched.get("status_page", "status.html")

        # Midday dead zone — no new entries, but positions are still managed.
        self._dead_zone_start: dtime = self._parse_time(sched.get("no_trade_window_start", "11:30"))
        self._dead_zone_end:   dtime = self._parse_time(sched.get("no_trade_window_end",   "14:00"))

        # Pre-close entry cutoff — block new entries N minutes before the
        # close buffer (15:50 − 30 min = no new entries after 15:20 ET).
        # Prevents zero-P&L "last bar" trades that have no time to move.
        _no_entry_before_close = int(sched.get("no_entry_before_close_minutes", 30))
        _close_cutoff_dt = (
            datetime.combine(datetime.today(), dtime(16, 0))
            - timedelta(minutes=self.close_buffer_min + _no_entry_before_close)
        )
        self._entry_cutoff: dtime = _close_cutoff_dt.time()   # e.g. 15:20 ET

        # Relative strength filter config.
        _scfg = config.get("signals", {})
        self._rs_min:      float = float(_scfg.get("relative_strength_min", 0.0))
        self._rs_lookback: int   = int(_scfg.get("rs_lookback_bars", 6))

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
            self.trader.begin_day()

            dynamic_tickers = self.universe.select_tickers()
            # Ensure we keep monitoring any open positions so we can exit them
            open_positions = self.broker.get_positions()
            current_holdings = list(open_positions.keys())

            # Combine and remove duplicates while keeping order
            combined = dynamic_tickers + [t for t in current_holdings if t not in dynamic_tickers]
            self.tickers = combined

            log.info("Target Universe updated for today: %s", self.tickers)
            self._last_day_started = today

        # End-of-day flatten
        if self._near_close(now_ny):
            log.info("Approaching market close - flattening all positions.")
            self.trader.flatten_all("EOD flatten")
            return {}

        # ── Dead-zone gate: no new entries, positions still managed ──────────
        in_dead_zone = self._in_dead_zone(now_ny)
        if in_dead_zone:
            log.info(
                "DEAD ZONE (%s ET) — managing open positions only; no new entries until %s.",
                now_ny.strftime("%H:%M"),
                self._dead_zone_end.strftime("%H:%M"),
            )

        # 1. Collect signals
        spy_bars = self.market.get_bars("SPY")
        market_multiplier, macro_reason = self.macro.evaluate(spy_bars)

        # ── Market regime detection ────────────────────────────────────────
        regime, regime_meta = self.regime.detect(spy_bars)
        regime_adj = REGIME_ADJUSTMENTS.get(regime, REGIME_ADJUSTMENTS["choppy"])
        # Push regime size multiplier to risk manager (affects all new entries this cycle).
        self.risk.regime_size_mult = regime_adj["position_size_mult"]
        
        # ── Pass 1: fetch bars + run technical signal for every ticker ─────────
        # We need tech scores up-front so MLSignal.prepare_cycle() can pick the
        # top-N symbols worth a live LLM call this cycle.
        signals: List[Signal] = []
        bars_by_sym: dict = {}
        tech_by_sym: dict = {}

        for sym in self.tickers:
            bars = self.market.get_bars(sym)
            bars_by_sym[sym] = bars

            # Feed today's first bar open to the daily-trend filter (once per day).
            if not bars.empty:
                try:
                    bar_index = bars.index
                    if hasattr(bar_index, "tz_convert"):
                        today_bars = bars[bar_index.tz_convert(NY).date == today]
                    else:
                        today_bars = bars[bar_index.date == today]
                    if not today_bars.empty and "open" in today_bars.columns:
                        self.trader.update_day_open(sym, float(today_bars["open"].iloc[0]))
                except Exception as _e:
                    log.debug("Could not extract day-open for %s: %s", sym, _e)

            t = self.tech.evaluate(sym, bars)
            tech_by_sym[sym] = t
            if t:
                signals.append(t)

        # Tell MLSignal which symbols are candidates for a live call this cycle.
        ml_candidates = {
            sym: abs(t.score)
            for sym, t in tech_by_sym.items()
            if t is not None and abs(t.score) >= self.ml.tech_threshold
        }
        self.ml.prepare_cycle(ml_candidates)

        # ── Pass 2: all remaining signals (ML, sentiment, fundamental, ORB, VWAP)
        for sym in self.tickers:
            bars = bars_by_sym[sym]
            t = tech_by_sym.get(sym)
            s = self.sent.evaluate(sym)
            if s: signals.append(s)
            f = self.fund.evaluate(sym)
            if f: signals.append(f)
            m = self.ml.evaluate(sym, bars, tech_signal=t)
            if m: signals.append(m)
            o = self.orb.evaluate(sym, bars)
            if o: signals.append(o)
            v = self.vwap_bounce.evaluate(sym, bars)
            if v: signals.append(v)

        # 2. Aggregate — pass regime weight overrides so choppy/bear days
        #    automatically de-emphasise ORB/momentum and boost VWAP-bounce.
        decisions = self.agg.aggregate(
            signals,
            market_multiplier=market_multiplier,
            weight_overrides=regime_adj.get("weight_overrides") or None,
        )
        for sym, d in decisions.items():
            log.info(
                "DECISION %s | score=%+.3f conf=%.2f action=%s regime=%s comp=%s",
                sym, d.score, d.confidence, d.action, regime,
                {k: round(v, 3) for k, v in d.components.items()},
            )

        # 3. Manage open positions (uses latest decisions)
        self.trader.manage_open_positions(decisions)

        # 4. Consider new entries — subject to dead zone, regime, and RS filter.
        for sym, d in decisions.items():
            if d.action == "HOLD":
                continue

            # Gate A: midday dead zone — no new entries.
            if in_dead_zone:
                log.debug("SKIP entry %s — dead zone active.", sym)
                continue

            # Gate A2: pre-close entry cutoff — no new entries after 15:20 ET.
            # Prevents entering positions too late in the day to move meaningfully.
            if now_ny.time() >= self._entry_cutoff:
                log.debug("SKIP entry %s — past pre-close cutoff (%s ET).", sym, self._entry_cutoff.strftime("%H:%M"))
                continue

            # Gate B: relative strength filter for LONG entries only.
            #         Only trade stocks that are outperforming SPY by at least
            #         rs_min over the last rs_lookback bars (~30 min on 5m).
            if d.action == "BUY":
                rs = self._compute_rs(sym, bars_by_sym, spy_bars)
                if rs < self._rs_min:
                    log.info(
                        "SKIP %s long — RS %.4f < threshold %.4f (underperforming SPY).",
                        sym, rs, self._rs_min,
                    )
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

    def _in_dead_zone(self, now_ny: datetime) -> bool:
        """Return True during the midday low-volume window where new entries are blocked."""
        t = now_ny.time()
        return self._dead_zone_start <= t < self._dead_zone_end

    def _compute_rs(self, symbol: str, bars_by_sym: dict, spy_bars) -> float:
        """Relative strength of symbol vs SPY over the last rs_lookback bars.

        Positive → symbol outperforming SPY (bullish for entry).
        Returns 0.0 when data is unavailable (safe default, won't block entries).
        """
        sym_bars = bars_by_sym.get(symbol)
        if sym_bars is None or sym_bars.empty or spy_bars is None or spy_bars.empty:
            return 0.0
        n = self._rs_lookback + 1
        sym_close = sym_bars["Close"].astype(float)
        spy_close = spy_bars["Close"].astype(float)
        if len(sym_close) < n or len(spy_close) < n:
            return 0.0
        try:
            sym_ret = (float(sym_close.iloc[-1]) - float(sym_close.iloc[-n])) / float(sym_close.iloc[-n])
            spy_ret = (float(spy_close.iloc[-1]) - float(spy_close.iloc[-n])) / float(spy_close.iloc[-n])
            return round(sym_ret - spy_ret, 6)
        except (ZeroDivisionError, IndexError):
            return 0.0

    @staticmethod
    def _parse_time(t_str: str) -> dtime:
        """Parse 'HH:MM' string into a datetime.time object."""
        try:
            h, m = map(int, str(t_str).split(":"))
            return dtime(h, m)
        except Exception:
            return dtime(11, 30)  # safe fallback
