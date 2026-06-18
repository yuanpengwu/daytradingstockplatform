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
from typing import Dict, List, Optional
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
from .signals.finrl_signal import FinRLSignal
from .signals.ml_model import MLSignal
from .signals.orb import ORBSignal
from .signals.regime import RegimeDetector, REGIME_ADJUSTMENTS, MarketRegimeDetector, MarketRegime
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
            feed=d.get("feed", "iex"),
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
        self.finrl = FinRLSignal(scfg.get("finrl", {}))
        self.orb = ORBSignal(scfg.get("orb", {}))
        self.vwap_bounce = VWAPBounceSignal(scfg.get("vwap_bounce", {}))
        self.macro = MacroSignal()
        self.regime = RegimeDetector()

        # Per-symbol ADX-based regime detectors for adaptive exit routing.
        # One instance per symbol — each keeps its own state / transition log.
        self._sym_regime_detectors: Dict[str, MarketRegimeDetector] = {}
        self.agg = SignalAggregator(
            weights=scfg.get("weights", {}),
            enter_long=scfg.get("enter_long_threshold", 0.35),
            enter_short=scfg.get("enter_short_threshold", -0.35),
            exit_thresh=scfg.get("exit_threshold", 0.10),
            min_confidence=scfg.get("min_confidence", 0.25),
            dead_signal_cycles=int(scfg.get("dead_signal_cycles", 3)),
            require_ml_finrl_agreement=scfg.get("require_ml_finrl_agreement", False),
        )

        # Regimes in which short entries are permitted.  In a bull regime, short
        # signals fire randomly and lose reliably — gate them out entirely.
        rcfg = config.get("risk", {})
        self._short_regimes: list = rcfg.get("short_regimes", ["trending_bear", "high_volatility"])

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
        self._last_universe_refresh: Optional[datetime] = None
        # Serialised decision list preserved across cycles and restarts so the
        # status page keeps showing last known scores when the market is closed.
        self._last_decisions: list = self._load_cached_decisions()
        # How often to run the intraday sector re-score during market hours.
        # Read from universe.universe_refresh_hours (not schedule.*) so it lives
        # next to the other universe config rather than being buried in schedule.
        # 0 or negative disables the periodic refresh entirely.
        _ucfg = config.get("universe", {})
        self._universe_refresh_hours: float = float(_ucfg.get("universe_refresh_hours", 2.0))
        self._cycle_count = 0
        self._started_at = datetime.now()

        # Train local ML model on startup if no saved model exists yet.
        # This runs once at launch (takes ~2s) so the model is ready by
        # the first trading cycle — even if the engine starts before market open.
        if self.ml._needs_retrain():
            log.info("ML startup training — fetching historical bars …")
            try:
                train_bars = {sym: self.market.get_bars(sym) for sym in self.tickers}
                ok = self.ml.train(train_bars)
                if ok:
                    log.info("ML startup training complete.")
                else:
                    log.warning("ML startup training produced no model — will retry at day start.")
            except Exception as _e:
                log.warning("ML startup training failed: %s", _e)

        # Train FinRL PPO agent on startup if no saved model exists yet.
        # Takes ~1-3 min on CPU or ~30s on GPU — runs once at launch.
        if self.finrl._needs_retrain():
            log.info("FinRL startup training — fetching historical bars …")
            try:
                # Reuse bars already fetched above (or fetch fresh if ml didn't run)
                if not self.ml._needs_retrain():
                    train_bars = {sym: self.market.get_bars(sym) for sym in self.tickers}
                ok_rl = self.finrl.train(train_bars)
                if ok_rl:
                    log.info("FinRL startup training complete.")
                else:
                    log.warning("FinRL startup training failed — signal will return score=0.")
            except Exception as _e:
                log.warning("FinRL startup training failed: %s", _e)

        # ── Next-day cooloff after stop-loss ──────────────────────────────────
        # _cooloff_until[sym] = date (exclusive) through which sym is banned.
        rcfg = config.get("risk", {})
        self._use_next_day_cooloff: bool = bool(rcfg.get("next_day_cooloff", True))
        self._cooloff_until: Dict[str, object] = {}   # sym → date

    # ---------- decision cache helpers ----------

    def _load_cached_decisions(self) -> list:
        """Seed _last_decisions from status.json so a restart doesn't clear scores."""
        try:
            import json
            p = _PROJECT_ROOT / "status.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                cached = data.get("decisions", [])
                if cached:
                    log.info("Engine: loaded %d cached decisions from status.json.", len(cached))
                    return cached
        except Exception:
            pass
        return []

    def _serialise_decisions(self, decisions: dict) -> list:
        """Convert {sym: AggregatedDecision} → list of dicts for JSON/status page."""
        out = []
        for sym, d in decisions.items():
            out.append({
                "symbol":     sym,
                "score":      d.score,
                "confidence": d.confidence,
                "action":     d.action,
                "components": {k: round(v, 4) for k, v in d.components.items()}
                              if hasattr(d, "components") else {},
                "raw_scores": {k: round(v, 4) for k, v in d.raw_scores.items()}
                              if hasattr(d, "raw_scores") else {},
            })
        return out

    # ---------- main loop ----------
    def run_forever(self) -> None:
        # Log the exact git commit so we always know which version is running.
        try:
            import subprocess as _sp
            _git = _sp.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            _commit = _git.stdout.strip() if _git.returncode == 0 else "unknown"
            _msg = _sp.run(
                ["git", "log", "-1", "--format=%s"],
                capture_output=True, text=True, timeout=5,
            )
            _subject = _msg.stdout.strip() if _msg.returncode == 0 else ""
        except Exception:
            _commit, _subject = "unknown", ""

        log.info(
            "Engine starting | broker=%s | tickers=%s | commit=%s (%s)",
            self.config["broker"]["name"],
            ",".join(self.tickers),
            _commit,
            _subject[:60],
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
            # Preserve last known decisions so the dashboard keeps showing scores
            # instead of going blank every time the market closes.
            write_status_page(
                self.status_path,
                broker=self.broker,
                decisions=self._last_decisions,
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
            # Pass the previous trading day so the performance tracker can run
            # its end-of-day evaluation (dynamic exclusion of weak symbols).
            _prev_day = (
                self._last_day_started.strftime("%Y-%m-%d")
                if self._last_day_started else None
            )
            self.trader.begin_day(_prev_day)

            # Train / retrain the local ML model at day-start using the most
            # recent historical bars.  This is a one-time ~2s cost per day;
            # inference during the day is free and sub-millisecond.
            _day_train_bars: dict = {}
            if self.ml._needs_retrain() or self.finrl._needs_retrain():
                log.info("Fetching bars for day-start ML/FinRL training …")
                try:
                    _day_train_bars = {sym: self.market.get_bars(sym) for sym in self.tickers}
                except Exception as _fe:
                    log.warning("Bar fetch for training failed: %s", _fe)

            if self.ml._needs_retrain() and _day_train_bars:
                try:
                    ok = self.ml.train(_day_train_bars)
                    if not ok:
                        log.warning("ML training failed — signal will return score=0 today.")
                except Exception as _train_err:
                    log.warning("ML training error: %s", _train_err)

            if self.finrl._needs_retrain() and _day_train_bars:
                try:
                    ok_rl = self.finrl.train(_day_train_bars)
                    if not ok_rl:
                        log.warning("FinRL training failed — signal will return score=0 today.")
                except Exception as _rl_err:
                    log.warning("FinRL training error: %s", _rl_err)

            # Expire cooloffs that have passed (ban_date is exclusive upper bound).
            if self._use_next_day_cooloff:
                expired = [s for s, d in self._cooloff_until.items() if d <= today]
                for s in expired:
                    del self._cooloff_until[s]
                    log.info("Cooloff expired for %s — eligible for entries today.", s)
                if self._cooloff_until:
                    log.info("Symbols still in cooloff: %s", list(self._cooloff_until.keys()))

            dynamic_tickers = self.universe.select_tickers()
            # Keep monitoring any open stock positions so we can exit them.
            # get_stock_positions() excludes crypto at the broker level — this
            # engine never needs to know that crypto positions exist.
            current_holdings = list(self.broker.get_stock_positions().keys())

            # Combine and remove duplicates while keeping order
            combined = dynamic_tickers + [t for t in current_holdings if t not in dynamic_tickers]
            self.tickers = combined

            log.info("Target Universe updated for today: %s", self.tickers)
            self._last_day_started = today
            self._last_universe_refresh = now_ny   # day-start counts as first refresh

        # ── Periodic intraday universe refresh ────────────────────────────────
        # Re-scores sector ETFs on intraday 5-min bars and rotates the stock
        # pool to whichever sectors are moving right now.
        # Frequency: universe.universe_refresh_hours in config.yaml (default 2h).
        elif self._should_refresh_universe(now_ny):
            self._intraday_universe_refresh(now_ny)

        # End-of-day flatten — TRENDING positions are skipped (they run overnight);
        # CHOPPY and NEUTRAL positions are always closed before market close.
        if self._near_close(now_ny):
            log.info("Approaching market close - flattening EOD-eligible positions.")
            self.trader.flatten_eod_eligible("eod_flatten")
            return {}

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
                    if not today_bars.empty and "Open" in today_bars.columns:
                        self.trader.update_day_open(sym, float(today_bars["Open"].iloc[0]))
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

        # ── Pass 2: all remaining signals (ML, FinRL, sentiment, fundamental, ORB, VWAP)
        for sym in self.tickers:
            bars = bars_by_sym[sym]
            t = tech_by_sym.get(sym)
            s = self.sent.evaluate(sym)
            if s: signals.append(s)
            f = self.fund.evaluate(sym)
            if f: signals.append(f)
            m = self.ml.evaluate(sym, bars, tech_signal=t)
            if m: signals.append(m)
            rl = self.finrl.evaluate(sym, bars, tech_signal=t)
            if rl: signals.append(rl)
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

        # Propagate the active-weight ratio to RiskManager so position sizes
        # are scaled up to compensate for structurally-suppressed confidence
        # when ML / sentiment / fundamental are offline.
        self.risk.update_dead_signal_ratio(self.agg.threshold_ratio)

        _dead  = self.agg.dead_sources
        _ratio = self.agg.threshold_ratio
        for sym, d in decisions.items():
            _r          = d.raw_scores
            _score_ok   = "✓" if (d.score >= d.enter_long or d.score <= d.enter_short) else "✗"
            _conf_ok    = "✓" if d.confidence >= d.min_confidence else "✗"
            _agree_tag  = "✓" if d.agreement_ok else "✗DISAGREE"
            _dead_str   = f" dead={sorted(_dead)} ratio={_ratio:.0%}" if _dead else ""
            _scaled_str = " [scaled]" if _ratio < 1.0 - 1e-4 else ""
            log.info(
                "DECISION %s | %s"
                "  score=%+.3f(gate=%+.3f%s)"
                "  conf=%.2f(gate=%.2f%s)"
                "  agree=%s"
                "  |  tech=%+.3f  sent=%+.3f  fund=%+.3f  ml=%+.3f"
                "  finrl=%+.3f  orb=%+.3f  vwap=%+.3f"
                "  |  regime=%s  macro=%.2fx%s%s",
                sym, d.action,
                d.score, d.enter_long, _score_ok,
                d.confidence, d.min_confidence, _conf_ok,
                _agree_tag,
                _r.get("technical",   0.0), _r.get("sentiment",   0.0),
                _r.get("fundamental", 0.0), _r.get("ml",          0.0),
                _r.get("finrl",       0.0), _r.get("orb",         0.0),
                _r.get("vwap_bounce", 0.0),
                regime, market_multiplier,
                _dead_str, _scaled_str,
            )

        # 3. Manage open positions (uses latest decisions)
        self.trader.manage_open_positions(decisions)

        # Register next-day cooloffs for any stop-losses that just fired.
        if self._use_next_day_cooloff and self.trader.recent_stop_losses:
            from datetime import date as _date
            import bisect as _bisect
            for sym in self.trader.recent_stop_losses:
                # Ban the symbol for the rest of today + the next trading day.
                # ban_until is exclusive (expiry check: d <= today lifts it).
                # Friday stops skip the weekend so Monday is still banned.
                _days_ahead = 4 if today.weekday() == 4 else 2
                ban_until = today + timedelta(days=_days_ahead)
                self._cooloff_until[sym] = ban_until
                log.info(
                    "COOLOFF: %s stopped out — banned until %s (next trading day).",
                    sym, ban_until,
                )

        # 4. Consider new entries — subject to regime and RS filter.
        for sym, d in decisions.items():
            if d.action == "HOLD":
                continue

            # Gate A: pre-close entry cutoff — no new entries after 15:20 ET.
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

            # Gate D: regime-gated short selling.
            #         Shorting in a bull market fires randomly and loses reliably.
            #         Only allow SELL entries when the regime is one of the
            #         configured short_regimes (trending_bear, high_volatility).
            if d.action == "SELL":
                if regime not in self._short_regimes:
                    log.info(
                        "SKIP %s short — regime '%s' not in allowed short regimes %s.",
                        sym, regime, self._short_regimes,
                    )
                    continue
                # For shorts: also apply relative-strength filter in reverse —
                # only short stocks that are *underperforming* SPY.
                rs = self._compute_rs(sym, bars_by_sym, spy_bars)
                if rs > self._rs_min:
                    log.info(
                        "SKIP %s short — RS %.4f > %.4f (outperforming SPY, not a short candidate).",
                        sym, rs, self._rs_min,
                    )
                    continue

            # Gate C: next-day cooloff — skip symbols that stopped out yesterday.
            if self._use_next_day_cooloff and sym in self._cooloff_until:
                log.info(
                    "SKIP %s — in cooloff until %s after prior stop-loss.",
                    sym, self._cooloff_until[sym],
                )
                continue

            atr = None
            tech_sig = next((s for s in signals if s.symbol == sym and s.source.value == "technical"), None)
            if tech_sig:
                atr = tech_sig.metadata.get("atr")

            # ── Per-symbol adaptive regime params ─────────────────────────
            sym_regime_params = self._detect_sym_regime_params(
                sym, bars_by_sym.get(sym)
            )
            self.trader.handle_decision(d, atr=atr, regime_params=sym_regime_params)

        # 5. Write the live status page (never let this break the loop).
        self._cycle_count += 1
        if decisions:
            self._last_decisions = self._serialise_decisions(decisions)
        write_status_page(
            self.status_path,
            broker=self.broker,
            decisions=self._last_decisions,   # always use serialised list
            cycle_count=self._cycle_count,
            started_at=self._started_at,
            refresh_seconds=max(10, self.poll_seconds),
            kill_switch=self.risk.kill_switch_engaged(),
            regime=regime,
        )

        return decisions

    # ---------- adaptive regime helpers ----------
    def _detect_sym_regime_params(self, sym: str, bars) -> dict:
        """Return per-position exit params for *sym* based on its own ADX regime.

        TRENDING  → no partial profit, hold overnight (eod_flatten=False)
        CHOPPY /
        NEUTRAL   → partial profit at +1 % / +2.5 %, close at EOD (eod_flatten=True)
        """
        if bars is None or bars.empty:
            # No data — default to conservative (CHOPPY) behaviour.
            return {"regime": "neutral", "pp1_pct": 0.010, "pp2_pct": 0.025, "eod_flatten": True}

        if sym not in self._sym_regime_detectors:
            _rcfg = self.config.get("regime", {})
            self._sym_regime_detectors[sym] = MarketRegimeDetector(
                adx_trend_thresh=float(_rcfg.get("adx_trend_thresh", 25.0)),
                adx_choppy_thresh=float(_rcfg.get("adx_choppy_thresh", 20.0)),
            )

        try:
            regime = self._sym_regime_detectors[sym].detect(bars)
        except Exception as _e:
            log.debug("Regime detection failed for %s: %s — defaulting NEUTRAL", sym, _e)
            regime = MarketRegime.NEUTRAL

        # Expose ADX value so the trader's entry gate can use it.
        adx_val = self._sym_regime_detectors[sym]._last_adx

        if regime == MarketRegime.TRENDING:
            return {
                "regime":      "trending",
                "pp1_pct":     9999.0,   # partial profit disabled — let winner run
                "pp2_pct":     9999.0,
                "eod_flatten": False,    # position may run overnight
                "adx":         adx_val,
            }
        else:
            return {
                "regime":      regime.value,
                "pp1_pct":     float(self.config.get("risk", {}).get("partial_profit_1_pct", 0.010)),
                "pp2_pct":     float(self.config.get("risk", {}).get("partial_profit_2_pct", 0.025)),
                "eod_flatten": True,     # close before market close as usual
                "adx":         adx_val,
            }

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

    def _should_refresh_universe(self, now_ny: datetime) -> bool:
        """True when enough market-hours time has passed for a universe refresh.

        Returns False immediately when:
          • universe_refresh_hours ≤ 0  (feature disabled in config)
          • no previous refresh timestamp recorded yet
        """
        if self._universe_refresh_hours <= 0 or self._last_universe_refresh is None:
            return False
        elapsed_h = (now_ny - self._last_universe_refresh).total_seconds() / 3600
        return elapsed_h >= self._universe_refresh_hours

    def _intraday_universe_refresh(self, now_ny: datetime) -> None:
        """Rotate the ticker pool based on current intraday sector momentum.

        Tickers with open positions are always kept in the pool even if they
        drop out of the fresh selection — they are never force-exited here.
        New tickers that enter the selection become eligible for entries immediately.
        """
        log.info(
            "Universe refresh at %s ET (every %.0fh) — re-scoring intraday sectors …",
            now_ny.strftime("%H:%M"),
            self._universe_refresh_hours,
        )
        new_tickers = self.universe.select_tickers_intraday()
        if not new_tickers:
            log.warning("Intraday refresh returned no tickers — keeping current pool.")
            self._last_universe_refresh = now_ny
            return

        # Carry over any tickers with open positions that didn't make the fresh cut.
        # They stay in self.tickers so the engine keeps fetching their bars and can
        # fire stops / TPs / trailing exits normally.
        # Stock positions ONLY — get_positions() would leak crypto pairs
        # (e.g. LINK/USD) into the stock universe, letting this engine apply
        # stock exit rules to the crypto engine's positions.
        open_syms = list(self.broker.get_stock_positions().keys())
        retained  = sorted(s for s in open_syms if s not in new_tickers)
        combined  = new_tickers + retained

        # Diff against the OLD universe (compare against fresh selection only —
        # not the carry-over set — so "removed" means truly dropped from next cycle).
        old_set   = set(self.tickers)
        new_set   = set(new_tickers)
        added     = sorted(new_set - old_set)
        removed   = sorted(old_set - new_set - set(retained))
        unchanged = sorted(old_set & new_set)

        if retained:
            log.info(
                "Universe refreshed: added %s, removed %s, unchanged %s"
                " (retained for open positions: %s)",
                added, removed, unchanged, retained,
            )
        else:
            log.info(
                "Universe refreshed: added %s, removed %s, unchanged %s",
                added, removed, unchanged,
            )

        self.tickers = combined
        self._last_universe_refresh = now_ny

    @staticmethod
    def _parse_time(t_str: str) -> dtime:
        """Parse 'HH:MM' string into a datetime.time object."""
        try:
            h, m = map(int, str(t_str).split(":"))
            return dtime(h, m)
        except Exception:
            return dtime(11, 30)  # safe fallback
