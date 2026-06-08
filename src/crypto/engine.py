"""24/7 Crypto trading engine.

Runs alongside the main stock TradingEngine in a background daemon thread.
Uses the same Alpaca broker account but operates completely independently —
no shared trading logic, no market-hours gate, no regime detection.

Architecture
────────────
CryptoEngine (this file) — orchestrates the per-cycle loop:
  1. Reconcile broker state       → CryptoTrader.reconcile_positions()
  2. Manage open positions        → CryptoTrader.manage_position()
  3. Evaluate new-entry signals   → CryptoTrader.place_entry()

CryptoTrader  (trader.py)  — position state + order submission
CryptoUniverse (universe.py) — which pairs to trade

The stock engine (src/engine.py) and stock Trader (src/execution/trader.py)
know nothing about crypto — they call broker.get_stock_positions() and only
ever see equities.
"""
from __future__ import annotations

import time
from datetime import date
from typing import Optional

from ..brokers.base import BrokerBase
from ..data.market_data import MarketData
from ..signals.aggregator import SignalAggregator
from ..signals.ml_model import MLSignal
from ..signals.technical import TechnicalSignal
from ..utils.logger import get_logger
from ..utils.status_page import write_crypto_decisions
from .trader import CryptoTrader
from .universe import CryptoUniverse

log = get_logger(__name__)


class CryptoEngine:
    """Lightweight 24/7 crypto trading loop.

    Two dedicated ML models
    ───────────────────────
      models/local_lgbm.pkl   — trained on stock bars  (TradingEngine)
      models/crypto_lgbm.pkl  — trained on crypto bars (CryptoEngine)
    """

    def __init__(self, broker: BrokerBase, config: dict):
        self.broker = broker
        ccfg = config.get("crypto", {})

        self.universe     = CryptoUniverse(ccfg)
        self.trader       = CryptoTrader(broker, config)
        self.poll_seconds = int(ccfg.get("poll_seconds", 60))
        self.enabled      = bool(ccfg.get("enabled", True))

        # Market data — crypto routing handled inside MarketData
        d = config.get("data", {})
        self.market = MarketData(
            provider=d.get("provider", "alpaca"),
            interval=ccfg.get("bar_interval", "1m"),
            lookback_days=int(ccfg.get("lookback_days", 10)),
            feed=d.get("feed", "iex"),
        )

        # Technical signal (shared indicator set, works on any OHLCV series)
        tech_cfg = config.get("signals", {}).get("technical", {})
        self._tech = TechnicalSignal(tech_cfg)

        # Dedicated crypto ML model — never trained on stock data
        crypto_ml_cfg = dict(ccfg.get("ml", {}))
        crypto_ml_cfg.setdefault("mode", "local")
        crypto_ml_cfg.setdefault("model_path", "models/crypto_lgbm.pkl")
        crypto_ml_cfg.setdefault("retrain_days", 1)
        crypto_ml_cfg.setdefault("prediction_horizon_minutes", 30)
        crypto_ml_cfg.setdefault("label_method", "fixed")
        crypto_ml_cfg.setdefault("min_confidence", float(ccfg.get("min_confidence", 0.45)))
        crypto_ml_cfg.setdefault("tech_threshold", 0.05)
        self._ml = MLSignal(crypto_ml_cfg)
        self._ml_last_trained: Optional[date] = None

        entry_thresh  = float(ccfg.get("enter_long_threshold",  0.35))
        entry_short   = float(ccfg.get("enter_short_threshold", -entry_thresh))
        min_conf      = float(ccfg.get("min_confidence", 0.45))
        self._agg = SignalAggregator(
            weights={"technical": 0.50, "ml": 0.50},
            enter_long=entry_thresh,
            enter_short=entry_short,
            min_confidence=min_conf,
        )
        self._short_notional: float = float(ccfg.get("short_max_notional", 300))

        # Entry gates (evaluated in _evaluate_entry)
        self._tech_gate:    float = float(ccfg.get("tech_score_gate", 0.05))
        self._max_exposure: float = float(ccfg.get("max_total_exposure_pct", 0.15))
        self._max_notional: float = float(ccfg.get("max_position_notional", 500))

        log.info(
            "CryptoEngine initialised | tickers=%s | notional=$%.0f | "
            "stop=%.0f%% tp=%.0f%% trail=%.0f%% | shorts=%s | ml_model=%s",
            self.universe.tickers,
            self._max_notional,
            self.trader._stop_pct * 100,
            self.trader._tp_pct * 100,
            self.trader._trail_pct * 100,
            "ENABLED" if self.trader.shorting_enabled else "disabled (Alpaca crypto = cash-only)",
            crypto_ml_cfg["model_path"],
        )

    # ── ML training ───────────────────────────────────────────────────────────

    def _train_crypto_ml(self) -> None:
        """Fetch crypto bars and train the dedicated LightGBM model.

        Bars for all configured pairs are pooled so the model learns patterns
        shared across BTC, ETH, SOL, AVAX, and LINK.
        """
        log.info(
            "CryptoEngine: training crypto ML model on %d pairs …",
            len(self.universe.tickers),
        )
        bars_by_sym = {}
        for sym in self.universe.tickers:
            try:
                df = self.market.get_bars(sym, force_refresh=True)
                if df is not None and not df.empty:
                    bars_by_sym[sym] = df
                    log.info("  %s: %d bars", sym, len(df))
            except Exception as e:
                log.warning("  %s: fetch failed — %s", sym, e)

        if not bars_by_sym:
            log.warning("CryptoEngine: no crypto bars fetched — skipping ML training.")
            return

        ok = self._ml.train(bars_by_sym)
        if ok:
            self._ml_last_trained = date.today()
            log.info("CryptoEngine: crypto ML model trained successfully.")
        else:
            log.warning("CryptoEngine: crypto ML training failed.")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        if not self.enabled:
            log.info("CryptoEngine disabled in config — exiting.")
            return

        self._train_crypto_ml()
        log.info("CryptoEngine started (24/7) | pairs=%s", self.universe.tickers)

        while True:
            try:
                today = date.today()
                if self._ml_last_trained != today:
                    self._train_crypto_ml()
                self._cycle()
            except Exception as e:
                log.error("CryptoEngine cycle error: %s", e, exc_info=True)
            time.sleep(self.poll_seconds)

    # ── One cycle ─────────────────────────────────────────────────────────────

    def _cycle(self) -> None:
        # get_crypto_positions() returns only crypto — no filtering needed
        positions = self.broker.get_crypto_positions()

        # Sync tracking with actual broker state (handles restarts)
        self.trader.reconcile_positions(positions)

        # Manage existing positions (stop/TP/trailing-stop checks)
        for sym in list(self.trader.tracked_symbols):
            pos = positions.get(sym)
            if pos is None:
                self.trader.cleanup(sym)
            else:
                self.trader.manage_position(pos)

        # ── Pass 1: evaluate signals for ALL pairs (for dashboard visibility) ──
        # Collect ALL signals first, then aggregate once — same pattern as the
        # stock engine. Calling aggregate() per ticker breaks dead-signal
        # detection (counters flip within a single cycle, causing false recovery
        # messages on every iteration).
        equity      = self.broker.get_equity()
        all_decisions: dict = {}
        bars_cache:  dict = {}
        all_signals: list  = []

        for sym in self.universe.tickers:
            bars = self.market.get_bars(sym)
            bars_cache[sym] = bars
            if bars is None or len(bars) < 20:
                continue
            try:
                tech_sig = self._tech.evaluate(sym, bars)
                ml_sig   = self._ml.evaluate(sym, bars, tech_signal=tech_sig)
                for s in (tech_sig, ml_sig):
                    if s is not None and s.confidence > 0:
                        all_signals.append(s)
            except Exception as e:
                log.debug("CryptoEngine signal eval %s: %s", sym, e)

        if all_signals:
            all_decisions = self._agg.aggregate(all_signals)

        # Write crypto signals to crypto_status.json for the dashboard
        write_crypto_decisions("status.json", all_decisions)

        # Per-cycle heartbeat — keeps the LIVE LOG alive between ML cache refreshes
        # (ML scores are cached for 30 min; without this the log goes silent).
        held = list(self.trader.tracked_symbols)
        dec_parts = []
        for sym in self.universe.tickers:
            d = all_decisions.get(sym)
            if d:
                tag = "[H] " if sym in held else ""
                dec_parts.append(f"{tag}{sym} {d.action} {d.score:+.2f}")
        log.info("CryptoEngine cycle | %s", "  |  ".join(dec_parts) if dec_parts else "no signals")

        # ── Pass 2: place new entries for untracked pairs ─────────────────────
        crypto_exposure  = sum(pos.market_value for pos in positions.values())
        max_exposure_usd = equity * self._max_exposure

        for sym in self.universe.tickers:
            if self.trader.is_tracking(sym):
                continue  # already in a position (long or short)
            if crypto_exposure >= max_exposure_usd:
                log.debug(
                    "Crypto exposure limit reached (%.0f/%.0f) — no new entries.",
                    crypto_exposure, max_exposure_usd,
                )
                break

            dec = all_decisions.get(sym)
            if dec is None:
                continue

            # Re-check tech gate
            tech_sig = self._tech.evaluate(sym, bars_cache.get(sym)) \
                       if bars_cache.get(sym) is not None else None
            if tech_sig is None or abs(tech_sig.score) < self._tech_gate:
                continue

            price = self.broker.get_last_price(sym)
            if price <= 0:
                continue

            try:
                if dec.action == "BUY":
                    notional = min(self._max_notional, equity * 0.05)
                    if notional >= 10:
                        self.trader.place_entry(sym, price, notional, dec)

                elif dec.action == "SELL" and self.trader.shorting_enabled:
                    notional = min(self._short_notional, equity * 0.03)
                    if notional >= 10:
                        self.trader.place_short_entry(sym, price, notional, dec)

            except Exception as e:
                log.warning("CryptoEngine entry failed for %s: %s", sym, e, exc_info=True)
