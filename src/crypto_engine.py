"""24/7 Crypto trading engine.

Runs alongside the main stock TradingEngine in a background thread.
Uses the same Alpaca broker account, same signal stack (technical + ML),
but with crypto-specific risk parameters and no market-hours gate.

Crypto differences vs stocks
─────────────────────────────
  • 24/7 markets — runs on weekends and after-hours
  • Orders sized by notional $ (not shares)
  • Wider stops / larger take-profits (higher volatility)
  • GTC time-in-force (DAY is invalid for crypto)
  • No PDT restriction
  • No regime detection (crypto doesn't correlate to NYSE schedule)
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Dict, List, Optional

from .brokers.base import BrokerBase, Order, OrderSide, OrderType, Position, is_crypto_symbol
from .data.market_data import MarketData
from .signals.technical import TechnicalSignal
from .signals.ml_model import MLSignal
from .signals.aggregator import SignalAggregator, AggregatedDecision
from .utils.logger import get_logger
from .utils.notifications import notify_order_entry, notify_order_exit

log = get_logger(__name__)


class CryptoEngine:
    """Lightweight 24/7 crypto trading loop.

    Polls every *poll_seconds* (default 60), evaluates technical + ML signals
    for each configured crypto pair, and places notional market orders when
    conviction is high enough.

    Position management:
      • Trailing stop at *trailing_stop_pct* (default 4 %)
      • Take-profit at *take_profit_pct* (default 10 %)
      • Stop-loss at *per_trade_stop_loss_pct* (default 5 %)
    """

    def __init__(self, broker: BrokerBase, config: dict):
        self.broker = broker
        ccfg = config.get("crypto", {})

        self.tickers: List[str]    = ccfg.get("tickers", ["BTC/USD", "ETH/USD"])
        self.poll_seconds: int     = int(ccfg.get("poll_seconds", 60))
        self.enabled: bool         = bool(ccfg.get("enabled", True))

        # Market data (reuses existing MarketData with crypto routing)
        d = config.get("data", {})
        self.market = MarketData(
            provider=d.get("provider", "alpaca"),
            interval=ccfg.get("bar_interval", "1m"),
            lookback_days=int(ccfg.get("lookback_days", 10)),
            feed=d.get("feed", "iex"),
        )

        # Risk params
        self._max_notional:   float = float(ccfg.get("max_position_notional", 500))
        self._max_exposure:   float = float(ccfg.get("max_total_exposure_pct", 0.15))
        self._stop_pct:       float = float(ccfg.get("per_trade_stop_loss_pct", 0.05))
        self._tp_pct:         float = float(ccfg.get("take_profit_pct", 0.10))
        self._trail_pct:      float = float(ccfg.get("trailing_stop_pct", 0.04))
        self._min_conf:       float = float(ccfg.get("min_confidence", 0.30))
        self._entry_thresh:   float = float(ccfg.get("enter_long_threshold", 0.35))
        self._tech_gate:      float = float(ccfg.get("tech_score_gate", 0.05))
        self._min_adx:        float = float(ccfg.get("min_entry_adx", 20))

        # Signals — technical primary, ML secondary
        # Note: the ML model was trained on stocks → low confidence on crypto is
        # expected and normal.  We use a technical-heavy weighting (80/20) and
        # a lower min_confidence gate so the technical signal can still trigger
        # entries when the market is genuinely trending.
        sig_cfg = config.get("signals", {})
        tech_cfg = sig_cfg.get("technical", {})
        self._tech = TechnicalSignal(tech_cfg)

        ml_cfg = sig_cfg.get("ml", {})
        self._ml = MLSignal(ml_cfg)

        # Aggregator — technical-heavy for crypto (ML less reliable on non-stock data)
        self._agg = SignalAggregator(
            weights={"technical": 0.80, "ml": 0.20},
            enter_long=self._entry_thresh,
            enter_short=-self._entry_thresh,
            min_confidence=self._min_conf,
        )

        # Per-position tracking
        self._trail_high: Dict[str, float] = {}
        self._stops:      Dict[str, tuple] = {}   # (stop_price, tp_price)
        self._entry_time: Dict[str, datetime] = {}
        self._entry_price: Dict[str, float] = {}

        # Notification channels
        notif_cfg = config.get("notifications", {})
        self._channels = list(notif_cfg.get("channels", ["console"]))

        log.info(
            "CryptoEngine initialised | tickers=%s | notional=$%.0f | "
            "stop=%.0f%% tp=%.0f%% trail=%.0f%%",
            self.tickers, self._max_notional,
            self._stop_pct * 100, self._tp_pct * 100, self._trail_pct * 100,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        if not self.enabled:
            log.info("CryptoEngine disabled in config — exiting.")
            return
        log.info("CryptoEngine started (24/7) | pairs=%s", self.tickers)
        while True:
            try:
                self._cycle()
            except Exception as e:
                log.error("CryptoEngine cycle error: %s", e, exc_info=True)
            time.sleep(self.poll_seconds)

    def _cycle(self) -> None:
        positions = self.broker.get_positions()

        # ── Manage existing crypto positions ──────────────────────────────────
        for sym in list(self._trail_high.keys()):
            pos = positions.get(sym)
            if pos is None:
                # Position closed externally — clean up
                self._cleanup(sym)
                continue
            self._manage_position(pos)

        # ── Evaluate new entries ──────────────────────────────────────────────
        equity = self.broker.get_equity()
        crypto_exposure = sum(
            pos.market_value for sym, pos in positions.items() if is_crypto_symbol(sym)
        )
        max_exposure_usd = equity * self._max_exposure

        for sym in self.tickers:
            if sym in self._trail_high:
                continue   # already holding
            if crypto_exposure >= max_exposure_usd:
                log.debug("Crypto exposure limit reached (%.0f/%.0f) — skipping new entries.",
                          crypto_exposure, max_exposure_usd)
                break

            bars = self.market.get_bars(sym)
            if bars is None or len(bars) < 20:
                continue

            self._evaluate_entry(sym, bars, equity)

    # ── Entry evaluation ──────────────────────────────────────────────────────

    def _evaluate_entry(self, sym: str, bars, equity: float) -> None:
        try:
            tech_sig = self._tech.evaluate(sym, bars)
            if tech_sig is None or abs(tech_sig.score) < self._tech_gate:
                return  # weak technical signal

            ml_sig = self._ml.evaluate(sym, bars, tech_signal=tech_sig)

            # aggregate() expects List[Signal] and returns Dict[symbol, AggregatedDecision]
            signal_list = [s for s in [tech_sig, ml_sig]
                           if s is not None and hasattr(s, "confidence") and s.confidence > 0]
            if not signal_list:
                return

            decisions = self._agg.aggregate(signal_list)
            dec = decisions.get(sym)
            if dec is None or dec.action != "BUY":   # crypto long-only
                return

            price = self.broker.get_last_price(sym)
            if price <= 0:
                return

            # Size by notional — cap at config max
            notional = min(self._max_notional, equity * 0.05)  # max 5% per trade
            if notional < 10:
                log.debug("Crypto %s: notional $%.2f too small — skipping.", sym, notional)
                return

            self._place_entry(sym, price, notional, dec)
        except Exception as e:
            log.warning("CryptoEngine entry eval failed for %s: %s", sym, e, exc_info=True)

    def _place_entry(self, sym: str, price: float, notional: float, dec: AggregatedDecision) -> None:
        stop_price = price * (1 - self._stop_pct)
        tp_price   = price * (1 + self._tp_pct)

        order = Order(
            symbol=sym,
            side=OrderSide.BUY,
            qty=0.0,             # crypto uses notional instead
            notional=notional,
            type=OrderType.MARKET,
        )
        result = self.broker.submit_order(order)
        if result.status.value != "filled":
            return

        fill_price = result.filled_avg_price or price
        self._trail_high[sym]  = fill_price
        self._stops[sym]       = (stop_price, tp_price)
        self._entry_time[sym]  = datetime.now()
        self._entry_price[sym] = fill_price

        log.info(
            "CRYPTO ENTRY %s | notional=$%.2f fill=$%.4f SL=$%.4f TP=$%.4f "
            "score=%+.3f conf=%.2f",
            sym, notional, fill_price, stop_price, tp_price,
            dec.score, dec.confidence,
        )
        notify_order_entry(
            symbol=sym, side="buy",
            qty=notional / fill_price,
            price=fill_price,
            agg_score=dec.score, confidence=dec.confidence,
            min_confidence=self._min_conf, agreement_ok=True,
            components=dec.components,
            stop_loss=round(stop_price, 4),
            take_profit=round(tp_price, 4),
            channels=self._channels,
        )

    # ── Position management ───────────────────────────────────────────────────

    def _manage_position(self, pos: Position) -> None:
        sym   = pos.symbol
        price = pos.current_price or self.broker.get_last_price(sym)
        if price <= 0:
            return

        # Update trailing high
        self._trail_high[sym] = max(self._trail_high.get(sym, price), price)
        peak = self._trail_high[sym]
        stop_price, tp_price = self._stops.get(sym, (None, None))

        reason = None
        if tp_price and price >= tp_price:
            reason = "take_profit"
        elif stop_price and price <= stop_price:
            reason = "stop_loss"
        elif price <= peak * (1 - self._trail_pct):
            reason = "trailing_stop"

        if reason:
            self._close_position(pos, reason)

    def _close_position(self, pos: Position, reason: str) -> None:
        order = Order(
            symbol=pos.symbol,
            side=OrderSide.SELL,
            qty=abs(pos.qty),
            type=OrderType.MARKET,
        )
        result = self.broker.submit_order(order)
        if result.status.value != "filled":
            return

        exit_price  = result.filled_avg_price or pos.current_price
        entry_price = self._entry_price.get(pos.symbol, pos.avg_entry_price)
        pnl         = (exit_price - entry_price) * abs(pos.qty)
        pnl_pct     = (exit_price - entry_price) / entry_price if entry_price else 0.0

        log.info(
            "CRYPTO EXIT %s | reason=%s exit=$%.4f pnl=%+.2f (%+.2f%%)",
            pos.symbol, reason, exit_price, pnl, pnl_pct * 100,
        )
        notify_order_exit(
            symbol=pos.symbol, side="sell",
            qty=abs(pos.qty), pnl=pnl, pnl_pct=pnl_pct,
            reason=reason,
            entry_price=entry_price, exit_price=exit_price,
            held_since=self._entry_time.get(pos.symbol),
            channels=self._channels,
        )
        self._cleanup(pos.symbol)

    def _cleanup(self, sym: str) -> None:
        self._trail_high.pop(sym, None)
        self._stops.pop(sym, None)
        self._entry_time.pop(sym, None)
        self._entry_price.pop(sym, None)
