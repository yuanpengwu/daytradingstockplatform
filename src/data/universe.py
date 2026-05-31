"""Smart universe selection.

Three-layer sector intelligence:
  1. Multi-factor quantitative scoring  — momentum + relative-strength vs SPY
                                          + volume trend + trend health
  2. Gemini forward-looking prediction  — 1 API call per day, asks which
                                          sectors are most likely to move today
                                          based on macro / news / calendar
  3. Stock ranking within sectors       — ranks candidates by volume surge
                                          + ATR + momentum; picks the most
                                          active names rather than static lists

Config keys (under `universe:`):
  max_sectors        : 2        # pick top N sectors
  gemini_sector_call : true     # enable daily Gemini sector call
  stock_rank_by      : volume_atr  # (reserved for future modes)
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..utils.logger import get_logger

log = get_logger(__name__)

# ── Sector ETF universe ───────────────────────────────────────────────────────
SECTORS: Dict[str, str] = {
    "XLK":  "Technology",
    "XLV":  "Health Care",
    "XLF":  "Financials",
    "XLY":  "Consumer Discretionary",
    "XLC":  "Communication Services",
    "XLI":  "Industrials",
    "XLP":  "Consumer Staples",
    "XLE":  "Energy",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLB":  "Materials",
}

# High-liquidity stock pool per sector (ranked roughly by market cap)
SECTOR_STOCKS: Dict[str, List[str]] = {
    "XLK":  ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "AMD", "QCOM", "INTC", "TXN"],
    "XLV":  ["UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY"],
    "XLF":  ["JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "C", "BLK", "AXP"],
    "XLY":  ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG", "TJX", "TGT"],
    "XLC":  ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "CHTR", "TMUS", "EA"],
    "XLI":  ["HON", "UPS", "UNP", "BA", "RTX", "CAT", "LMT", "GE", "DE", "MMM"],
    "XLP":  ["PG", "PEP", "KO", "WMT", "COST", "PM", "MO", "TGT", "DG", "KMB"],
    "XLE":  ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HES"],
    "XLU":  ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "PEG"],
    "XLRE": ["AMT", "PLD", "CCI", "EQIX", "PSA", "O", "SPG", "WELL", "DLR", "AVB"],
    "XLB":  ["LIN", "APD", "SHW", "NEM", "ECL", "FCX", "CTVA", "DOW", "NUE", "VMC"],
}


class DynamicUniverse:
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("dynamic", False))
        self.max_tickers = int(cfg.get("max_tickers", 10))
        self.max_sectors = int(cfg.get("max_sectors", 2))
        self.method = cfg.get("method", "sector_intelligence")
        self.fallback: List[str] = cfg.get("fallback_tickers", ["SPY", "QQQ"])
        self.use_gemini = bool(cfg.get("gemini_sector_call", True))
        self._gemini_key: Optional[str] = os.getenv("GEMINI_API_KEY")

        # Daily Gemini cache — sector prediction
        self._gemini_picks: List[str] = []
        self._gemini_date: Optional[date] = None

    # ── public ────────────────────────────────────────────────────────────────

    def select_tickers_intraday(self) -> List[str]:
        """Hourly intraday sector rotation using 5-min bars (no Gemini call).

        Scores each sector ETF on the last 60 minutes of intraday action:
          • 1-hour return relative to SPY  (50 % weight)
          • Raw 1-hour return              (30 % weight)
          • Volume surge vs prior hour     (20 % weight)

        Then re-ranks stocks within the top sectors by intraday momentum
        and volume.  Returns an empty list on failure so the engine keeps
        the current pool unchanged.
        """
        if not self.enabled:
            return []
        try:
            return self._select_intraday_momentum()
        except Exception as exc:
            log.warning("Intraday universe refresh failed (%s) — keeping current pool.", exc)
            return []

    def select_tickers(self) -> List[str]:
        if not self.enabled:
            return self.fallback

        log.info("Dynamic Universe: running '%s' selection …", self.method)
        try:
            return self._select_sector_intelligence()
        except Exception as exc:
            log.error("Universe selection failed (%s). Falling back to: %s", exc, self.fallback)
            return self.fallback

    # ── Step 1: fetch ETF bars ────────────────────────────────────────────────

    def _fetch_daily_bars(self, symbols: List[str], lookback: int = 35) -> pd.DataFrame:
        """Return a multi-index (symbol, timestamp) daily-bar DataFrame."""
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed

        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError("Alpaca API keys missing — cannot fetch bar data.")

        client = StockHistoricalDataClient(key, secret)
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=lookback + 10)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX,
        )
        df = client.get_stock_bars(req).df
        return df if not df.empty else pd.DataFrame()

    # ── Step 2: multi-factor sector scoring ───────────────────────────────────

    def _score_sectors(
        self,
        etf_bars: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Score each sector on 4 factors; return (scores, 5d_returns)."""

        def _get_series(df: pd.DataFrame, symbol: str) -> Optional[pd.DataFrame]:
            if etf_bars.empty:
                return None
            idx = etf_bars.index
            # Handle both MultiIndex (symbol, ts) and flat index
            if isinstance(idx, pd.MultiIndex):
                if symbol not in idx.get_level_values(0):
                    return None
                return etf_bars.loc[symbol]
            return None

        # SPY 5-day return as the benchmark
        spy_df = _get_series(etf_bars, "SPY")
        spy_5d = 0.0
        if spy_df is not None and len(spy_df) >= 6:
            spy_5d = float(spy_df["close"].iloc[-1] / spy_df["close"].iloc[-6] - 1)

        scores: Dict[str, float] = {}
        returns: Dict[str, float] = {}

        for etf in SECTORS:
            df = _get_series(etf_bars, etf)
            if df is None or len(df) < 6:
                continue

            close = df["close"]
            volume = df["volume"]

            # Factor 1 — 5-day momentum
            momentum = float(close.iloc[-1] / close.iloc[-6] - 1)
            returns[etf] = momentum

            # Factor 2 — relative strength vs SPY
            rel_strength = momentum - spy_5d

            # Factor 3 — volume trend: 5-day avg vs 20-day avg
            if len(df) >= 21:
                vol_5 = float(volume.tail(5).mean())
                vol_20 = float(volume.tail(20).mean())
                vol_trend = (vol_5 / vol_20 - 1) if vol_20 > 0 else 0.0
            else:
                vol_trend = 0.0
            # Clip to prevent outliers dominating; scale to return-like range
            vol_trend_norm = max(-0.05, min(0.05, vol_trend * 0.05))

            # Factor 4 — trend health: is sector above its 20-day SMA?
            if len(df) >= 21:
                sma20 = float(close.tail(20).mean())
                trend_flag = 0.01 if float(close.iloc[-1]) > sma20 else -0.01
            else:
                trend_flag = 0.0

            # Weighted combination
            score = (
                0.35 * rel_strength
                + 0.25 * momentum
                + 0.25 * vol_trend_norm
                + 0.15 * trend_flag
            )
            scores[etf] = score

        return scores, returns

    # ── Step 3: Gemini forward-looking prediction ─────────────────────────────

    def _gemini_sector_prediction(
        self,
        scores: Dict[str, float],
        returns: Dict[str, float],
    ) -> List[str]:
        """Ask Gemini which sectors will be most active today. Cached per day."""
        today = date.today()

        # Serve from daily cache
        if self._gemini_date == today and self._gemini_picks:
            log.info("Gemini sector call: using today's cached picks %s.", self._gemini_picks)
            return self._gemini_picks

        if not self._gemini_key:
            log.debug("Gemini sector call skipped — no GEMINI_API_KEY.")
            return []

        # Build a ranked summary for the prompt
        ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        lines = []
        for etf in ranked:
            ret = returns.get(etf, 0.0)
            sc = scores.get(etf, 0.0)
            lines.append(
                f"  {SECTORS[etf]:30s} ({etf}): "
                f"5d_return={ret:+.2%}  factor_score={sc:+.4f}"
            )

        prompt = (
            f"Today is {today.strftime('%A, %B %d %Y')}.\n"
            f"Below are the 11 US stock-market sectors ranked by a quantitative score\n"
            f"(relative strength vs SPY + momentum + volume trend + trend health):\n\n"
            + "\n".join(lines)
            + "\n\n"
            "As an expert market analyst, considering:\n"
            "  • Today's macro environment (Fed policy, rates, dollar)\n"
            "  • Any notable earnings reports or economic data due this week\n"
            "  • Typical sector rotation patterns for this day of the week\n"
            "  • Current geopolitical or sector-specific news themes\n\n"
            f"Which {self.max_sectors} sectors are most likely to show the STRONGEST "
            "intraday price movement and trading volume today?\n\n"
            "Respond with ONLY valid JSON — no markdown fences:\n"
            '{"sectors": ["XLK", "XLF"], "reasoning": "one concise sentence"}\n\n'
            f"Valid ETF tickers: {list(SECTORS.keys())}"
        )

        try:
            from google import genai
            client = genai.Client(api_key=self._gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash", contents=prompt
            )
            raw = response.text.strip()
            # Strip any accidental markdown fences
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw.strip())
            picks = [s for s in result.get("sectors", []) if s in SECTORS]
            reasoning = result.get("reasoning", "")
            if picks:
                log.info(
                    "Gemini sector prediction → %s | %s",
                    [f"{s} ({SECTORS[s]})" for s in picks],
                    reasoning,
                )
                self._gemini_picks = picks
                self._gemini_date = today
                return picks
            log.warning("Gemini sector prediction returned no valid tickers.")
        except Exception as exc:
            log.warning("Gemini sector prediction failed: %s", exc)

        return []

    # ── Step 4: rank stocks within a sector ───────────────────────────────────

    def _rank_stocks_in_sector(
        self,
        etf: str,
        stock_bars: pd.DataFrame,
    ) -> List[str]:
        """Rank sector stocks by volume surge + ATR% + momentum.

        Higher rank = more likely to have strong intraday moves today.
        Falls back to static name order if bar data is missing.
        """
        candidates = SECTOR_STOCKS.get(etf, [])
        if stock_bars.empty or not candidates:
            return candidates

        scores: Dict[str, float] = {}
        for sym in candidates:
            try:
                if not isinstance(stock_bars.index, pd.MultiIndex):
                    scores[sym] = 0.0
                    continue
                if sym not in stock_bars.index.get_level_values(0):
                    scores[sym] = 0.0
                    continue
                df = stock_bars.loc[sym]
                if len(df) < 5:
                    scores[sym] = 0.0
                    continue

                close = df["close"]
                volume = df["volume"]
                high = df["high"]
                low = df["low"]

                # Volume surge: last day vs 20-day mean (clipped at 3×)
                vol_mean = float(volume.mean())
                vol_surge = float(volume.iloc[-1]) / vol_mean if vol_mean > 0 else 1.0
                vol_surge = min(vol_surge, 3.0) / 3.0  # normalise to [0,1]

                # ATR% (5-day average true range as % of price)
                n = min(6, len(df))
                hi = high.tail(n).values
                lo = low.tail(n).values
                cp = close.tail(n + 1).iloc[:-1].values   # previous closes, same length
                min_len = min(len(hi), len(lo), len(cp))
                hi, lo, cp = hi[:min_len], lo[:min_len], cp[:min_len]
                tr_df = pd.DataFrame({
                    "hl": hi - lo,
                    "hc": abs(hi - cp),
                    "lc": abs(lo - cp),
                })
                atr_pct = float(tr_df.max(axis=1).mean()) / float(close.iloc[-1]) if float(close.iloc[-1]) > 0 else 0.0
                atr_norm = min(atr_pct * 50, 1.0)  # 2% ATR → score 1.0

                # 5-day momentum (capped)
                period = min(5, len(df) - 1)
                mom = float(close.iloc[-1] / close.iloc[-period - 1] - 1) if period > 0 else 0.0
                mom_norm = max(-1.0, min(1.0, mom * 10))  # 10% → 1.0

                scores[sym] = 0.40 * vol_surge + 0.40 * atr_norm + 0.20 * mom_norm

            except Exception as exc:
                log.debug("Stock ranking failed for %s: %s", sym, exc)
                scores[sym] = 0.0

        ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        log.info(
            "Stock ranking [%s]: %s",
            SECTORS.get(etf, etf),
            [(s, f"{scores[s]:.2f}") for s in ranked[:5]],
        )
        return ranked

    # ── Intraday 5-min bar fetch ──────────────────────────────────────────────

    def _fetch_intraday_bars(
        self,
        symbols: List[str],
        lookback_hours: int = 4,
    ) -> Dict[str, "pd.DataFrame"]:
        """Fetch the last `lookback_hours` of 5-min bars for each symbol."""
        from datetime import timezone
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import DataFeed

        key    = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_API_SECRET")
        if not key or not secret:
            raise RuntimeError("Alpaca API keys missing")

        client   = StockHistoricalDataClient(key, secret)
        end_dt   = datetime.now(tz=timezone.utc)
        start_dt = end_dt - timedelta(hours=lookback_hours)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start_dt,
            end=end_dt,
            feed=DataFeed.IEX,
        )
        raw = client.get_stock_bars(req).df
        result: Dict[str, pd.DataFrame] = {}
        if raw.empty:
            return result
        for sym in symbols:
            try:
                if isinstance(raw.index, pd.MultiIndex):
                    if sym not in raw.index.get_level_values(0):
                        continue
                    result[sym] = raw.loc[sym].copy()
                else:
                    result[sym] = raw.copy()
            except Exception:
                pass
        return result

    # ── Intraday momentum selection ───────────────────────────────────────────

    def _select_intraday_momentum(self) -> List[str]:
        """Score sectors on last 60 min of 5-min bars; re-rank stocks intraday."""
        etf_symbols = list(SECTORS.keys()) + ["SPY"]
        log.info(
            "Intraday sector refresh: fetching 4h 5-min bars for %d sector ETFs …",
            len(SECTORS),
        )
        bars = self._fetch_intraday_bars(etf_symbols, lookback_hours=4)
        if not bars:
            raise RuntimeError("No intraday ETF bars returned.")

        # SPY 1-hour benchmark return
        spy_df = bars.get("SPY")
        spy_1h = 0.0
        if spy_df is not None and len(spy_df) >= 12:
            spy_1h = float(spy_df["close"].iloc[-1] / spy_df["close"].iloc[-12] - 1)

        # Score each sector ETF
        etf_scores: Dict[str, float] = {}
        for etf in SECTORS:
            df = bars.get(etf)
            if df is None or len(df) < 12:
                continue
            close  = df["close"]
            volume = df["volume"]

            ret_1h = float(close.iloc[-1] / close.iloc[-12] - 1)
            rs     = ret_1h - spy_1h

            # Volume surge: last 12 bars vs prior 12 bars
            vol_now   = float(volume.tail(12).mean())
            vol_prior = float(volume.iloc[-24:-12].mean()) if len(volume) >= 24 else vol_now
            vol_surge = (vol_now / vol_prior - 1) if vol_prior > 0 else 0.0
            vol_surge = max(-0.5, min(0.5, vol_surge))

            etf_scores[etf] = 0.50 * rs + 0.30 * ret_1h + 0.20 * vol_surge

        if not etf_scores:
            raise RuntimeError("No sectors could be scored from intraday bars.")

        ranked_etfs = sorted(etf_scores, key=etf_scores.get, reverse=True)  # type: ignore[arg-type]
        top_sectors = ranked_etfs[: self.max_sectors]
        log.info(
            "Intraday sector rotation → top %d: %s",
            self.max_sectors,
            [(s, SECTORS[s], f"{etf_scores[s]:+.4f}") for s in top_sectors],
        )

        # Fetch intraday bars for stock candidates
        stock_pool: List[str] = []
        for etf in top_sectors:
            stock_pool.extend(SECTOR_STOCKS.get(etf, []))
        seen: set = set()
        stock_pool = [s for s in stock_pool if not (s in seen or seen.add(s))]  # type: ignore[func-returns-value]

        try:
            stock_bars = self._fetch_intraday_bars(stock_pool, lookback_hours=4)
        except Exception as exc:
            log.warning("Stock intraday bar fetch failed (%s); using sector order.", exc)
            stock_bars = {}

        # Rank stocks within each sector by 1-hour return + volume surge
        stocks_per_sector = max(1, self.max_tickers // len(top_sectors))
        chosen: List[str] = []
        for etf in top_sectors:
            candidates = SECTOR_STOCKS.get(etf, [])
            sym_scores: Dict[str, float] = {}
            for sym in candidates:
                df = stock_bars.get(sym)
                if df is None or len(df) < 6:
                    sym_scores[sym] = 0.0
                    continue
                close  = df["close"]
                volume = df["volume"]
                n      = min(12, len(close))
                ret    = float(close.iloc[-1] / close.iloc[-n] - 1)
                vol_m  = float(volume.mean()) or 1.0
                vsurge = float(volume.iloc[-1]) / vol_m - 1
                sym_scores[sym] = 0.60 * ret + 0.40 * min(vsurge, 2.0) / 2.0

            ranked_stocks = sorted(sym_scores, key=sym_scores.get, reverse=True)  # type: ignore[arg-type]
            log.info(
                "Intraday stock ranking [%s]: %s",
                SECTORS.get(etf, etf),
                [(s, f"{sym_scores[s]:+.3f}") for s in ranked_stocks[:5]],
            )
            chosen.extend(ranked_stocks[:stocks_per_sector])

        # Always keep SPY and QQQ
        for t in ["SPY", "QQQ"]:
            if t not in chosen:
                chosen.append(t)

        log.info("Intraday universe (%d tickers): %s", len(chosen), chosen)
        return chosen

    # ── Main selection flow ───────────────────────────────────────────────────

    def _select_sector_intelligence(self) -> List[str]:
        # ── 1. Fetch ETF + SPY daily bars ──────────────────────────────────
        etf_symbols = list(SECTORS.keys()) + ["SPY"]
        log.info("Fetching %d-day daily bars for %d sector ETFs …", 30, len(SECTORS))
        etf_bars = self._fetch_daily_bars(etf_symbols, lookback=30)
        if etf_bars.empty:
            raise RuntimeError("ETF bar fetch returned empty DataFrame.")

        # ── 2. Multi-factor scoring ────────────────────────────────────────
        scores, returns = self._score_sectors(etf_bars)
        if not scores:
            raise RuntimeError("Could not compute sector scores (insufficient bar history).")

        quant_ranking = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
        log.info(
            "Quantitative sector ranking (top 5): %s",
            [(s, SECTORS[s], f"{scores[s]:+.4f}") for s in quant_ranking[:5]],
        )

        # ── 3. Gemini forward-looking prediction ───────────────────────────
        gemini_picks: List[str] = []
        if self.use_gemini and self._gemini_key:
            gemini_picks = self._gemini_sector_prediction(scores, returns)

        # ── 4. Blend: Gemini picks from top-6 quant candidates ────────────
        top6 = quant_ranking[:6]
        if gemini_picks:
            # Gemini picks that are quantitatively supported → front of list
            selected = [s for s in gemini_picks if s in top6]
            # Fill remaining slots with top quant sectors not already chosen
            for s in quant_ranking:
                if len(selected) >= self.max_sectors:
                    break
                if s not in selected:
                    selected.append(s)
        else:
            selected = quant_ranking[: self.max_sectors]

        selected = selected[: self.max_sectors]
        log.info(
            "Selected sectors: %s",
            [(s, SECTORS[s], f"score={scores.get(s, 0):+.4f}") for s in selected],
        )

        # ── 5. Fetch stock bars for ranking ────────────────────────────────
        stock_pool = []
        for etf in selected:
            stock_pool.extend(SECTOR_STOCKS.get(etf, []))
        # Deduplicate while preserving order
        seen: set = set()
        stock_pool = [s for s in stock_pool if not (s in seen or seen.add(s))]  # type: ignore

        log.info("Fetching 25-day daily bars for %d candidate stocks …", len(stock_pool))
        try:
            stock_bars = self._fetch_daily_bars(stock_pool, lookback=25)
        except Exception as exc:
            log.warning("Stock bar fetch failed (%s); falling back to name order.", exc)
            stock_bars = pd.DataFrame()

        # ── 6. Rank and select stocks from each sector ─────────────────────
        stocks_per_sector = max(1, self.max_tickers // len(selected))
        chosen: List[str] = []
        for etf in selected:
            ranked = self._rank_stocks_in_sector(etf, stock_bars)
            chosen.extend(ranked[:stocks_per_sector])

        # ── 7. Add fallback tickers (SPY / QQQ) and cap ────────────────────
        for t in self.fallback:
            if t not in chosen:
                chosen.append(t)
        chosen = chosen[: self.max_tickers + len(self.fallback)]

        log.info("Final universe (%d tickers): %s", len(chosen), chosen)
        return chosen
