# DayTradingBot — Regime-Adaptive Automated Day-Trading Platform

A fully-automated, multi-signal day-trading platform for **US Stocks & ETFs**.
It combines **8 signal sources** — including two trained ML models — into a
weighted decision engine that routes each trade through a **market-regime
detector**, applying different exit strategies depending on whether the stock is
trending or choppy. A daily **Serenity supply-chain bottleneck screen** injects
niche "Invisible Champion" stocks alongside standard sector picks. Executes
through a pluggable broker layer (Alpaca or built-in paper simulator) with
strict risk controls.

---

## Architecture

```
              ┌──────────────── Data Layer ────────────────────┐
              │  MarketData  (yfinance / Polygon / Alpaca)      │
              │  NewsFeed    (NewsAPI / Finnhub)                │
              │  SECFilings  (EDGAR RSS)                        │
              └──────────────────────┬──────────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────────┐
              │                 Signal Engines                   │
              │                                                  │
              │  Rule-based                                      │
              │    TechnicalSignal  — RSI · MACD · BB · EMA ·   │
              │                       ATR · Volume              │
              │    ORBSignal        — Opening Range Breakout     │
              │    VWAPBounceSignal — VWAP bounce / cross        │
              │    SentimentSignal  — VADER NLP on headlines     │
              │    FundamentalSignal— SEC 8-K keyword scoring    │
              │    MacroSignal      — SPY trend multiplier       │
              │                                                  │
              │  Trained ML models (retrained daily/weekly)      │
              │    MLSignal   — LightGBM classifier (AUC ~0.72) │
              │    FinRLSignal— PPO reinforcement-learning agent │
              │                 (incremental daily update)       │
              │                                                  │
              │  Each source emits a normalised [-1, +1] score   │
              └──────────────────────┬──────────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────────┐
              │         SignalAggregator — weighted vote         │
              │  Configurable weights per source; dead-signal   │
              │  detection auto-reduces threshold when fewer    │
              │  sources are active                             │
              └──────────────────────┬──────────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────────┐
              │      MarketRegimeDetector (per-symbol ADX)       │
              │  TRENDING  → multi-day hold, no intraday close  │
              │  CHOPPY    → same-day close at 3:55 pm,         │
              │              partial profits at +1% / +2.5%     │
              │  NEUTRAL   → same-day close (conservative)      │
              └──────────────────────┬──────────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────────┐
              │   RiskManager  (Kelly + ATR stops)               │
              │   Per-trade stop-loss · take-profit · trailing  │
              │   Daily-loss kill switch · PDT-rule guard       │
              │   Max concurrent positions · max exposure %     │
              └──────────────────────┬──────────────────────────┘
                                     │
              ┌──────────────────────▼──────────────────────────┐
              │           Execution / Broker layer               │
              │   Alpaca (recommended) · Paper (built-in)        │
              └─────────────────────────────────────────────────┘
```

---

## Signal stack

| # | Signal | Type | Notes |
|---|--------|------|-------|
| 1 | **TechnicalSignal** | Rule-based | RSI, MACD, Bollinger Bands, EMA cross, ATR, volume surge — 7 sub-scores averaged |
| 2 | **ORBSignal** | Rule-based | Opening Range Breakout — price breaks first-30-min range with volume |
| 3 | **VWAPBounceSignal** | Rule-based | Bounce off VWAP + volume confirmation; VWAP momentum cross |
| 4 | **SentimentSignal** | NLP / VADER | Recent news headlines scored with VADER lexicon (FinBERT optional) |
| 5 | **FundamentalSignal** | Rule-based | SEC EDGAR RSS — 8-K, buybacks, earnings surprise, insider keywords |
| 6 | **MacroSignal** | Rule-based | SPY short-term trend multiplier — scales all scores up/down |
| 7 | **MLSignal** | LightGBM | Supervised binary classifier on 8 technical features. Retrain every 7 days (~2 s) |
| 8 | **FinRLSignal** | PPO (RL) | Stable-Baselines3 PPO agent; action-probability spread = score. **Incremental daily retrain** (~30 s GPU) |

---

## Market regime detection

Each stock is classified independently at entry using its own **ADX + vol-ratio** voting:

| Vote source | TRENDING | CHOPPY |
|-------------|----------|--------|
| ADX (14-day) > 25 | +2 | — |
| ADX < 20 | — | +2 |
| Vol-ratio (5d/20d) > 1.20 | +1 | — |
| Vol-ratio < 0.80 | — | +1 |

**votes_trending ≥ 2 → TRENDING** — position is held multi-day; no intraday EOD
close; partial-profit targets disabled so momentum runs to the +6 % take-profit.

**votes_choppy ≥ 2 → CHOPPY** — position closes at 3:55 pm same day; partial
profits taken at +1 % and +2.5 % to lock in gains before EOD mean-reversion.

**Otherwise → NEUTRAL** — same-day close, conservative.

A second VIX-based `RegimeDetector` adjusts the signal aggregator's entry
threshold for broad market conditions (trending_bull, choppy, high_volatility).

---

## LLM API usage — zero during trading

| Feature | Calls/day | API key |
|---------|-----------|---------|
| Sector prediction (Gemini) | 1 | `GEMINI_API_KEY` |
| ML signal (`mode: local`) | **0** | — |
| FinRL PPO (local GPU) | **0** | — |
| **Total** | **1 call/day** | ~$0.0001/day |

The ML signal runs `mode: local` — pure LightGBM inference, zero API calls during
market hours. Switch to `mode: hybrid` in `config.yaml` to re-enable LLM fallback
for borderline signals (≤ 5 extra calls/day).

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in API keys (see .env.example)
cp config.yaml.example config.yaml

# 1. Paper trade first — no real money
python main.py --broker paper

# 2. Backtest on 6 months of 5-min bars (walk-forward, A/B/C comparison)
python run_backtest_regime.py

# 3. Full out-of-sample backtest
python run_backtest_oos.py

# 4. Live dashboard (React + FastAPI — open separate terminal)
cd frontend && npm run dev        # dev mode
# or serve the pre-built bundle:
python -m uvicorn src.dashboard.api:app --port 8000

# 5. Go live (Alpaca paper or live account)
python main.py --broker alpaca
```

---

## Backtest modes

| Script | Description |
|--------|-------------|
| `run_backtest_regime.py` | **Recommended.** 6-month walk-forward A/B/C test: ADAPTIVE vs BEST vs OLD exit strategies on all 15 symbols. Trains LightGBM + PPO on first 50%, tests on last 50%. |
| `run_backtest_oos.py` | Out-of-sample backtest — trains on a fixed window, tests on held-out period |
| `run_backtest.py` | Simple single-pass backtest (no ML retraining) |

---

## ML model retraining schedule

| Model | Trigger | Duration | Method |
|-------|---------|---------|--------|
| LightGBM (`MLSignal`) | Every 7 days at market open | ~2 s | Full retrain on 180-day bars |
| PPO FinRL (`FinRLSignal`) | **Every day** at market open | ~30 s (GPU) | Incremental — `set_env()` + `learn(reset_num_timesteps=False)`. Full retrain from scratch only on first run. |

Both retrains happen automatically inside the engine's daily kick-off loop —
no cron job or external scheduler needed.

---

## Configuration

All knobs live in `config.yaml`:

| Section | Key options |
|---------|------------|
| `universe` | `dynamic`, `max_tickers`, `max_sectors`, `gemini_sector_call` |
| `signals.weights` | Per-source weight (technical, sentiment, fundamental, ml, finrl, orb, vwap_bounce, macro) |
| `signals.ml` | `mode: local/llm/hybrid`, `retrain_days`, `model_path` |
| `signals.finrl` | `retrain_days: 1`, `total_timesteps`, `daily_timesteps`, `min_confidence` |
| `risk` | `max_position_pct`, `daily_loss_limit_pct`, `take_profit_pct`, `trailing_stop_pct`, `use_atr_stops` |
| `broker` | `name: alpaca/paper`, Alpaca base-URL |
| `schedule` | `poll_seconds`, `market_hours_only` |

---

## Project layout

```
src/
├── brokers/
│   ├── alpaca_broker.py     Alpaca REST + WebSocket broker
│   ├── paper_broker.py      Built-in paper-trading simulator
│   └── base.py              BrokerBase interface
├── data/
│   ├── market_data.py       OHLCV bars (yfinance / Alpaca)
│   ├── news_feed.py         NewsAPI / Finnhub headlines
│   ├── sec_filings.py       EDGAR RSS 8-K / 4 parser
│   ├── universe.py          Dynamic ticker selection: sector intelligence
│   └── sectors.py           Sector / ETF mapping
├── signals/
│   ├── technical.py         RSI, MACD, BB, EMA, ATR, volume
│   ├── orb.py               Opening Range Breakout
│   ├── vwap_bounce.py       VWAP bounce / cross
│   ├── sentiment.py         VADER / FinBERT NLP
│   ├── fundamental.py       SEC filing keyword scorer
│   ├── macro.py             SPY trend multiplier
│   ├── ml_model.py          LightGBM classifier (local, GPU)
│   ├── finrl_signal.py      PPO RL agent (Stable-Baselines3)
│   ├── regime.py            VIX RegimeDetector + ADX MarketRegimeDetector
│   ├── aggregator.py        Weighted vote + dead-signal detection
│   └── base.py              Signal / SignalSource dataclasses
├── risk/
│   ├── risk_manager.py      Kelly sizing, ATR stops, kill switch
│   └── performance_tracker.py  Daily Sharpe / drawdown tracking
├── execution/
│   └── trader.py            Order placement, partial exits, regime-aware EOD flatten
├── backtest/
│   └── backtester.py        Walk-forward backtester with per-position exit params
├── dashboard/
│   ├── api.py               FastAPI backend (REST endpoints)
│   └── app.py               Dashboard server entry point
├── utils/
│   ├── logger.py            Structured logging
│   ├── notifications.py     Discord webhook alerts
│   ├── status_page.py       JSON status for dashboard
│   └── trade_history.py     Trade log persistence
└── engine.py                Main orchestration loop

frontend/                    React + Vite live dashboard
models/                      Persisted ML model files (gitignored)
run_backtest_regime.py       6-month A/B/C regime backtest (main backtest script)
run_backtest_oos.py          Out-of-sample backtest
main.py                      Entry point
config.yaml                  All configuration
```

---

## ⚠️ Important warnings

1. **Pattern Day Trader rule.** US brokers require ≥ $25,000 equity for accounts
   that make 4+ day trades within 5 business days. The risk manager blocks trades
   that would trigger this rule on sub-$25k accounts (`pdt_protection: true`).
2. **TRENDING positions are held overnight.** When the regime detector classifies
   a stock as TRENDING, the position is not closed at 3:55 pm — it may be held
   for multiple days. Ensure your account has overnight margin / buying power.
3. **This is not financial advice.** Markets are adversarial. Paper-trade for at
   least 2–4 weeks, backtest exhaustively, and never risk capital you cannot
   afford to lose. Start with the smallest position sizes possible.
4. **API keys and credentials** are stored in `.env` (never committed). Copy
   `.env.example` and fill in your Alpaca, NewsAPI, and optional LLM keys.
