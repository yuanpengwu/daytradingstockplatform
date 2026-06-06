# DayTradingBot — Regime-Adaptive Automated Trading Platform

A fully-automated, multi-signal trading platform for **US Stocks, ETFs, and
Crypto (24/7)**. It combines **8 signal sources** — including two trained ML
models — into a weighted decision engine that routes each trade through a
**market-regime detector**, applying different exit strategies depending on
whether the asset is trending or choppy. Executes through Alpaca with strict
risk controls, real-time Discord alerts, and a live terminal status monitor.

---

## What's new (latest session)

| Feature | Detail |
|---------|--------|
| **24/7 Crypto engine** | BTC/USD · ETH/USD · SOL/USD · AVAX/USD · LINK/USD in a daemon thread alongside the stock engine |
| **Dedicated crypto ML model** | `models/crypto_lgbm.pkl` — trained exclusively on crypto bars (AUC 0.832 vs ~0.45 when trained on stocks) |
| **ADX entry gate** | Blocks entries when symbol ADX < 18 (directionless noise) |
| **Weak-trend zone B+C filter** | ADX 18–25 → requires score ≥ 0.55 AND caps at 1 loss/day |
| **FinRL v2 improvements** | Fixed seed (reproducible), 30-min reward horizon, drawdown penalty, [256,128] network, 500k timesteps |
| **FinRL batch precompute** | O(1) score lookup per bar instead of O(N²) — backtest 15 min → 3 min |
| **Enriched Discord notifications** | Entry: regime, ADX, score gate, conf gate, ML×FinRL agree, signal breakdown with bar chart. Exit: emoji reason, entry→exit price, hold duration |
| **EOD win-rate report** | Day / Week / Month / 3M / 6M / 1Y windows with dollar-weighted avg win/loss, posted to Discord daily |
| **Auto-scheduler** | Windows Task Scheduler at 6:20 AM PDT Mon–Fri — starts engine, stops at 4:05 PM ET, sends EOD report |
| **Live terminal monitor** | `scripts/engine_status.py` — auto-opens on every `main.py` launch; shows running commit, stale warning, open positions, today's trades |
| **Custom backtest date range** | `python run_backtest_regime.py --start 2025/05/01 --end 2025/08/31` |

---

## Architecture

```
main.py
├── TradingEngine  (main thread — US stocks & ETFs, Mon–Fri 9:30–4:00 ET)
│   │
│   ├── Data Layer
│   │   ├── MarketData         OHLCV bars (Alpaca 1m / yfinance / Polygon)
│   │   ├── DynamicUniverse    Sector intelligence → top 10 stocks daily
│   │   │     ├── ETF momentum scoring  (XLK, XLE, XLF …)
│   │   │     └── Gemini sector call    (1 call/day, ~$0.0001)
│   │   ├── NewsFeed           NewsAPI + Finnhub headlines
│   │   └── SECFilings         EDGAR RSS 8-K / insider trade parser
│   │
│   ├── Signal Stack (8 sources → normalised [-1, +1] score each)
│   │   ├── TechnicalSignal    RSI · MACD · Bollinger · EMA · ATR · Volume
│   │   ├── ORBSignal          Opening Range Breakout (first 15 min)
│   │   ├── VWAPBounceSignal   VWAP bounce / cross + volume confirmation
│   │   ├── SentimentSignal    VADER NLP on recent headlines
│   │   ├── FundamentalSignal  SEC 8-K keyword scoring
│   │   ├── MacroSignal        SPY trend multiplier (scales all scores)
│   │   ├── MLSignal           LightGBM stock model  models/local_lgbm.pkl
│   │   └── FinRLSignal        PPO RL agent          models/finrl_ppo.zip
│   │         (incremental GPU retrain daily, seed=42, 30-min horizon)
│   │
│   ├── ADX Entry Gate
│   │   ├── ADX < 18           Hard block — no trend
│   │   ├── ADX 18–25          Weak-trend zone: score ≥ 0.55, ≤ 1 loss/day
│   │   └── ADX ≥ 25           Full trend — normal rules
│   │
│   ├── MarketRegimeDetector   (per-symbol ADX + vol-ratio)
│   │   ├── TRENDING           Hold multi-day, no EOD close, no partial exits
│   │   ├── CHOPPY             Close by 3:55 PM, partial profit at +1%/+2.5%
│   │   └── NEUTRAL            Same-day close (conservative default)
│   │
│   └── RiskManager + Trader
│         Kelly sizing · ATR stops · trailing stop · PDT guard
│         Dynamic symbol exclusion (win rate < 30% for 3 days → banned)
│
└── CryptoEngine  (daemon thread — 24/7, no market-hours gate)
      ├── Pairs: BTC/USD · ETH/USD · SOL/USD · AVAX/USD · LINK/USD
      ├── Signal: TechnicalSignal (60%) + CryptoMLSignal (40%)
      │     └── models/crypto_lgbm.pkl — trained on crypto bars only (AUC 0.832)
      ├── Sizing: notional $500/trade, max 15% portfolio in crypto
      ├── Risk: 5% stop · 10% take-profit · 4% trailing stop · GTC orders
      └── Retrains daily at midnight UTC on fresh crypto bars
```

---

## Signal stack

| # | Signal | Type | Notes |
|---|--------|------|-------|
| 1 | **TechnicalSignal** | Rule-based | RSI, MACD, BB, EMA cross, ATR, volume surge — 7 sub-scores |
| 2 | **ORBSignal** | Rule-based | 15-min opening range breakout with volume confirmation |
| 3 | **VWAPBounceSignal** | Rule-based | Bounce off VWAP + volume surge; VWAP momentum cross |
| 4 | **SentimentSignal** | NLP/VADER | Recent headlines scored — VADER lexicon, FinBERT optional |
| 5 | **FundamentalSignal** | Rule-based | SEC EDGAR RSS — 8-K, buybacks, earnings surprise, insider buys |
| 6 | **MacroSignal** | Rule-based | SPY SMA20 + momentum multiplier — scales all other scores |
| 7 | **MLSignal** | LightGBM (stock) | Trained on stock OHLCV bars. AUC ~0.727. Retrain every 7 days |
| 8 | **FinRLSignal** | PPO RL (stock) | Stable-Baselines3 PPO, seed=42, 30-min horizon, 500k steps initial. Incremental GPU update daily (~30 s) |
| C | **CryptoMLSignal** | LightGBM (crypto) | Trained on crypto bars **only**. AUC 0.832. Daily retrain at midnight |

---

## Market regime detection

Each stock is classified independently at entry using its own **ADX + vol-ratio** voting:

| Vote source | TRENDING | CHOPPY |
|-------------|----------|--------|
| ADX (14-day) ≥ 25 | +2 | — |
| ADX < 20 | — | +2 |
| Vol-ratio (5d/20d) > 1.20 | +1 | — |
| Vol-ratio < 0.80 | — | +1 |

**TRENDING** (votes ≥ 2) — multi-day hold, no EOD close, no partial profit. Let winners run to the +6% take-profit or signal reversal.

**CHOPPY / NEUTRAL** (otherwise) — same-day close at 3:55 PM ET, partial profits at +1% and +2.5% to lock in gains before mean-reversion.

---

## Entry quality filters

On top of signal scoring, three additional gates protect against bad entries:

| Filter | Description |
|--------|-------------|
| **ADX hard gate** | `min_entry_adx: 18` — blocks entries when the symbol's 14-day ADX is below 18 (no directional trend → pure noise) |
| **Weak-trend zone** | ADX 18–25 → requires aggregated score ≥ 0.55 AND limits to 1 losing trade/day for that symbol |
| **Dynamic exclusion** | Symbols with win rate < 30% for 3 consecutive trading days are banned until they recover |

---

## 24/7 Crypto engine

A dedicated `CryptoEngine` runs in a background thread alongside the stock engine:

```yaml
crypto:
  tickers: [BTC/USD, ETH/USD, SOL/USD, AVAX/USD, LINK/USD]
  poll_seconds: 60
  max_position_notional: 500    # max $ per trade
  per_trade_stop_loss_pct: 0.05 # 5% stop (wider — crypto is more volatile)
  take_profit_pct: 0.10
  trailing_stop_pct: 0.04
```

Key differences from the stock engine:

| | Stocks | Crypto |
|---|---|---|
| Market hours | 9:30 AM – 4:00 PM ET | 24/7 |
| Order sizing | Shares (integer) | Notional $ (fractional) |
| Time-in-force | DAY | GTC |
| ML model | `local_lgbm.pkl` (stocks) | `crypto_lgbm.pkl` (crypto only) |
| Stop loss | 2.5% | 5% |
| PDT rule | Yes | No |

---

## Discord notifications

Every trade fires a rich Discord embed:

**Entry card** includes: price · qty · aggregated score vs gate · confidence vs gate · ML×FinRL agreement · stop loss ($ and %) · take profit ($ and %) · regime · ADX · EOD-flatten flag · signal breakdown with visual bar chart (strongest contributor first)

**Exit card** includes: exit reason with emoji (🎯 Take Profit / 🌙 EOD Flatten / 🛑 Stop Loss / 🔄 Signal Reversed / etc.) · realised P&L in $ and % · entry→exit price comparison · hold duration (Xh Ym)

**EOD win-rate report** (sent daily after market close):

| Window | Metrics |
|--------|---------|
| Day | trades · WR% · net P&L · avg win $ · avg loss $ |
| Week | same |
| Month | same |
| 3 Months | same |
| 6 Months | same |
| 1 Year | same |

---

## Auto-scheduler

The engine starts and stops automatically every trading day:

```
Windows Task Scheduler — fires 6:20 AM PDT (Mon–Fri)
  └── scripts/market_runner.py
        1. Kill any stale processes from yesterday
        2. Log current git commit
        3. Clear __pycache__ (force recompile)
        4. Start main.py  →  opens status monitor terminal automatically
        5. Sleep until 4:05 PM ET
        6. Stop engine
        7. Post EOD win-rate report to Discord
```

---

## Live terminal monitor

`scripts/engine_status.py` opens automatically every time `main.py` starts:

```
+========================================================================+
| DayTradingBot  |  09:45:22 AM ET  |  Market OPEN                     |
+------------------------------------------------------------------------+
| ENGINE  [RUNNING]  PID 84256  Uptime 1h 15m  commit c3041c4  [GREEN]  |
+------------------------------------------------------------------------+
| Equity $99,112  Cash $96,522                                           |
+------------------------------------------------------------------------+
| OPEN POSITIONS                                                         |
|   AMD  5 sh  entry $507.54  now $515.30  unrealized +$38.80 (+1.53%) |
+------------------------------------------------------------------------+
| TODAY'S CLOSED TRADES                                                  |
|   3 trades  67% WR  (2W / 1L)  P&L +$47.30                          |
|   [W] 09:40  NVDA  $892.10→$900.50  +$25.20 (take_profit)           |
+------------------------------------------------------------------------+
| LIVE LOG                                                               |
|   09:44  engine  DECISION AMD | score=+0.45 action=BUY               |
+========================================================================+
  Refreshed 09:45:22  |  Ctrl+C to exit
```

The commit is shown **green** when up to date, **yellow [STALE]** if code has changed and a restart is needed.

---

## Backtest

```bash
# Default: last 180 days (Dec 2025 → Jun 2026)
python run_backtest_regime.py

# Custom date range
python run_backtest_regime.py --start 2025/05/01 --end 2025/08/31

# Skip crypto engine
python main.py --no-crypto
```

Best confirmed result (Dec 2025 → Jun 2026, ADAPTIVE strategy):

| Metric | Value |
|--------|-------|
| Net P&L | +$187.20 (+1.87%) |
| Sharpe ratio | 5.22 |
| Win rate | 49.3% |
| Avg win | +1.79% |
| Avg loss | −0.71% |
| Profit factor | 1.91x |

Three backtest modes:

| Mode | Description |
|------|-------------|
| **ADAPTIVE** | Regime-routed exits — TRENDING → let winners run, CHOPPY/NEUTRAL → partial profit + EOD close |
| **BEST** | Always partial profit + trailing stop |
| **OLD** | Baseline — signal reversal or take-profit only, no partial exits |

---

## ML model retraining schedule

| Model | Trigger | Duration | Data |
|-------|---------|---------|------|
| Stock LightGBM (`MLSignal`) | Every 7 days at market open | ~2 s | Last 180 days of stock bars |
| Stock PPO FinRL (`FinRLSignal`) | Every day at market open | ~30 s GPU | Incremental update on latest bars |
| Crypto LightGBM (`CryptoMLSignal`) | Every day at midnight UTC | ~5 s | Last 10 days of crypto bars (54k+ samples) |

---

## LLM API usage

| Feature | Calls/day | API key |
|---------|-----------|---------|
| Sector prediction (Gemini) | 1 | `GEMINI_API_KEY` |
| Stock ML (`mode: local`) | **0** | — |
| Crypto ML (LightGBM) | **0** | — |
| FinRL PPO (GPU) | **0** | — |
| **Total** | **1/day** | ~$0.0001/day |

---

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in ALPACA_API_KEY, ALPACA_API_SECRET, DISCORD_WEBHOOK_URL
cp config.yaml.example config.yaml

# Paper trade (stocks only)
python main.py --broker paper

# Paper trade (stocks + crypto 24/7)
python main.py --broker alpaca

# Paper trade (stocks only, no crypto)
python main.py --broker alpaca --no-crypto

# Backtest (last 180 days)
python run_backtest_regime.py

# Backtest (custom date range)
python run_backtest_regime.py --start 2025/05/01 --end 2025/08/31

# Win-rate report (prints + optional Discord post)
python scripts/win_rate_report.py --discord

# Live status monitor (opens automatically, or run manually)
python scripts/engine_status.py
```

---

## Configuration

All knobs live in `config.yaml`:

| Section | Key options |
|---------|------------|
| `universe` | `dynamic`, `max_tickers`, `max_sectors`, `gemini_sector_call` |
| `regime` | `adx_trend_thresh` (default 25), `adx_choppy_thresh` (default 20) |
| `signals.weights` | Per-source weight (technical, sentiment, fundamental, ml, finrl, orb, vwap_bounce) |
| `signals.ml` | `mode: local/llm/hybrid`, `retrain_days`, `model_path`, `seed` |
| `signals.finrl` | `retrain_days: 1`, `total_timesteps: 500000`, `reward_horizon: 6`, `seed: 42` |
| `risk` | `min_entry_adx: 18`, `weak_trend_adx_max: 25`, `per_trade_stop_loss_pct`, `trailing_stop_pct` |
| `crypto` | `enabled`, `tickers`, `max_position_notional`, `per_trade_stop_loss_pct: 0.05` |
| `broker` | `name: alpaca/paper` |
| `schedule` | `poll_seconds`, `market_open_buffer_minutes` |
| `notifications` | `channels: [console, discord, email]` |

---

## Project layout

```
src/
├── brokers/
│   ├── alpaca_broker.py     Alpaca REST broker (stocks + crypto)
│   ├── paper_broker.py      Built-in paper-trading simulator
│   └── base.py              BrokerBase interface; is_crypto_symbol()
├── data/
│   ├── market_data.py       OHLCV bars — Alpaca stock + CryptoHistoricalDataClient
│   ├── news_feed.py         NewsAPI / Finnhub headlines
│   ├── sec_filings.py       EDGAR RSS 8-K / 4 parser
│   ├── universe.py          Dynamic universe: sector ETF scoring + Gemini call
│   └── sectors.py           Sector / ETF mapping
├── signals/
│   ├── technical.py         RSI, MACD, BB, EMA, ATR, volume
│   ├── orb.py               Opening Range Breakout
│   ├── vwap_bounce.py       VWAP bounce / cross
│   ├── sentiment.py         VADER / FinBERT NLP
│   ├── fundamental.py       SEC filing keyword scorer
│   ├── macro.py             SPY trend multiplier
│   ├── ml_model.py          LightGBM classifier (stock + crypto instances)
│   ├── finrl_signal.py      PPO RL agent; batch precompute for backtest
│   ├── regime.py            ADX MarketRegimeDetector (per-symbol)
│   ├── aggregator.py        Weighted vote + dead-signal detection
│   └── base.py              Signal / SignalSource dataclasses
├── risk/
│   ├── risk_manager.py      Kelly sizing, ATR stops, daily loss kill switch
│   └── performance_tracker.py  Dynamic symbol exclusion tracker
├── execution/
│   └── trader.py            Order placement, ADX gates, partial exits, regime routing
├── backtest/
│   └── backtester.py        Walk-forward backtester
├── utils/
│   ├── logger.py            Structured logging
│   ├── notifications.py     Discord embeds (entry + exit + win-rate report)
│   ├── status_page.py       JSON status for dashboard
│   └── trade_history.py     Trade log persistence
├── engine.py                Stock trading engine (main thread)
└── crypto_engine.py         Crypto trading engine (daemon thread, 24/7)

scripts/
├── market_runner.py         Daily auto-start/stop + EOD report
├── engine_status.py         Live terminal monitor (auto-opens on launch)
├── win_rate_report.py       Win-rate report across 6 windows → Discord
├── start_bot.ps1            Manual start script (PowerShell)
└── stop_bot.ps1             Manual stop script (PowerShell)

models/                      Persisted ML model files (gitignored)
  ├── local_lgbm.pkl         Stock LightGBM model
  ├── finrl_ppo.zip          Stock PPO RL model
  └── crypto_lgbm.pkl        Crypto LightGBM model (AUC 0.832)

run_backtest_regime.py       A/B/C regime backtest (ADAPTIVE vs BEST vs OLD)
main.py                      Entry point (starts stock + crypto engines)
config.yaml                  All configuration
```

---

## ⚠️ Important warnings

1. **Pattern Day Trader rule.** US brokers require ≥ $25,000 equity for accounts
   that make 4+ day trades within 5 business days. The risk manager blocks trades
   that would trigger this rule on sub-$25k accounts (`pdt_protection: true`).

2. **TRENDING positions are held overnight.** When the regime detector classifies
   a stock as TRENDING, the position is not closed at 3:55 PM — it may run for
   multiple days. Ensure your account has overnight margin / buying power.

3. **Crypto trades 24/7.** The crypto engine runs on weekends and overnight.
   Positions can move significantly while you sleep. The 5% stop and 4% trailing
   stop limit downside, but crypto is inherently more volatile than stocks.

4. **This is not financial advice.** Markets are adversarial. Paper-trade for at
   least 2–4 weeks, backtest exhaustively, and never risk capital you cannot
   afford to lose. Start with the smallest position sizes possible.

5. **API keys and credentials** are stored in `.env` (never committed). Copy
   `.env.example` and fill in your Alpaca, DISCORD_WEBHOOK_URL, and optional
   Gemini / NewsAPI keys.
