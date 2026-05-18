# DayTradingBot — Advanced Automated Day-Trading Platform

A fully-automated, multi-signal day-trading platform for **US Stocks & ETFs**.
It combines **technical indicators**, **news/sentiment NLP**, **SEC filings**,
and a **machine-learning predictor** into a single weighted decision engine,
then executes trades through a pluggable broker layer (Robinhood, Alpaca, or
a built-in paper-trading simulator) with strict risk controls.

## Architecture

```
                ┌──────────── Data Layer ────────────┐
                │  MarketData (yfinance / Polygon)   │
                │  NewsFeed   (NewsAPI / Finnhub)    │
                │  SECFilings (EDGAR RSS)            │
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │            Signal Engines          │
                │  Technical • Sentiment • Fundamental • ML
                │  (each emits a normalized [-1, +1] score)
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │      Aggregator → weighted vote     │
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │       RiskManager (Kelly + ATR     │
                │   stops, daily-loss kill switch,   │
                │   PDT-rule guard, max exposure)    │
                └────────────────┬───────────────────┘
                                 │
                ┌────────────────▼───────────────────┐
                │        Execution / Broker          │
                │  Robinhood • Alpaca • Paper        │
                └────────────────────────────────────┘
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env             # fill in API keys
cp config.yaml.example config.yaml

# 1. Validate everything with the paper-trading simulator first
python main.py --broker paper

# 2. Backtest the strategy on history before risking real money
python run_backtest.py --start 2024-01-01 --end 2024-12-31

# 3. Start the dashboard (see full tutorial below)
uvicorn src.dashboard.api:app --host 127.0.0.1 --port 8000

# 4. Go live (after thorough paper testing)
python main.py --broker alpaca         # recommended
python main.py --broker robinhood      # unofficial API — see warning below
```

## Starting the dashboard

The dashboard is a **React frontend** served by a **FastAPI backend**.
You have two options depending on whether you want to develop the UI or just run it.

### Option A — Production (recommended, single command)

Build the React app once, then FastAPI serves everything on one port:

```bash
# Step 1 — build the frontend (only needed after UI changes)
cd frontend
npm install
npm run build
cd ..

# Step 2 — start the API server (also serves the built UI)
uvicorn src.dashboard.api:app --host 127.0.0.1 --port 8000
```

Open **http://localhost:8000** in your browser.
The trading bot must also be running to see live data:

```bash
# In a second terminal
python main.py --broker alpaca
```

### Option B — Development (hot-reload UI)

Run the backend and the Vite dev server side-by-side so UI changes
appear instantly without rebuilding:

```bash
# Terminal 1 — FastAPI backend
uvicorn src.dashboard.api:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2 — React dev server (proxies /api → localhost:8000)
cd frontend
npm install       # skip after first run
npm run dev

# Terminal 3 — trading bot
python main.py --broker alpaca
```

Open **http://localhost:5173** for the hot-reloading dev UI.

### Environment variables

Copy `.env.example` to `.env` and fill in your keys before starting:

| Variable | Required | Description |
|---|---|---|
| `ALPACA_API_KEY` | Yes (live/paper) | Alpaca trading key |
| `ALPACA_SECRET_KEY` | Yes (live/paper) | Alpaca secret |
| `GEMINI_API_KEY` | Recommended | Primary LLM signal provider |
| `ANTHROPIC_API_KEY` | Recommended | Claude fallback when Gemini quota is exhausted |
| `NEWSAPI_KEY` | Optional | News sentiment headlines |
| `FINNHUB_API_KEY` | Optional | Additional news source |

## ⚠️ Important warnings

1. **Robinhood has no official trading API.** This bot uses the community
   `robin_stocks` library, which scrapes Robinhood's private endpoints.
   Fully-automated trading may violate Robinhood's Terms of Service, the
   library can break without notice, and 2FA flows are fragile.
   **Alpaca is the recommended live broker** — real public API, free paper
   trading, identical trading semantics. Swap with one line in `config.yaml`.
2. **Pattern Day Trader rule.** US brokers require ≥ $25,000 equity for
   accounts that make 4+ day trades in 5 business days. The risk manager
   blocks trades that would trip this rule on sub-$25k accounts.
3. **This is not financial advice.** Markets are adversarial. Backtest
   exhaustively, paper-trade for weeks, and never risk capital you can't
   afford to lose. Start with the smallest position sizes possible.

## Configuration

All knobs live in `config.yaml`:

- `universe` — tickers to watch
- `signals.weights` — how much each signal contributes to the final score
- `risk.*` — position sizing, stops, daily loss limit
- `schedule.poll_seconds` — how often the loop runs
- `broker.name` — robinhood | alpaca | paper

## Project layout

```
src/
├── brokers/        Robinhood, Alpaca, Paper adapters (BrokerBase interface)
├── data/           Market data, news, SEC filings
├── signals/        Technical, sentiment, fundamental, ML signals + aggregator
├── risk/           Position sizing, stops, kill switches
├── execution/      Order placement, position tracking
├── backtest/       Vectorized historical evaluation
├── dashboard/      FastAPI backend + React frontend (served from frontend/dist)
├── utils/          Logging, notifications
└── engine.py       Main orchestration loop
```

See each module's docstring for details.
