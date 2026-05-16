"""Streamlit live dashboard.

Run with:
    streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.brokers import get_broker
from src.data.market_data import MarketData
from src.data.universe import DynamicUniverse
from src.signals.aggregator import SignalAggregator
from src.signals.fundamental import FundamentalSignal
from src.signals.macro import MacroSignal
from src.signals.ml_model import MLSignal
from src.signals.sentiment import SentimentSignal
from src.signals.technical import TechnicalSignal

load_dotenv()
st.set_page_config(page_title="DayTradingBot", layout="wide", page_icon="chart")

# ----------- config -----------
@st.cache_resource
def load_cfg():
    p = Path("config.yaml")
    if not p.exists():
        p = Path("config.yaml.example")
    with open(p) as f:
        return yaml.safe_load(f)


cfg = load_cfg()
broker = get_broker(cfg["broker"]["name"], cfg["broker"])
md = MarketData(
    provider=cfg["data"].get("provider", "yfinance"),
    interval=cfg["data"].get("bar_interval", "5m"),
    lookback_days=cfg["data"].get("lookback_days", 30),
)
tech = TechnicalSignal(cfg["signals"]["technical"])
sent = SentimentSignal(cfg["signals"]["sentiment"])
fund = FundamentalSignal(cfg["signals"]["fundamental"])
ml = MLSignal(cfg["signals"]["ml"])
macro = MacroSignal()
agg = SignalAggregator(weights=cfg["signals"]["weights"])

# ----------- header -----------
st.title("DayTradingBot - Live Monitor")

# Fetch macro status first to display it
spy_bars = md.get_bars("SPY")
market_multiplier, macro_reason = macro.evaluate(spy_bars)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Equity", f"${broker.get_equity():,.2f}")
col2.metric("Cash", f"${broker.get_cash():,.2f}")
col3.metric("Buying Power", f"${broker.get_buying_power():,.2f}")
col4.metric("Broker", cfg["broker"]["name"])
col5.metric("Market Conf", f"{market_multiplier:.2f}x")
st.caption(f"Macro Regime: {macro_reason}")

# ----------- positions -----------
colA, colB = st.columns([4, 1])
with colA:
    st.subheader("Open positions")
with colB:
    if st.button("🚨 EMERGENCY SELL ALL", width='stretch'):
        from src.execution.trader import Trader
        from src.risk.risk_manager import RiskManager
        risk = RiskManager(cfg.get("risk", {}))
        trader = Trader(broker, risk, notify_channels=cfg.get("notifications", {}).get("channels", ["console"]))
        trader.flatten_all("Manual dashboard emergency sell")
        st.success("Liquidation orders sent!")
        time.sleep(1)
        st.rerun()

positions = broker.get_positions()
if positions:
    rows = []
    for sym, p in positions.items():
        rows.append({
            "Symbol": sym,
            "Qty": p.qty,
            "Avg Entry": p.avg_entry_price,
            "Last": p.current_price,
            "MV": p.market_value,
            "PnL $": p.unrealized_pnl,
            "PnL %": f"{p.unrealized_pnl_pct*100:+.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), width='stretch')
else:
    st.info("No open positions.")

# ----------- signals -----------
st.subheader("Live signal scores")
sym_rows = []

# Fetch the dynamic universe
universe = DynamicUniverse(cfg["universe"])
active_tickers = universe.select_tickers()

# Ensure open positions are also monitored
open_positions = broker.get_positions()
for sym in open_positions.keys():
    if sym not in active_tickers:
        active_tickers.append(sym)

for sym in active_tickers:
    bars = md.get_bars(sym)
    sigs = []
    
    t = tech.evaluate(sym, bars)
    s = sent.evaluate(sym)
    f = fund.evaluate(sym)
    m = ml.evaluate(sym, bars, tech_signal=t)
    
    for sig in (t, s, f, m):
        if sig is not None:
            sigs.append(sig)
    decisions = agg.aggregate(sigs, market_multiplier=market_multiplier)
    d = decisions.get(sym)
    if d is None:
        continue
    row = {"Symbol": sym, "Score": round(d.score, 3), "Confidence": round(d.confidence, 2), "Action": d.action}
    for k, v in d.components.items():
        row[k] = round(v, 3)
    sym_rows.append(row)
if sym_rows:
    st.dataframe(pd.DataFrame(sym_rows).sort_values("Score", ascending=False), width='stretch')

st.caption("Auto-refresh: hit R or rerun.")
