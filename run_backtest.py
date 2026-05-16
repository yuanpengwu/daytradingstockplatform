"""CLI entry-point: backtest the configured strategy on historical bars.

Data sources:
    --data yfinance   (default) free Yahoo data; intraday limited to ~60 days
    --data alpaca     Alpaca historical bars (uses your ALPACA_* keys in .env);
                      more reliable, no rate-limiting

Usage:
    python run_backtest.py --data alpaca --interval 5m --start 2026-03-15 --end 2026-05-14
    python run_backtest.py --data alpaca --interval 1d --start 2025-05-13 --end 2026-05-13

Outputs:
    - console summary + last 20 trades
    - backtest_equity.html  (self-contained equity-curve chart)
    - backtest_equity.csv   (raw equity curve)
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from src.backtest.backtester import Backtester, BacktestResult
from src.utils.logger import get_logger

log = get_logger("backtest")


# ----------------------------------------------------------------------
#  Equity-curve chart writer
# ----------------------------------------------------------------------
def _write_equity_chart(result: BacktestResult, html_path: Path, csv_path: Path,
                        title: str) -> None:
    """Write a self-contained HTML line chart of total capital over the run."""
    ec = result.equity_curve
    if not ec:
        log.warning("Empty equity curve — skipping chart.")
        return

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bar_index", "equity"])
        for i, v in enumerate(ec):
            w.writerow([i, round(v, 2)])

    W, H = 960, 420
    pad_l, pad_r, pad_t, pad_b = 70, 20, 40, 40
    plot_w, plot_h = W - pad_l - pad_r, H - pad_t - pad_b
    lo, hi = min(ec), max(ec)
    if hi == lo:
        hi = lo + 1.0
    n = len(ec)

    def x(i: int) -> float:
        return pad_l + (i / max(1, n - 1)) * plot_w

    def y(v: float) -> float:
        return pad_t + (1 - (v - lo) / (hi - lo)) * plot_h

    pts = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(ec))
    start_v, end_v = ec[0], ec[-1]
    ret_pct = (end_v - start_v) / start_v * 100 if start_v else 0.0
    line_color = "#16a34a" if end_v >= start_v else "#dc2626"
    baseline_y = y(start_v)

    ticks = []
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        ticks.append((v, y(v)))
    grid = "".join(
        f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{W-pad_r}" y2="{gy:.1f}" '
        f'stroke="#e5e7eb" stroke-width="1"/>'
        f'<text x="{pad_l-8}" y="{gy+4:.1f}" text-anchor="end" '
        f'font-size="11" fill="#6b7280">${v:,.0f}</text>'
        for v, gy in ticks
    )

    svg = f'''<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
  <rect width="{W}" height="{H}" fill="white"/>
  {grid}
  <line x1="{pad_l}" y1="{baseline_y:.1f}" x2="{W-pad_r}" y2="{baseline_y:.1f}"
        stroke="#9ca3af" stroke-width="1" stroke-dasharray="4 3"/>
  <polyline points="{pts}" fill="none" stroke="{line_color}" stroke-width="2"/>
  <text x="{pad_l}" y="24" font-size="15" font-weight="600" fill="#111827">{title}</text>
  <text x="{W-pad_r}" y="24" text-anchor="end" font-size="13" fill="{line_color}">
    ${start_v:,.0f} &#8594; ${end_v:,.0f} ({ret_pct:+.2f}%)
  </text>
</svg>'''

    html = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,Arial,sans-serif;margin:24px;color:#111827}}
.stats{{margin:12px 0;font-size:14px}} .stats b{{color:#111827}}</style></head>
<body>
<h2>Backtest — Total Capital Over Time</h2>
<div class="stats">{result.summary()}</div>
{svg}
<div class="stats">Equity curve has {len(ec)} points. Raw data: backtest_equity.csv</div>
</body></html>'''

    html_path.write_text(html, encoding="utf-8")
    log.info("Equity chart written to %s", html_path)
    log.info("Equity curve CSV written to %s", csv_path)


# ----------------------------------------------------------------------
#  Data fetchers
# ----------------------------------------------------------------------
def _fetch_yfinance_bars(tickers, interval, start, end):
    import yfinance as yf

    bars = {}
    for sym in tickers:
        df = yf.download(sym, start=start, end=end, interval=interval,
                         progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            log.warning("No bars for %s", sym)
            continue
        bars[sym] = df.rename(columns=str.title)[["Open", "High", "Low", "Close", "Volume"]]
    return bars


def _alpaca_timeframe(interval):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    unit = interval[-1]
    amt = int(interval[:-1]) if interval[:-1].isdigit() else 1
    if unit == "m":
        return TimeFrame(amt, TimeFrameUnit.Minute)
    if unit == "h":
        return TimeFrame(amt, TimeFrameUnit.Hour)
    if unit == "d":
        return TimeFrame(amt, TimeFrameUnit.Day)
    log.warning("Unknown interval %s — defaulting to 5-minute bars.", interval)
    return TimeFrame(5, TimeFrameUnit.Minute)


def _fetch_alpaca_bars(tickers, interval, start, end):
    """Fetch historical bars from Alpaca's data API.

    Uses the IEX feed (available on free / paper accounts). Requires
    ALPACA_API_KEY / ALPACA_API_SECRET in .env.
    """
    import os

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.enums import DataFeed
    except ImportError as e:
        raise RuntimeError("alpaca-py is not installed. Run setup.bat.") from e

    key = os.getenv("ALPACA_API_KEY")
    secret = os.getenv("ALPACA_API_SECRET")
    if not (key and secret):
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET missing from .env")

    # Default window: last 60 days if not given.
    end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
    start_dt = datetime.fromisoformat(start) if start else end_dt - timedelta(days=60)

    client = StockHistoricalDataClient(key, secret)
    tf = _alpaca_timeframe(interval)
    req = StockBarsRequest(
        symbol_or_symbols=list(tickers),
        timeframe=tf,
        start=start_dt,
        end=end_dt,
        feed=DataFeed.IEX,          # free-tier feed
    )
    log.info("Requesting Alpaca bars: %s %s %s -> %s", tickers, interval, start_dt, end_dt)
    resp = client.get_stock_bars(req)
    df = resp.df  # MultiIndex (symbol, timestamp)
    if df is None or df.empty:
        log.error("Alpaca returned no data for the requested window.")
        return {}

    bars = {}
    symbols_in = df.index.get_level_values(0).unique()
    for sym in tickers:
        if sym not in symbols_in:
            log.warning("No Alpaca bars for %s", sym)
            continue
        sub = df.loc[sym].copy()
        sub = sub.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        # Index is already a tz-aware DatetimeIndex of bar timestamps.
        bars[sym] = sub[["Open", "High", "Low", "Close", "Volume"]]
    return bars


# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data", default="yfinance", choices=["yfinance", "alpaca"],
                        help="Historical data source")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default=None)
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--cash", type=float, default=10_000)
    args = parser.parse_args()

    load_dotenv()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml.example")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    tickers = args.tickers or cfg["universe"].get("fallback_tickers", ["SPY", "QQQ"])
    interval = args.interval or cfg["data"].get("bar_interval", "5m")
    log.info("Backtesting %s on %s %s bars from %s to %s",
             tickers, args.data, interval, args.start, args.end)

    if args.data == "alpaca":
        bars = _fetch_alpaca_bars(tickers, interval, args.start, args.end)
    else:
        bars = _fetch_yfinance_bars(tickers, interval, args.start, args.end)

    if not bars:
        log.error("No data — aborting. (yfinance: intraday limited to ~60 days; "
                  "alpaca: check your date window and that the market had data.)")
        return

    bt = Backtester(cfg, starting_cash=args.cash, interval=interval)
    result = bt.run(bars)

    print("=" * 70)
    print(f"Data source: {args.data}")
    print(result.summary())
    print("=" * 70)
    for t in result.trades[-20:]:
        print(
            f"{t.symbol:6s}  {t.entry_time}  -> {t.exit_time}  "
            f"qty={t.qty}  pnl=${t.pnl:+.2f} ({t.pnl_pct*100:+.2f}%)  {t.reason}"
        )

    title = f"{'+'.join(bars.keys())}  |  {args.data} {interval}  |  {args.start or 'start'} -> {args.end or 'end'}"
    _write_equity_chart(
        result,
        html_path=Path("backtest_equity.html"),
        csv_path=Path("backtest_equity.csv"),
        title=title,
    )
    print("=" * 70)
    print("Equity-curve chart saved to: backtest_equity.html  (open it in your browser)")
    print("Raw equity data saved to:   backtest_equity.csv")


if __name__ == "__main__":
    main()
