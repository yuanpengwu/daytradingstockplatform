"""Writes a self-contained, auto-refreshing status.html every cycle.

No web server, no dependencies — the data is embedded directly in the file
and the page reloads itself via a <meta refresh> tag. Open status.html in
any browser and leave it open to watch the bot run.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .logger import get_logger

log = get_logger(__name__)


def _esc(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def write_status_page(
    path: str,
    broker,
    decisions: Optional[Dict] = None,
    cycle_count: int = 0,
    started_at: Optional[datetime] = None,
    refresh_seconds: int = 30,
    kill_switch: bool = False,
    regime: str = "",
) -> None:
    """Render the current bot state to a standalone HTML file.

    Wrapped so a rendering failure can never break the trading loop.
    """
    try:
        now = datetime.now()
        equity = broker.get_equity()
        cash = broker.get_cash()
        positions = broker.get_positions()
        market_value = sum(p.market_value for p in positions.values())
        unrealized = sum(p.unrealized_pnl for p in positions.values())
        uptime = ""
        if started_at:
            secs = int((now - started_at).total_seconds())
            uptime = f"{secs // 3600}h {(secs % 3600) // 60}m"

        # ----- positions table -----
        if positions:
            rows = ""
            for sym, p in sorted(positions.items()):
                cls = "pos" if p.unrealized_pnl >= 0 else "neg"
                rows += (
                    f"<tr><td><b>{_esc(sym)}</b></td>"
                    f"<td>{p.qty:g}</td>"
                    f"<td>${p.avg_entry_price:,.2f}</td>"
                    f"<td>${p.current_price:,.2f}</td>"
                    f"<td>${p.market_value:,.2f}</td>"
                    f"<td class='{cls}'>${p.unrealized_pnl:+,.2f} "
                    f"({p.unrealized_pnl_pct*100:+.2f}%)</td></tr>"
                )
            positions_html = (
                "<table><tr><th>Symbol</th><th>Qty</th><th>Entry</th>"
                "<th>Last</th><th>Value</th><th>Unrealized P&amp;L</th></tr>"
                f"{rows}</table>"
            )
        else:
            positions_html = "<p class='muted'>No open positions.</p>"

        # ----- decisions table -----
        decisions = decisions or {}
        if decisions:
            drows = ""
            for sym, d in sorted(decisions.items(), key=lambda kv: -kv[1].score):
                act = d.action
                acls = {"BUY": "pos", "SELL": "neg", "HOLD": "muted"}.get(act, "")
                drows += (
                    f"<tr><td><b>{_esc(sym)}</b></td>"
                    f"<td>{d.score:+.3f}</td>"
                    f"<td>{d.confidence:.2f}</td>"
                    f"<td class='{acls}'>{act}</td></tr>"
                )
            decisions_html = (
                "<table><tr><th>Symbol</th><th>Score</th>"
                f"<th>Confidence</th><th>Action</th></tr>{drows}</table>"
            )
        else:
            decisions_html = "<p class='muted'>No signals yet this cycle.</p>"

        banner = ""
        if kill_switch:
            banner = ("<div class='kill'>KILL SWITCH ENGAGED &mdash; "
                      "daily loss limit hit, trading halted for the day.</div>")

        pnl_cls = "pos" if unrealized >= 0 else "neg"

        html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{refresh_seconds}">
<title>DayTradingBot status</title>
<style>
 body{{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;padding:24px;
   background:#0f1115;color:#e6e6e6}}
 h1{{font-size:20px;margin:0 0 4px}} .muted{{color:#8b8f98}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
   gap:12px;margin:18px 0}}
 .card{{background:#1a1d24;border-radius:10px;padding:14px 16px}}
 .card .label{{font-size:12px;color:#8b8f98}}
 .card .value{{font-size:22px;font-weight:600;margin-top:4px}}
 table{{width:100%;border-collapse:collapse;margin:8px 0 20px}}
 th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #262a33;font-size:14px}}
 th{{color:#8b8f98;font-weight:500}}
 .pos{{color:#3fb950}} .neg{{color:#f85149}}
 .kill{{background:#3d1418;color:#f85149;padding:12px 16px;border-radius:8px;
   margin-bottom:16px;font-weight:600}}
 h2{{font-size:15px;margin:18px 0 6px;color:#c9d1d9}}
</style></head><body>
<h1>DayTradingBot &mdash; live status</h1>
<div class="muted">Updated {now:%Y-%m-%d %H:%M:%S} &nbsp;|&nbsp;
  cycle #{cycle_count} &nbsp;|&nbsp; uptime {uptime or "&mdash;"} &nbsp;|&nbsp;
  page auto-refreshes every {refresh_seconds}s</div>
{banner}
<div class="grid">
  <div class="card"><div class="label">Equity</div>
    <div class="value">${equity:,.2f}</div></div>
  <div class="card"><div class="label">Cash</div>
    <div class="value">${cash:,.2f}</div></div>
  <div class="card"><div class="label">Invested</div>
    <div class="value">${market_value:,.2f}</div></div>
  <div class="card"><div class="label">Open P&amp;L</div>
    <div class="value {pnl_cls}">${unrealized:+,.2f}</div></div>
  <div class="card"><div class="label">Open positions</div>
    <div class="value">{len(positions)}</div></div>
</div>
<h2>Open positions</h2>
{positions_html}
<h2>Latest signal scores</h2>
{decisions_html}
</body></html>"""

        Path(path).write_text(html, encoding="utf-8")
        
        # ----- write JSON for frontend -----
        import json
        state = {
            "updated_at": now.isoformat(),
            "cycle_count": cycle_count,
            "uptime_seconds": int((now - started_at).total_seconds()) if started_at else 0,
            "equity": equity,
            "cash": cash,
            "market_value": market_value,
            "unrealized_pnl": unrealized,
            "kill_switch": kill_switch,
            "regime": regime,
            "positions": [
                {
                    "symbol": sym,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "current_price": p.current_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                    "unrealized_pnl_pct": p.unrealized_pnl_pct
                } for sym, p in positions.items()
            ],
            "decisions": [
                {
                    "symbol": sym,
                    "score": d.score,
                    "confidence": d.confidence,
                    "action": d.action,
                    "components": {k: round(v, 4) for k, v in d.components.items()} if hasattr(d, "components") else {}
                } for sym, d in (decisions or {}).items()
            ]
        }
        Path(path.replace(".html", ".json")).write_text(json.dumps(state), encoding="utf-8")
        
    except Exception as e:
        log.warning("Failed to write status page: %s", e)
