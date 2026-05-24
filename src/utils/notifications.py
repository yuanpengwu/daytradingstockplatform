"""Trade and alert notifications via console / Telegram / Discord / Email."""
from __future__ import annotations

import os
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, Iterable, Optional

import requests

from .logger import get_logger

log = get_logger(__name__)

# Discord embed colours
_COLOR_BUY  = 0x2ECC71   # green
_COLOR_SELL = 0xE74C3C   # red
_COLOR_EXIT = 0x95A5A6   # grey

# Emoji per signal source
_SOURCE_EMOJI = {
    "technical":   "📊",
    "sentiment":   "📰",
    "fundamental": "📄",
    "ml":          "🤖",
    "orb":         "🔔",
    "vwap_bounce": "📈",
}


def _notify_console(msg: str) -> None:
    log.info("NOTIFY: %s", msg)


def _notify_telegram(msg: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat = os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat, "text": msg},
            timeout=5,
        )
    except Exception as e:
        log.warning("Telegram notify failed: %s", e)


def _notify_discord(msg: str) -> None:
    """Send a plain-text Discord message."""
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": msg}, timeout=5)
    except Exception as e:
        log.warning("Discord notify failed: %s", e)


def _notify_discord_embed(payload: dict) -> None:
    """Send a rich Discord embed payload."""
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        r = requests.post(url, json=payload, timeout=5)
        if r.status_code not in (200, 204):
            log.warning("Discord embed returned %s: %s", r.status_code, r.text[:200])
    except Exception as e:
        log.warning("Discord embed notify failed: %s", e)


def notify_order_entry(
    *,
    symbol: str,
    side: str,                          # "buy" | "sell"
    qty: float,
    price: float,
    agg_score: float,
    confidence: float,
    components: Dict[str, float],       # per-source weighted contributions from aggregator
    stop_loss: Optional[float],
    take_profit: Optional[float],
    channels: Iterable[str] = ("console",),
) -> None:
    """Rich order-entry notification with per-signal breakdown."""
    is_buy = side.lower() == "buy"
    arrow  = "🟢 BUY" if is_buy else "🔴 SELL"
    color  = _COLOR_BUY if is_buy else _COLOR_SELL

    # Plain-text fallback (console / telegram / email)
    signal_lines = "  ".join(
        f"{src}={v:+.3f}" for src, v in sorted(components.items())
    )
    plain = (
        f"[{arrow}] {qty:.0f} {symbol} @ ~${price:.2f} | "
        f"score={agg_score:+.3f} conf={confidence:.2f} | "
        f"SL=${stop_loss} TP=${take_profit} | {signal_lines}"
    )

    # Discord embed
    fields = [
        {"name": "Price",       "value": f"`${price:.2f}`",       "inline": True},
        {"name": "Qty",         "value": f"`{qty:.0f}`",           "inline": True},
        {"name": "Score",       "value": f"`{agg_score:+.3f}`",    "inline": True},
        {"name": "Confidence",  "value": f"`{confidence:.0%}`",    "inline": True},
        {"name": "Stop Loss",   "value": f"`${stop_loss}`" if stop_loss else "`—`",   "inline": True},
        {"name": "Take Profit", "value": f"`${take_profit}`" if take_profit else "`—`", "inline": True},
    ]

    if components:
        breakdown = "\n".join(
            f"{_SOURCE_EMOJI.get(src, '•')} **{src}** `{v:+.4f}`"
            for src, v in sorted(components.items(), key=lambda x: -abs(x[1]))
        )
        fields.append({"name": "Signal Breakdown", "value": breakdown, "inline": False})

    embed_payload = {
        "embeds": [{
            "title": f"{arrow}  {symbol}",
            "color": color,
            "fields": fields,
            "footer": {"text": "DayTradingBot"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    }

    for ch in channels:
        if ch == "console":
            _notify_console(plain)
        elif ch == "telegram":
            _notify_telegram(plain)
        elif ch == "email":
            _notify_email(plain)
        elif ch == "discord":
            _notify_discord_embed(embed_payload)


def notify_order_exit(
    *,
    symbol: str,
    side: str,
    qty: float,
    pnl: float,
    pnl_pct: float,
    reason: str,
    channels: Iterable[str] = ("console",),
) -> None:
    """Rich exit notification with P&L."""
    won   = pnl >= 0
    icon  = "✅" if won else "❌"
    color = _COLOR_BUY if won else _COLOR_SELL

    plain = (
        f"[EXIT {icon}] {side.upper()} {qty:.0f} {symbol} | "
        f"PnL={pnl:+.2f} ({pnl_pct*100:+.2f}%) | reason={reason}"
    )

    embed_payload = {
        "embeds": [{
            "title": f"{icon} EXIT  {symbol}",
            "color": color,
            "fields": [
                {"name": "Side",   "value": f"`{side.upper()}`",          "inline": True},
                {"name": "Qty",    "value": f"`{qty:.0f}`",               "inline": True},
                {"name": "P&L",    "value": f"`{pnl:+.2f} ({pnl_pct*100:+.2f}%)`", "inline": True},
                {"name": "Reason", "value": f"`{reason}`",                "inline": False},
            ],
            "footer": {"text": "DayTradingBot"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }]
    }

    for ch in channels:
        if ch == "console":
            _notify_console(plain)
        elif ch == "telegram":
            _notify_telegram(plain)
        elif ch == "email":
            _notify_email(plain)
        elif ch == "discord":
            _notify_discord_embed(embed_payload)


def _notify_email(msg: str) -> None:
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USERNAME")
    smtp_pass = os.getenv("SMTP_PASSWORD")
    email_to = os.getenv("EMAIL_TO")
    email_from = os.getenv("EMAIL_FROM", smtp_user or "bot@localhost")

    if not (smtp_server and smtp_user and smtp_pass and email_to):
        return

    try:
        mime_msg = MIMEText(msg)
        mime_msg["Subject"] = "DayTradingBot Alert"
        mime_msg["From"] = email_from
        mime_msg["To"] = email_to

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(mime_msg)
    except Exception as e:
        log.warning("Email notify failed: %s", e)


def notify(msg: str, channels: Iterable[str] = ("console",)) -> None:
    """Fan out a plain-text notification to every requested channel."""
    for ch in channels:
        if ch == "console":
            _notify_console(msg)
        elif ch == "telegram":
            _notify_telegram(msg)
        elif ch == "discord":
            _notify_discord(msg)
        elif ch == "email":
            _notify_email(msg)
