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
_COLOR_BUY   = 0x2ECC71   # green
_COLOR_SELL  = 0xE74C3C   # red
_COLOR_EXIT  = 0x95A5A6   # grey
_COLOR_WARN  = 0xF39C12   # orange (partial exit)

# Emoji per signal source
_SOURCE_EMOJI = {
    "technical":   "📊",
    "sentiment":   "📰",
    "fundamental": "📄",
    "ml":          "🤖",
    "finrl":       "🧠",
    "orb":         "🔔",
    "vwap_bounce": "📈",
    "macro":       "🌐",
}

# Regime label → display string
_REGIME_DISPLAY = {
    "trending": "📈 Trending",
    "choppy":   "〰️ Choppy",
    "neutral":  "⚖️ Neutral",
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


def _fmt_duration(seconds: float) -> str:
    """Format elapsed seconds as 'Xh Ym' or 'Ym Zs'."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def notify_order_entry(
    *,
    symbol: str,
    side: str,                           # "buy" | "sell"
    qty: float,
    price: float,
    agg_score: float,
    confidence: float,
    min_confidence: float = 0.0,         # effective confidence gate used
    agreement_ok: bool = True,           # ML + FinRL agreement gate
    components: Dict[str, float],        # per-source weighted contributions
    stop_loss: Optional[float],
    take_profit: Optional[float],
    regime_params: Optional[dict] = None,  # regime, adx, pp1_pct, eod_flatten …
    enter_threshold: Optional[float] = None,  # effective entry threshold used
    channels: Iterable[str] = ("console",),
) -> None:
    """Rich order-entry notification with per-signal breakdown and regime context."""
    is_buy = side.lower() == "buy"
    arrow  = "🟢 BUY" if is_buy else "🔴 SHORT"
    color  = _COLOR_BUY if is_buy else _COLOR_SELL

    # Compute stop/TP as percentages
    sl_pct = ((stop_loss   - price) / price * 100) if stop_loss   else None
    tp_pct = ((take_profit - price) / price * 100) if take_profit else None

    # Regime context
    regime_str = None
    adx_str    = None
    eod_str    = None
    if regime_params:
        raw_regime = regime_params.get("regime", "")
        regime_str = _REGIME_DISPLAY.get(raw_regime, raw_regime.capitalize())
        adx_val    = regime_params.get("adx")
        if adx_val is not None and adx_val == adx_val:  # not NaN
            adx_str = f"`{adx_val:.1f}`"
        eod_flatten = regime_params.get("eod_flatten")
        if eod_flatten is not None:
            eod_str = "`Yes`" if eod_flatten else "`No (trending)`"

    # ── Plain-text fallback ───────────────────────────────────────────────
    signal_lines = "  ".join(
        f"{src}={v:+.3f}" for src, v in sorted(components.items())
    )
    regime_part = ""
    if regime_str:
        regime_part = f" | regime={regime_params.get('regime','?')}"
        if adx_str:
            regime_part += f" ADX={regime_params.get('adx', 0):.1f}"
    agree_tag = "" if agreement_ok else " [DISAGREE]"
    plain = (
        f"[{arrow}] {qty:.0f} {symbol} @ ~${price:.2f} | "
        f"score={agg_score:+.3f}(gate={enter_threshold:+.3f}) "
        f"conf={confidence:.0%}(gate={min_confidence:.0%}){agree_tag} | "
        f"SL=${stop_loss}{f'({sl_pct:+.1f}%)' if sl_pct else ''} "
        f"TP=${take_profit}{f'({tp_pct:+.1f}%)' if tp_pct else ''}"
        f"{regime_part} | {signal_lines}"
    )

    # ── Discord embed ─────────────────────────────────────────────────────
    fields = [
        # Row 1: price & qty
        {"name": "Price",  "value": f"`${price:.2f}`",  "inline": True},
        {"name": "Qty",    "value": f"`{qty:.0f} sh`",  "inline": True},
        {"name": "​", "value": "​",            "inline": True},  # spacer

        # Row 2: score & confidence (actual vs gate)
        {"name": "Agg Score",    "value": f"`{agg_score:+.4f}`",  "inline": True},
        {
            "name":  "Score Gate",
            "value": f"`{enter_threshold:+.3f}`" if enter_threshold is not None else "`default`",
            "inline": True,
        },
        {"name": "​", "value": "​", "inline": True},  # spacer

        # Row 3: confidence actual vs gate + ML/FinRL agreement
        {"name": "Confidence",     "value": f"`{confidence:.0%}`",        "inline": True},
        {"name": "Conf Gate",      "value": f"`{min_confidence:.0%}`",     "inline": True},
        {"name": "ML×FinRL Agree", "value": "`✅ Yes`" if agreement_ok else "`⚠️ No`", "inline": True},

        # Row 3: stop & take-profit (absolute + %)
        {
            "name":   "Stop Loss",
            "value":  (f"`${stop_loss:.2f}` `({sl_pct:+.1f}%)`" if stop_loss and sl_pct is not None
                       else "`—`"),
            "inline": True,
        },
        {
            "name":   "Take Profit",
            "value":  (f"`${take_profit:.2f}` `({tp_pct:+.1f}%)`" if take_profit and tp_pct is not None
                       else "`—`"),
            "inline": True,
        },
        {"name": "​", "value": "​", "inline": True},  # spacer
    ]

    # Row 4: regime context (only if available)
    if regime_str or adx_str or eod_str:
        fields += [
            {"name": "Regime",      "value": regime_str or "`—`",  "inline": True},
            {"name": "ADX",         "value": adx_str    or "`—`",  "inline": True},
            {"name": "EOD Flatten", "value": eod_str    or "`—`",  "inline": True},
        ]

    # Signal breakdown sorted by abs contribution (strongest first)
    if components:
        breakdown_lines = []
        for src, v in sorted(components.items(), key=lambda x: -abs(x[1])):
            bar_len = int(abs(v) * 20)          # visual bar up to 20 chars
            bar     = ("█" * bar_len).ljust(10)
            sign    = "▲" if v >= 0 else "▼"
            emoji   = _SOURCE_EMOJI.get(src, "•")
            breakdown_lines.append(
                f"{emoji} **{src:<12}** {sign} `{v:+.4f}` `{bar}`"
            )
        fields.append({
            "name":   "━━ Signal Breakdown ━━",
            "value":  "\n".join(breakdown_lines),
            "inline": False,
        })

    embed_payload = {
        "embeds": [{
            "title":     f"{arrow}  **{symbol}**",
            "color":     color,
            "fields":    fields,
            "footer":    {"text": "DayTradingBot • entry"},
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


# Human-readable exit reason labels
_REASON_DISPLAY: Dict[str, str] = {
    "signal_reversed":    "🔄 Signal Reversed",
    "eod_flatten":        "🌙 EOD Flatten",
    "take_profit":        "🎯 Take Profit",
    "partial_profit_1":   "💰 Partial Profit #1",
    "partial_profit_2":   "💰 Partial Profit #2",
    "stop":               "🛑 Stop Loss",
    "stop_loss":          "🛑 Stop Loss",
    "breakeven_stop":     "⚖️ Breakeven Stop",
    "max_hold":           "⏱️ Max Hold Time",
    "stale_position":     "⏰ Stale Position",
    "trailing_stop":      "📉 Trailing Stop",
}


def notify_order_exit(
    *,
    symbol: str,
    side: str,
    qty: float,
    pnl: float,
    pnl_pct: float,
    reason: str,
    entry_price: Optional[float] = None,
    exit_price:  Optional[float] = None,
    held_since:  Optional[datetime] = None,
    partial_num: Optional[int] = None,     # 1 = first partial, 2 = second, None = full
    channels: Iterable[str] = ("console",),
) -> None:
    """Rich exit notification with P&L, duration, and trade summary."""
    won        = pnl >= 0
    is_partial = partial_num is not None
    if is_partial:
        icon  = "🔶"
        color = _COLOR_WARN
        label = f"PARTIAL #{partial_num} EXIT"
    else:
        icon  = "✅" if won else "❌"
        color = _COLOR_BUY if won else _COLOR_SELL
        label = "EXIT"

    # Human-readable reason (fall back to raw code if not in map)
    reason_display = _REASON_DISPLAY.get(reason.lower(), f"📋 {reason.replace('_', ' ').title()}")

    # Duration
    duration_str = None
    if held_since is not None:
        elapsed = (datetime.now() - held_since).total_seconds()
        duration_str = _fmt_duration(elapsed)

    # ── Plain-text fallback ───────────────────────────────────────────────
    price_part = ""
    if entry_price and exit_price:
        price_part = f" @ ${entry_price:.2f}→${exit_price:.2f}"
    dur_part = f" held={duration_str}" if duration_str else ""
    plain = (
        f"[{icon} {label}] {side.upper()} {qty:.0f} {symbol}"
        f"{price_part} | "
        f"PnL={pnl:+.2f} ({pnl_pct*100:+.2f}%) | "
        f"reason={reason}{dur_part}"
    )

    # ── Discord embed ─────────────────────────────────────────────────────
    # Row 1 — prominent exit reason (full width so it reads at a glance)
    fields: list = [
        {
            "name":   "Exit Reason",
            "value":  f"**{reason_display}**  `{reason}`",
            "inline": False,
        },
    ]

    # Row 2 — P&L
    pnl_sign = "▲" if won else "▼"
    fields.append({
        "name":   "Realised P&L",
        "value":  f"`{pnl_sign} {pnl:+.2f}` `({pnl_pct*100:+.2f}%)`",
        "inline": True,
    })

    # Row 2 cont — entry → exit price (inline with P&L)
    if entry_price is not None and exit_price is not None:
        fields.append({
            "name":   "Price",
            "value":  f"`${entry_price:.2f}` → `${exit_price:.2f}`",
            "inline": True,
        })

    # Row 2 cont — hold duration
    if duration_str:
        fields.append({
            "name":   "Held",
            "value":  f"`{duration_str}`",
            "inline": True,
        })

    # Row 3 — side & qty (lower priority info)
    fields += [
        {"name": "Side", "value": f"`{side.upper()}`",  "inline": True},
        {"name": "Qty",  "value": f"`{qty:.0f} sh`",    "inline": True},
    ]

    embed_payload = {
        "embeds": [{
            "title":     f"{icon} {label}  **{symbol}**",
            "color":     color,
            "fields":    fields,
            "footer":    {"text": f"DayTradingBot • {'partial exit' if is_partial else 'full exit'}"},
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
