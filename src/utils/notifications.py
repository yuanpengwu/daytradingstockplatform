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


def _fmt_price(price: float, is_crypto: bool = False) -> str:
    """Format a dollar price. Crypto uses more decimal places for sub-$100 coins."""
    if not is_crypto or price >= 100:
        return f"${price:.2f}"
    if price >= 1:
        return f"${price:.4f}"
    return f"${price:.6f}"


def _fmt_qty(qty: float, symbol: str) -> str:
    """Format quantity: fractional coin units for crypto, integer shares for stocks."""
    if "/" in symbol:                          # e.g. BTC/USD, ETH/USD
        base = symbol.split("/")[0]            # "BTC"
        if qty >= 1:
            return f"{qty:.4f} {base}"
        if qty >= 0.0001:
            return f"{qty:.6f} {base}"
        return f"{qty:.8f} {base}"
    return f"{qty:.0f} sh"


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
    components: Dict[str, float],        # per-source weighted contributions (for log)
    raw_scores: Optional[Dict[str, float]] = None,  # per-source raw scores [-1,+1] (for display)
    stop_loss: Optional[float],
    take_profit: Optional[float],
    regime_params: Optional[dict] = None,  # regime, adx, pp1_pct, eod_flatten …
    enter_threshold: Optional[float] = None,  # effective entry threshold used
    position_size_info: Optional[dict] = None,  # sizing details from RiskManager
    channels: Iterable[str] = ("console",),
) -> None:
    """Rich order-entry notification with per-signal breakdown and regime context."""
    is_buy    = side.lower() == "buy"
    is_crypto = "/" in symbol
    arrow     = "🟢 BUY" if is_buy else "🔴 SHORT"
    color     = _COLOR_BUY if is_buy else _COLOR_SELL

    # Formatted price / qty strings (crypto-aware)
    price_str = _fmt_price(price, is_crypto)
    qty_str   = _fmt_qty(qty, symbol)

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
    # Use raw scores (full [-1,+1] range) so the signal line is readable at a
    # glance.  The weighted components are tiny fractions (weight × conf × score)
    # and convey nothing useful in a one-line notification.
    # Fall back to components only when raw_scores aren't provided.
    _pt_scores = raw_scores if raw_scores else components
    signal_lines = "  ".join(
        f"{src}={v:+.3f}"
        for src, v in sorted(_pt_scores.items(), key=lambda x: -abs(x[1]))
        if abs(v) > 1e-4   # omit sources that returned no signal (dead / offline)
    )
    regime_part = ""
    if regime_str:
        regime_part = f" | regime={regime_params.get('regime','?')}"
        if adx_str:
            regime_part += f" ADX={regime_params.get('adx', 0):.1f}"
    agree_tag = "" if agreement_ok else " [DISAGREE]"
    # Guard against enter_threshold=None — it has a default of None in the
    # function signature and older callers may not pass it.
    gate_str = f"(gate={enter_threshold:+.3f})" if enter_threshold is not None else ""
    # Position-size rationale line
    size_part = ""
    if position_size_info:
        _si = position_size_info
        _ratio_tag = (
            f" signal_ratio={_si['signal_ratio']:.0%}" if _si.get("signal_ratio", 1.0) < 0.999 else ""
        )
        size_part = (
            f" | size: ${_si.get('notional', 0):.0f} of ${_si.get('equity', 0):.0f}"
            f" (kelly={_si.get('kelly_pct', 0):.1f}%→target={_si.get('target_pct', 0):.1f}%"
            f" eff_kelly={_si.get('eff_kelly', 0):.3f}"
            f" regime_mult={_si.get('regime_mult', 1):.2f}x{_ratio_tag})"
        )
    plain = (
        f"[{arrow}] {qty_str} {symbol} @ ~{price_str} | "
        f"score={agg_score:+.3f}{gate_str} "
        f"conf={confidence:.0%}(gate={min_confidence:.0%}){agree_tag} | "
        f"SL={_fmt_price(stop_loss, is_crypto) if stop_loss else '—'}{f'({sl_pct:+.1f}%)' if sl_pct else ''} "
        f"TP={_fmt_price(take_profit, is_crypto) if take_profit else '—'}{f'({tp_pct:+.1f}%)' if tp_pct else ''}"
        f"{regime_part} | {signal_lines}{size_part}"
    )

    # ── Discord embed ─────────────────────────────────────────────────────
    fields = [
        # Row 1: price & qty
        {"name": "Price",  "value": f"`{price_str}`",  "inline": True},
        {"name": "Qty",    "value": f"`{qty_str}`",    "inline": True},
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

        # Row 4: stop & take-profit (absolute + %)
        {
            "name":   "Stop Loss",
            "value":  (f"`{_fmt_price(stop_loss, is_crypto)}` `({sl_pct:+.1f}%)`"
                       if stop_loss and sl_pct is not None else "`—`"),
            "inline": True,
        },
        {
            "name":   "Take Profit",
            "value":  (f"`{_fmt_price(take_profit, is_crypto)}` `({tp_pct:+.1f}%)`"
                       if take_profit and tp_pct is not None else "`—`"),
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

    # Position-size row (only when sizing info is available)
    if position_size_info:
        _si = position_size_info
        _scale_note = (
            f" *(signal_ratio={_si['signal_ratio']:.0%})*"
            if _si.get("signal_ratio", 1.0) < 0.999 else ""
        )
        fields.append({
            "name":   "━━ Position Size ━━",
            "value":  (
                f"**${_si.get('notional', 0):.0f}** of ${_si.get('equity', 0):.0f} equity"
                f" · kelly `{_si.get('kelly_pct', 0):.1f}%` → target `{_si.get('target_pct', 0):.1f}%`"
                f" · eff_kelly `{_si.get('eff_kelly', 0):.3f}`"
                f" · regime_mult `{_si.get('regime_mult', 1):.2f}x`{_scale_note}"
            ),
            "inline": False,
        })

    # Signal breakdown — use raw scores for bars (full [-1,+1] range → readable bars)
    # Fall back to weighted components if raw_scores not available.
    display = raw_scores if raw_scores else components
    if display:
        breakdown_lines = []
        # Sort by absolute raw score descending (strongest signal first)
        for src, v in sorted(display.items(), key=lambda x: -abs(x[1])):
            bar_len = int(abs(v) * 16)                  # 0–16 blocks for [-1,+1]
            bar     = ("█" * bar_len).ljust(16, "░")    # filled vs empty blocks
            sign    = "▲" if v >= 0 else "▼"
            emoji   = _SOURCE_EMOJI.get(src, "•")
            # Also show the weighted contribution in parentheses for context
            contrib = components.get(src, 0.0)
            breakdown_lines.append(
                f"{emoji} **{src:<12}** {sign} `{v:+.3f}` `{bar}` `w={contrib:+.3f}`"
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


# Human-readable exit reason labels.  Matched by substring (ordered — first
# hit wins) because reasons arrive in two formats: snake_case codes from the
# crypto trader ("stop_loss") and free-text sentences from the stock
# RiskManager ("stop price hit (px=…)").  Specific patterns come before the
# generic "stop" catch-all.
_REASON_KEYWORDS: list = [
    ("breakeven",        "⚖️ Breakeven Stop"),
    ("trailing",         "📉 Trailing Stop"),
    ("partial_profit_1", "💰 Partial Profit #1"),
    ("partial_profit_2", "💰 Partial Profit #2"),
    ("time-decayed",     "🎯 Take Profit (time-decayed)"),
    ("take_profit",      "🎯 Take Profit"),
    ("take-profit",      "🎯 Take Profit"),
    ("take profit",      "🎯 Take Profit"),
    ("stop",             "🛑 Stop Loss"),
    ("signal_reversed",  "🔄 Signal Reversed"),
    ("signal reversed",  "🔄 Signal Reversed"),
    ("eod",              "🌙 EOD Flatten"),
    ("end-of-day",       "🌙 EOD Flatten"),
    ("max_hold",         "⏱️ Max Hold Time"),
    ("max hold",         "⏱️ Max Hold Time"),
    ("stale",            "⏰ Stale Position"),
]


def _display_reason(reason: str) -> str:
    r = reason.lower()
    for key, label in _REASON_KEYWORDS:
        if key in r:
            return label
    return f"📋 {reason.replace('_', ' ').title()}"


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
    exit_scores: Optional[Dict[str, float]] = None,  # raw signal scores at exit time
    channels: Iterable[str] = ("console",),
) -> None:
    """Rich exit notification with P&L, duration, and trade summary."""
    won        = pnl >= 0
    is_crypto  = "/" in symbol
    qty_str    = _fmt_qty(qty, symbol)
    is_partial = partial_num is not None
    if is_partial:
        icon  = "🔶"
        color = _COLOR_WARN
        label = f"PARTIAL #{partial_num} EXIT"
    else:
        icon  = "✅" if won else "❌"
        color = _COLOR_BUY if won else _COLOR_SELL
        label = "EXIT"

    # Human-readable reason (fall back to raw code if no keyword matches)
    reason_display = _display_reason(reason)

    # Duration — guard against negative elapsed (clock drift / DST edge case)
    if held_since is not None:
        elapsed = max(0.0, (datetime.now() - held_since).total_seconds())
        duration_str: str = _fmt_duration(elapsed)
    else:
        # Entry time unknown — position was reconciled after a bot restart
        # where the fill happened between the in-memory write and the file
        # write, or was placed manually outside the bot.  Show "?" so it
        # is visible in the notification rather than silently absent.
        duration_str = "unknown"

    # ── Plain-text fallback ───────────────────────────────────────────────
    price_part = ""
    if entry_price and exit_price:
        price_part = f" @ {_fmt_price(entry_price, is_crypto)}→{_fmt_price(exit_price, is_crypto)}"
    dur_part = f" held={duration_str}"
    # Exit signal scores — compact inline list sorted by magnitude
    exit_sig_part = ""
    if exit_scores:
        _sig_items = "  ".join(
            f"{src[:4]}={v:+.3f}"
            for src, v in sorted(exit_scores.items(), key=lambda x: -abs(x[1]))
            if abs(v) > 1e-4
        )
        if _sig_items:
            exit_sig_part = f" | exit_signals: {_sig_items}"
    plain = (
        f"[{icon} {label}] {side.upper()} {qty_str} {symbol}"
        f"{price_part} | "
        f"PnL={pnl:+.2f} ({pnl_pct*100:+.2f}%) | "
        f"reason={reason}{dur_part}{exit_sig_part}"
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
            "value":  f"`{_fmt_price(entry_price, is_crypto)}` → `{_fmt_price(exit_price, is_crypto)}`",
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
        {"name": "Qty",  "value": f"`{qty_str}`",       "inline": True},
    ]

    # Exit signal scores row (when available)
    if exit_scores:
        _esc_lines = []
        for src, v in sorted(exit_scores.items(), key=lambda x: -abs(x[1])):
            if abs(v) <= 1e-4:
                continue
            bar_len = int(abs(v) * 16)
            bar     = ("█" * bar_len).ljust(16, "░")
            sign    = "▲" if v >= 0 else "▼"
            emoji   = _SOURCE_EMOJI.get(src, "•")
            _esc_lines.append(f"{emoji} **{src:<12}** {sign} `{v:+.3f}` `{bar}`")
        if _esc_lines:
            fields.append({
                "name":   "━━ Signals at Exit ━━",
                "value":  "\n".join(_esc_lines),
                "inline": False,
            })

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
