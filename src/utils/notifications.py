"""Trade and alert notifications via console / Telegram / Discord."""
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from typing import Iterable

import requests

from .logger import get_logger

log = get_logger(__name__)


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
    except Exception as e:  # pragma: no cover
        log.warning("Telegram notify failed: %s", e)


def _notify_discord(msg: str) -> None:
    url = os.getenv("DISCORD_WEBHOOK_URL")
    if not url:
        return
    try:
        requests.post(url, json={"content": msg}, timeout=5)
    except Exception as e:  # pragma: no cover
        log.warning("Discord notify failed: %s", e)


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
    except Exception as e:  # pragma: no cover
        log.warning("Email notify failed: %s", e)


def notify(msg: str, channels: Iterable[str] = ("console",)) -> None:
    """Fan out a notification to every requested channel."""
    for ch in channels:
        if ch == "console":
            _notify_console(msg)
        elif ch == "telegram":
            _notify_telegram(msg)
        elif ch == "discord":
            _notify_discord(msg)
        elif ch == "email":
            _notify_email(msg)
