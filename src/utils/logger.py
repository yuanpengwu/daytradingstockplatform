"""Centralized logger setup.

Configures the root logger ONCE so that every module-level logger created
with `logging.getLogger(__name__)` (or this helper) propagates to the same
console + rotating-file handlers.
"""
from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def _make_console_utf8_safe() -> None:
    """Force stdout/stderr to UTF-8.

    Windows consoles default to the legacy cp1252 codec, which raises
    'charmap' codec errors the moment anything non-ASCII (a ticker symbol,
    a file path containing non-Latin characters, a yfinance message) is
    printed. Reconfiguring to UTF-8 with a forgiving error handler makes
    every handler that writes to these streams safe.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, ValueError):
            # Older Python or a stream that doesn't support reconfigure.
            pass


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    _make_console_utf8_safe()

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(level)
    root.addHandler(sh)

    # Rotating file (always UTF-8 so non-ASCII content never breaks it)
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        fh = RotatingFileHandler(
            log_dir / "bot.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        fh.setLevel(level)
        root.addHandler(fh)
    except Exception:
        # If the filesystem is read-only (e.g. some sandboxes), console-only is fine.
        pass

    # yfinance logs retryable fetch failures at ERROR level and is very noisy.
    # MarketData already handles failures gracefully and logs its own clean
    # warning, so silence yfinance's internal logger.
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)

    _CONFIGURED = True


def get_logger(name: str = "daytradingbot") -> logging.Logger:
    """Return a logger that writes to the shared console + file handlers."""
    _configure_root()
    return logging.getLogger(name)
