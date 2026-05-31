"""Per-symbol win-rate tracker with dynamic exclusion logic.

Rules
─────
1. Every time a trade closes, record whether it was a win or a loss for that
   calendar day.
2. At end-of-day (first bar of the next trading day), evaluate each symbol
   that had at least one closed trade:
     • If that day's win rate ≥ min_win_rate  → reset its bad-day streak to 0.
     • If that day's win rate <  min_win_rate  → increment its bad-day streak.
       – streak == 1 or 2 → tighten max_daily_losses to 1 (early warning).
       – streak >= bad_day_streak_limit        → add to dynamic exclusion list
                                                 (no new entries for this symbol).
3. A symbol can recover from the exclusion list once it posts a winning day
   while in "watch" mode (configurable: may require N consecutive good days).

Config
──────
risk:
  dynamic_exclusion_win_rate:    0.33   # daily win rate threshold
  dynamic_exclusion_streak_days: 3      # consecutive bad days before exclusion
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set

from ..utils.logger import get_logger

log = get_logger(__name__)


class SymbolPerformanceTracker:
    """Track per-symbol daily win rates and manage the dynamic exclusion list.

    Usage (backtester / engine)
    ───────────────────────────
    1. Call record_trade(symbol, day_str, is_win) immediately after each trade closes.
    2. Call end_of_day(day_str) once at the start of each new trading day
       (before evaluating signals for that new day).
    3. Call is_excluded(symbol) and daily_loss_cap(symbol, default) at entry time.
    """

    def __init__(
        self,
        min_win_rate: float = 0.33,
        streak_limit: int = 3,
        recovery_days: int = 1,
        min_trades_per_day: int = 3,
    ):
        self.min_win_rate       = min_win_rate
        self.streak_limit       = streak_limit
        self.recovery_days      = recovery_days        # consecutive good days needed to exit exclusion
        self.min_trades_per_day = min_trades_per_day   # skip win-rate eval if fewer trades than this

        # symbol → day_str → [True/False, ...]
        self._daily: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))

        # Consecutive days below threshold
        self._bad_streak: Dict[str, int]  = {}
        # Consecutive good days since exclusion (for recovery)
        self._good_since:  Dict[str, int] = {}

        self.exclusion_list: Set[str] = set()

    # ── Trade recording ────────────────────────────────────────────────────

    def record_trade(self, symbol: str, day_str: str, is_win: bool) -> None:
        """Record one closed trade outcome for *symbol* on *day_str* ('YYYY-MM-DD')."""
        self._daily[symbol][day_str].append(is_win)

    # ── End-of-day evaluation ──────────────────────────────────────────────

    def end_of_day(self, day_str: str) -> None:
        """Evaluate all symbols that traded on *day_str* and update streaks/exclusions.

        Call this once at the beginning of each new trading day (after detecting a
        date change in the bar stream) so that entries on the new day already
        reflect the updated exclusion state.
        """
        for sym, days in self._daily.items():
            results = days.get(day_str, [])
            if not results:
                # Symbol had no closed trades this day — skip (don't penalise it
                # for a quiet day; only count days with actual trade outcomes).
                continue

            trades   = len(results)
            wins     = sum(results)

            # Skip win-rate evaluation for days with very few trades —
            # a single loss on 2 trades shouldn't trigger a bad-day penalty.
            if trades < self.min_trades_per_day:
                log.debug(
                    "PerformanceTracker: %s skipped on %s (%d trade(s) < min %d).",
                    sym, day_str, trades, self.min_trades_per_day,
                )
                continue

            win_rate = wins / trades

            if win_rate >= self.min_win_rate:
                # Good day — reset bad streak.
                prev_streak = self._bad_streak.get(sym, 0)
                self._bad_streak[sym] = 0

                if sym in self.exclusion_list:
                    # Count recovery days.
                    self._good_since[sym] = self._good_since.get(sym, 0) + 1
                    if self._good_since[sym] >= self.recovery_days:
                        self.exclusion_list.discard(sym)
                        self._good_since.pop(sym, None)
                        log.info(
                            "PerformanceTracker: %s REINSTATED after %d good day(s).",
                            sym, self.recovery_days,
                        )
                elif prev_streak > 0:
                    log.info(
                        "PerformanceTracker: %s bad streak RESET (day=%s  win=%d/%d  %.0f%%).",
                        sym, day_str, wins, trades, win_rate * 100,
                    )
            else:
                # Bad day — increment streak.
                self._good_since.pop(sym, None)
                self._bad_streak[sym] = self._bad_streak.get(sym, 0) + 1
                streak = self._bad_streak[sym]

                if sym not in self.exclusion_list and streak >= self.streak_limit:
                    self.exclusion_list.add(sym)
                    log.warning(
                        "PerformanceTracker: %s EXCLUDED — %d consecutive days with "
                        "win rate < %.0f%% (day=%s  win=%d/%d  %.0f%%).",
                        sym, streak, self.min_win_rate * 100,
                        day_str, wins, trades, win_rate * 100,
                    )
                else:
                    log.info(
                        "PerformanceTracker: %s bad day %d/%d "
                        "(day=%s  win=%d/%d  %.0f%%) — daily-loss cap → 1.",
                        sym, streak, self.streak_limit,
                        day_str, wins, trades, win_rate * 100,
                    )

    # ── Query helpers ──────────────────────────────────────────────────────

    def is_excluded(self, symbol: str) -> bool:
        """True if *symbol* is on the dynamic exclusion list (no new entries)."""
        return symbol in self.exclusion_list

    def daily_loss_cap(self, symbol: str, default_cap: int) -> int:
        """Return the effective max-daily-losses cap for *symbol*.

        • Excluded symbol  → 0  (never enter)
        • Bad streak ≥ 1   → 1  (tightened early-warning cap)
        • No bad streak    → default_cap  (from config)
        """
        if symbol in self.exclusion_list:
            return 0
        if self._bad_streak.get(symbol, 0) >= 1:
            return 1
        return default_cap

    # ── Status ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "excluded":   sorted(self.exclusion_list),
            "bad_streak": {s: v for s, v in self._bad_streak.items() if v > 0},
        }
