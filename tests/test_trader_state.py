"""Trader state persistence + restart reconciliation tests.

Covers the two restart failure modes seen on 2026-06-16:
  1. trader_state.json round-trips stops / entry time / partial progress /
     regime params so an ATR-stopped position survives a restart.
  2. reconcile_positions() rebuilds ATR-or-% stops and an entry time for a
     broker-held position that has no in-memory tracking (empty state file),
     and persists the result.
"""
from __future__ import annotations

import json
from datetime import datetime

import src.execution.trader as trader_mod
from src.brokers.base import Position
from src.execution.trader import Trader
from src.risk.risk_manager import RiskManager


class _FakeBroker:
    """Minimal broker stub — Trader only stores it and reads positions."""

    def __init__(self, positions=None):
        self._positions = positions or {}

    def get_positions(self):
        return dict(self._positions)

    def get_stock_positions(self):
        return dict(self._positions)

    def get_last_price(self, sym):
        p = self._positions.get(sym)
        return p.current_price if p else 0.0


def _make_trader(tmp_path, monkeypatch, positions=None):
    """Build a Trader with _STATE_PATH redirected into tmp_path."""
    monkeypatch.setattr(trader_mod, "_STATE_PATH", tmp_path / "trader_state.json")
    return Trader(_FakeBroker(positions), RiskManager({}))


def test_save_load_round_trip(tmp_path, monkeypatch):
    state_path = tmp_path / "trader_state.json"
    monkeypatch.setattr(trader_mod, "_STATE_PATH", state_path)

    t = Trader(_FakeBroker(), RiskManager({}))
    t._stops["AAPL"] = (95.0, 110.0)
    t._entry_time["AAPL"] = datetime(2026, 6, 16, 10, 30, 0)
    t._entry_qty["AAPL"] = 12.0
    t._partial_exits["AAPL"] = 1
    t._regime_params["AAPL"] = {"regime": "trending", "eod_flatten": False}
    t._save_state()

    # File is populated (not the empty {} that caused the bug).
    assert state_path.exists()
    on_disk = json.loads(state_path.read_text(encoding="utf-8"))
    assert on_disk["AAPL"]["stop"] == 95.0
    assert on_disk["AAPL"]["tp"] == 110.0

    # A fresh Trader (simulating restart) restores everything via _load_state.
    t2 = Trader(_FakeBroker(), RiskManager({}))
    assert t2._stops["AAPL"] == (95.0, 110.0)
    assert t2._entry_qty["AAPL"] == 12.0
    assert t2._partial_exits["AAPL"] == 1
    assert t2._regime_params["AAPL"]["eod_flatten"] is False
    assert t2._entry_time["AAPL"] == datetime(2026, 6, 16, 10, 30, 0)


def test_reconcile_rebuilds_atr_stops_and_persists(tmp_path, monkeypatch):
    state_path = tmp_path / "trader_state.json"
    pos = Position(symbol="PG", qty=10, avg_entry_price=150.0, current_price=151.0)
    t = _make_trader(tmp_path, monkeypatch, positions={"PG": pos})

    # State file empty + no entry time → exactly the 2026-06-16 PG scenario.
    assert "PG" not in t._stops
    assert "PG" not in t._entry_time

    t.reconcile_positions({"PG": pos}, atr_by_sym={"PG": 1.0})

    # ATR stop = 150 - 2.5*1 = 147.5 (tighter than 2% floor 147.0 → ATR wins).
    # ATR TP   = 150 + 3.0*1 = 153.0 (looser than 4% floor 156.0 → floor wins).
    stop, tp = t._stops["PG"]
    assert stop == 147.5
    assert tp == 156.0
    assert "PG" in t._entry_time          # entry time restored (was None)
    assert t._trail_high["PG"] == 151.0   # watermark seeded from current
    assert t._entry_qty["PG"] == 10.0

    # Reconstructed state is persisted so the NEXT restart reads it back.
    assert state_path.exists()
    on_disk = json.loads(state_path.read_text(encoding="utf-8"))
    assert on_disk["PG"]["stop"] == 147.5
    assert on_disk["PG"]["tp"] == 156.0


def test_reconcile_falls_back_to_pct_without_atr(tmp_path, monkeypatch):
    pos = Position(symbol="PG", qty=10, avg_entry_price=150.0, current_price=150.0)
    t = _make_trader(tmp_path, monkeypatch, positions={"PG": pos})

    t.reconcile_positions({"PG": pos})  # no ATR supplied

    stop, tp = t._stops["PG"]
    assert stop == 147.0   # 150 * (1 - 0.02)
    assert tp == 156.0     # 150 * (1 + 0.04)


def test_reconcile_skips_tracked_symbol(tmp_path, monkeypatch):
    pos = Position(symbol="PG", qty=10, avg_entry_price=150.0, current_price=150.0)
    t = _make_trader(tmp_path, monkeypatch, positions={"PG": pos})

    t._stops["PG"] = (140.0, 165.0)   # already tracking a wider, custom stop
    t.reconcile_positions({"PG": pos}, atr_by_sym={"PG": 1.0})

    assert t._stops["PG"] == (140.0, 165.0)   # untouched


def test_reconcile_short_position(tmp_path, monkeypatch):
    pos = Position(symbol="XYZ", qty=-10, avg_entry_price=100.0, current_price=99.0)
    t = _make_trader(tmp_path, monkeypatch, positions={"XYZ": pos})

    t.reconcile_positions({"XYZ": pos}, atr_by_sym={"XYZ": 5.0})

    # Short: stop = min(atr_stop, pct_stop) — lower price is the tighter stop.
    #   atr_stop = 100 + 2.5*5 = 112.5 ; pct_stop = 100*1.02 = 102.0 → 102.0 wins.
    #   (a small ATR would put atr_stop below the floor and win instead.)
    # TP: min(atr_tp, pct_tp) — atr_tp = 100 - 3*5 = 85.0 ; pct_tp = 96.0 → 85.0.
    stop, tp = t._stops["XYZ"]
    assert stop == 102.0
    assert tp == 85.0
    assert t._trail_high["XYZ"] == 99.0   # short watermark = min(entry, current)
