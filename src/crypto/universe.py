"""Crypto pair universe — which symbols the CryptoEngine trades."""
from __future__ import annotations

from typing import List


class CryptoUniverse:
    """Holds the configured list of crypto pairs.

    Intentionally minimal — just a typed wrapper around the config list so
    the engine has a single place to query the active pairs.
    """

    _DEFAULTS: List[str] = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD"]

    def __init__(self, cfg: dict):
        self.tickers: List[str] = cfg.get("tickers", self._DEFAULTS)

    def get_tickers(self) -> List[str]:
        return list(self.tickers)
