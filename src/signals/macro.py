"""Macro Regime / Market Confidence Indicator.

Evaluates a broad market index (e.g. SPY) to determine overall market health.
Returns a multiplier that scales individual stock signal scores.
"""
from __future__ import annotations

from typing import Tuple, Optional
import pandas as pd
import numpy as np

from ..utils.logger import get_logger

log = get_logger(__name__)


class MacroSignal:
    def __init__(self, index_ticker: str = "SPY"):
        self.index_ticker = index_ticker

    def evaluate(self, bars: Optional[pd.DataFrame]) -> Tuple[float, str]:
        """
        Evaluate market health and return a (multiplier, reason) tuple.
        multiplier < 1.0 reduces individual stock scores (weak market).
        multiplier > 1.0 boosts individual stock scores (strong market).
        """
        if bars is None or len(bars) < 20:
            return 1.0, "Insufficient market data for SPY."
            
        close = bars['Close']
        current_price = close.iloc[-1]
        
        # 1. Trend: Price vs 20-period SMA
        sma_20 = close.rolling(20).mean().iloc[-1]
        trend_is_up = current_price > sma_20
        
        # 2. Momentum: 5-period return
        ret_5 = (current_price / close.iloc[-6]) - 1.0 if len(close) > 5 else 0.0
        
        multiplier = 1.0
        reasons = []
        
        if trend_is_up:
            multiplier += 0.15
            reasons.append("SPY above SMA20 (Bullish)")
        else:
            multiplier -= 0.25
            reasons.append("SPY below SMA20 (Bearish)")
            
        if ret_5 > 0.002: # +0.2% in last ~25 mins
            multiplier += 0.15
            reasons.append("Strong short-term momentum")
        elif ret_5 < -0.002:
            multiplier -= 0.15
            reasons.append("Weak short-term momentum")
            
        # Clip multiplier to a reasonable range [0.5, 1.5]
        multiplier = float(np.clip(multiplier, 0.5, 1.5))
        reason_str = ", ".join(reasons)
        
        log.info(f"Macro Market Confidence: {multiplier:.2f}x ({reason_str})")
        return multiplier, reason_str
