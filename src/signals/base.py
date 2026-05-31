"""Shared signal data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict


class SignalSource(str, Enum):
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    ML = "ml"
    ORB = "orb"
    VWAP_BOUNCE = "vwap_bounce"
    FINRL = "finrl"


@dataclass
class Signal:
    symbol: str
    source: SignalSource
    score: float                 # normalized in [-1, +1]; +1 = strong buy, -1 = strong sell
    confidence: float = 0.5      # [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        # Clamp defensively.
        self.score = max(-1.0, min(1.0, float(self.score)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
