"""
Joint Venture Boting – Signal Protocol
Das Kommunikationsformat zwischen JV-Bots und Prime.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class JVSignal:
    """Ein Signal von einem JV-Bot an den Prime."""

    bot_id: str               # "momentum", "volume", "regime", "sentiment"
    timestamp: str            # ISO 8601 UTC
    direction: str            # "long" | "short" | "neutral"
    confidence: float         # 0.0-1.0 (wie sicher)
    reasoning: str            # Menschenlesbare Begründung
    features: dict = field(default_factory=dict)   # Key-Metriken
    ttl_candles: int = 3      # Gültigkeit in 4H-Kerzen (default 12h)
    price_at_signal: float = 0.0  # Preis zum Zeitpunkt des Signals

    def __post_init__(self):
        assert self.direction in ("long", "short", "neutral"), \
            f"Invalid direction: {self.direction}"
        self.confidence = max(0.0, min(1.0, self.confidence))

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return JVSignal(**{k: v for k, v in d.items()
                          if k in JVSignal.__dataclass_fields__})

    def save(self, signals_dir):
        """Speichert Signal als JSON-Datei."""
        path = os.path.join(signals_dir, f"{self.bot_id}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @staticmethod
    def load(signals_dir, bot_id):
        """Lädt Signal eines Bots."""
        path = os.path.join(signals_dir, f"{bot_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return JVSignal.from_dict(json.load(f))

    @staticmethod
    def create(bot_id, direction, confidence, reasoning,
               price, features=None, ttl_candles=3):
        """Factory-Methode für einfache Signal-Erstellung."""
        return JVSignal(
            bot_id=bot_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            direction=direction,
            confidence=confidence,
            reasoning=reasoning,
            features=features or {},
            ttl_candles=ttl_candles,
            price_at_signal=price,
        )
