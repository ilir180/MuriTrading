"""
Joint Venture Boting – Base Bot
Abstrakte Klasse für alle JV-Bots.

Jeder Bot hat eine Aufgabe: observe() aufrufen und ein JVSignal zurückgeben.
Der Bot weiss nichts über Credits, den Prime oder andere Bots.
"""

from abc import ABC, abstractmethod
from src.jv.signal_protocol import JVSignal


class JVBot(ABC):
    """Abstrakte Basis für alle JV-Bots."""

    def __init__(self, bot_id: str):
        self.bot_id = bot_id

    @abstractmethod
    def observe(self, market_data: dict) -> JVSignal:
        """
        Analysiert den aktuellen Markt und gibt ein Signal zurück.

        Args:
            market_data: Dict mit vorberechneten Daten:
                - price: float (aktueller Preis)
                - df_1h: DataFrame mit 1H-Indikatoren
                - df_4h: DataFrame mit 4H-Indikatoren
                - df_1d: DataFrame mit 1D-Indikatoren
                - latest_1h: Series (letzte 1H-Zeile)
                - latest_4h: Series (letzte 4H-Zeile)
                - latest_1d: Series (letzte 1D-Zeile)
                - exchange: CCXT Exchange Instance
                - atr_4h: float (4H ATR)

        Returns:
            JVSignal mit direction, confidence, reasoning
        """
        ...

    def neutral(self, price: float, reason: str = "Keine klaren Bedingungen") -> JVSignal:
        """Convenience: Neutral-Signal erzeugen."""
        return JVSignal.create(
            bot_id=self.bot_id,
            direction="neutral",
            confidence=0.0,
            reasoning=reason,
            price=price,
        )
