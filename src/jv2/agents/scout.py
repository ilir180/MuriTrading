"""
JV Boting v2 – Scout Agent
Sondierer: Analysiert verpasste Chancen und erkennt neue Marktphasen.
"""

import json
import os
from datetime import datetime, timezone

from src.jv2.config import SCOUT_REPORT


class ScoutAgent:
    def __init__(self):
        self.missed_opportunities = []
        self._load()

    def analyze_missed_moves(self, df_4h, bots, last_n_candles=6):
        """Findet signifikante Moves die kein Bot getradet hat."""
        if df_4h is None or len(df_4h) < last_n_candles + 2:
            return []

        missed = []

        for i in range(-last_n_candles, -1):
            try:
                candle = df_4h.iloc[i]
                next_candle = df_4h.iloc[i + 1]
            except IndexError:
                continue

            move_pct = (next_candle["close"] - candle["close"]) / candle["close"]
            if abs(move_pct) < 0.012:  # < 1.2% = nicht interessant
                continue

            move_dir = "long" if move_pct > 0 else "short"
            candle_time = candle.name if hasattr(candle, 'name') else str(i)

            # War ein Bot positioniert?
            positioned = False
            for bot in bots:
                if bot.state.position and bot.state.position.direction == move_dir:
                    positioned = True
                    break

            if not positioned:
                phase = self._detect_phase(candle)
                missed.append({
                    "timestamp": str(candle_time),
                    "move_pct": round(move_pct * 100, 2),
                    "direction": move_dir,
                    "phase": phase,
                    "why_missed": self._hypothesize(candle),
                })

        self.missed_opportunities.extend(missed)
        # Nur letzte 50 behalten
        self.missed_opportunities = self.missed_opportunities[-50:]
        self._save()
        return missed

    def _detect_phase(self, row):
        adx = row.get("4h_adx", 20) if hasattr(row, 'get') else 20
        chop = row.get("4h_chop", 0.5) if hasattr(row, 'get') else 0.5
        vol_z = row.get("4h_vol_regime", 0) if hasattr(row, 'get') else 0

        if adx > 25 and chop < 0.45:
            return "trending"
        elif adx < 18 and chop > 0.55:
            return "ranging"
        elif vol_z > 1.5:
            return "volatile"
        else:
            return "transitional"

    def _hypothesize(self, row):
        reasons = []
        adx = row.get("4h_adx", 20) if hasattr(row, 'get') else 20
        rsi = row.get("4h_rsi_14", 50) if hasattr(row, 'get') else 50
        bb_sq = row.get("4h_bb_squeeze", 1.0) if hasattr(row, 'get') else 1.0

        if 18 < adx < 25:
            reasons.append("ADX Dead-Zone (18-25)")
        if 35 < rsi < 65:
            reasons.append("RSI neutral (kein Extrem)")
        if 0.7 < bb_sq < 1.2:
            reasons.append("Kein BB Squeeze")

        return "; ".join(reasons) if reasons else "Unbekannt"

    def get_gap_analysis(self):
        if not self.missed_opportunities:
            return {"total_missed": 0, "by_phase": {}, "gaps": []}

        by_phase = {}
        for m in self.missed_opportunities:
            phase = m["phase"]
            by_phase.setdefault(phase, []).append(m)

        gaps = []
        if len(by_phase.get("transitional", [])) >= 3:
            gaps.append("Kein Bot deckt Transitions-Phasen (ADX 18-25)")
        if len(by_phase.get("ranging", [])) >= 3:
            gaps.append("Range-Moves werden verpasst")

        total_missed_pct = sum(abs(m["move_pct"]) for m in self.missed_opportunities)

        return {
            "total_missed": len(self.missed_opportunities),
            "total_missed_pct": round(total_missed_pct, 1),
            "by_phase": {k: len(v) for k, v in by_phase.items()},
            "gaps": gaps,
            "worst_miss": max(self.missed_opportunities, key=lambda x: abs(x["move_pct"])) if self.missed_opportunities else None,
        }

    def _save(self):
        try:
            with open(SCOUT_REPORT, "w") as f:
                json.dump({
                    "missed": self.missed_opportunities,
                    "analysis": self.get_gap_analysis(),
                    "updated": datetime.now(timezone.utc).isoformat(),
                }, f, indent=2, default=str)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(SCOUT_REPORT):
                with open(SCOUT_REPORT) as f:
                    data = json.load(f)
                self.missed_opportunities = data.get("missed", [])
        except Exception:
            self.missed_opportunities = []
