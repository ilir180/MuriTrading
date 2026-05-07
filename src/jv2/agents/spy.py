"""
JV Boting v2 – Spy Agent
Info-Broker: Verteilt selektive Intel zwischen Bots.
Kein Konsens, keine Kontrolle — nur Information.
"""


class SpyAgent:
    def __init__(self):
        self.last_intel = {}

    def compile_intel(self, bots, current_price):
        """
        Kompiliert für jeden Bot spezifische Intel von den anderen.
        Jeder Bot bekommt nur das, was für ihn relevant ist.
        """
        # Alle aktuellen Signale und Positionen sammeln
        signals = {}
        positions = {}
        for bot in bots:
            if bot.state.last_signal and bot.state.last_signal.direction != "neutral":
                signals[bot.bot_id] = {
                    "direction": bot.state.last_signal.direction,
                    "confidence": bot.state.last_signal.confidence,
                }
            if bot.state.position:
                positions[bot.bot_id] = {
                    "direction": bot.state.position.direction,
                    "pnl": bot.state.position.unrealized_pnl(current_price),
                }

        bulls = sum(1 for s in signals.values() if s["direction"] == "long")
        bears = sum(1 for s in signals.values() if s["direction"] == "short")
        bots_in_loss = sum(1 for p in positions.values() if p["pnl"] < 0)

        def _get_dir(bot_id):
            return signals.get(bot_id, {}).get("direction")

        def _get_conf(bot_id):
            return signals.get(bot_id, {}).get("confidence", 0)

        intel = {}
        for bot in bots:
            base = {
                "bulls": bulls,
                "bears": bears,
                "total_positioned": len(positions),
                "bots_in_loss": bots_in_loss,
            }

            # Spezifische Intel pro Bot-Typ
            if bot.bot_id == "trend_rider":
                base["whale_direction"] = _get_dir("flow_tracker")
                base["momentum_confirms"] = _get_dir("momentum_surfer")

            elif bot.bot_id == "mean_reverter":
                base["trend_strength"] = _get_conf("trend_rider")
                base["momentum_strength"] = _get_conf("momentum_surfer")

            elif bot.bot_id == "breakout_hunter":
                base["whale_direction"] = _get_dir("flow_tracker")
                base["vol_fader_active"] = _get_dir("volatility_fader") is not None

            elif bot.bot_id == "contrarian":
                base["whale_direction"] = _get_dir("flow_tracker")
                base["flow_confidence"] = _get_conf("flow_tracker")

            elif bot.bot_id == "flow_tracker":
                base["trend_direction"] = _get_dir("trend_rider")
                base["momentum_direction"] = _get_dir("momentum_surfer")

            elif bot.bot_id == "momentum_surfer":
                base["trend_direction"] = _get_dir("trend_rider")
                base["whale_direction"] = _get_dir("flow_tracker")

            elif bot.bot_id == "level_bouncer":
                base["trend_direction"] = _get_dir("trend_rider")
                base["trend_strength"] = _get_conf("trend_rider")

            elif bot.bot_id == "volatility_fader":
                base["trend_strength"] = _get_conf("trend_rider")
                base["momentum_direction"] = _get_dir("momentum_surfer")

            intel[bot.bot_id] = base

        self.last_intel = intel
        return intel

    def get_summary(self):
        """Kompakte Zusammenfassung für Logging."""
        if not self.last_intel:
            return {}
        sample = next(iter(self.last_intel.values()), {})
        return {
            "bulls": sample.get("bulls", 0),
            "bears": sample.get("bears", 0),
            "positioned": sample.get("total_positioned", 0),
        }
