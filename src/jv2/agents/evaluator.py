"""
JV Boting v2 – Evaluator
Trennt These, Execution und Edge für jeden Bot.

1. These-Score:  Hat der Bot den Markt richtig gelesen?
2. Execution-Score:  Hat er das Maximum rausgeholt?
3. Edge-Score:  Besser als Zufall?
"""

import math
import random
from collections import defaultdict
from datetime import datetime, timezone


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


class Evaluator:

    def evaluate_signals(self, signals_with_outcomes):
        """
        Input: Liste von dicts mit:
          - bot_id, direction, confidence, price_at_signal
          - future_prices: [price_1c, price_2c, price_4c, price_8c]
          - trade_pnl: float or None (wenn kein Trade)
          - max_favorable: max Bewegung in richtige Richtung
          - max_adverse: max Bewegung gegen Position

        Returns: dict {bot_id: {these_score, exec_score, edge_score, details}}
        """
        by_bot = defaultdict(list)
        for s in signals_with_outcomes:
            by_bot[s["bot_id"]].append(s)

        results = {}
        for bot_id, sigs in by_bot.items():
            results[bot_id] = self._evaluate_bot(bot_id, sigs)
        return results

    def _evaluate_bot(self, bot_id, signals):
        these_correct = 0
        these_total = 0
        these_missed = 0  # NEUTRAL aber Markt bewegt sich >1.5%

        exec_scores = []
        trade_count = 0
        edge_wins = 0
        edge_total = 0

        for sig in signals:
            direction = sig["direction"]
            price = sig["price_at_signal"]
            future = sig.get("future_prices", [])
            if not future or not any(f > 0 for f in future):
                continue

            # ── 1. THESE-SCORE ──
            # Hat sich der Markt in die vorhergesagte Richtung bewegt?
            # Messen bei 2 Candles (8h) und 4 Candles (16h)
            moves = []
            for fp in future:
                if fp > 0:
                    moves.append((fp - price) / price)

            if not moves:
                continue

            max_move = max(moves)  # Bester Moment
            min_move = min(moves)  # Schlechtester Moment
            move_at_2c = moves[1] if len(moves) > 1 else moves[0]

            if direction == "neutral":
                # Bot hat geschwiegen — war das richtig?
                if abs(max_move) > 0.015 or abs(min_move) > 0.015:
                    these_missed += 1  # Verpasste Chance
                continue

            these_total += 1
            if direction == "long":
                these_ok = max_move > 0.003  # >0.3% in die richtige Richtung
            else:
                these_ok = min_move < -0.003

            if these_ok:
                these_correct += 1

            # ── 2. EXECUTION-SCORE ──
            # Wie viel der verfügbaren Bewegung hat der Bot eingefangen?
            if sig.get("trade_pnl") is not None:
                trade_count += 1
                max_favorable = sig.get("max_favorable", 0)
                actual_pnl_pct = sig.get("trade_return_pct", 0)

                if max_favorable > 0.001:
                    # Ratio: was hat er gefangen vs. was war möglich
                    exec_ratio = actual_pnl_pct / (max_favorable * 100)
                    exec_ratio = max(-1.0, min(1.0, exec_ratio))
                    exec_scores.append(exec_ratio)
                elif sig["trade_pnl"] < 0:
                    exec_scores.append(-1.0)  # These falsch + Verlust

            # ── 3. EDGE-SCORE ──
            # Simuliere 100 zufällige Entries mit gleichen Parametern
            # War der Bot besser als Zufall?
            if sig.get("trade_pnl") is not None and len(moves) >= 2:
                edge_total += 1
                # Einfacher Test: Hat der Bot in >50% der zukünftigen
                # Candles die richtige Richtung erwischt?
                if direction == "long":
                    favorable_candles = sum(1 for m in moves if m > 0)
                else:
                    favorable_candles = sum(1 for m in moves if m < 0)

                if favorable_candles > len(moves) / 2:
                    edge_wins += 1

        # Scores berechnen
        these_score = these_correct / these_total if these_total > 0 else None
        exec_score = sum(exec_scores) / len(exec_scores) if exec_scores else None
        edge_score = edge_wins / edge_total if edge_total > 0 else None

        return {
            "these_score": round(these_score, 3) if these_score is not None else None,
            "exec_score": round(exec_score, 3) if exec_score is not None else None,
            "edge_score": round(edge_score, 3) if edge_score is not None else None,
            "these_correct": these_correct,
            "these_total": these_total,
            "these_missed": these_missed,
            "trade_count": trade_count,
            "edge_wins": edge_wins,
            "edge_total": edge_total,
        }


def build_signal_outcomes(df_4h_dict, signals_csv_path, trades_csv_path):
    """
    Verknüpft Signale mit tatsächlichen Marktbewegungen und Trade-Ergebnissen.

    df_4h_dict: {symbol: DataFrame} — 4H Candle-Daten pro Asset
    signals_csv_path: Pfad zur signals.csv
    trades_csv_path: Pfad zur trades.csv

    Returns: Liste von dicts für Evaluator.evaluate_signals()
    """
    import csv
    import pandas as pd

    # Signale laden
    signals = []
    # errors="replace": Alt-Daten aus der Windows-Ära enthalten cp1252-Bytes
    with open(signals_csv_path, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            signals.append(row)

    # Trades laden
    trades = []
    try:
        with open(trades_csv_path, encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                trades.append(row)
    except FileNotFoundError:
        pass

    # Trades Index: bot_id → [trades]
    trades_by_bot = defaultdict(list)
    for t in trades:
        trades_by_bot[t["bot_id"]].append(t)

    results = []
    for sig in signals:
        bot_id = sig["bot_id"]
        direction = sig["direction"]
        price = float(sig["price"])
        timestamp = sig["timestamp"]
        confidence = float(sig["confidence"])

        if price <= 0:
            continue

        # Symbol erkennen aus bot_id
        parts = bot_id.rsplit("_", 1)
        if len(parts) == 2:
            sym_short = parts[1]
        else:
            sym_short = "XRP"  # Fallback für alte Signale

        sym_map = {"XRP": "XRP/USDT", "BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT"}
        symbol = sym_map.get(sym_short)
        if not symbol or symbol not in df_4h_dict:
            continue

        df = df_4h_dict[symbol]

        # Finde den Candle-Index für dieses Signal
        try:
            sig_time = pd.Timestamp(timestamp)
            # Finde nächsten Candle nach dem Signal
            future_mask = df.index > sig_time
            future_df = df[future_mask]
            if len(future_df) < 2:
                continue
        except Exception:
            continue

        # Future Prices: 1, 2, 4, 8 Candles nach Signal
        future_prices = []
        for offset in [1, 2, 4, 8]:
            if len(future_df) > offset - 1:
                future_prices.append(float(future_df.iloc[offset - 1]["close"]))
            else:
                future_prices.append(0)

        # Max favorable/adverse move in den nächsten 8 candles
        check_candles = min(8, len(future_df))
        if check_candles > 0:
            future_highs = [float(future_df.iloc[i]["high"]) for i in range(check_candles)]
            future_lows = [float(future_df.iloc[i]["low"]) for i in range(check_candles)]
            max_high = max(future_highs)
            min_low = min(future_lows)

            if direction == "long":
                max_favorable = (max_high - price) / price
                max_adverse = (price - min_low) / price
            elif direction == "short":
                max_favorable = (price - min_low) / price
                max_adverse = (max_high - price) / price
            else:
                max_favorable = max((max_high - price) / price, (price - min_low) / price)
                max_adverse = 0
        else:
            max_favorable = 0
            max_adverse = 0

        # Trade-Ergebnis finden (wenn vorhanden)
        trade_pnl = None
        trade_return_pct = None
        bot_trades = trades_by_bot.get(bot_id, [])
        for t in bot_trades:
            # Trade innerhalb von 1h nach Signal = gehört zusammen
            try:
                t_time = pd.Timestamp(t["timestamp"])
                sig_t = pd.Timestamp(timestamp)
                diff_hours = abs((t_time - sig_t).total_seconds()) / 3600
                if diff_hours < 24 and t.get("direction") == direction:
                    trade_pnl = float(t["pnl"])
                    trade_return_pct = float(t["net_return_pct"])
                    break
            except Exception:
                continue

        results.append({
            "bot_id": bot_id,
            "direction": direction,
            "confidence": confidence,
            "price_at_signal": price,
            "future_prices": future_prices,
            "max_favorable": max_favorable,
            "max_adverse": max_adverse,
            "trade_pnl": trade_pnl,
            "trade_return_pct": trade_return_pct,
        })

    return results
