"""
MuriTrading – Phase 3b: Realistischer Backtest
Testet das Ensemble mit echten Kosten:
- Binance Futures Fees (Taker: 0.04% pro Seite)
- Slippage (0.02% konservativ)
- Position Sizing (1% Risiko pro Trade)
- Max Drawdown Tracking
Input:  data/processed/features_1h.csv + models/
Output: Backtest-Report
"""

import pandas as pd
import numpy as np
import os
import pickle
import json

DATA_PATH  = os.path.expanduser("~/MuriTrading/data/processed/features_1h.csv")
MODEL_DIR  = os.path.expanduser("~/MuriTrading/models")

# ── Kosten-Parameter (Binance Futures) ──────────────────────────
TAKER_FEE    = 0.0004   # 0.04% pro Seite
SLIPPAGE     = 0.0002   # 0.02% konservativ
ROUND_TRIP   = (TAKER_FEE + SLIPPAGE) * 2   # Einstieg + Ausstieg

# ── Risiko-Parameter ────────────────────────────────────────────
INITIAL_CAPITAL  = 1000.0   # Startkapital in USDT
RISK_PER_TRADE   = 0.01     # 1% des Kapitals pro Trade riskiert
CONFIDENCE_THRESH= 0.65     # Mindest-Confidence für Signal
MAX_TRADES_PER_DAY = 4      # Limitiert Overtrading
STOP_LOSS_MULT   = 1.5      # ATR-Multiplikator für Stop-Loss


def load_models_and_data():
    """Lädt Modelle und Feature-Daten."""
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"),  "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        meta = json.load(f)

    df = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)

    # Nur Test-Periode (Out-of-Sample)
    test_start = pd.Timestamp(meta["test_start"], tz="UTC")
    df_test = df[df.index >= test_start].copy()

    feature_cols = meta["feature_cols"]
    return rf_model, xgb_model, df_test, feature_cols


def get_ensemble_probs(rf_model, xgb_model, X):
    """Berechnet Ensemble-Wahrscheinlichkeiten."""
    rf_prob  = rf_model.predict_proba(X)[:, 1]
    xgb_prob = xgb_model.predict_proba(X)[:, 1]
    return (rf_prob + xgb_prob) / 2


def run_backtest(df_test, probs, feature_cols):
    """
    Simuliert Trading mit realistischen Kosten und Risikomanagement.
    Jeder Trade hält genau HORIZON=2 Kerzen (2 Stunden).
    """
    capital    = INITIAL_CAPITAL
    equity     = [capital]
    trades     = []
    peak       = capital
    max_dd     = 0.0

    probs_series = pd.Series(probs, index=df_test.index)

    # Tages-Trade-Counter
    daily_trades = {}

    for i, (ts, row) in enumerate(df_test.iterrows()):
        prob = probs_series.iloc[i]

        # Signal-Filterung
        is_long  = prob >= CONFIDENCE_THRESH
        is_short = prob <= (1 - CONFIDENCE_THRESH)

        if not (is_long or is_short):
            equity.append(capital)
            continue

        # Max Trades pro Tag
        day = ts.date()
        daily_trades[day] = daily_trades.get(day, 0)
        if daily_trades[day] >= MAX_TRADES_PER_DAY:
            equity.append(capital)
            continue
        daily_trades[day] += 1

        # ATR-basierter Stop-Loss
        atr_col = "1h_atr_rel"
        atr_rel = row.get(atr_col, 0.005) if atr_col in df_test.columns else 0.005
        stop_loss_pct = atr_rel * STOP_LOSS_MULT

        # Position Sizing: riskiere RISK_PER_TRADE % des Kapitals
        position_size = (capital * RISK_PER_TRADE) / max(stop_loss_pct, 0.001)
        position_size = min(position_size, capital * 0.20)  # Max 20% des Kapitals

        # Tatsächlicher Return (aus Label)
        raw_return = row.get("label_med", 0)

        # Richtung anwenden
        if is_short:
            raw_return = -raw_return

        # Kosten abziehen
        net_return = raw_return - ROUND_TRIP

        # PnL berechnen
        pnl = position_size * net_return

        # Kapital updaten
        capital += pnl
        capital  = max(capital, 0.01)  # Kein Negativkapital

        # Drawdown tracken
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak
        max_dd = max(max_dd, dd)

        equity.append(capital)

        trades.append({
            "timestamp":    ts,
            "direction":    "long" if is_long else "short",
            "confidence":   round(prob, 4),
            "raw_return":   round(raw_return * 100, 4),
            "net_return":   round(net_return * 100, 4),
            "pnl":          round(pnl, 4),
            "capital":      round(capital, 2),
            "stop_loss_pct":round(stop_loss_pct * 100, 4),
        })

    return pd.DataFrame(trades), pd.Series(equity), max_dd


def print_report(trades_df, equity, max_dd, initial_capital):
    """Druckt vollständigen Backtest-Report."""
    print("\n" + "=" * 55)
    print("REALISTISCHER BACKTEST-REPORT")
    print("=" * 55)

    if len(trades_df) == 0:
        print("Keine Trades ausgeführt.")
        return

    final_capital = equity.iloc[-1]
    total_return  = (final_capital - initial_capital) / initial_capital

    # Basis-Statistiken
    wins   = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0

    print(f"\n── Kapital ──────────────────────────────────────────")
    print(f"   Startkapital  : ${initial_capital:>10,.2f}")
    print(f"   Endkapital    : ${final_capital:>10,.2f}")
    print(f"   Total Return  : {total_return:>+10.1%}")
    print(f"   Max Drawdown  : {max_dd:>10.1%}")

    print(f"\n── Trades ───────────────────────────────────────────")
    print(f"   Anzahl Total  : {len(trades_df):>6,}")
    print(f"   Long          : {(trades_df['direction']=='long').sum():>6,}")
    print(f"   Short         : {(trades_df['direction']=='short').sum():>6,}")
    print(f"   Win Rate      : {win_rate:>10.1%}")

    print(f"\n── Returns ──────────────────────────────────────────")
    print(f"   Ø Gewinn/Trade: {wins['pnl'].mean():>+10.4f} USDT")
    print(f"   Ø Verlust/Trade: {losses['pnl'].mean():>+10.4f} USDT")
    print(f"   Ø Net Return  : {trades_df['net_return'].mean():>+10.4f}%")
    print(f"   Gesamte Fees  : ~${len(trades_df) * initial_capital * 0.0002 * ROUND_TRIP * 100:.2f} USDT (geschätzt)")

    if len(losses) > 0 and losses["pnl"].mean() != 0:
        pf = abs(wins["pnl"].sum() / losses["pnl"].sum()) if losses["pnl"].sum() != 0 else 0
        print(f"   Profit Factor : {pf:>10.2f}")

    # Sharpe Ratio (vereinfacht)
    daily_pnl = trades_df.set_index("timestamp")["pnl"].resample("1D").sum()
    if daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
        print(f"   Sharpe Ratio  : {sharpe:>10.2f}")

    # Kosten-Analyse
    print(f"\n── Kosten-Analyse ───────────────────────────────────")
    print(f"   Fees pro Trade: {ROUND_TRIP*100:.3f}% (Round-Trip)")
    print(f"   Trades Total  : {len(trades_df):,}")
    total_fees_pct = len(trades_df) * ROUND_TRIP * 100
    print(f"   Fees Gesamt   : ~{total_fees_pct:.1f}% des eingesetzten Kapitals")

    # Monatsweise Performance
    print(f"\n── Monatsweise Performance ──────────────────────────")
    monthly = trades_df.set_index("timestamp")["pnl"].resample("1ME").agg(
        total_pnl="sum", n_trades="count", win_rate=lambda x: (x > 0).mean()
    )
    print(f"   {'Monat':>10} | {'PnL (USDT)':>12} | {'Trades':>7} | {'Win%':>6}")
    print(f"   {'-'*10}-+-{'-'*12}-+-{'-'*7}-+-{'-'*6}")
    for month, row in monthly.iterrows():
        print(f"   {str(month.date()):>10} | {row['total_pnl']:>+12.2f} | {int(row['n_trades']):>7} | {row['win_rate']:>5.1%}")

    # Fazit
    print(f"\n── Fazit ────────────────────────────────────────────")
    if total_return > 0.5 and max_dd < 0.20 and win_rate > 0.55:
        print("   VIELVERSPRECHEND – System zeigt echte Stärke.")
        print("   Empfehlung: Live-Paper-Trading auf Binance Testnet starten.")
    elif total_return > 0.1 and max_dd < 0.30:
        print("   SOLIDE – System ist profitabel aber verbesserungswürdig.")
        print("   Empfehlung: Feature-Engineering verfeinern, dann Paper-Trading.")
    elif total_return > 0:
        print("   MARGINAL – System ist leicht profitabel.")
        print("   Empfehlung: Confidence-Schwelle erhöhen, Trades reduzieren.")
    else:
        print("   NICHT PROFITABEL nach realistischen Kosten.")
        print("   Empfehlung: Zurück zu Feature-Analyse, andere Signale suchen.")


def main():
    print("MuriTrading – Realistischer Backtest")
    print("=" * 45)
    print(f"\nKosten-Annahmen:")
    print(f"  Taker Fee    : {TAKER_FEE*100:.3f}% pro Seite")
    print(f"  Slippage     : {SLIPPAGE*100:.3f}% pro Seite")
    print(f"  Round-Trip   : {ROUND_TRIP*100:.3f}% total")
    print(f"  Risiko/Trade : {RISK_PER_TRADE*100:.1f}% des Kapitals")
    print(f"  Confidence   : > {CONFIDENCE_THRESH}")
    print(f"  Max Trades/Tag: {MAX_TRADES_PER_DAY}")

    print("\nLade Modelle und Daten...")
    rf_model, xgb_model, df_test, feature_cols = load_models_and_data()
    print(f"Test-Periode: {df_test.index[0].date()} → {df_test.index[-1].date()}")
    print(f"Kerzen: {len(df_test):,}")

    print("\nBerechne Ensemble-Signale...")
    X_test = df_test[feature_cols]
    probs  = get_ensemble_probs(rf_model, xgb_model, X_test)

    print("Simuliere Trading...")
    trades_df, equity, max_dd = run_backtest(df_test, probs, feature_cols)

    print_report(trades_df, equity, max_dd, INITIAL_CAPITAL)

    # Speichern
    out_path = os.path.expanduser("~/MuriTrading/data/processed/backtest_results.csv")
    trades_df.to_csv(out_path, index=False)
    print(f"\nTrade-Log gespeichert: {out_path}")


if __name__ == "__main__":
    main()
