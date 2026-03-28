"""
MuriTrading – Phase 2b: Event-driven Feature Analysis
Beantwortet: Welche Features haben echte Vorhersagekraft?
Geht RÜCKWÄRTS: Erst erfolgreiche Signale finden, dann Muster suchen.
Input:  data/processed/features_1h.csv
Output: Analyse-Report im Terminal + data/processed/feature_importance.csv
"""

import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.expanduser("~/MuriTrading/data/processed/features_1h.csv")
OUTPUT_DIR = os.path.expanduser("~/MuriTrading/data/processed")

# Schwellwert: ab welchem Return gilt ein Signal als "erfolgreich"?
WIN_THRESHOLD = 0.003   # +0.3% in 2h = bullischer Win
LOSS_THRESHOLD = -0.003 # -0.3% in 2h = bärischer Win (Short)


def load_features():
    df = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)
    print(f"Geladen: {df.shape[0]:,} Zeilen, {df.shape[1]} Spalten")
    return df


def classify_outcomes(df):
    """Teilt alle Kerzen in 3 Outcome-Klassen ein."""
    df["outcome"] = "neutral"
    df.loc[df["label_med"] >= WIN_THRESHOLD,  "outcome"] = "bull_win"
    df.loc[df["label_med"] <= LOSS_THRESHOLD, "outcome"] = "bear_win"

    counts = df["outcome"].value_counts()
    total  = len(df)
    print(f"\n── Outcome-Verteilung ({'±'}{WIN_THRESHOLD*100:.1f}% Schwelle) ──")
    for outcome, count in counts.items():
        print(f"   {outcome:12s}: {count:6,}  ({count/total*100:.1f}%)")
    return df


def analyze_feature_power(df):
    """
    Für jedes numerische Feature:
    Wie stark unterscheiden sich Bull-Win vs Bear-Win vs Neutral?
    Berechnet: Trennkraft (Cohen's d) zwischen Bull-Win und Bear-Win.
    """
    feature_cols = [c for c in df.columns if c not in [
        "open", "high", "low", "close", "volume",
        "label_min", "label_med", "label_max", "label_dir", "outcome"
    ] and df[c].dtype in [np.float64, np.int64]]

    bull = df[df["outcome"] == "bull_win"]
    bear = df[df["outcome"] == "bear_win"]

    results = []
    for col in feature_cols:
        b_vals = bull[col].dropna()
        s_vals = bear[col].dropna()

        if len(b_vals) < 30 or len(s_vals) < 30:
            continue

        # Cohen's d: Effektgrösse der Trennung
        pooled_std = np.sqrt((b_vals.std()**2 + s_vals.std()**2) / 2)
        if pooled_std < 1e-10:
            continue
        cohens_d = abs(b_vals.mean() - s_vals.mean()) / pooled_std

        # Richtung: positiv = Feature höher bei Bull-Win
        direction = "bull_higher" if b_vals.mean() > s_vals.mean() else "bear_higher"

        results.append({
            "feature":     col,
            "cohens_d":    round(cohens_d, 4),
            "direction":   direction,
            "bull_mean":   round(b_vals.mean(), 4),
            "bear_mean":   round(s_vals.mean(), 4),
            "bull_std":    round(b_vals.std(), 4),
            "bear_std":    round(s_vals.std(), 4),
        })

    return pd.DataFrame(results).sort_values("cohens_d", ascending=False)


def analyze_confluence(df):
    """Analysiert: Wie gut sagt der Confluence-Score den Ausgang voraus?"""
    print("\n── Confluence Score vs Outcome ──────────────────────")
    print(f"   {'Score':>8} | {'Bull%':>7} | {'Bear%':>7} | {'Neutral%':>9} | {'n':>6}")
    print(f"   {'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*6}")

    for score in sorted(df["confluence_net"].unique()):
        sub = df[df["confluence_net"] == score]
        n = len(sub)
        if n < 20:
            continue
        bull_pct    = (sub["outcome"] == "bull_win").mean() * 100
        bear_pct    = (sub["outcome"] == "bear_win").mean() * 100
        neutral_pct = (sub["outcome"] == "neutral").mean() * 100
        print(f"   {score:>8.0f} | {bull_pct:>6.1f}% | {bear_pct:>6.1f}% | {neutral_pct:>8.1f}% | {n:>6,}")


def analyze_rsi_zones(df):
    """Analysiert RSI-Zonen vs Outcomes auf verschiedenen Timeframes."""
    print("\n── RSI-Zonen Analyse ────────────────────────────────")
    for tf in ["15m", "1h", "4h", "1d"]:
        col = f"{tf}_rsi_14"
        if col not in df.columns:
            continue

        zones = pd.cut(df[col], bins=[0, 30, 45, 55, 70, 100],
                       labels=["oversold", "low", "neutral", "high", "overbought"])
        df["_rsi_zone"] = zones

        print(f"\n   {tf} RSI:")
        print(f"   {'Zone':>12} | {'Bull%':>7} | {'Bear%':>7} | {'n':>6}")
        for zone in ["oversold", "low", "neutral", "high", "overbought"]:
            sub = df[df["_rsi_zone"] == zone]
            if len(sub) < 20:
                continue
            bull_pct = (sub["outcome"] == "bull_win").mean() * 100
            bear_pct = (sub["outcome"] == "bear_win").mean() * 100
            print(f"   {zone:>12} | {bull_pct:>6.1f}% | {bear_pct:>6.1f}% | {len(sub):>6,}")

    df.drop(columns=["_rsi_zone"], inplace=True, errors="ignore")


def print_top_features(importance_df, n=20):
    """Zeigt die Top-N Features nach Trennkraft."""
    print(f"\n── Top {n} Features nach Vorhersagekraft (Cohen's d) ──")
    print(f"   {'Feature':45s} | {'Cohen d':>8} | {'Richtung'}")
    print(f"   {'-'*45}-+-{'-'*8}-+-{'-'*12}")
    for _, row in importance_df.head(n).iterrows():
        strength = "stark" if row["cohens_d"] > 0.3 else "mittel" if row["cohens_d"] > 0.1 else "schwach"
        print(f"   {row['feature']:45s} | {row['cohens_d']:>8.4f} | {row['direction']} ({strength})")


def main():
    print("MuriTrading – Event-driven Feature Analysis")
    print("=" * 50)

    df = load_features()
    df = classify_outcomes(df)

    print("\n1. Feature-Trennkraft berechnen...")
    importance = analyze_feature_power(df)

    print_top_features(importance, n=25)

    # Speichern
    out_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    importance.to_csv(out_path, index=False)
    print(f"\n   Vollständige Liste gespeichert: {out_path}")

    # Confluence Analyse
    analyze_confluence(df)

    # RSI Zonen
    analyze_rsi_zones(df)

    # Zusammenfassung
    top10 = importance.head(10)
    strong = importance[importance["cohens_d"] > 0.3]
    medium = importance[(importance["cohens_d"] > 0.1) & (importance["cohens_d"] <= 0.3)]
    weak   = importance[importance["cohens_d"] <= 0.1]

    print(f"\n── Zusammenfassung ──────────────────────────────────")
    print(f"   Starke Features  (d > 0.3): {len(strong):3d}")
    print(f"   Mittlere Features(d > 0.1): {len(medium):3d}")
    print(f"   Schwache Features(d ≤ 0.1): {len(weak):3d}")
    print(f"\n   Top-3 Timeframe-Präfix in Top-20:")

    top20 = importance.head(20)
    for prefix in ["15m_", "1h_", "4h_", "1d_"]:
        count = top20["feature"].str.startswith(prefix).sum()
        print(f"   {prefix:5s}: {count} Features unter Top-20")

    print("\nAnalyse abgeschlossen.")
    print("→ Nächster Schritt: Nur starke + mittlere Features ins Modell")


if __name__ == "__main__":
    main()
