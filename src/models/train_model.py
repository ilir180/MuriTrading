"""
MuriTrading – Phase 3: Modell-Training
Ensemble aus Random Forest + XGBoost
Trainiert auf den starken/mittleren Features (Cohen's d > 0.1)
Input:  data/processed/features_1h.csv + feature_importance.csv
Output: models/ (gespeicherte Modelle + Evaluation)
"""

import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime

DATA_PATH       = os.path.expanduser("~/MuriTrading/data/processed/features_1h.csv")
IMPORTANCE_PATH = os.path.expanduser("~/MuriTrading/data/processed/feature_importance.csv")
MODEL_DIR       = os.path.expanduser("~/MuriTrading/models")

# Nur Features mit Cohen's d > 0.1 verwenden
MIN_COHENS_D = 0.1

# Train/Test Split: letzte 20% als Out-of-Sample Test
TEST_RATIO = 0.20


def load_data():
    df = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)
    importance = pd.read_csv(IMPORTANCE_PATH)
    return df, importance


def select_features(df, importance):
    """Wählt nur Features mit ausreichender Vorhersagekraft."""
    strong_features = importance[importance["cohens_d"] >= MIN_COHENS_D]["feature"].tolist()
    available = [f for f in strong_features if f in df.columns]
    print(f"   Features gewählt: {len(available)} von {len(importance)} total")
    return available


def prepare_xy(df, feature_cols):
    """Bereitet Feature-Matrix X und Label-Vektor y vor."""
    X = df[feature_cols].copy()
    y_dir = df["label_dir"].copy()          # Richtung: 1=bull, 0=bear
    y_med = df["label_med"].copy()          # Median-Return
    y_min = df["label_min"].copy()          # Min-Return (10. Pz.)
    y_max = df["label_max"].copy()          # Max-Return (90. Pz.)
    return X, y_dir, y_med, y_min, y_max


def time_split(X, y_dir, y_med, y_min, y_max, test_ratio):
    """Zeitbasierter Split – keine zufällige Mischung (verhindert Data Leakage)."""
    split_idx = int(len(X) * (1 - test_ratio))
    return (
        X.iloc[:split_idx],   X.iloc[split_idx:],
        y_dir.iloc[:split_idx], y_dir.iloc[split_idx:],
        y_med.iloc[:split_idx], y_med.iloc[split_idx:],
        y_min.iloc[:split_idx], y_min.iloc[split_idx:],
        y_max.iloc[:split_idx], y_max.iloc[split_idx:],
    )


def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=30,    # Verhindert Overfitting
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    try:
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=20,  # Verhindert Overfitting
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)
        return model
    except ImportError:
        print("   XGBoost nicht installiert – wird übersprungen")
        print("   → pip install xgboost")
        return None


def evaluate_classifier(model, X_test, y_test, name):
    """Berechnet Klassifikations-Metriken."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n   {name}:")
    print(f"   Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   Precision: {prec:.4f}  (von allen Bull-Signalen: wie viele korrekt?)")
    print(f"   Recall   : {rec:.4f}  (wie viele echten Bulls erkannt?)")
    print(f"   F1-Score : {f1:.4f}")

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "probs": y_prob}


def ensemble_vote(results, X_test, y_test):
    """Kombiniert Modell-Wahrscheinlichkeiten per Durchschnitt."""
    all_probs = np.array([r["probs"] for r in results if r is not None])
    ensemble_probs = all_probs.mean(axis=0)
    ensemble_pred  = (ensemble_probs >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score
    acc  = accuracy_score(y_test, ensemble_pred)
    prec = precision_score(y_test, ensemble_pred, zero_division=0)

    print(f"\n   Ensemble (Durchschnitt):")
    print(f"   Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"   Precision: {prec:.4f}")

    return ensemble_probs, ensemble_pred


def confidence_analysis(ensemble_probs, ensemble_pred, y_test):
    """
    Analysiert: Wie gut ist das Modell wenn es sehr sicher ist?
    Hohe Confidence = Wahrscheinlichkeit > 0.65 oder < 0.35
    """
    from sklearn.metrics import accuracy_score

    probs = pd.Series(ensemble_probs, index=y_test.index)
    pred  = pd.Series(ensemble_pred,  index=y_test.index)

    print(f"\n── Confidence-Filter Analyse ────────────────────────")
    print(f"   {'Schwelle':>10} | {'Accuracy':>9} | {'Abdeckung':>10} | {'n':>6}")
    print(f"   {'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}")

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    for thresh in thresholds:
        high_conf = (probs >= thresh) | (probs <= (1 - thresh))
        n = high_conf.sum()
        if n < 10:
            break
        acc = accuracy_score(y_test[high_conf], pred[high_conf])
        coverage = n / len(y_test)
        print(f"   {thresh:>10.2f} | {acc:>8.4f}% | {coverage:>9.1%} | {n:>6,}")


def trading_simulation(ensemble_probs, y_med, y_test):
    """
    Einfache Backtest-Simulation:
    Kaufe wenn Confidence > 0.60, halte für 2h, messe Return.
    Kein Leverage, kein Short in dieser ersten Version.
    """
    probs   = pd.Series(ensemble_probs, index=y_test.index)
    returns = y_med.loc[y_test.index]

    HIGH_CONF = 0.60

    long_signals  = probs >= HIGH_CONF
    short_signals = probs <= (1 - HIGH_CONF)

    long_returns  = returns[long_signals]
    short_returns = (-returns[short_signals])   # Short = invertierter Return

    all_trades = pd.concat([long_returns, short_returns])

    print(f"\n── Einfache Trading-Simulation (Confidence > {HIGH_CONF}) ──")
    print(f"   Long Trades : {long_signals.sum():,}")
    print(f"   Short Trades: {short_signals.sum():,}")
    print(f"   Total Trades: {len(all_trades):,}")

    if len(all_trades) > 0:
        win_rate  = (all_trades > 0).mean()
        avg_win   = all_trades[all_trades > 0].mean() if (all_trades > 0).any() else 0
        avg_loss  = all_trades[all_trades < 0].mean() if (all_trades < 0).any() else 0
        total_ret = all_trades.sum()

        print(f"   Win Rate    : {win_rate:.1%}")
        print(f"   Ø Gewinn    : {avg_win*100:+.4f}%")
        print(f"   Ø Verlust   : {avg_loss*100:+.4f}%")
        print(f"   Total Return: {total_ret*100:+.2f}% (uncompounded, ohne Fees)")

        if avg_loss != 0:
            profit_factor = abs(avg_win / avg_loss)
            print(f"   Profit Factor: {profit_factor:.2f}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("MuriTrading – Phase 3: Modell-Training")
    print("=" * 45)

    # Pakete prüfen
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("ERROR: scikit-learn nicht installiert")
        print("→ pip install scikit-learn")
        return

    # Laden
    print("\n1. Daten laden...")
    df, importance = load_data()
    feature_cols = select_features(df, importance)

    # Vorbereiten
    print("\n2. Train/Test Split (zeitbasiert)...")
    X, y_dir, y_med, y_min, y_max = prepare_xy(df, feature_cols)
    (X_train, X_test,
     y_dir_train, y_dir_test,
     y_med_train, y_med_test,
     y_min_train, y_min_test,
     y_max_train, y_max_test) = time_split(X, y_dir, y_med, y_min, y_max, TEST_RATIO)

    print(f"   Train: {len(X_train):,} Kerzen ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"   Test : {len(X_test):,}  Kerzen ({X_test.index[0].date()} → {X_test.index[-1].date()})")

    # Training
    print("\n3. Modelle trainieren...")
    print("   Random Forest...", end="", flush=True)
    rf_model = train_random_forest(X_train, y_dir_train)
    print(" fertig")

    print("   XGBoost...", end="", flush=True)
    xgb_model = train_xgboost(X_train, y_dir_train)
    print(" fertig" if xgb_model else " übersprungen")

    # Evaluation
    print("\n4. Evaluation auf Test-Daten (Out-of-Sample)...")
    results = []
    rf_result = evaluate_classifier(rf_model, X_test, y_dir_test, "Random Forest")
    results.append(rf_result)

    if xgb_model:
        xgb_result = evaluate_classifier(xgb_model, X_test, y_dir_test, "XGBoost")
        results.append(xgb_result)

    # Ensemble
    print("\n5. Ensemble...")
    ensemble_probs, ensemble_pred = ensemble_vote(results, X_test, y_dir_test)

    # Confidence Analyse
    confidence_analysis(ensemble_probs, ensemble_pred, y_dir_test)

    # Trading Simulation
    trading_simulation(ensemble_probs, y_med_test, y_dir_test)

    # Modelle speichern
    print("\n6. Modelle speichern...")
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    if xgb_model:
        with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)

    meta = {
        "trained_at":    datetime.now().isoformat(),
        "feature_cols":  feature_cols,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
        "train_end":     str(X_train.index[-1].date()),
        "test_start":    str(X_test.index[0].date()),
    }
    with open(os.path.join(MODEL_DIR, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   Gespeichert in: {MODEL_DIR}")
    print("\nPhase 3 abgeschlossen.")
    print("→ Nächster Schritt: Phase 4 – Web-App Dashboard")


if __name__ == "__main__":
    main()
