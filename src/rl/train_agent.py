"""
MuriTrading – RL Agent Training v2 (Regime-Aware)
Trainiert einen PPO-Agent der lernt XRP profitabel zu traden.

Der Agent lernt:
  - WANN einsteigen (Timing)
  - WIE VIEL riskieren (Position Sizing)
  - WELCHE RICHTUNG (Long/Short)
  - WANN NICHTS TUN (das Wichtigste!)
  - MARKT-REGIME erkennen (Trending vs. Seitwärts)

Start: python src/rl/train_agent.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.rl.environment import XRPTradingEnv

# ── Pfade ──────────────────────────────────────────────────────
DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "processed", "features_1h.csv")
IMPORT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "feature_importance.csv")
MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")
RL_DIR      = os.path.join(MODEL_DIR, "rl")

# ── Konfiguration ─────────────────────────────────────────────
MIN_COHENS_D   = 0.1
TEST_RATIO     = 0.20
TOTAL_TIMESTEPS= 800_000       # Mehr Steps für Regime-Learning
LEARNING_RATE  = 0.0002        # Etwas niedriger für Stabilität
N_STEPS        = 2048
BATCH_SIZE     = 256
N_EPOCHS       = 10
GAMMA          = 0.995         # Höher: langfristigere Perspektive
GAE_LAMBDA     = 0.95
ENT_COEF       = 0.015         # Mehr Exploration für "do nothing"


# ═══════════════════════════════════════════════════════════════
#  CALLBACK
# ═══════════════════════════════════════════════════════════════

class TradingCallback(BaseCallback):
    """Loggt Training-Fortschritt mit Regime-Info."""

    def __init__(self, eval_env, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_reward = -np.inf
        self.best_pnl = -np.inf
        self.results = []

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            obs, _ = self.eval_env.reset()
            total_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                total_reward += reward
                done = terminated or truncated

            env = self.eval_env
            # Trades in Trend vs. Seitwärts analysieren
            trend_trades = [t for t in env.trade_log if t.get("is_trending", 1)]
            side_trades = [t for t in env.trade_log if not t.get("is_trending", 1)]
            trend_pnl = sum(t["pnl"] for t in trend_trades)
            side_pnl = sum(t["pnl"] for t in side_trades)

            result = {
                "step": self.n_calls,
                "reward": round(total_reward, 4),
                "capital": round(env.capital, 2),
                "pnl": round(env.total_pnl, 2),
                "trades": env.n_trades,
                "win_rate": round(env.win_rate, 3),
                "fees": round(env.total_fees, 2),
                "sharpe": round(env.sharpe_ratio, 2),
                "trend_trades": len(trend_trades),
                "side_trades": len(side_trades),
                "trend_pnl": round(trend_pnl, 2),
                "side_pnl": round(side_pnl, 2),
            }
            self.results.append(result)

            wr = f"{env.win_rate:.0%}" if env.n_trades > 0 else "–"
            pnl_sign = "+" if env.total_pnl >= 0 else ""
            print(f"  Step {self.n_calls:>7,}  │  "
                  f"PnL: {pnl_sign}${env.total_pnl:.2f}  │  "
                  f"Trades: {env.n_trades} (T:{len(trend_trades)}/S:{len(side_trades)})  │  "
                  f"WR: {wr}  │  "
                  f"Sharpe: {env.sharpe_ratio:.2f}  │  "
                  f"Fees: ${env.total_fees:.2f}", flush=True)

            # Bestes Modell speichern (nach PnL)
            if env.total_pnl > self.best_pnl and env.n_trades > 10:
                self.best_pnl = env.total_pnl
                self.model.save(os.path.join(RL_DIR, "ppo_xrp_best"))
                print(f"         → Neues bestes Modell gespeichert (PnL: ${env.total_pnl:.2f})", flush=True)

            if total_reward > self.best_reward:
                self.best_reward = total_reward

        return True


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(RL_DIR, exist_ok=True)

    print("═" * 60)
    print("  MuriTrading – RL Agent Training v2 (Regime-Aware)")
    print("═" * 60)

    # Daten laden
    print("\n1. Daten laden...")
    df = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)
    importance = pd.read_csv(IMPORT_PATH)

    # Feature-Selektion (gleich wie ML-Modell)
    strong_features = importance[importance["cohens_d"] >= MIN_COHENS_D]["feature"].tolist()
    feature_cols = [f for f in strong_features if f in df.columns]
    print(f"   Features: {len(feature_cols)}")

    # Regime-Features vorhanden?
    regime_cols = [c for c in XRPTradingEnv.REGIME_COLS if c in df.columns]
    if regime_cols:
        print(f"   Regime-Features: {regime_cols}")
    else:
        print("   WARNUNG: Keine Regime-Features gefunden!")
        print("   Bitte erst 'python src/features/build_features.py' ausführen.")
        return

    # Regime-Statistik
    if "1h_regime_trend" in df.columns:
        trend_pct = df["1h_regime_trend"].mean() * 100
        print(f"   Markt-Regime: {trend_pct:.1f}% Trend / {100-trend_pct:.1f}% Seitwärts")
    if "1h_adx" in df.columns:
        print(f"   ADX Durchschnitt: {df['1h_adx'].mean():.1f}")

    # Train/Test Split (zeitbasiert)
    split_idx = int(len(df) * (1 - TEST_RATIO))
    df_train = df.iloc[:split_idx].copy()
    df_test  = df.iloc[split_idx:].copy()
    print(f"   Train: {len(df_train):,} Kerzen ({df_train.index[0].date()} → {df_train.index[-1].date()})")
    print(f"   Test : {len(df_test):,}  Kerzen ({df_test.index[0].date()} → {df_test.index[-1].date()})")

    # Environments erstellen
    print("\n2. Environments erstellen...")
    train_env = XRPTradingEnv(df_train, feature_cols)
    test_env  = XRPTradingEnv(df_test, feature_cols)

    print(f"   Observation Space: {train_env.observation_space.shape}")
    print(f"   Action Space: {train_env.action_space.shape}")
    print(f"   Regime-Features: {train_env.n_regime}")

    # PPO Agent
    print(f"\n3. PPO Training ({TOTAL_TIMESTEPS:,} Steps)...")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Gamma: {GAMMA}")
    print(f"   Entropy Coef: {ENT_COEF}")
    print()
    print(f"  {'Step':>10}  │  {'PnL':>12}  │  {'Trades':>18}  │  "
          f"{'WR':>5}  │  {'Sharpe':>7}  │  {'Fees':>8}")
    print(f"  {'─'*10}──┼──{'─'*12}──┼──{'─'*18}──┼──"
          f"{'─'*5}──┼──{'─'*7}──┼──{'─'*8}")

    callback = TradingCallback(test_env, eval_freq=10000)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=0.2,
        ent_coef=ENT_COEF,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        seed=42,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128]),  # Grösseres Netzwerk
        ),
    )

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # Finale Evaluation
    print("\n4. Finale Evaluation auf Test-Daten...")
    obs, _ = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

    ret = (test_env.capital - test_env.initial_capital) / test_env.initial_capital
    wr = f"{test_env.win_rate:.1%}" if test_env.n_trades > 0 else "–"

    # Regime-basierte Analyse
    trend_trades = [t for t in test_env.trade_log if t.get("is_trending", 1)]
    side_trades = [t for t in test_env.trade_log if not t.get("is_trending", 1)]
    trend_pnl = sum(t["pnl"] for t in trend_trades)
    side_pnl = sum(t["pnl"] for t in side_trades)
    trend_wins = sum(1 for t in trend_trades if t["pnl"] > 0)
    side_wins = sum(1 for t in side_trades if t["pnl"] > 0)

    print(f"\n{'═' * 60}")
    print(f"  ERGEBNIS")
    print(f"{'═' * 60}")
    print(f"  Startkapital : ${test_env.initial_capital:,.2f}")
    print(f"  Endkapital   : ${test_env.capital:,.2f}")
    print(f"  Total Return : {ret:+.1%}")
    print(f"  PnL          : ${test_env.total_pnl:+.2f}")
    print(f"  Trades       : {test_env.n_trades}")
    print(f"  Win Rate     : {wr}")
    print(f"  Sharpe Ratio : {test_env.sharpe_ratio:.2f}")
    print(f"  Fees         : ${test_env.total_fees:.2f}")
    print(f"{'─' * 60}")
    print(f"  REGIME-ANALYSE:")
    print(f"  Trend-Trades : {len(trend_trades):>5}  "
          f"PnL: ${trend_pnl:+.2f}  "
          f"WR: {trend_wins/max(len(trend_trades),1):.0%}")
    print(f"  Seite-Trades : {len(side_trades):>5}  "
          f"PnL: ${side_pnl:+.2f}  "
          f"WR: {side_wins/max(len(side_trades),1):.0%}")
    print(f"{'═' * 60}")

    # Modell speichern (letztes)
    print("\n5. Modell speichern...")
    model_path = os.path.join(RL_DIR, "ppo_xrp")
    model.save(model_path)

    # Feature-Stats speichern (für Live-Normalisierung)
    stats = {
        "mean": train_env.feat_mean.tolist(),
        "std": train_env.feat_std.tolist(),
    }
    # Regime-Stats hinzufügen
    if train_env.n_regime > 0:
        stats["regime_mean"] = train_env.regime_mean.tolist()
        stats["regime_std"] = train_env.regime_std.tolist()
        stats["regime_cols"] = train_env.regime_cols

    with open(os.path.join(RL_DIR, "feature_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    meta = {
        "trained_at": datetime.now().isoformat(),
        "version": "v2_regime_aware",
        "algorithm": "PPO",
        "feature_cols": feature_cols,
        "regime_cols": regime_cols,
        "n_features": len(feature_cols),
        "n_regime": len(regime_cols),
        "n_obs": train_env.n_obs,
        "n_train": len(df_train),
        "n_test": len(df_test),
        "total_timesteps": TOTAL_TIMESTEPS,
        "hyperparams": {
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "ent_coef": ENT_COEF,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "net_arch": "pi=[128,128], vf=[128,128]",
        },
        "test_return": round(ret, 4),
        "test_pnl": round(test_env.total_pnl, 2),
        "test_trades": test_env.n_trades,
        "test_win_rate": round(test_env.win_rate, 4),
        "test_sharpe": round(test_env.sharpe_ratio, 2),
        "test_trend_trades": len(trend_trades),
        "test_side_trades": len(side_trades),
        "test_trend_pnl": round(trend_pnl, 2),
        "test_side_pnl": round(side_pnl, 2),
        "training_progress": callback.results,
    }
    with open(os.path.join(RL_DIR, "rl_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"   Gespeichert: {model_path}")

    # Bestes Modell als Default?
    best_path = os.path.join(RL_DIR, "ppo_xrp_best.zip")
    if os.path.exists(best_path) and callback.best_pnl > test_env.total_pnl:
        print(f"   Bestes Modell (PnL ${callback.best_pnl:.2f}) > Letztes (${test_env.total_pnl:.2f})")
        print(f"   → Verwende bestes Modell")
        import shutil
        shutil.copy(best_path, model_path + ".zip")

    print(f"\nRL Agent v2 Training abgeschlossen.")
    print(f"→ Nächster Schritt: Bot neustarten mit neuem Modell")


if __name__ == "__main__":
    main()
