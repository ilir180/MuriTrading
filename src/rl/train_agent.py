"""
MuriTrading – RL Agent Training v3 (Walk-Forward)
Trainiert auf 4H-Daten mit Walk-Forward Validation.

Kernänderungen vs v2:
  - Walk-Forward: Train auf 60 Tage (360 4H-Kerzen), Test auf 14 Tage (84 Kerzen)
  - Mehrere Fenster durchlaufen, nur deployen wenn Mehrheit profitabel
  - Quality Gate: Mindest-Win-Rate und positive PnL auf OOS
  - Reduzierte Timesteps pro Fenster (200k statt 800k = weniger Overfitting)
  - 4H Regime-Features statt 1H

Start: python src/rl/train_agent.py
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import shutil
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

# ── Walk-Forward Konfiguration ────────────────────────────────
MIN_COHENS_D       = 0.1
TRAIN_DAYS         = 60        # 60 Tage Training pro Fenster
TEST_DAYS          = 14        # 14 Tage OOS Test
CANDLES_PER_DAY_4H = 6         # 6 Kerzen × 4H = 24H
TRAIN_CANDLES      = TRAIN_DAYS * CANDLES_PER_DAY_4H   # 360
TEST_CANDLES       = TEST_DAYS * CANDLES_PER_DAY_4H     # 84
STEP_CANDLES       = TEST_CANDLES                        # Fenster um Test-Periode verschieben

# Training
TIMESTEPS_PER_WINDOW = 200_000  # Weniger = weniger Overfitting
LEARNING_RATE  = 0.0003
N_STEPS        = 1024           # Kleiner als v2 (passend für weniger Daten)
BATCH_SIZE     = 128
N_EPOCHS       = 8
GAMMA          = 0.99           # Etwas niedriger: weniger Zukunfts-Bias
GAE_LAMBDA     = 0.95
ENT_COEF       = 0.02           # Mehr Exploration

# Quality Gate
MIN_WIN_RATE       = 0.45       # Mindestens 45% Win Rate auf OOS
MIN_PROFIT_FACTOR  = 1.0        # Profit Factor >= 1 (breakeven)
MIN_POSITIVE_WINDOWS = 0.5      # Mindestens 50% der Fenster profitabel
MAX_OOS_DD         = 0.15       # Max 15% Drawdown auf OOS


# ═══════════════════════════════════════════════════════════════
#  CALLBACK
# ═══════════════════════════════════════════════════════════════

class WalkForwardCallback(BaseCallback):
    """Minimaler Callback: loggt Fortschritt, speichert bestes Modell."""

    def __init__(self, eval_env, eval_freq=25000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.best_pnl = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            result = evaluate_agent(self.model, self.eval_env)

            pnl_sign = "+" if result["pnl"] >= 0 else ""
            wr = f"{result['win_rate']:.0%}" if result["n_trades"] > 0 else "–"
            print(f"    Step {self.n_calls:>7,}  │  "
                  f"PnL: {pnl_sign}${result['pnl']:.2f}  │  "
                  f"Trades: {result['n_trades']}  │  WR: {wr}  │  "
                  f"DD: {result['max_dd']:.1%}  │  "
                  f"PF: {result['profit_factor']:.2f}")

            if result["pnl"] > self.best_pnl and result["n_trades"] >= 3:
                self.best_pnl = result["pnl"]
                self.model.save(os.path.join(RL_DIR, "ppo_xrp_wf_best"))

        return True


# ═══════════════════════════════════════════════════════════════
#  HELPER
# ═══════════════════════════════════════════════════════════════

def evaluate_agent(model, env, deterministic=True):
    """Evaluiert Agent auf einer Umgebung, gibt Metriken zurück."""
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    trend_trades = [t for t in env.trade_log if t.get("is_trending", 1)]
    side_trades = [t for t in env.trade_log if not t.get("is_trending", 1)]

    return {
        "capital": float(env.capital),
        "pnl": round(float(env.total_pnl), 2),
        "return_pct": round(float((env.capital - env.initial_capital) / env.initial_capital * 100), 2),
        "n_trades": int(env.n_trades),
        "win_rate": float(env.win_rate),
        "wins": int(env.wins),
        "losses": int(env.losses),
        "total_fees": round(float(env.total_fees), 2),
        "sharpe": round(float(env.sharpe_ratio), 2),
        "max_dd": float(env.max_drawdown),
        "profit_factor": round(float(env.profit_factor), 3),
        "trend_trades": len(trend_trades),
        "side_trades": len(side_trades),
        "trend_pnl": round(float(sum(t["pnl"] for t in trend_trades)), 2),
        "side_pnl": round(float(sum(t["pnl"] for t in side_trades)), 2),
        "avg_hold": round(float(np.mean([t.get("hold_steps", 0) for t in env.trade_log])), 1) if env.trade_log else 0,
        "time_exits": int(sum(1 for t in env.trade_log if t.get("exit_reason") == "time_exit")),
    }


def resample_to_4h(df_1h):
    """Resampled 1H-Daten auf 4H für das Environment."""
    # Nur OHLCV Spalten resampen
    df_4h = df_1h.resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    # Feature-Spalten: nimm den letzten Wert pro 4H-Fenster
    feature_cols = [c for c in df_1h.columns if c not in ["open", "high", "low", "close", "volume"]]
    for col in feature_cols:
        df_4h[col] = df_1h[col].resample("4h").last()

    return df_4h.dropna()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(RL_DIR, exist_ok=True)

    print("=" * 60)
    print("  MuriTrading – RL Agent Training v3 (Walk-Forward)")
    print("=" * 60)

    # ── 1. Daten laden ────────────────────────────────────────
    print("\n1. Daten laden und auf 4H resampen...")
    df_1h = pd.read_csv(DATA_PATH, index_col="datetime", parse_dates=True)
    importance = pd.read_csv(IMPORT_PATH)

    # Feature-Selektion
    strong_features = importance[importance["cohens_d"] >= MIN_COHENS_D]["feature"].tolist()
    feature_cols = [f for f in strong_features if f in df_1h.columns]
    print(f"   Features: {len(feature_cols)}")

    # Auf 4H resampen
    df_4h = resample_to_4h(df_1h)
    print(f"   1H Kerzen: {len(df_1h):,}")
    print(f"   4H Kerzen: {len(df_4h):,}")
    print(f"   Zeitraum: {df_4h.index[0].date()} → {df_4h.index[-1].date()}")

    # Regime-Features
    regime_cols_4h = [c for c in XRPTradingEnv.REGIME_COLS if c in df_4h.columns]
    regime_cols_1h = [c for c in XRPTradingEnv.REGIME_COLS_1H if c in df_4h.columns]
    regime_cols = regime_cols_4h if regime_cols_4h else regime_cols_1h
    if regime_cols:
        print(f"   Regime-Features: {regime_cols}")
    else:
        print("   WARNUNG: Keine Regime-Features!")

    # ── 2. Walk-Forward Windows ───────────────────────────────
    print(f"\n2. Walk-Forward Validation...")
    print(f"   Train: {TRAIN_DAYS} Tage ({TRAIN_CANDLES} Kerzen)")
    print(f"   Test:  {TEST_DAYS} Tage ({TEST_CANDLES} Kerzen)")
    print(f"   Step:  {TEST_DAYS} Tage")

    total_candles = len(df_4h)
    min_needed = TRAIN_CANDLES + TEST_CANDLES

    if total_candles < min_needed:
        print(f"   FEHLER: Zu wenig Daten ({total_candles} < {min_needed})")
        return

    # Windows berechnen
    windows = []
    start = 0
    while start + TRAIN_CANDLES + TEST_CANDLES <= total_candles:
        train_end = start + TRAIN_CANDLES
        test_end = train_end + TEST_CANDLES
        windows.append((start, train_end, test_end))
        start += STEP_CANDLES

    # Nehme maximal die letzten 6 Fenster (für Geschwindigkeit)
    if len(windows) > 6:
        windows = windows[-6:]

    print(f"   Windows: {len(windows)}")
    for i, (s, te, end) in enumerate(windows):
        print(f"     W{i+1}: Train {df_4h.index[s].date()}→{df_4h.index[te-1].date()} | "
              f"Test {df_4h.index[te].date()}→{df_4h.index[min(end-1, total_candles-1)].date()}")

    # ── 3. Walk-Forward Training ──────────────────────────────
    print(f"\n3. Training ({len(windows)} Fenster × {TIMESTEPS_PER_WINDOW:,} Steps)...")

    all_results = []
    best_overall_pnl = -np.inf
    best_model_path = None

    for i, (train_start, train_end, test_end) in enumerate(windows):
        print(f"\n  ╔══ Window {i+1}/{len(windows)} ══════════════════════════════════╗")

        df_train = df_4h.iloc[train_start:train_end].copy()
        df_test = df_4h.iloc[train_end:min(test_end, total_candles)].copy()

        # Environments
        train_env = XRPTradingEnv(
            df_train, feature_cols,
            max_hold_steps=18,           # 72h
            min_position_change=0.30,
            overtrade_penalty=0.003,
        )
        test_env = XRPTradingEnv(
            df_test, feature_cols,
            max_hold_steps=18,
            min_position_change=0.30,
            overtrade_penalty=0.003,
        )

        print(f"  ║ Train: {len(df_train)} candles  Test: {len(df_test)} candles")
        print(f"  ║ Obs Space: {train_env.observation_space.shape}  Regime: {train_env.n_regime}")

        # Callback
        callback = WalkForwardCallback(test_env, eval_freq=25000)

        # PPO Agent (frisch für jedes Fenster = keine Überanpassung)
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
            seed=42 + i,
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),  # Kleiner = weniger Overfitting
            ),
        )

        # Training
        model.learn(total_timesteps=TIMESTEPS_PER_WINDOW, callback=callback)

        # Finale OOS Evaluation
        result = evaluate_agent(model, test_env)
        all_results.append(result)

        wr = f"{result['win_rate']:.0%}" if result["n_trades"] > 0 else "–"
        pnl_sign = "+" if result["pnl"] >= 0 else ""
        print(f"  ║")
        print(f"  ║ OOS Result: PnL {pnl_sign}${result['pnl']:.2f}  "
              f"WR: {wr}  Trades: {result['n_trades']}  "
              f"DD: {result['max_dd']:.1%}  PF: {result['profit_factor']:.2f}")
        print(f"  ║ Regime: Trend={result['trend_trades']}(${result['trend_pnl']:+.2f}) "
              f"Side={result['side_trades']}(${result['side_pnl']:+.2f})")
        print(f"  ║ Avg Hold: {result['avg_hold']:.0f} steps ({result['avg_hold']*4:.0f}h)  "
              f"Time Exits: {result['time_exits']}")

        status = "PASS" if result["pnl"] > 0 else "FAIL"
        status_color = "\033[92m" if result["pnl"] > 0 else "\033[91m"
        print(f"  ╚══ {status_color}{status}\033[0m ════════════════════════════════════╝")

        # Bestes Modell tracken
        if result["pnl"] > best_overall_pnl:
            best_overall_pnl = result["pnl"]
            temp_path = os.path.join(RL_DIR, f"ppo_xrp_wf_{i}")
            model.save(temp_path)
            best_model_path = temp_path

    # ── 4. Quality Gate ───────────────────────────────────────
    print(f"\n4. Quality Gate...")
    print(f"   {'='*50}")

    n_windows = len(all_results)
    n_profitable = sum(1 for r in all_results if r["pnl"] > 0)
    avg_pnl = np.mean([r["pnl"] for r in all_results])
    avg_wr = np.mean([r["win_rate"] for r in all_results])
    avg_dd = np.mean([r["max_dd"] for r in all_results])
    avg_pf = np.mean([r["profit_factor"] for r in all_results])
    avg_trades = np.mean([r["n_trades"] for r in all_results])

    print(f"   Profitable Windows: {n_profitable}/{n_windows} ({n_profitable/n_windows:.0%})")
    print(f"   Avg PnL:           ${avg_pnl:+.2f}")
    print(f"   Avg Win Rate:      {avg_wr:.0%}")
    print(f"   Avg Max DD:        {avg_dd:.1%}")
    print(f"   Avg Profit Factor: {avg_pf:.2f}")
    print(f"   Avg Trades/Window: {avg_trades:.0f}")

    # Quality Checks
    checks = []
    checks.append(("Profitable Windows >= 50%",
                    n_profitable / n_windows >= MIN_POSITIVE_WINDOWS))
    checks.append(("Avg Win Rate >= 45%",
                    avg_wr >= MIN_WIN_RATE))
    checks.append(("Avg Profit Factor >= 1.0",
                    avg_pf >= MIN_PROFIT_FACTOR))
    checks.append(("Avg Max DD <= 15%",
                    avg_dd <= MAX_OOS_DD))

    all_pass = True
    for check_name, passed in checks:
        icon = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"   [{icon}] {check_name}")
        if not passed:
            all_pass = False

    # ── 5. Deploy Entscheidung ────────────────────────────────
    print(f"\n5. Deploy-Entscheidung...")

    deploy = all_pass and best_model_path is not None

    if deploy:
        # Letztes Fenster auch nochmal trainieren (mit allen Daten bis zum Schluss)
        print("   DEPLOY: Quality Gate bestanden!")
        print(f"   Trainiere finales Modell auf letztem Fenster...")

        # Finales Training: letzte TRAIN_CANDLES Kerzen
        final_start = max(0, total_candles - TRAIN_CANDLES - TEST_CANDLES)
        final_split = total_candles - TEST_CANDLES
        df_final_train = df_4h.iloc[final_start:final_split].copy()
        df_final_test = df_4h.iloc[final_split:].copy()

        final_train_env = XRPTradingEnv(
            df_final_train, feature_cols,
            max_hold_steps=18,
            min_position_change=0.30,
            overtrade_penalty=0.003,
        )
        final_test_env = XRPTradingEnv(
            df_final_test, feature_cols,
            max_hold_steps=18,
            min_position_change=0.30,
            overtrade_penalty=0.003,
        )

        final_callback = WalkForwardCallback(final_test_env, eval_freq=25000)
        final_model = PPO(
            "MlpPolicy",
            final_train_env,
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
            seed=99,
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            ),
        )
        final_model.learn(total_timesteps=TIMESTEPS_PER_WINDOW, callback=final_callback)

        # Finale Evaluation
        final_result = evaluate_agent(final_model, final_test_env)
        wr = f"{final_result['win_rate']:.0%}" if final_result["n_trades"] > 0 else "–"
        print(f"\n   Final OOS: PnL ${final_result['pnl']:+.2f}  WR: {wr}  "
              f"Trades: {final_result['n_trades']}  DD: {final_result['max_dd']:.1%}")

        # Modell speichern
        model_path = os.path.join(RL_DIR, "ppo_xrp")
        final_model.save(model_path)
        print(f"   Gespeichert: {model_path}")

        # Auch als best speichern
        shutil.copy(model_path + ".zip", os.path.join(RL_DIR, "ppo_xrp_best.zip"))

        # Feature-Stats für Live-Normalisierung
        stats = {
            "mean": final_train_env.feat_mean.tolist(),
            "std": final_train_env.feat_std.tolist(),
        }
        if final_train_env.n_regime > 0:
            stats["regime_mean"] = final_train_env.regime_mean.tolist()
            stats["regime_std"] = final_train_env.regime_std.tolist()
            stats["regime_cols"] = final_train_env.regime_cols

        with open(os.path.join(RL_DIR, "feature_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

    else:
        print("   KEIN DEPLOY: Quality Gate nicht bestanden!")
        print("   Behalte altes Modell.")
        final_result = {"pnl": 0, "n_trades": 0, "win_rate": 0, "max_dd": 0}

    # ── 6. Metadata speichern ─────────────────────────────────
    meta = {
        "trained_at": datetime.now().isoformat(),
        "version": "v3_walk_forward",
        "algorithm": "PPO",
        "feature_cols": feature_cols,
        "regime_cols": regime_cols,
        "n_features": len(feature_cols),
        "n_regime": len(regime_cols),
        "walk_forward": {
            "n_windows": n_windows,
            "n_profitable": n_profitable,
            "train_days": TRAIN_DAYS,
            "test_days": TEST_DAYS,
            "timesteps_per_window": TIMESTEPS_PER_WINDOW,
            "results": all_results,
        },
        "quality_gate": {
            "passed": all_pass,
            "deployed": deploy,
            "avg_pnl": round(avg_pnl, 2),
            "avg_win_rate": round(avg_wr, 4),
            "avg_max_dd": round(avg_dd, 4),
            "avg_profit_factor": round(avg_pf, 3),
            "checks": {name: bool(passed) for name, passed in checks},
        },
        "final_result": final_result if deploy else None,
        "hyperparams": {
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "ent_coef": ENT_COEF,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "net_arch": "pi=[64,64], vf=[64,64]",
        },
    }
    with open(os.path.join(RL_DIR, "rl_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Cleanup: temporäre Modelle löschen ────────────────────
    for i in range(len(windows)):
        temp = os.path.join(RL_DIR, f"ppo_xrp_wf_{i}.zip")
        if os.path.exists(temp):
            os.remove(temp)
    wf_best = os.path.join(RL_DIR, "ppo_xrp_wf_best.zip")
    if os.path.exists(wf_best):
        os.remove(wf_best)

    # ── Zusammenfassung ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RL Agent v3 Training abgeschlossen")
    print(f"  Walk-Forward: {n_profitable}/{n_windows} Windows profitabel")
    print(f"  Quality Gate: {'BESTANDEN' if all_pass else 'NICHT BESTANDEN'}")
    print(f"  Deploy: {'JA' if deploy else 'NEIN (altes Modell behalten)'}")
    print(f"{'=' * 60}")

    return deploy


if __name__ == "__main__":
    main()
