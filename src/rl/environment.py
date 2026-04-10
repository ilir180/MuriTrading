"""
MuriTrading – RL Trading Environment v2
Custom Gymnasium Umgebung für XRP Trading mit Regime-Detection.

Der Agent entscheidet bei jedem Step:
  - Action [-1, 1]: -1 = Full Short, 0 = Flat, +1 = Full Long
  - Er lernt WANN, WIE VIEL und in welche RICHTUNG
  - NEU: Er wird belohnt wenn er in Seitwärtsphasen NICHTS tut

State = Markt-Features + Regime-Features + Position-Info + Portfolio-Info
Reward = Realisierter PnL + Regime-basierte Anreize
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class XRPTradingEnv(gym.Env):
    """
    XRP Trading Environment v2 mit Regime-Awareness.

    Observations:
      - Normalisierte Markt-Features (46+)
      - Aktuelle Position (-1 bis 1)
      - Unrealisierter PnL
      - Zeit-Features (Stunde, Wochentag)
      - Regime-Features (ADX, Choppiness, Volatility, Trend-Score)

    Actions:
      - Continuous [-1, 1]: Zielposition
        -1.0 = maximaler Short
         0.0 = keine Position (flat)
        +1.0 = maximaler Long

    Reward:
      - Realisierter PnL pro Step (nach Fees)
      - Bonus für flat bleiben in Seitwärtsmärkten
      - Penalty für Trading in choppy Märkten
      - Bonus für Trading in Trends
    """

    metadata = {"render_modes": ["human"]}

    # Regime-Feature-Spalten (werden automatisch gesucht)
    REGIME_COLS = ["1h_adx", "1h_chop", "1h_vol_regime", "1h_bb_squeeze",
                   "1h_trend_consistency", "1h_regime_trend"]

    def __init__(self, df, feature_cols, initial_capital=1000.0,
                 max_position_pct=0.20, fee_rate=0.0006):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.fee_rate = fee_rate

        self.n_features = len(feature_cols)

        # Regime-Features erkennen
        self.regime_cols = [c for c in self.REGIME_COLS if c in df.columns]
        self.n_regime = len(self.regime_cols)

        # Observation: features + position + unrealized_pnl + hour + weekday
        #            + regime features (ADX, chop, vol_regime, bb_squeeze, trend_consistency, regime_trend)
        self.n_obs = self.n_features + 4 + self.n_regime
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_obs,), dtype=np.float32,
        )

        # Action: target position [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,), dtype=np.float32,
        )

        # Pre-compute normalized features
        self._precompute_features()

    def _precompute_features(self):
        """Normalisiert Features einmal vorab (z-score)."""
        raw = self.df[self.feature_cols].values
        self.feat_mean = np.nanmean(raw, axis=0)
        self.feat_std = np.nanstd(raw, axis=0) + 1e-8
        self.features_norm = (raw - self.feat_mean) / self.feat_std
        self.features_norm = np.nan_to_num(self.features_norm, 0.0)

        # Regime-Features separat normalisieren
        if self.n_regime > 0:
            raw_regime = self.df[self.regime_cols].values
            self.regime_mean = np.nanmean(raw_regime, axis=0)
            self.regime_std = np.nanstd(raw_regime, axis=0) + 1e-8
            self.regime_norm = (raw_regime - self.regime_mean) / self.regime_std
            self.regime_norm = np.nan_to_num(self.regime_norm, 0.0)

            # Unnormalisierte Regime-Werte für Reward-Berechnung
            self.adx_values = self.df["1h_adx"].values if "1h_adx" in self.df.columns else np.full(len(self.df), 25.0)
            self.chop_values = self.df["1h_chop"].values if "1h_chop" in self.df.columns else np.full(len(self.df), 0.5)
            self.regime_values = self.df["1h_regime_trend"].values if "1h_regime_trend" in self.df.columns else np.ones(len(self.df))
        else:
            self.regime_norm = np.zeros((len(self.df), 0))
            self.adx_values = np.full(len(self.df), 25.0)
            self.chop_values = np.full(len(self.df), 0.5)
            self.regime_values = np.ones(len(self.df))

        # Preise
        self.prices = self.df["close"].values
        self.n_steps = len(self.df)

        # Zeit-Features
        if hasattr(self.df.index, 'hour'):
            self.hours = self.df.index.hour / 23.0
            self.weekdays = self.df.index.weekday / 6.0
        else:
            self.hours = np.zeros(self.n_steps)
            self.weekdays = np.zeros(self.n_steps)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.capital = self.initial_capital
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.n_trades = 0
        self.wins = 0
        self.losses = 0
        self.steps_flat = 0  # Zähler für flat-Steps

        # Tracking
        self.equity_curve = [self.capital]
        self.trade_log = []

        return self._get_obs(), {}

    def _get_obs(self):
        """Baut Observation-Vektor mit Regime-Features."""
        features = self.features_norm[self.current_step]

        extra = np.array([
            self.position,
            self._unrealized_pnl(),
            self.hours[self.current_step] if self.current_step < len(self.hours) else 0,
            self.weekdays[self.current_step] if self.current_step < len(self.weekdays) else 0,
        ], dtype=np.float32)

        parts = [features, extra]

        # Regime-Features hinzufügen
        if self.n_regime > 0:
            regime = self.regime_norm[self.current_step]
            parts.append(regime.astype(np.float32))

        return np.concatenate(parts).astype(np.float32)

    def _unrealized_pnl(self):
        """Berechnet unrealisierten PnL der aktuellen Position."""
        if abs(self.position) < 0.01 or self.entry_price == 0:
            return 0.0
        current_price = self.prices[self.current_step]
        if self.position > 0:
            return self.position_size * (current_price - self.entry_price) / self.entry_price
        else:
            return abs(self.position_size) * (self.entry_price - current_price) / self.entry_price

    def _get_regime_state(self):
        """Gibt aktuelle Regime-Informationen zurück."""
        step = min(self.current_step, self.n_steps - 1)
        adx = float(self.adx_values[step])
        chop = float(self.chop_values[step])
        is_trending = int(self.regime_values[step])
        return adx, chop, is_trending

    def step(self, action):
        target_position = float(np.clip(action[0], -1.0, 1.0))

        current_price = self.prices[self.current_step]
        reward = 0.0

        # Regime-State für Reward-Shaping
        adx, chop, is_trending = self._get_regime_state()

        # Position-Änderung berechnen
        position_change = target_position - self.position

        # Nur handeln wenn Änderung signifikant (> 10% Schritt)
        if abs(position_change) > 0.10:
            # Alte Position schliessen (falls vorhanden)
            if abs(self.position) > 0.01:
                pnl = self._unrealized_pnl()
                fee = abs(self.position_size) * self.fee_rate
                net_pnl = pnl - fee
                self.capital += net_pnl
                self.total_pnl += net_pnl
                self.total_fees += fee

                # PnL-Reward skaliert mit Regime
                pnl_reward = net_pnl / self.initial_capital
                if is_trending:
                    # Trend: normaler PnL-Reward (leicht verstärkt)
                    reward += pnl_reward * 1.2
                else:
                    # Seitwärts: Verluste werden härter bestraft
                    if net_pnl < 0:
                        reward += pnl_reward * 1.5  # 50% mehr Strafe
                    else:
                        reward += pnl_reward

                if net_pnl > 0:
                    self.wins += 1
                elif net_pnl < 0:
                    self.losses += 1
                self.n_trades += 1

                self.trade_log.append({
                    "step": self.current_step,
                    "price": current_price,
                    "direction": "long" if self.position > 0 else "short",
                    "pnl": net_pnl,
                    "capital": self.capital,
                    "adx": adx,
                    "is_trending": is_trending,
                })

            # Neue Position eröffnen
            if abs(target_position) > 0.05:
                self.position = target_position
                self.position_size = abs(target_position) * self.capital * self.max_position_pct
                self.entry_price = current_price
                fee = self.position_size * self.fee_rate
                self.capital -= fee
                self.total_fees += fee

                # Strafe für das Eröffnen in Seitwärtsmärkten
                if not is_trending and adx < 20:
                    # Starke Strafe: Seitwärtsmarkt + schwacher Trend
                    reward -= 0.002 * abs(target_position)
                elif not is_trending:
                    # Moderate Strafe: nicht klar trending
                    reward -= 0.001 * abs(target_position)

                self.steps_flat = 0
            else:
                self.position = 0.0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.steps_flat = 0
        else:
            # Agent hat sich entschieden NICHTS zu tun
            if abs(self.position) < 0.05:
                # Flat und bleibt flat
                self.steps_flat += 1
                if not is_trending:
                    # Belohnung für Nichtstun im Seitwärtsmarkt!
                    # Kleine aber konsistente Belohnung
                    reward += 0.0003
                # Keine Strafe für flat in Trends - der Agent soll
                # selbst lernen wann er einsteigen soll

        # Unrealisierter PnL Bonus (klein, regime-skaliert)
        unrealized = self._unrealized_pnl()
        if is_trending:
            # In Trends: stärkerer Bonus für offene Gewinner (ride the trend)
            reward += unrealized / self.initial_capital * 0.15
        else:
            # Seitwärts: kaum Bonus für offene Positionen
            reward += unrealized / self.initial_capital * 0.05

        # Nächster Step
        self.current_step += 1
        self.equity_curve.append(self.capital + unrealized)

        terminated = self.current_step >= self.n_steps - 1
        truncated = self.capital <= self.initial_capital * 0.5

        if truncated:
            reward -= 1.0

        info = {
            "capital": self.capital,
            "position": self.position,
            "total_pnl": self.total_pnl,
            "n_trades": self.n_trades,
            "total_fees": self.total_fees,
            "adx": adx,
            "is_trending": is_trending,
        }

        return self._get_obs(), reward, terminated, truncated, info

    @property
    def win_rate(self):
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0

    @property
    def sharpe_ratio(self):
        """Annualisierter Sharpe Ratio der Equity-Kurve."""
        eq = np.array(self.equity_curve)
        returns = np.diff(eq) / eq[:-1]
        if returns.std() == 0:
            return 0.0
        return (returns.mean() / returns.std()) * np.sqrt(8760)
