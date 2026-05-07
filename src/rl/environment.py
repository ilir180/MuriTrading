"""
MuriTrading – RL Trading Environment v3
Trainiert auf 4H-Kerzen mit realistischen Bedingungen.

Kernänderungen vs v2:
  - 4H Steps statt 1H (grössere Preisbewegungen pro Step)
  - Höhere Fee-Strafe (realistisch: 0.12% Round-Trip)
  - Bonus für Halten profitabler Positionen
  - Harte Strafe für Overtrading (neue Positions-Eröffnungen kosten)
  - Walk-Forward-Ready: akzeptiert beliebige DataFrames

State = Markt-Features + Regime-Features + Position-Info
Reward = Realisierter PnL + Halte-Bonus + Regime-Anreize - Overtrade-Penalty
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class XRPTradingEnv(gym.Env):
    """
    XRP Trading Environment v3 – 4H Regime-Aware.

    Observations:
      - Normalisierte Markt-Features
      - Position State [-1, 1]
      - Unrealisierter PnL
      - Haltezeit (normalisiert)
      - Regime-Features (ADX, Choppiness, etc.)

    Actions:
      - Continuous [-1, 1]: Zielposition
        -1.0 = maximaler Short
         0.0 = flat
        +1.0 = maximaler Long

    Reward:
      - Realisierter PnL nach Fees
      - Halte-Bonus für profitable Positionen
      - Bonus für flat im Seitwärtsmarkt
      - Penalty für Overtrading
      - Penalty für Verluste im Seitwärtsmarkt
    """

    metadata = {"render_modes": ["human"]}

    # Regime-Features (4H statt 1H)
    REGIME_COLS = ["4h_adx", "4h_chop", "4h_vol_regime", "4h_bb_squeeze",
                   "4h_trend_consistency", "4h_regime_trend"]
    # Fallback auf 1h falls 4h nicht vorhanden
    REGIME_COLS_1H = ["1h_adx", "1h_chop", "1h_vol_regime", "1h_bb_squeeze",
                      "1h_trend_consistency", "1h_regime_trend"]

    def __init__(self, df, feature_cols, initial_capital=1000.0,
                 max_position_pct=0.20, fee_rate=0.0006,
                 max_hold_steps=18,   # 18 × 4h = 72h
                 min_position_change=0.30,  # Höherer Threshold (war 0.10)
                 overtrade_penalty=0.003):   # Strafe pro Trade
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.fee_rate = fee_rate
        self.max_hold_steps = max_hold_steps
        self.min_position_change = min_position_change
        self.overtrade_penalty = overtrade_penalty

        self.n_features = len(feature_cols)

        # Regime-Features: bevorzuge 4h, Fallback auf 1h
        self.regime_cols = [c for c in self.REGIME_COLS if c in df.columns]
        if not self.regime_cols:
            self.regime_cols = [c for c in self.REGIME_COLS_1H if c in df.columns]
        self.n_regime = len(self.regime_cols)

        # Observation: features + position + unrealized_pnl + hold_time + extra
        #            + regime features
        self.n_extra = 4  # position, unrealized_pnl, hold_time_norm, trades_today_norm
        self.n_obs = self.n_features + self.n_extra + self.n_regime

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_obs,), dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(1,), dtype=np.float32,
        )

        self._precompute_features()

    def _precompute_features(self):
        """Normalisiert Features vorab (z-score)."""
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

            # Unnormalisierte Regime-Werte für Reward
            adx_col = "4h_adx" if "4h_adx" in self.df.columns else "1h_adx"
            chop_col = "4h_chop" if "4h_chop" in self.df.columns else "1h_chop"
            regime_col = "4h_regime_trend" if "4h_regime_trend" in self.df.columns else "1h_regime_trend"

            self.adx_values = self.df[adx_col].values if adx_col in self.df.columns else np.full(len(self.df), 25.0)
            self.chop_values = self.df[chop_col].values if chop_col in self.df.columns else np.full(len(self.df), 0.5)
            self.regime_values = self.df[regime_col].values if regime_col in self.df.columns else np.ones(len(self.df))
        else:
            self.regime_norm = np.zeros((len(self.df), 0))
            self.adx_values = np.full(len(self.df), 25.0)
            self.chop_values = np.full(len(self.df), 0.5)
            self.regime_values = np.ones(len(self.df))

        self.prices = self.df["close"].values
        self.n_steps = len(self.df)

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
        self.steps_flat = 0
        self.steps_in_position = 0
        self.peak_equity = self.initial_capital
        self.trades_in_episode = 0  # Gesamtanzahl Trades

        # Tracking
        self.equity_curve = [self.capital]
        self.trade_log = []

        return self._get_obs(), {}

    def _get_obs(self):
        """Observation-Vektor mit Regime-Features."""
        features = self.features_norm[self.current_step]

        # Haltezeit normalisiert
        hold_norm = self.steps_in_position / max(self.max_hold_steps, 1)

        # Trades-Rate (wie viel getradet relativ zu Steps)
        trade_rate = self.trades_in_episode / max(self.current_step + 1, 1) * 10

        extra = np.array([
            self.position,
            self._unrealized_pnl_pct(),
            min(hold_norm, 2.0),
            min(trade_rate, 2.0),
        ], dtype=np.float32)

        parts = [features, extra]

        if self.n_regime > 0:
            regime = self.regime_norm[self.current_step]
            parts.append(regime.astype(np.float32))

        return np.concatenate(parts).astype(np.float32)

    def _unrealized_pnl_pct(self):
        """Unrealisierter PnL als Prozent des Kapitals."""
        if abs(self.position) < 0.01 or self.entry_price == 0:
            return 0.0
        current_price = self.prices[self.current_step]
        if self.position > 0:
            ret = (current_price - self.entry_price) / self.entry_price
        else:
            ret = (self.entry_price - current_price) / self.entry_price
        return ret * abs(self.position)

    def _unrealized_pnl(self):
        """Unrealisierter PnL in Dollar."""
        if abs(self.position) < 0.01 or self.entry_price == 0:
            return 0.0
        current_price = self.prices[self.current_step]
        if self.position > 0:
            return self.position_size * (current_price - self.entry_price) / self.entry_price
        else:
            return abs(self.position_size) * (self.entry_price - current_price) / self.entry_price

    def _get_regime_state(self):
        step = min(self.current_step, self.n_steps - 1)
        adx = float(self.adx_values[step])
        chop = float(self.chop_values[step])
        is_trending = int(self.regime_values[step])
        if np.isnan(adx): adx = 25.0
        if np.isnan(chop): chop = 0.5
        return adx, chop, is_trending

    def _close_position(self, reason="signal"):
        """Schliesst aktuelle Position und berechnet PnL."""
        current_price = self.prices[self.current_step]
        pnl = self._unrealized_pnl()
        fee = abs(self.position_size) * self.fee_rate
        net_pnl = pnl - fee

        self.capital += net_pnl
        self.total_pnl += net_pnl
        self.total_fees += fee

        if net_pnl > 0:
            self.wins += 1
        elif net_pnl < 0:
            self.losses += 1
        self.n_trades += 1

        adx, chop, is_trending = self._get_regime_state()
        self.trade_log.append({
            "step": self.current_step,
            "price": current_price,
            "direction": "long" if self.position > 0 else "short",
            "pnl": net_pnl,
            "capital": self.capital,
            "hold_steps": self.steps_in_position,
            "adx": adx,
            "is_trending": is_trending,
            "exit_reason": reason,
        })

        old_pos = self.position
        self.position = 0.0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.steps_in_position = 0

        return net_pnl, old_pos

    def step(self, action):
        target_position = float(np.clip(action[0], -1.0, 1.0))
        current_price = self.prices[self.current_step]
        reward = 0.0

        adx, chop, is_trending = self._get_regime_state()
        position_change = target_position - self.position

        # ── FORCED TIME EXIT ──────────────────────────────────
        if abs(self.position) > 0.05 and self.steps_in_position >= self.max_hold_steps:
            net_pnl, _ = self._close_position("time_exit")
            pnl_reward = net_pnl / self.initial_capital

            reward += pnl_reward
            # Härtere Strafe für time_exit: Agent soll lernen selbst rauszugehen
            reward -= 0.005
            if net_pnl < 0:
                reward -= 0.003

            # Position Change neu berechnen (jetzt flat)
            position_change = target_position - 0.0

        # ── HALTEZEIT TRACKEN ─────────────────────────────────
        if abs(self.position) > 0.05:
            self.steps_in_position += 1

        # ── POSITION CHANGES ─────────────────────────────────
        if abs(position_change) > self.min_position_change:
            # Alte Position schliessen
            if abs(self.position) > 0.01:
                net_pnl, old_pos = self._close_position("signal")
                pnl_reward = net_pnl / self.initial_capital

                if is_trending:
                    reward += pnl_reward * 1.3  # Trend: leicht verstärkt
                else:
                    if net_pnl < 0:
                        reward += pnl_reward * 2.0  # Seitwärts-Verlust: DOPPELT bestraft
                    else:
                        reward += pnl_reward * 0.8  # Seitwärts-Gewinn: etwas weniger

            # Neue Position eröffnen
            if abs(target_position) > 0.20:  # Mindestens 20% Überzeugung
                self.position = target_position
                self.position_size = abs(target_position) * self.capital * self.max_position_pct
                self.entry_price = current_price
                fee = self.position_size * self.fee_rate
                self.capital -= fee
                self.total_fees += fee
                self.trades_in_episode += 1

                # Overtrade Penalty: jeder Trade kostet
                reward -= self.overtrade_penalty

                # Extra Strafe in Seitwärtsmärkten
                if not is_trending and adx < 20:
                    reward -= 0.004 * abs(target_position)
                elif not is_trending:
                    reward -= 0.002 * abs(target_position)

                self.steps_flat = 0
                self.steps_in_position = 0
            else:
                self.position = 0.0
                self.position_size = 0.0
                self.entry_price = 0.0
                self.steps_flat = 0
                self.steps_in_position = 0

        else:
            # Agent hält Position oder bleibt flat
            if abs(self.position) < 0.05:
                # FLAT
                self.steps_flat += 1
                if not is_trending:
                    # Belohnung für Nichtstun im Seitwärtsmarkt
                    # Grösser als v2 (0.0005 statt 0.0003)
                    reward += 0.0005
            else:
                # POSITION HALTEN
                unrealized_pct = self._unrealized_pnl_pct()

                if unrealized_pct > 0:
                    # Halte-Bonus für profitable Positionen
                    # Je profitabler, desto mehr Bonus (ride the trend!)
                    if is_trending:
                        reward += unrealized_pct * 0.3  # Starker Bonus im Trend
                    else:
                        reward += unrealized_pct * 0.1  # Moderater Bonus seitwärts
                else:
                    # Leichte Strafe für Halten unprofitabler Positionen
                    reward += unrealized_pct * 0.05  # Negativ * positiv = negativ

        # ── EQUITY TRACKING ───────────────────────────────────
        unrealized = self._unrealized_pnl()
        current_equity = self.capital + unrealized
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # Nächster Step
        self.current_step += 1
        self.equity_curve.append(current_equity)

        terminated = self.current_step >= self.n_steps - 1
        truncated = self.capital <= self.initial_capital * 0.5

        if truncated:
            reward -= 2.0  # Härter als v2

        # Ruin-Warnung (unter 75% Kapital)
        if self.capital < self.initial_capital * 0.75:
            reward -= 0.001

        info = {
            "capital": self.capital,
            "equity": current_equity,
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
        """Annualisierter Sharpe (angepasst für 4H-Steps: 2190 Steps/Jahr)."""
        eq = np.array(self.equity_curve)
        if len(eq) < 2:
            return 0.0
        returns = np.diff(eq) / eq[:-1]
        if returns.std() == 0:
            return 0.0
        # 2190 ≈ 365 * 6 (6 vier-Stunden-Kerzen pro Tag)
        return (returns.mean() / returns.std()) * np.sqrt(2190)

    @property
    def max_drawdown(self):
        """Maximaler Drawdown der Equity-Kurve."""
        eq = np.array(self.equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / peak
        return float(np.max(dd)) if len(dd) > 0 else 0.0

    @property
    def profit_factor(self):
        """Profit Factor = Gross Profit / Gross Loss."""
        gross_profit = sum(t["pnl"] for t in self.trade_log if t["pnl"] > 0)
        gross_loss = abs(sum(t["pnl"] for t in self.trade_log if t["pnl"] < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
