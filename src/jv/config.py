"""
Joint Venture Boting – Konfiguration
Alle Konstanten an einem Ort.
"""

import os

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")

# ── Pfade ─────────────────────────────────────────────
JV_DIR        = os.path.join(PROJECT_ROOT, "data", "bot", "jv")
SIGNALS_DIR   = os.path.join(JV_DIR, "signals")
LEDGER_FILE   = os.path.join(JV_DIR, "credit_ledger.json")
PRIME_STATE    = os.path.join(JV_DIR, "prime_state.json")
JV_TRADES     = os.path.join(JV_DIR, "jv_trades.csv")
JV_EQUITY     = os.path.join(JV_DIR, "jv_equity.csv")
JV_SIG_HIST   = os.path.join(JV_DIR, "jv_signals_history.csv")
JV_DASHBOARD  = os.path.join(JV_DIR, "dashboard_status.json")
MODEL_DIR     = os.path.join(PROJECT_ROOT, "models")

# ── Asset ─────────────────────────────────────────────
SYMBOL         = "XRP/USDT"
PRIMARY_TF     = "4h"
CHECK_INTERVAL = 60            # Sekunden zwischen Ticker-Checks
CANDLE_WAIT_MIN= 2             # Minuten nach 4H-Schluss warten

# ── Credit System ─────────────────────────────────────
INITIAL_CREDITS    = 100.0     # Startguthaben pro Bot
MAX_CREDITS        = 200.0     # Obergrenze
MIN_CREDITS        = 0.0       # Untergrenze (Bot kann sich erholen)
MIN_INFLUENCE      = 50.0      # Unter 50 Credits: kein Einfluss auf Prime

CREDIT_REWARD      = 5.0       # Richtig × confidence
CREDIT_PENALTY     = -8.0      # Falsch × confidence (Asymmetrie!)
MAGNITUDE_BONUS    = 1.5       # Extra-Bonus für grosse Moves (× min(move/atr, 2))
DECAY_FACTOR       = 0.99      # Pro 4H-Kerze (~12 Tage Halbwertszeit)

EVAL_MIN_MOVE      = 0.001     # 0.1% Mindestbewegung für "korrekt"

# ── Prime Bot ─────────────────────────────────────────
INITIAL_CAPITAL    = 1000.0
RISK_PER_TRADE     = 0.015     # 1.5% Risiko
MIN_CONSENSUS      = 0.05      # Weich: fast jedes Signal darf traden
MIN_AGREEMENT      = 0.0       # Kein Agreement nötig — einzelne Bots reichen
LEADER_MUST_AGREE  = False     # Leader muss nicht bestätigen

# Position Management
SL_ATR_MULT        = 2.0
TRAILING_ATR_MULT  = 1.5
TRAILING_ACTIVATE  = 0.8       # ATR-Vielfaches für Trailing-Aktivierung
PARTIAL_TP_RR      = 1.0       # Partial TP bei 1:1
PARTIAL_SIZE_PCT   = 0.50      # 50% schliessen
BREAKEVEN_BUFFER   = 0.001     # 0.1% über Entry
MAX_HOLD_CANDLES   = 18        # 72h
MAX_TRADES_PER_WEEK= 5
MAX_MONTHLY_DD     = 0.08      # 8%
CONSEC_LOSS_LIMIT  = 3
COOLDOWN_HOURS     = 24

# Fees
TAKER_FEE          = 0.0004
SLIPPAGE           = 0.0002
ROUND_TRIP         = (TAKER_FEE + SLIPPAGE) * 2

# ── Telegram ──────────────────────────────────────────
TG_BOT_TOKEN = "8503143803:AAH-7DPWX-bXq-ITRGpw4TwkDTDtIsRzQt8"
TG_CHAT_ID   = "7704168743"

# ── Daily Report ──────────────────────────────────────
DAILY_REPORT_HOUR  = 22        # UTC
