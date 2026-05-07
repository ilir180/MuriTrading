"""
JV Boting v2 – Konfiguration
"""

import os

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")

# ── Pfade ─────────────────────────────────────────────
JV2_DIR         = os.path.join(PROJECT_ROOT, "data", "bot", "jv2")
STATE_FILE      = os.path.join(JV2_DIR, "state.json")
TRADES_CSV      = os.path.join(JV2_DIR, "trades.csv")
EQUITY_CSV      = os.path.join(JV2_DIR, "equity.csv")
SIGNALS_CSV     = os.path.join(JV2_DIR, "signals.csv")
SPY_LOG_CSV     = os.path.join(JV2_DIR, "spy_log.csv")
SCOUT_REPORT    = os.path.join(JV2_DIR, "scout_report.json")
REPORTS_DIR     = os.path.join(JV2_DIR, "daily_reports")

# ── Assets ────────────────────────────────────────────
SYMBOLS = {
    "XRP/USDT":  {"binance_id": "XRPUSDT",  "short": "XRP",  "emoji": "\U0001F4A7"},
    "BTC/USDT":  {"binance_id": "BTCUSDT",   "short": "BTC",  "emoji": "\U0001FA99"},
    "ETH/USDT":  {"binance_id": "ETHUSDT",   "short": "ETH",  "emoji": "\U0001F4CE"},
    "SOL/USDT":  {"binance_id": "SOLUSDT",   "short": "SOL",  "emoji": "\u2600\uFE0F"},
}
PRIMARY_TF      = "4h"
CHECK_INTERVAL  = 60          # Sekunden
CANDLE_WAIT_MIN = 2           # Minuten nach 4H-Schluss warten

# ── Kapital ───────────────────────────────────────────
CAPITAL_PER_ASSET = 1000.0    # $1000 pro Asset (Paper)
NUM_BOTS        = 8
INITIAL_ALLOC   = CAPITAL_PER_ASSET / NUM_BOTS   # 125.0
MIN_ALLOC       = 50.0
MAX_ALLOC       = 250.0
REBALANCE_DAY   = 0           # Montag

# ── Risiko (Defaults, überschrieben durch BOT_RISK_PROFILES) ──
RISK_PER_TRADE  = 0.02
SL_ATR_MULT     = 2.0
TP_ATR_MULT     = 3.0
TRAILING_ATR_MULT = 1.5
TRAILING_ACTIVATE = 0.8
MAX_HOLD_CANDLES  = 18        # 72h
CONSEC_LOSS_LIMIT = 3
COOLDOWN_HOURS    = 24
MIN_CONFIDENCE    = 0.20

# ── Bot-Risikoprofile ─────────────────────────────────
# Jeder Bot hat sein eigenes Profil passend zur These.
# risk: % des Kapitals pro Trade
# leverage: Hebel (1x = kein Hebel, 5x = 5-fache Position)
# sl_atr: Stop-Loss in ATR-Vielfachen
# tp_atr: Take-Profit in ATR-Vielfachen
# max_hold: Max Candles halten
BOT_RISK_PROFILES = {
    "trend_rider": {        # Reitet Trends — moderater Hebel, lässt laufen
        "risk": 0.04, "leverage": 3, "sl_atr": 2.5, "tp_atr": 5.0, "max_hold": 24,
    },
    "mean_reverter": {      # Präzise Entries — hoher Hebel, enge Stops
        "risk": 0.03, "leverage": 5, "sl_atr": 1.5, "tp_atr": 2.0, "max_hold": 12,
    },
    "breakout_hunter": {    # Alles oder nichts — max Hebel, schnell raus
        "risk": 0.05, "leverage": 5, "sl_atr": 1.5, "tp_atr": 3.0, "max_hold": 8,
    },
    "contrarian": {         # Konservativ — kein Hebel, unsicherstes Signal
        "risk": 0.02, "leverage": 1, "sl_atr": 2.5, "tp_atr": 3.5, "max_hold": 18,
    },
    "flow_tracker": {       # Whale-Follow — moderater Hebel
        "risk": 0.04, "leverage": 3, "sl_atr": 1.5, "tp_atr": 2.5, "max_hold": 10,
    },
    "momentum_surfer": {    # Momentum — hoher Hebel, Trailing sichert ab
        "risk": 0.05, "leverage": 5, "sl_atr": 2.0, "tp_atr": 4.0, "max_hold": 16,
    },
    "level_bouncer": {      # Präzise Levels — hoher Hebel weil enger Stop
        "risk": 0.03, "leverage": 5, "sl_atr": 1.0, "tp_atr": 2.0, "max_hold": 12,
    },
    "volatility_fader": {   # Vola-Fade — moderater Hebel, schnell rein/raus
        "risk": 0.05, "leverage": 3, "sl_atr": 1.5, "tp_atr": 2.0, "max_hold": 6,
    },
}

# ── Signal-Overrides pro Bot×Asset ─────────────────────
# invert: Signal umdrehen (These ist konsistent falsch = invertiert gut)
# exec_override: Execution-Parameter überschreiben für Bots mit guter These aber schlechter Exec
BOT_OVERRIDES = {
    # === INVERTIEREN: These konsistent falsch ===
    "trend_rider_SOL":   {"invert": True},     # These 0% → invertiert 100%
    "trend_rider_BTC":   {"invert": True},     # These 0%
    "flow_tracker_SOL":  {"invert": True},     # These 33% → invertiert 67%
    "flow_tracker_ETH":  {"invert": True},     # These 50% → invertiert 50% (Test)

    # === EXECUTION FIX: These gut (>75%) aber Exec negativ ===
    # Problem: Stops zu eng → mehr Raum geben
    "contrarian_XRP":    {"exec_override": {"sl_atr": 3.5, "max_hold": 24}},
    "contrarian_BTC":    {"exec_override": {"sl_atr": 3.5, "max_hold": 24}},
    "contrarian_ETH":    {"exec_override": {"sl_atr": 3.5, "max_hold": 24}},
    "contrarian_SOL":    {"exec_override": {"sl_atr": 3.5, "max_hold": 24}},
    "level_bouncer_XRP": {"exec_override": {"sl_atr": 1.8, "max_hold": 18}},
    "level_bouncer_SOL": {"exec_override": {"sl_atr": 1.8, "max_hold": 18}},
    "level_bouncer_ETH": {"exec_override": {"sl_atr": 1.8, "max_hold": 18}},
    "level_bouncer_BTC": {"exec_override": {"sl_atr": 1.8, "max_hold": 18}},
    "mean_reverter_XRP": {"exec_override": {"sl_atr": 2.0, "max_hold": 16}},
}

# ── Fees ──────────────────────────────────────────────
TAKER_FEE       = 0.0004
SLIPPAGE        = 0.0002
ROUND_TRIP      = (TAKER_FEE + SLIPPAGE) * 2

# ── Telegram ──────────────────────────────────────────
TG_BOT_TOKEN    = "8503143803:AAH-7DPWX-bXq-ITRGpw4TwkDTDtIsRzQt8"
TG_CHAT_ID      = "7704168743"
DAILY_REPORT_HOUR = 22        # UTC

# ── Bot-Configs ───────────────────────────────────────
BOT_CONFIGS = {
    "trend_rider":      {"emoji": "\U0001F3C4", "label": "Trend Rider"},
    "mean_reverter":    {"emoji": "\U0001F504", "label": "Mean Reverter"},
    "breakout_hunter":  {"emoji": "\U0001F4A5", "label": "Breakout Hunter"},
    "contrarian":       {"emoji": "\U0001F914", "label": "Contrarian"},
    "flow_tracker":     {"emoji": "\U0001F40B", "label": "Flow Tracker"},
    "momentum_surfer":  {"emoji": "\U0001F680", "label": "Momentum Surfer"},
    "level_bouncer":    {"emoji": "\U0001F3AF", "label": "Level Bouncer"},
    "volatility_fader": {"emoji": "\U0001F32A", "label": "Volatility Fader"},
}
