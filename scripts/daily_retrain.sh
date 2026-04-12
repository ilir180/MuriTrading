#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  MuriTrading – Daily Retrain Pipeline
#  Läuft täglich um 04:00 Schweizer Zeit (02:00 UTC)
#
#  1. Frische Daten von Binance holen
#  2. Features berechnen (mit Regime-Detection)
#  3. Feature-Analyse (Cohen's d)
#  4. ML-Modelle trainieren (RF + XGBoost)
#  5. RL Agent trainieren (PPO mit Regime-Awareness)
#  6. Bot neustarten mit neuem Modell
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

PROJECT_ROOT="$HOME/MuriTrading"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/retrain_$(date +%Y%m%d_%H%M%S).log"
LOCK_FILE="$PROJECT_ROOT/.retrain.lock"

# Python Setup
export PATH="$HOME/.pyenv/versions/3.11.9/bin:$HOME/.pyenv/shims:$PATH"
eval "$(pyenv init -)" 2>/dev/null || true
pyenv shell 3.11.9 2>/dev/null || true

# Threading Fix (PyTorch + sklearn)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Telegram Notification
TG_BOT_TOKEN="8503143803:AAH-7DPWX-bXq-ITRGpw4TwkDTDtIsRzQt8"
TG_CHAT_ID="7704168743"

tg_send() {
    curl -s -X POST "https://api.telegram.org/bot${TG_BOT_TOKEN}/sendMessage" \
        -d chat_id="${TG_CHAT_ID}" \
        -d parse_mode="HTML" \
        -d text="$1" > /dev/null 2>&1 || true
}

# Logging
mkdir -p "$LOG_DIR"
exec >> "$LOG_FILE" 2>&1

echo "═══════════════════════════════════════════════════════"
echo "  MuriTrading – Daily Retrain $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════"

# Lock (verhindert parallele Runs)
if [ -f "$LOCK_FILE" ]; then
    echo "ABBRUCH: Retrain läuft bereits (Lock-Datei existiert)"
    exit 1
fi
trap 'rm -f "$LOCK_FILE"' EXIT
touch "$LOCK_FILE"

START_TIME=$(date +%s)
tg_send "🔄 <b>Daily Retrain gestartet</b> – $(date '+%H:%M %Z')"

# ── 1. Daten holen ────────────────────────────────────────────
echo ""
echo "── STEP 1: Frische Daten von Binance ──────────────────"
cd "$PROJECT_ROOT"

if python src/data/fetch_data.py; then
    echo "✓ Daten aktualisiert"
else
    tg_send "❌ <b>Retrain fehlgeschlagen</b> – Daten-Download Fehler"
    exit 1
fi

# ── 2. Features berechnen ─────────────────────────────────────
echo ""
echo "── STEP 2: Features berechnen (mit Regime-Detection) ──"

if python src/features/build_features.py; then
    echo "✓ Features berechnet"
else
    tg_send "❌ <b>Retrain fehlgeschlagen</b> – Feature-Engineering Fehler"
    exit 1
fi

# ── 3. Feature-Analyse ────────────────────────────────────────
echo ""
echo "── STEP 3: Feature-Analyse (Cohen's d) ────────────────"

if python src/features/analyze_features.py; then
    echo "✓ Feature-Analyse abgeschlossen"
else
    tg_send "❌ <b>Retrain fehlgeschlagen</b> – Feature-Analyse Fehler"
    exit 1
fi

# ── 4. ML-Modelle trainieren ──────────────────────────────────
echo ""
echo "── STEP 4: ML-Modelle trainieren (RF + XGBoost) ───────"

if python src/models/train_model.py; then
    echo "✓ ML-Modelle trainiert"
else
    tg_send "❌ <b>Retrain fehlgeschlagen</b> – ML-Training Fehler"
    exit 1
fi

# ── 5. RL Agent trainieren ────────────────────────────────────
echo ""
echo "── STEP 5: RL Agent trainieren (PPO Regime-Aware) ─────"

if python src/rl/train_agent.py; then
    echo "✓ RL Agent trainiert"
else
    tg_send "❌ <b>Retrain fehlgeschlagen</b> – RL-Training Fehler"
    exit 1
fi

# ── 6. Bot neustarten ─────────────────────────────────────────
echo ""
echo "── STEP 6: Bot neustarten ─────────────────────────────"

# Bot stoppen und warten bis er wirklich tot ist
BOT_PID=$(pgrep -f "paper_trader.py" 2>/dev/null || true)
if [ -n "$BOT_PID" ]; then
    kill -TERM "$BOT_PID" 2>/dev/null || true
    # Warten bis Prozess wirklich beendet ist (max 15s)
    for i in $(seq 1 15); do
        if ! pgrep -f "paper_trader.py" > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done
    # Falls immer noch da, force kill
    if pgrep -f "paper_trader.py" > /dev/null 2>&1; then
        kill -9 $(pgrep -f "paper_trader.py") 2>/dev/null || true
        sleep 2
    fi
    echo "  Alter Bot gestoppt (PID: $BOT_PID)"
fi

# Bot neu starten via launchd (unload/load garantiert sauberen Neustart)
PLIST="$HOME/Library/LaunchAgents/com.muritrading.papertrader.plist"
launchctl unload "$PLIST" 2>/dev/null || true
sleep 2
launchctl load "$PLIST" 2>/dev/null || true
sleep 5

NEW_PID=$(pgrep -f "paper_trader.py" 2>/dev/null || true)
if [ -n "$NEW_PID" ]; then
    echo "✓ Bot gestartet (PID: $NEW_PID)"
else
    echo "⚠ Bot konnte nicht gestartet werden!"
    tg_send "⚠ <b>Bot konnte nach Retrain nicht gestartet werden!</b>"
fi

# ── Zusammenfassung ───────────────────────────────────────────
END_TIME=$(date +%s)
DURATION=$(( (END_TIME - START_TIME) / 60 ))

# RL Ergebnisse lesen
RL_META="$PROJECT_ROOT/models/rl/rl_meta.json"
if [ -f "$RL_META" ]; then
    RL_PNL=$(python -c "import json; d=json.load(open('$RL_META')); print(f\"{d['test_pnl']:+.2f}\")")
    RL_WR=$(python -c "import json; d=json.load(open('$RL_META')); print(f\"{d['test_win_rate']:.0%}\")")
    RL_TRADES=$(python -c "import json; d=json.load(open('$RL_META')); print(d['test_trades'])")
    RL_TREND=$(python -c "import json; d=json.load(open('$RL_META')); print(f\"T:{d.get('test_trend_trades',0)}/S:{d.get('test_side_trades',0)}\")")
else
    RL_PNL="?"
    RL_WR="?"
    RL_TRADES="?"
    RL_TREND="?"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  FERTIG in ${DURATION} Minuten"
echo "  RL: PnL \$${RL_PNL}  WR: ${RL_WR}  Trades: ${RL_TRADES} (${RL_TREND})"
echo "═══════════════════════════════════════════════════════"

tg_send "$(cat <<EOF
✅ <b>Daily Retrain abgeschlossen</b>

⏱ Dauer: <code>${DURATION} min</code>
📅 Daten bis: <code>$(date '+%Y-%m-%d')</code>

🤖 <b>RL Agent (Test):</b>
  PnL: <code>\$${RL_PNL}</code>
  Win Rate: <code>${RL_WR}</code>
  Trades: <code>${RL_TRADES}</code> (${RL_TREND})

Bot wurde automatisch neugestartet.
EOF
)"

# Alte Logs aufräumen (behalte letzte 14 Tage)
find "$LOG_DIR" -name "retrain_*.log" -mtime +14 -delete 2>/dev/null || true

echo "Daily Retrain Pipeline abgeschlossen."
