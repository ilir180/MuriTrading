# MuriTrading – Claude Instructions

## Working with Ilir — Standing Orders
- **Full autonomy granted.** Don't ask for permission before acting. Don't ask "soll ich X?", "möchtest du Y?", "ja oder nein?". Don't soften with "falls du noch Saft hast", "wenn du willst", "wir können auch X". These all read as permission asks even without a question mark.
- **Don't pre-narrate.** No "ich werde jetzt X tun", "Plan für heute: A, B, C", "starte ich jetzt manuell um zu prüfen". Just run the work and report results. Future-tense plan announcements are forbidden.
- **The bar is revolutionary, not incremental.** Default to the structurally ambitious move when offered a choice; build it in stages, but build toward the big thing. The endgame is Counterfactual Replay + Market Map + Coach + automatic promotion/demotion. Every decision should serve that spine.
- **Confirm only on truly destructive ops** (force-push to main, dropping data, deleting unrelated work). Everything else: ship it.
- **Evening sessions, kids at home.** End at clean checkpoints. Don't dump 5 hours of unreviewed work into one session.

## Projekt-Überblick
XRP/USDT Paper-Trading-System mit JV Boting v2 Architektur.

## Plattform
- **OS**: Windows (Migration von macOS, Mai 2026)
- **Python**: 3.11.x via pyenv-win oder conda
- **Scheduling**: Windows Task Scheduler (ersetzt macOS launchd)
- **Scripts**: PowerShell (.ps1) — keine .sh Shell-Scripts mehr
- **Pfade**: Windows-Pfade (`C:\Users\...`), im Python-Code `pathlib.Path` verwenden
- **Sleep deaktiviert**: `standby-timeout-ac/dc = 0`, `hibernate-timeout-ac/dc = 0`. PC läuft 24/7. Wenn `powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE` etwas anderes als `0x0` zeigt, neu setzen.

## Aktive Systeme

### JV Boting v2 (`src/jv2/`)
- **8 unabhängige Trading-Bots**, jeder mit eigener These, eigenem Kapital, eigenen Trades
- **3 Agents**: Spy (Info-Broker), Scout (verpasste Chancen), Analyst (täglicher HTML-Report)
- **Kein Konsens, kein Voting** — jeder Bot tradet selbstständig
- Kapital: $1000 total, $125 pro Bot, wöchentliches Performance-Rebalancing
- Timeframe: 4H Candles, Exit-Check alle 60s
- Scheduling: Windows Task Scheduler — Task `MuriTrading-JV2` (Logon-Trigger)
- Logs: `data/bot/jv2/jv2_output.log`
- **Heartbeat**: `runner.py` schreibt `data/bot/jv2/heartbeat.txt` jede Iteration

### Watchdog (`scripts/watchdog_jv2.ps1`)
- Task `MuriTrading-JV2-Watchdog` läuft alle 5 Min und garantiert Self-Healing
- Restartet JV2 wenn: Task nicht Running ODER kein `runner.py`-Prozess ODER `jv2_output.log` > 15 Min still ODER `heartbeat.txt` > 3 Min alt
- 3-Min Cold-Start Grace nach Bot-Start (für Imports + Binance-Init)
- Log: `data/bot/jv2/watchdog.log` (Heartbeat 1×/Stunde, sonst nur bei Restart)
- **Wenn Bot tot/inaktiv erscheint**: zuerst `watchdog.log` + Task-State checken, dann `LastTaskResult` (Hex). `0xC000013A` = Strg+C/Shutdown.

### Die 8 Bots
| Bot | Datei | These |
|-----|-------|-------|
| TrendRider | `bots/trend_rider.py` | Trend läuft weiter (ADX, EMA) |
| MeanReverter | `bots/mean_reverter.py` | Alles kehrt zur Mitte (RSI, BB) |
| BreakoutHunter | `bots/breakout_hunter.py` | Kompression → Explosion (BB Squeeze) |
| Contrarian | `bots/contrarian.py` | Masse liegt falsch (Fear&Greed) |
| FlowTracker | `bots/flow_tracker.py` | Folge Smart Money (Whale-Orderflow) |
| MomentumSurfer | `bots/momentum_surfer.py` | Stärke wird stärker (Velocity, Accel) |
| LevelBouncer | `bots/level_bouncer.py` | S/R hält (Pivot-Levels) |
| VolatilityFader | `bots/volatility_fader.py` | Vola normalisiert sich |

### Die 3 Agents
- **Spy** (`agents/spy.py`): Verteilt selektive Intel zwischen Bots (±0.05 Confidence max)
- **Scout** (`agents/scout.py`): Findet verpasste Moves, kategorisiert Marktphasen, erkennt Lücken
- **Analyst** (`agents/analyst.py`): Täglicher HTML-Report via Telegram um 22:00 UTC

## Veraltete/Deaktivierte Systeme
- `src/jv/` — Altes JV v1 (Konsortium-Architektur, deaktiviert)
- `src/bot/paper_trader.py` — Alter Paper Trader v3 (deaktiviert)

## Wiederverwendete Infrastruktur
- `src/features/build_features.py` → `add_indicators(df, prefix)` — 40+ technische Indikatoren
- `src/features/whale_features.py` → `compute_whale_features()` — Orderbook-Analyse
- `src/features/sentiment.py` → `compute_sentiment_features()` — Fear&Greed, etc.
- `src/features/cross_asset.py` → `build_cross_asset_features(exchange)` — BTC/ETH Korrelation

## Wichtige Dateien
- `src/jv2/config.py` — Alle Konstanten (inkl. `HEARTBEAT_FILE`)
- `src/jv2/models.py` — Datentypen (JV2Signal, BotPosition, BotState, TradeRecord)
- `src/jv2/base_bot.py` — Abstract Base Bot mit Position-Management
- `src/jv2/runner.py` — Hauptloop, schreibt `heartbeat.txt` pro Iteration
- `scripts/run_jv2.ps1` — Runner-Launcher (vom Task `MuriTrading-JV2` aufgerufen)
- `scripts/watchdog_jv2.ps1` — Self-Healing-Watchdog (vom Task `MuriTrading-JV2-Watchdog` aufgerufen)
- `data/bot/jv2/state.json` — Alle Bot-States
- `data/bot/jv2/trades.csv` — Trade-History
- `data/bot/jv2/equity.csv` — Equity-Kurven (wird pro 4H-Kerze geschrieben, NICHT pro Tick)
- `data/bot/jv2/heartbeat.txt` — Watchdog-Heartbeat (pro Tick, ~60s)
- `data/bot/jv2/watchdog.log` — Watchdog-Aktivität

## Technische Hinweise
- Python 3.11.x via pyenv-win oder conda
- `OMP_NUM_THREADS=1` als System-Umgebungsvariable setzen (PPO + sklearn Konflikt)
- Telegram: Token/Chat-ID in `src/jv2/config.py`
- models.py statt types.py (Python-Namenskonflikt vermeiden)
- Im Python-Code `pathlib.Path` für Pfade, keine hardcoded `/` oder `\`

## Philosophie
- JV = Joint Venture, nicht Konsortium. Jeder Bot ist selbständig.
- Ziel: Daten generieren, Stärken/Schwächen sichtbar machen
- Später: Teams/Firmen aus komplementären Bots bilden → echtes JV
- Win Rate Ziel: 73% minimum, 80% = "Muri-worthy"
