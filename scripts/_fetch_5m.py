# 5m-OHLCV von Binance für Exit-Replay (Apr 25 - jetzt), 4 Symbole
import csv
import time
from pathlib import Path

import requests

OUT = Path(__file__).resolve().parents[1] / "data" / "_live_snapshot"
SYMBOLS = ["XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]
START_MS = 1777413600000  # 2026-04-25 00:00 UTC ca.
HOSTS = ["https://api.binance.com", "https://api1.binance.com", "https://data-api.binance.vision"]

def fetch(symbol):
    rows = []
    start = START_MS
    while True:
        for host in HOSTS:
            try:
                r = requests.get(f"{host}/api/v3/klines",
                                 params=dict(symbol=symbol, interval="5m", startTime=start, limit=1000),
                                 timeout=20)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                print(f"{symbol}: {host} failed: {e}")
                batch = None
        if batch is None:
            raise RuntimeError(f"{symbol}: alle Hosts down")
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][6] + 1  # close_time + 1
        if len(batch) < 1000:
            break
        time.sleep(0.3)
    return rows

for sym in SYMBOLS:
    rows = fetch(sym)
    out = OUT / f"klines_5m_{sym}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["open_time", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow([r[0], r[1], r[2], r[3], r[4], r[5]])
    print(f"{sym}: {len(rows)} Kerzen -> {out.name}")
print("DONE")
