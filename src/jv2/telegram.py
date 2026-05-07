"""
JV Boting v2 – Telegram
"""

import requests
from src.jv2.config import TG_BOT_TOKEN, TG_CHAT_ID, BOT_CONFIGS


def tg_send(text, parse_mode="HTML"):
    """Sende Nachricht an Telegram. Splittet bei >4096 Zeichen."""
    try:
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            requests.post(
                f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
                json={"chat_id": TG_CHAT_ID, "text": chunk, "parse_mode": parse_mode},
                timeout=10,
            )
    except Exception:
        pass


def tg_send_document(file_path, caption=""):
    """Sende Datei an Telegram."""
    try:
        with open(file_path, "rb") as f:
            requests.post(
                f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendDocument",
                data={"chat_id": TG_CHAT_ID, "caption": caption},
                files={"document": f},
                timeout=30,
            )
    except Exception:
        pass


def fmt_trade_open(bot_id, pos):
    cfg = BOT_CONFIGS.get(bot_id, {})
    emoji = cfg.get("emoji", "")
    label = cfg.get("label", bot_id)
    d = "\U00002B06" if pos.direction == "long" else "\U00002B07"
    return (
        f"{emoji} <b>{label}</b> {d} {pos.direction.upper()}\n"
        f"Entry: ${pos.entry_price:.4f}\n"
        f"Size: ${pos.size_usd:.0f} | SL: ${pos.stop_loss:.4f} | TP: ${pos.take_profit:.4f}"
    )


def fmt_trade_close(record):
    cfg = BOT_CONFIGS.get(record.bot_id, {})
    emoji = cfg.get("emoji", "")
    label = cfg.get("label", record.bot_id)
    pnl_emoji = "\u2705" if record.pnl > 0 else "\u274C"
    return (
        f"{pnl_emoji} <b>{label}</b> CLOSE {record.direction.upper()} [{record.reason}]\n"
        f"PnL: ${record.pnl:+.2f} ({record.net_return_pct:+.2f}%)\n"
        f"Hold: {record.hold_candles} candles | Capital: ${record.bot_capital_after:.0f}"
    )
