"""
JV Boting v2 – Analyst Agent
Ilirs persönlicher Assistent: Täglicher HTML-Report via Telegram.
"""

import os
from datetime import datetime, timezone

from src.jv2.config import REPORTS_DIR, BOT_CONFIGS
from src.jv2.telegram import tg_send, tg_send_document


class AnalystAgent:

    def generate_daily_report(self, bots, scout_agent, prices, market_info=None, eval_results=None):
        """Generiert HTML-Report, speichert und sendet via TG.
        prices: dict {symbol: price} oder float.
        eval_results: dict {bot_id: {these_score, exec_score, edge_score}} oder None.
        """
        if isinstance(prices, (int, float)):
            prices = {"XRP/USDT": prices}
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")

        # Daten sammeln
        total_equity = sum(b.state.capital for b in bots)
        total_pnl = sum(b.state.total_pnl for b in bots)
        total_trades = sum(b.state.wins + b.state.losses for b in bots)
        total_wins = sum(b.state.wins for b in bots)
        wr = f"{total_wins/total_trades*100:.0f}%" if total_trades > 0 else "–"

        active_pos = sum(1 for b in bots if b.state.position)
        best_bot = max(bots, key=lambda b: b.state.total_pnl)
        worst_bot = min(bots, key=lambda b: b.state.total_pnl)

        # Scout Analyse
        gap = scout_agent.get_gap_analysis()

        def _get_price(bot):
            return prices.get(bot.symbol, 0)

        # ── TG-Nachricht ──
        lines = []
        lines.append(f"<b>\U0001F4CA JV Boting v2 — Daily Report</b>")
        price_str = " | ".join(f"{s.split('/')[0]}:${p:.2f}" for s, p in prices.items() if p > 0)
        lines.append(f"<b>{date_str}</b> | {price_str}\n")

        lines.append(f"<b>Portfolio:</b> ${total_equity:.0f} | PnL: ${total_pnl:+.2f} | WR: {wr} | Trades: {total_trades}")
        lines.append(f"Positionen offen: {active_pos}/{len(bots)}\n")

        lines.append("<b>Bot Performance:</b>")
        lines.append("<pre>")
        lines.append(f"{'Bot':<20} {'Cap':>5} {'PnL':>6} {'WR':>4} {'T':>2} {'Pos':>4}")
        lines.append("-" * 46)

        for bot in sorted(bots, key=lambda b: b.state.total_pnl, reverse=True):
            cfg = BOT_CONFIGS.get(bot.base_id, {})
            n = bot.state.wins + bot.state.losses
            bot_wr = f"{bot.state.wins/n*100:.0f}%" if n > 0 else "–"
            pos = ""
            if bot.state.position:
                p = _get_price(bot)
                pos = bot.state.position.direction[0].upper()
                if p > 0:
                    pos_pnl = bot.state.position.unrealized_pnl(p)
                    pos += f"{pos_pnl:+.0f}"
            label = bot.bot_id[:18]
            lines.append(f"{label:<20} ${bot.state.capital:>4.0f} {bot.state.total_pnl:>+5.1f} {bot_wr:>4} {n:>2} {pos:>4}")

        lines.append("</pre>")

        lines.append(f"\n\U0001F3C6 Best: <b>{best_bot.bot_id}</b> (${best_bot.state.total_pnl:+.1f})")
        lines.append(f"\u274C Worst: <b>{worst_bot.bot_id}</b> (${worst_bot.state.total_pnl:+.1f})")

        if market_info:
            lines.append(f"\n<b>Markt:</b> {market_info}")

        if gap["total_missed"] > 0:
            lines.append(f"\n<b>\U0001F50D Scout:</b> {gap['total_missed']} Moves verpasst ({gap['total_missed_pct']:.1f}% total)")
            if gap.get("by_phase"):
                phase_str = ", ".join(f"{k}:{v}" for k, v in gap["by_phase"].items())
                lines.append(f"  Phasen: {phase_str}")
            if gap.get("gaps"):
                for g in gap["gaps"]:
                    lines.append(f"  \u26A0 {g}")

        # Evaluator Scores
        if eval_results:
            lines.append(f"\n<b>\U0001F9E0 Bot Intelligence (These/Exec/Edge):</b>")
            lines.append("<pre>")
            lines.append(f"{'Bot':<22s} {'These':>5} {'Exec':>5} {'Edge':>5}")
            lines.append("-" * 42)

            # Aggregiere per base_id (über alle Assets)
            from collections import defaultdict
            agg = defaultdict(lambda: {"tc": 0, "tt": 0, "es": [], "ew": 0, "et": 0, "missed": 0})
            for bot_id, ev in eval_results.items():
                base = bot_id.rsplit("_", 1)[0]
                if ev["these_total"]:
                    agg[base]["tc"] += ev["these_correct"]
                    agg[base]["tt"] += ev["these_total"]
                if ev["exec_score"] is not None:
                    agg[base]["es"].append(ev["exec_score"])
                if ev["edge_total"]:
                    agg[base]["ew"] += ev["edge_wins"]
                    agg[base]["et"] += ev["edge_total"]
                agg[base]["missed"] += ev.get("these_missed", 0)

            for base in sorted(agg.keys()):
                a = agg[base]
                these = f"{a['tc']/a['tt']*100:.0f}%" if a["tt"] > 0 else "  –"
                ex = f"{sum(a['es'])/len(a['es'])*100:+.0f}%" if a["es"] else "  –"
                edge = f"{a['ew']/a['et']*100:.0f}%" if a["et"] > 0 else "  –"
                lines.append(f"{base:<22s} {these:>5} {ex:>5} {edge:>5}")

            lines.append("</pre>")
            lines.append("<i>These=Markt richtig gelesen | Exec=Move eingefangen | Edge=besser als Zufall</i>")

        # ── Shadow Challenger 3-way A/B (Champion / v1 / v2) ──
        try:
            from src.jv2.challenger import load_state, load_state_v2
            v1 = load_state()
            v2 = load_state_v2()
            v1n = v1.wins + v1.losses
            v1wr = v1.wins / v1n if v1n > 0 else 0.0
            v2n = v2.wins + v2.losses
            v2wr = v2.wins / v2n if v2n > 0 else 0.0
            cwr = total_wins / total_trades if total_trades > 0 else 0.0
            lines.append("\n<b>\U0001F9EA Shadow Challenger 3-way A/B</b>:")
            lines.append(f"  Champion: ${total_pnl:+.2f} "
                         f"({total_trades} trades, WR {cwr:.0%})")
            lines.append(f"  v1 boost: ${v1.total_pnl:+.2f} "
                         f"({v1n} closed/{v1.trades_taken} taken, WR {v1wr:.0%}) "
                         f"Δ ${v1.total_pnl - total_pnl:+.2f}")
            lines.append(f"  v2 inv:   ${v2.total_pnl:+.2f} "
                         f"({v2n} closed/{v2.trades_taken} taken, WR {v2wr:.0%}) "
                         f"Δ ${v2.total_pnl - total_pnl:+.2f}")
        except Exception:
            pass

        # ── Coach Verdict ──
        try:
            from src.jv2.coach import load_coach_state
            cs = load_coach_state()
            if cs:
                decisions = cs.get("decisions", {})
                from collections import defaultdict as _dd
                buckets = _dd(list)
                for bid, d in decisions.items():
                    buckets[d.get("action", "keep")].append(bid)
                lines.append(f"\n<b>\U0001F9E0 Coach Verdict</b> "
                             f"<i>({cs.get('last_update','')[:19]} UTC)</i>:")
                order = [("champion", "\U0001F451 Champion"),
                         ("promote", "\U0001F4C8 Promote"),
                         ("invert", "\U0001F501 Invert"),
                         ("demote", "\U0001F4C9 Demote"),
                         ("disable", "⛔ Disable")]
                for act, label in order:
                    cells = buckets.get(act, [])
                    if cells:
                        lines.append(f"  {label}: {len(cells)} — "
                                     f"{', '.join(c.replace('_XRP','').replace('_BTC','').replace('_ETH','').replace('_SOL','') + c[-4:] for c in sorted(cells)[:6])}"
                                     + (" …" if len(cells) > 6 else ""))
                n_kept = len(buckets.get("keep", []))
                if n_kept:
                    lines.append(f"  ➖ Keep: {n_kept}")
        except Exception:
            pass

        msg = "\n".join(lines)
        tg_send(msg)

        # ── HTML speichern ──
        html = self._build_html(date_str, prices, bots, gap, market_info, eval_results)
        html_path = os.path.join(REPORTS_DIR, f"report_{date_str}.html")
        try:
            with open(html_path, "w") as f:
                f.write(html)
            tg_send_document(html_path, caption=f"JV2 Report {date_str}")
        except Exception:
            pass

        return msg

    def _build_html(self, date_str, prices, bots, gap, market_info, eval_results=None):
        if isinstance(prices, (int, float)):
            prices = {"XRP/USDT": prices}

        rows = ""
        for bot in sorted(bots, key=lambda b: b.state.total_pnl, reverse=True):
            cfg = BOT_CONFIGS.get(bot.base_id, {})
            n = bot.state.wins + bot.state.losses
            wr = f"{bot.state.wins/n*100:.0f}%" if n > 0 else "–"
            pos = "–"
            if bot.state.position:
                p = prices.get(bot.symbol, 0)
                if p > 0:
                    pos_pnl = bot.state.position.unrealized_pnl(p)
                    pos = f"{bot.state.position.direction.upper()} ({pos_pnl:+.1f})"
                else:
                    pos = bot.state.position.direction.upper()
            color = "#4CAF50" if bot.state.total_pnl >= 0 else "#F44336"
            rows += f"""<tr>
                <td>{cfg.get('emoji','')} {bot.bot_id}</td>
                <td>${bot.state.capital:.0f}</td>
                <td style="color:{color}">${bot.state.total_pnl:+.2f}</td>
                <td>{wr}</td><td>{n}</td><td>{pos}</td>
            </tr>"""

        total_eq = sum(b.state.capital for b in bots)
        total_pnl = sum(b.state.total_pnl for b in bots)
        n_bots = len(bots)
        price_str = " | ".join(f"{s.split('/')[0]}: ${p:.2f}" for s, p in prices.items() if p > 0)

        scout_html = ""
        if gap["total_missed"] > 0:
            scout_html = f"""
            <h2>Scout Report</h2>
            <p>{gap['total_missed']} Moves verpasst ({gap['total_missed_pct']:.1f}% total)</p>
            <p>Phasen: {gap.get('by_phase', {})}</p>
            """
            if gap.get("gaps"):
                scout_html += "<ul>" + "".join(f"<li>{g}</li>" for g in gap["gaps"]) + "</ul>"

        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>JV Boting v2 Report {date_str}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #1a1a2e; color: #eee; }}
h1 {{ color: #e94560; }}
h2 {{ color: #0f3460; background: #16213e; padding: 8px 16px; border-radius: 4px; }}
table {{ width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 0.9em; }}
th, td {{ padding: 6px 10px; text-align: left; border-bottom: 1px solid #333; }}
th {{ background: #16213e; }}
.summary {{ display: flex; gap: 20px; flex-wrap: wrap; }}
.card {{ background: #16213e; padding: 16px; border-radius: 8px; min-width: 150px; }}
.card h3 {{ margin: 0 0 8px; color: #aaa; font-size: 0.9em; }}
.card p {{ margin: 0; font-size: 1.4em; font-weight: bold; }}
</style></head><body>
<h1>JV Boting v2 — Daily Report</h1>
<p>{date_str} | {price_str}</p>

<div class="summary">
    <div class="card"><h3>Total Equity</h3><p>${total_eq:.0f}</p></div>
    <div class="card"><h3>Total PnL</h3><p style="color:{'#4CAF50' if total_pnl >= 0 else '#F44336'}">${total_pnl:+.2f}</p></div>
    <div class="card"><h3>Aktive Positionen</h3><p>{sum(1 for b in bots if b.state.position)}/{n_bots}</p></div>
</div>

<h2>Bot Performance</h2>
<table>
<tr><th>Bot</th><th>Capital</th><th>PnL</th><th>WR</th><th>Trades</th><th>Position</th></tr>
{rows}
</table>

{scout_html}

<p style="color:#666; margin-top:40px; font-size:0.8em;">Generated by JV Boting v2 Analyst Agent</p>
</body></html>"""
