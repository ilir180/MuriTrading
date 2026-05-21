"""Run the Coach: read trades, emit coach_state.json, optionally send TG summary.

Usage:
    python scripts/run_coach.py           # silent run, writes file
    python scripts/run_coach.py --tg      # also send Telegram summary
    python scripts/run_coach.py --diff    # show diff vs current BOT_OVERRIDES
"""

import os
import sys
import argparse
from datetime import datetime, timezone

# Force UTF-8 on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

from src.jv2.coach import Coach
from src.jv2.config import SYMBOLS, BOT_RISK_PROFILES, BOT_OVERRIDES


def _all_cell_ids():
    """All 8x4 = 32 expected cells, used so every cell appears in coach_state even with zero trades."""
    out = []
    for base in BOT_RISK_PROFILES.keys():
        for sym_cfg in SYMBOLS.values():
            out.append(f"{base}_{sym_cfg['short']}")
    return out


def _diff_against_overrides(decisions):
    """Show where Coach disagrees with hand-curated BOT_OVERRIDES."""
    print("\n— Diff Coach vs BOT_OVERRIDES (manuelle Heuristik) —")
    agree, disagree = 0, 0
    for bid, d in sorted(decisions.items()):
        manual = BOT_OVERRIDES.get(bid, {})
        manual_invert = manual.get("invert", False)
        manual_exec = manual.get("exec_override") is not None
        c_invert = d.invert
        match_invert = manual_invert == c_invert
        if manual_invert or c_invert or manual_exec:
            tag = "OK " if match_invert else "DIFF"
            print(f"  {tag}  {bid:25s}  manual:invert={manual_invert}  coach:{d.action} invert={c_invert}")
            if match_invert:
                agree += 1
            else:
                disagree += 1
    print(f"\n  Agreement on invert flags: {agree} match / {disagree} diff")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tg", action="store_true", help="Send Telegram summary")
    p.add_argument("--diff", action="store_true", help="Show diff vs BOT_OVERRIDES")
    args = p.parse_args()

    coach = Coach()
    decisions = coach.evaluate(all_bot_ids=_all_cell_ids())
    payload = coach.write_state(decisions)

    summary = coach.summary(decisions)
    print(summary)
    print(f"\nWrote {len(decisions)} decisions to coach_state.json")
    print(f"Timestamp: {payload['last_update']}")

    if args.diff:
        _diff_against_overrides(decisions)

    if args.tg:
        from src.jv2.telegram import tg_send
        from collections import defaultdict
        by_action = defaultdict(list)
        for bid, d in decisions.items():
            by_action[d.action].append((bid, d))

        lines = [
            "\U0001F9E0 <b>Coach v1 — erste Entscheidung</b>",
            f"<i>{payload['last_update'][:19]} UTC</i>",
            "",
            f"<b>{len(decisions)} Cells analysiert</b>",
        ]
        order = [
            ("promote", "\U0001F4C8 Promote"),
            ("invert",  "\U0001F501 Invert"),
            ("exec_fix","\U0001F527 Exec-Fix"),
            ("keep",    "➖ Keep"),
            ("demote",  "\U0001F4C9 Demote"),
            ("disable", "⛔ Disable"),
        ]
        for act, label in order:
            items = sorted(by_action.get(act, []), key=lambda x: -x[1].confidence)
            if not items:
                continue
            lines.append(f"\n<b>{label} ({len(items)}):</b>")
            for bid, d in items[:8]:  # cap at 8 per category in TG
                ev = d.evidence[:120]
                lines.append(f"  • <code>{bid}</code> — {ev}")
            if len(items) > 8:
                lines.append(f"  … +{len(items)-8} weitere")

        tg_send("\n".join(lines))
        print("\nTelegram summary sent.")


if __name__ == "__main__":
    main()
