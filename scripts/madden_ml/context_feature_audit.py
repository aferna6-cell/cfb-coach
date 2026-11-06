"""Which pre-snap football context does CFB Coach actually have? Read-only.

Reports, per side and opponent kind, how often each context field is present
in the logged snaps and decision records, and which fields are only known
after the snap. Opponent identities are not printed.

    python scripts/madden_ml/context_feature_audit.py --db AUDIT_COPY.db
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model.strategic_context import POST_SNAP_FIELDS  # noqa: E402
from cfb_coach.madden.situation import parse_madden_situation  # noqa: E402
from cfb_coach.opponents import is_cpu_opponent  # noqa: E402

SNAP_FIELDS = ("down", "distance", "yardline", "quarter", "situation_raw", "game_clock", "score_for",
               "score_against", "drive_id", "formation", "play", "coverage_seen", "concept_seen", "result",
               "yards", "macro", "recommended_adjustment_id", "executed_adjustment_id")
PARSED_FIELDS = ("clock_seconds", "score_us", "score_them", "timeouts_us", "red_zone", "goal_line", "two_minute")
DECISION_CONTEXT_KEYS = ("game_clock", "clock_seconds", "score_us", "score_them", "quarter", "timeouts_us",
                         "personnel", "offense_personnel", "defense_personnel")


def _present(v: Any) -> bool:
    return v is not None and str(v).strip() != ""


def audit(conn: sqlite3.Connection) -> dict[str, Any]:
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(snaps)")}
    groups: dict[str, dict[str, Any]] = {}
    for r in conn.execute("SELECT * FROM snaps"):
        kind = "cpu" if is_cpu_opponent(str(r["opponent_id"] or "")) else "human"
        g = groups.setdefault(f"{r['side']}/{kind}", {"n": 0, "fields": {}})
        g["n"] += 1
        for f in SNAP_FIELDS:
            if f in cols and _present(r[f]):
                g["fields"][f] = g["fields"].get(f, 0) + 1
        sit = parse_madden_situation(str(r["situation_raw"] or ""), default_side=str(r["side"] or "offense"))
        extras = getattr(sit, "extras", None) or {}
        for f in PARSED_FIELDS:
            v = getattr(sit, f, None)
            if v is None:
                v = extras.get(f)
            if f in ("red_zone", "goal_line", "two_minute"):
                v = True if v else None
            if _present(v):
                g["fields"][f"parsed:{f}"] = g["fields"].get(f"parsed:{f}", 0) + 1
    decisions = {"n": 0, "keys": {}}
    try:
        for r in conn.execute("SELECT decision_json FROM ml_decisions"):
            decisions["n"] += 1
            try:
                dj = json.loads(r[0] or "{}")
            except json.JSONDecodeError:
                continue
            ctx = dj.get("pre_snap_action_context") or {}
            for k in DECISION_CONTEXT_KEYS:
                if _present(ctx.get(k)) or _present(dj.get(k)):
                    decisions["keys"][k] = decisions["keys"].get(k, 0) + 1
    except sqlite3.DatabaseError:
        pass
    for g in groups.values():
        g["share"] = {f: round(c / g["n"], 3) for f, c in sorted(g["fields"].items())}
        del g["fields"]
    return {"snaps": groups, "decision_context": decisions, "post_snap_only": sorted(POST_SNAP_FIELDS)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    args = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{Path(args.db).expanduser().resolve()}?mode=ro", uri=True)
    try:
        print(json.dumps(audit(conn), indent=1))
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
