"""Read-only historical comparison of the first-stage pick and joint pick.

Older decisions persist at most a subset of joint candidates; missing baseline
scores remain unknown. This report never constructs unexecuted yards or changes
the game DB. It is not a substitute for a full model-artifact replay.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def summarize(conn: sqlite3.Connection, *, model_contains: str = ".empirical") -> dict[str, Any]:
    counts: Counter[str] = Counter()
    cases: list[dict[str, Any]] = []
    rows = conn.execute(
        "SELECT session_id, model_version, decision_json FROM ml_decisions "
        "WHERE policy_source = 'model' ORDER BY id"
    ).fetchall()
    for session_id, version, raw in rows:
        if model_contains and model_contains not in str(version or ""):
            continue
        try:
            data = json.loads(raw or "{}")
        except (TypeError, json.JSONDecodeError):
            counts["bad_json"] += 1
            continue
        offense = data.get("experimental_offense") or {}
        anchor = (offense.get("selection_audit") or {}).get("top_selected")
        action = offense.get("offense_action") or {}
        joint = action.get("joint") or {}
        if not isinstance(anchor, list) or len(anchor) != 2 or not joint:
            counts["uninspectable"] += 1
            continue
        original = (str(anchor[0]), str(anchor[1]))
        winner = (str(joint.get("formation")), str(joint.get("play")))
        counts["inspectable"] += 1
        if original == winner:
            counts["same_play"] += 1
            continue
        counts["changed_play"] += 1
        if str(action.get("kind") or "none") != "none":
            counts["adjusted_play_change"] += 1
            continue
        counts["no_adjustment_changed_play"] += 1
        joint_rows = action.get("candidates") or []
        def score_for(key: tuple[str, str]) -> float | None:
            for candidate in joint_rows:
                if (str(candidate.get("formation")), str(candidate.get("play"))) == key and candidate.get("id") == "NO_ADJUSTMENT":
                    try:
                        return float(candidate["score"])
                    except (TypeError, ValueError, KeyError):
                        return None
            return None
        a, b = score_for(original), score_for(winner)
        if a is None or b is None:
            counts["score_pair_not_retained"] += 1
            continue
        counts["score_pair_retained"] += 1
        margin = round(b - a, 5)
        if margin < 0.025:
            counts["below_new_margin_with_retained_scores"] += 1
        cases.append({
            "session_id": session_id, "anchor": list(original), "joint": list(winner),
            "stored_score_gap": margin,
            "would_trigger_guard_if_these_were_final_scores": margin < 0.025,
        })
    return {
        "scope": f"model_version contains {model_contains!r}",
        "counts": dict(counts),
        "examples": sorted(cases, key=lambda x: x["stored_score_gap"])[:15],
        "limitations": [
            "Old decisions contain at most a small subset of joint candidates.",
            "Missing score pairs are unknown, not zero or a failed override.",
            "Archived scores are from the old policy and are not a complete new-policy replay.",
            "Unexecuted alternatives have unknown outcomes; this does not estimate yards or wins.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db", type=Path, help="Existing Madden database or backup; read only")
    parser.add_argument("--model-contains", default=".empirical")
    args = parser.parse_args(argv)
    with sqlite3.connect(f"file:{args.db.resolve()}?mode=ro", uri=True) as conn:
        report = summarize(conn, model_contains=args.model_contains)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
