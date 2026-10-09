"""Read-only postgame evaluation of the offensive coordinator.

A recommended adjustment is not an executed adjustment. Missing logs stay
missing. This module only runs SELECT statements.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from typing import Any, Mapping

from cfb_coach.madden.model.schema import MLStatus

BLOWOUT_MARGIN = 17
_SCORE_RE = re.compile(r"(\d+)\s*[-–]\s*(\d+)")


def missing_database_report(path: Any) -> dict[str, Any]:
    location = str(path)
    return {
        "database": {"path": location, "exists": False, "access": "read_only"},
        "games": [],
        "games_evaluated": 0,
        "last_two_cpu_games": [],
        "cpu_blowout_wins": [],
        "user_cpu_blowouts": "not_found_in_this_environment",
        "missing_data": [
            f"Madden database is not at {location}",
            "The two reported CPU blowout wins are not available in this environment.",
        ],
        "history_modified": False,
        "note": "No game history was read or written because the Madden database is absent.",
    }


def _names(db: Any) -> set[str]:
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _columns(db: Any, table: str) -> set[str]:
    if table not in {"ml_decisions", "ml_outcomes", "snaps", "game_sessions"}:
        return set()
    try:
        return {str(row[1]) for row in db.conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _load(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        loaded = json.loads(raw or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _score_pair(text: Any) -> tuple[int, int] | None:
    match = _SCORE_RE.search(str(text or ""))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _diversity(values: list[str]) -> dict[str, Any]:
    counts = Counter(value for value in values if value)
    return {"distinct": len(counts), "counts": dict(counts)}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def _session_row(db: Any, game_id: str, columns: set[str]) -> Mapping[str, Any] | None:
    if "game_sessions" not in _names(db) or "session_id" not in columns:
        return None
    try:
        return db.conn.execute(
            "SELECT * FROM game_sessions WHERE session_id = ?",
            (game_id,),
        ).fetchone()
    except sqlite3.Error:
        return None


def _select_game_ids(
    db: Any, *, game_id: str | None, limit: int,
) -> tuple[list[str], list[str]]:
    """Return (selected ids, notes). CPU games only unless one id was requested."""
    notes: list[str] = []
    if "ml_decisions" not in _names(db):
        return [], ["ml_decisions table is missing"]
    try:
        rows = list(db.conn.execute(
            "SELECT game_id, session_id, id FROM ml_decisions "
            "WHERE mode = ? ORDER BY id DESC",
            ( "experimental",),
        ))
    except sqlite3.Error as exc:
        return [], [f"ml_decisions could not be read: {exc}"]
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        gid = row["game_id"] or row["session_id"]
        if not gid or gid in seen:
            continue
        seen.add(gid)
        ordered.append(str(gid))
    if game_id:
        return [game_id], notes
    session_cols = _columns(db, "game_sessions")
    cpu: list[str] = []
    skipped = 0
    for gid in ordered:
        session = _session_row(db, gid, session_cols)
        opponent = None if session is None else session["opponent_id"] if "opponent_id" in session.keys() else None
        if opponent == "cpu":
            cpu.append(gid)
        elif opponent in (None, ""):
            skipped += 1
            notes.append(f"{gid}: opponent was not stored, so it was not treated as a CPU game")
        else:
            skipped += 1
    if skipped:
        notes.append(f"{skipped} non-CPU experimental game(s) left out of the default report")
    if limit > 0:
        cpu = cpu[:limit]
    if not cpu:
        notes.append("no CPU experimental games in this database")
    return cpu, notes


def _verified(outcome: Mapping[str, Any] | None) -> bool:
    if outcome is None:
        return False
    return (
        str(outcome["executed_verification"] or "") == "verified"
        and str(outcome["executed_status"] or "") == "identified"
    )


def _converted(outcome_payload: Mapping[str, Any], distance: Any) -> bool | None:
    from cfb_coach.outcome import parse_outcome

    parsed = parse_outcome(str(outcome_payload.get("result") or ""))
    if parsed.kind in ("td", "convert"):
        return True
    if parsed.kind in ("incomplete", "int", "fumble", "sack", "stop"):
        return False
    if parsed.yards is None or distance is None:
        return None
    try:
        return int(parsed.yards) >= int(distance)
    except (TypeError, ValueError):
        return None


def _success(outcome_payload: Mapping[str, Any]) -> bool | None:
    from cfb_coach.outcome import parse_outcome

    parsed = parse_outcome(str(outcome_payload.get("result") or ""))
    return parsed.success


def evaluate_saved_games(
    db: Any, *, game_id: str | None = None, limit: int = 2,
) -> dict[str, Any]:
    """Evaluate stored games. Does not write and does not invent missing fields."""
    names = _names(db)
    game_ids, notes = _select_game_ids(db, game_id=game_id, limit=limit)
    outcomes = _outcomes_by_snap(db) if "ml_outcomes" in names else {}
    if "ml_outcomes" not in names:
        notes.append("ml_outcomes table is missing")
    games = [
        _evaluate_game(db, gid, outcomes) for gid in game_ids
    ]
    blowouts = [
        game["game_id"] for game in games
        if game.get("stored_result_label") == "blowout_win"
    ]
    missing = list(notes)
    if not games:
        missing.append(
            "The two reported CPU blowout wins were not found in this database."
        )
    return {
        "database": {"exists": True, "access": "read_only"},
        "games_evaluated": len(games),
        "last_two_cpu_games": [game["game_id"] for game in games],
        "cpu_blowout_wins": blowouts,
        "user_cpu_blowouts": blowouts or "not_found_in_this_database",
        "games": games,
        "missing_data": missing,
        "history_modified": False,
        "note": (
            "Recommended adjustments are counted separately from adjustments "
            "the user explicitly confirmed. A missing field is not a zero."
        ),
    }


def _outcomes_by_snap(db: Any) -> dict[str, Any]:
    try:
        return {
            str(row["snap_id"]): row
            for row in db.conn.execute("SELECT * FROM ml_outcomes")
            if row["snap_id"]
        }
    except sqlite3.Error:
        return {}


def _snaps_for(db: Any, game_id: str) -> list[Any]:
    columns = _columns(db, "snaps")
    if "snaps" not in _names(db) or "session_id" not in columns:
        return []
    try:
        return list(db.conn.execute(
            "SELECT * FROM snaps WHERE session_id = ? ORDER BY id ASC",
            (game_id,),
        ))
    except sqlite3.Error:
        return []


def _evaluate_game(
    db: Any, game_id: str, outcomes: Mapping[str, Any],
) -> dict[str, Any]:
    from cfb_coach.madden.model.experimental_model import _play_concept
    from cfb_coach.madden.playcaller import coverage_class

    missing: list[str] = []
    session_cols = _columns(db, "game_sessions")
    session = _session_row(db, game_id, session_cols)
    score_text = None if session is None or "score" not in session_cols else session["score"]
    result_wl = None if session is None or "result_wl" not in session_cols else session["result_wl"]
    pair = _score_pair(score_text)
    if pair is None:
        missing.append("final score was not stored")
    label = None
    if pair is not None and pair[0] - pair[1] >= BLOWOUT_MARGIN:
        label = "blowout_win"
    elif pair is not None and pair[1] - pair[0] >= BLOWOUT_MARGIN:
        label = "blowout_loss"
    try:
        decisions = list(db.conn.execute(
            "SELECT * FROM ml_decisions WHERE mode = ? "
            "AND (game_id = ? OR session_id = ?) ORDER BY id ASC",
            ("experimental", game_id, game_id),
        ))
    except sqlite3.Error as exc:
        decisions = []
        missing.append(f"decisions unreadable: {exc}")
    snaps = _snaps_for(db, game_id)
    snap_by_ml = {}
    if snaps and "ml_snap_id" in _columns(db, "snaps"):
        snap_by_ml = {
            str(row["ml_snap_id"]): row for row in snaps if row["ml_snap_id"]
        }
    else:
        missing.append("snaps are not linked by ml_snap_id")

    recommended_forms: list[str] = []
    recommended_plays: list[str] = []
    verified_forms: list[str] = []
    verified_plays: list[str] = []
    recommended_adjustments = 0
    applied_adjustments = 0
    recommended_macros = 0
    applied_macros = 0
    custom_macros_applied = 0
    no_adjustment_recommended = 0
    no_adjustment_confirmed = 0
    third = {"decisions": 0, "verified_outcomes": 0, "verified_conversions": 0, "conversion_unknown": 0}
    red_zone = {"decisions": 0, "verified_outcomes": 0}
    clock = {"decisions": 0, "verified_outcomes": 0, "context_missing": 0}
    plays_considered: list[int] = []
    plans_compared: list[int] = []
    latencies: list[float] = []
    latency_missing = 0
    fallback = 0
    inventory_ids: list[str] = []
    inventory_confirmed = 0
    inventory_unconfirmed = 0
    inventory_missing = 0
    predicted: list[float] = []
    observed_success: list[int] = []
    prediction_missing = 0
    verified_executed = 0
    looks: list[str] = []

    for row in decisions:
        payload = _load(row["decision_json"])
        experimental = payload.get("experimental_offense") or {}
        action = experimental.get("offense_action") or {}
        presnap = experimental.get("pre_snap_action_context") or {}
        audit = {
            item.get("name"): item
            for item in (experimental.get("input_audit") or {}).get("fields") or []
            if isinstance(item, dict)
        }
        kind = str(action.get("kind") or "none")
        action_id = action.get("id")
        status = str(row["shadow_status"] or "")
        if status != MLStatus.OK.value:
            fallback += 1
        else:
            if row["final_formation"]:
                recommended_forms.append(str(row["final_formation"]))
            if row["final_play"]:
                recommended_plays.append(str(row["final_play"]))
            if kind == "adjustment" and action_id:
                recommended_adjustments += 1
            elif kind == "macro" and action_id:
                recommended_macros += 1
            else:
                no_adjustment_recommended += 1
        if action.get("plays_considered") is not None:
            plays_considered.append(int(action["plays_considered"]))
        elif experimental.get("inventory_eligible_count") is not None:
            plays_considered.append(int(experimental["inventory_eligible_count"]))
        if action.get("plans_compared") is not None:
            plans_compared.append(int(action["plans_compared"]))
        latency = row["latency_ms_model"]
        if latency is None:
            latency_missing += 1
        else:
            latencies.append(float(latency))
        inv = experimental.get("inventory_id")
        if inv:
            inventory_ids.append(str(inv))
            if experimental.get("inventory_confirmed") is True:
                inventory_confirmed += 1
            elif experimental.get("inventory_confirmed") is False:
                inventory_unconfirmed += 1
        else:
            inventory_missing += 1

        outcome = outcomes.get(str(row["snap_id"] or ""))
        outcome_payload = _load(outcome["outcome_json"]) if outcome is not None else {}
        verified = _verified(outcome)
        if verified and outcome is not None and outcome["executed_play"]:
            verified_executed += 1
            if outcome["executed_formation"]:
                verified_forms.append(str(outcome["executed_formation"]))
            verified_plays.append(str(outcome["executed_play"]))
        confirmed_action = bool(
            verified and outcome_payload.get("offense_action_explicitly_confirmed")
        )
        if kind == "adjustment" and confirmed_action and outcome_payload.get("executed_adjustment_id") == action_id:
            applied_adjustments += 1
        if kind == "macro" and confirmed_action and outcome_payload.get("executed_macro") == action_id:
            applied_macros += 1
            if str(action_id).startswith("ML-"):
                custom_macros_applied += 1
        if verified and outcome_payload.get("no_adjustment_explicitly_confirmed") is True:
            no_adjustment_confirmed += 1

        down = presnap.get("down")
        distance = presnap.get("distance")
        snap = snap_by_ml.get(str(row["snap_id"] or ""))
        if down is None and snap is not None:
            down = snap["down"]
            distance = distance if distance is not None else snap["distance"]
        if down == 3:
            third["decisions"] += 1
            if verified:
                third["verified_outcomes"] += 1
                converted = _converted(outcome_payload, distance)
                if converted is True:
                    third["verified_conversions"] += 1
                elif converted is None:
                    third["conversion_unknown"] += 1
        zone_known = None
        if "red_zone" in audit and audit["red_zone"].get("status") == "known":
            zone_known = bool(audit["red_zone"].get("value"))
        elif presnap.get("red_zone") is True:
            zone_known = True
        elif snap is not None and snap["yardline"] is not None and int(snap["yardline"]) >= 80:
            zone_known = True
        elif presnap.get("red_zone") is False and presnap.get("yardline") is not None:
            zone_known = False
        if zone_known is True:
            red_zone["decisions"] += 1
            if verified:
                red_zone["verified_outcomes"] += 1
        quarter = presnap.get("quarter")
        clock_seconds = presnap.get("clock_seconds")
        phase = ((action.get("joint") or {}).get("football") or {}).get("score_phase")
        two_minute = presnap.get("two_minute")
        clock_case = False
        if quarter is None and snap is not None:
            quarter = snap["quarter"]
        if quarter is None and clock_seconds is None and two_minute is None:
            clock["context_missing"] += 1
        else:
            late = quarter is not None and int(quarter) >= 4
            short_clock = clock_seconds is not None and int(clock_seconds) <= 300
            if late and (short_clock or two_minute is True or phase in ("protect_lead", "trailing")):
                clock_case = True
            elif two_minute is True:
                clock_case = True
        if clock_case:
            clock["decisions"] += 1
            if verified:
                clock["verified_outcomes"] += 1

        probability = experimental.get("probability")
        if probability is None:
            probability = payload.get("probability")
        observed = _success(outcome_payload) if verified else None
        if probability is None or observed is None:
            prediction_missing += 1
        else:
            predicted.append(float(probability))
            observed_success.append(int(bool(observed)))
        if snap is not None and snap["coverage_seen"]:
            classified = coverage_class(snap["coverage_seen"])
            if classified:
                looks.append(classified)

    if not plays_considered and decisions:
        missing.append("legal action opportunity counts were not stored on these decisions")
    if inventory_missing and decisions:
        missing.append("inventory fingerprint was not stored on every decision")
    if latency_missing and decisions:
        missing.append("model latency was missing on some decisions")
    if prediction_missing and decisions:
        missing.append(
            "model probability and a verified outcome were not both available "
            f"on {prediction_missing} decision(s)"
        )
    half = max(1, len(looks) // 2) if looks else 0
    early = Counter(looks[:half]) if looks else Counter()
    late = Counter(looks[half:]) if len(looks) > half else Counter()
    if len(looks) < 4:
        tendency = {
            "status": "insufficient",
            "observed_looks": len(looks),
            "note": "Fewer than four classified defensive observations.",
        }
    else:
        early_top = early.most_common(1)[0][0] if early else None
        late_top = late.most_common(1)[0][0] if late else None
        tendency = {
            "status": "observed",
            "observed_looks": len(looks),
            "early": dict(early),
            "late": dict(late),
            "modal_look_changed": early_top != late_top,
        }
    distinct_inventory = sorted(set(inventory_ids))
    return {
        "game_id": game_id,
        "opponent_id": None if session is None or "opponent_id" not in session.keys() else session["opponent_id"],
        "final_score": None if pair is None else {"us": pair[0], "them": pair[1], "stored": score_text},
        "result_wl": result_wl,
        "stored_result_label": label,
        "recommended_offensive_plays": len(recommended_plays),
        "verified_executed_offensive_plays": verified_executed,
        "formation_diversity": {
            "recommended": _diversity(recommended_forms),
            "verified_executed": _diversity(verified_forms),
        },
        "concept_diversity": {
            "recommended": _diversity([_play_concept(play) for play in recommended_plays]),
            "verified_executed": _diversity([_play_concept(play) for play in verified_plays]),
        },
        "third_down": third,
        "red_zone": red_zone,
        "clock_management": clock,
        "adjustments": {
            "recommended": recommended_adjustments,
            "verified_applied": applied_adjustments,
            "recommended_but_not_confirmed": recommended_adjustments - applied_adjustments,
            "note": "A displayed adjustment is not an executed adjustment.",
        },
        "no_adjustment": {
            "recommended": no_adjustment_recommended,
            "explicitly_confirmed": no_adjustment_confirmed,
            "note": "Unconfirmed snaps are not counted as NO ADJUSTMENT executions.",
        },
        "macros": {
            "recommended": recommended_macros,
            "verified_applied": applied_macros,
            "custom_verified_applied": custom_macros_applied,
        },
        "legal_action_opportunities": {
            "snaps_with_play_counts": len(plays_considered),
            "mean_plays_considered": _mean([float(n) for n in plays_considered]),
            "snaps_with_plan_counts": len(plans_compared),
            "mean_plans_compared": _mean([float(n) for n in plans_compared]),
        },
        "opponent_tendencies": tendency,
        "prediction_vs_outcome": {
            "comparable_snaps": len(predicted),
            "mean_predicted_probability": _mean(predicted),
            "observed_success_rate": _mean([float(v) for v in observed_success]),
        },
        "latency_ms": {
            "n": len(latencies),
            "mean": _mean(latencies),
            "max": round(max(latencies), 3) if latencies else None,
        },
        "fallback_count": fallback,
        "inventory": {
            "fingerprints": distinct_inventory,
            "consistent": len(distinct_inventory) <= 1,
            "confirmed_decisions": inventory_confirmed,
            "unconfirmed_decisions": inventory_unconfirmed,
            "missing_fingerprint": inventory_missing,
        },
        "missing_data": missing,
    }
