"""Training-row export and import. Owner: Data Engineering.

Rows keep recommendations, verified executions, and outcomes in separate
fields. Missing source columns stay unknown. Do not synthesize snaps.
A displayed recommendation is never treated as a verified execution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.learning import SUCCESS_NEED
from cfb_coach.madden.model.schema import (
    CONTRACT_VERSION,
    CSV_UNKNOWN,
    LEAKAGE_FIELDS,
    ExecutedStatus,
    Tri,
    Verification,
)
from cfb_coach.outcome import parse_outcome

_RECOMMENDATION_KEYS = (
    "recommended_formation",
    "recommended_play",
    "recommended_macro",
    "our_call",
)
_EXECUTION_KEYS = (
    "executed_formation",
    "executed_play",
    "executed_status",
    "executed_verification",
)
_LABEL_KEYS = ("success", "stop", "yards", "result", "label_available")

# Training-data eligibility. Supervised play-specific training uses VERIFIED_* only.
ELIGIBILITY_VERIFIED_EXECUTION = "verified_execution"
ELIGIBILITY_TRUSTED_VOD = "trusted_vod"
ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC = "outcome_known_execution_uncertain"
ELIGIBILITY_RECOMMENDATION_ONLY = "recommendation_only"
ELIGIBILITY_UNLABELED = "unlabeled"
ELIGIBILITY_EXCLUDED = "excluded"

SUPERVISED_ELIGIBLE = frozenset(
    {ELIGIBILITY_VERIFIED_EXECUTION, ELIGIBILITY_TRUSTED_VOD}
)


def classify_eligibility(row: Mapping[str, Any]) -> str:
    """Classify one row for supervised play-specific training eligibility."""
    if row.get("exclude") or row.get("invalid"):
        return ELIGIBILITY_EXCLUDED
    status = str(row.get("executed_status") or "").lower()
    verified = str(row.get("executed_verification") or "").lower() == "verified"
    has_exec = bool(row.get("executed_play")) and status == ExecutedStatus.IDENTIFIED.value and verified
    label_ok = bool(row.get("label_available"))
    provenance = str(row.get("provenance") or "")
    if has_exec and label_ok:
        return ELIGIBILITY_VERIFIED_EXECUTION
    # Trusted external VOD: explicit flag + valid play identity + outcome label.
    if bool(row.get("trusted_vod")) and label_ok:
        vod_play = (
            row.get("executed_play")
            or row.get("recommended_play")
            or row.get("play")
        )
        vod_form = (
            row.get("executed_formation")
            or row.get("recommended_formation")
            or row.get("formation")
        )
        prov_ok = (
            "vod" in provenance.lower()
            or provenance.startswith("csv:")
            or provenance.startswith("jsonl:")
            or provenance.startswith("json:")
            or bool(row.get("trusted_vod"))
        )
        if vod_play and vod_form and prov_ok:
            return ELIGIBILITY_TRUSTED_VOD
    if label_ok and not has_exec:
        if row.get("recommended_play"):
            return ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC
        return ELIGIBILITY_UNLABELED
    # Verified execution without a usable label stays unlabeled (not recommendation-only).
    if has_exec and not label_ok:
        return ELIGIBILITY_UNLABELED
    if row.get("recommended_play") and not label_ok:
        return ELIGIBILITY_RECOMMENDATION_ONLY
    return ELIGIBILITY_UNLABELED


def _blank(value: Any) -> bool:
    return value is None or value == "" or value == CSV_UNKNOWN


def _as_tri(value: Any) -> str:
    if value is None or value == "" or value == CSV_UNKNOWN:
        return Tri.UNKNOWN.value
    if isinstance(value, Tri):
        return value.value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "success"):
        return Tri.TRUE.value
    if text in ("0", "false", "no", "n", "fail", "failure"):
        return Tri.FALSE.value
    if text in ("true", "false", "unknown"):
        return text
    return Tri.UNKNOWN.value


def _success_from_yards(
    *,
    side: str,
    down: Any,
    distance: Any,
    yards: int | None,
    kind: str | None,
) -> str:
    if yards is None and not kind:
        return Tri.UNKNOWN.value
    side_l = (side or "offense").strip().lower()
    is_def = side_l.startswith("d")
    if kind in ("int", "interception", "fumble"):
        return Tri.FALSE.value if not is_def else Tri.TRUE.value
    if kind == "sack":
        return Tri.FALSE.value if not is_def else Tri.TRUE.value
    if yards is None:
        if kind == "incomplete":
            return Tri.FALSE.value if not is_def else Tri.TRUE.value
        return Tri.UNKNOWN.value
    try:
        d = int(down) if down is not None and down != "" else None
        dist = int(distance) if distance is not None and distance != "" else None
    except (TypeError, ValueError):
        return Tri.UNKNOWN.value
    if d is None or dist is None:
        return Tri.UNKNOWN.value
    need = SUCCESS_NEED.get(d, 1.0) * float(dist)
    ok = float(yards) >= need
    if is_def:
        return Tri.TRUE.value if not ok else Tri.FALSE.value
    return Tri.TRUE.value if ok else Tri.FALSE.value


def _normalize_row(raw: Mapping[str, Any], *, provenance: str) -> dict[str, Any]:
    row = {str(k): raw[k] for k in raw}
    side = str(row.get("side") or row.get("possession") or "offense").strip().lower()
    if side.startswith("d"):
        side = "defense"
    else:
        side = "offense"

    formation = row.get("formation") or row.get("recommended_formation")
    play = row.get("play") or row.get("recommended_play")
    recommended_formation = row.get("recommended_formation") or formation
    recommended_play = row.get("recommended_play") or play

    executed_status = str(row.get("executed_status") or ExecutedStatus.UNKNOWN.value).lower()
    executed_verification = str(
        row.get("executed_verification") or Verification.UNKNOWN.value
    ).lower()
    executed_formation = row.get("executed_formation")
    executed_play = row.get("executed_play")

    # Never promote a recommendation to a verified execution.
    if executed_status != ExecutedStatus.IDENTIFIED.value:
        executed_formation = None
        executed_play = None
        executed_verification = Verification.UNKNOWN.value
    elif executed_verification != Verification.VERIFIED.value:
        executed_status = ExecutedStatus.UNKNOWN.value
        executed_formation = None
        executed_play = None
        executed_verification = Verification.UNKNOWN.value

    result = row.get("result")
    yards = row.get("yards")
    kind = None
    if yards is None and result not in (None, ""):
        parsed = parse_outcome(str(result))
        yards = parsed.yards
        kind = parsed.kind
    try:
        yards_i = int(yards) if yards is not None and yards != "" else None
    except (TypeError, ValueError):
        yards_i = None

    success = row.get("success")
    if _blank(success):
        success = _success_from_yards(
            side=side,
            down=row.get("down"),
            distance=row.get("distance"),
            yards=yards_i,
            kind=kind,
        )
    else:
        success = _as_tri(success)

    stop = row.get("stop")
    if _blank(stop):
        if side == "defense" and success != Tri.UNKNOWN.value:
            stop = success
        elif side == "offense" and success == Tri.FALSE.value:
            stop = Tri.TRUE.value
        elif side == "offense" and success == Tri.TRUE.value:
            stop = Tri.FALSE.value
        else:
            stop = Tri.UNKNOWN.value
    else:
        stop = _as_tri(stop)

    label_available = success != Tri.UNKNOWN.value or stop != Tri.UNKNOWN.value
    behavior_propensity = row.get("behavior_propensity")
    if _blank(behavior_propensity):
        behavior_propensity = None

    # Prefer session / ml snap identity. Never collapse games onto opponent_id.
    session_id = row.get("session_id") or row.get("game_id")
    ml_snap = row.get("ml_snap_id") or row.get("snap_id")
    snap_id = ml_snap or row.get("id")
    if session_id:
        game_id = str(session_id)
    elif ml_snap and "-" in str(ml_snap):
        game_id = str(ml_snap).rsplit("-", 1)[0]
    elif row.get("source_game"):
        game_id = str(row.get("source_game"))
    elif row.get("source_path") and row.get("id") is not None:
        game_id = f"file:{Path(str(row['source_path'])).name}"
    else:
        game_id = None

    opp = row.get("opponent_id")
    opp_type = row.get("opponent_type")
    if _blank(opp_type) and opp:
        opp_l = str(opp).lower()
        opp_type = "cpu" if opp_l.startswith("cpu") or "cpu" in opp_l else "human"

    out = {
        "schema_version": CONTRACT_VERSION,
        "provenance": provenance,
        "snap_id": None if _blank(snap_id) else str(snap_id),
        "game_id": None if _blank(game_id) else str(game_id),
        "session_id": None if _blank(session_id) else str(session_id),
        "opponent_id": None if _blank(opp) else str(opp),
        "opponent_type": None if _blank(opp_type) else str(opp_type),
        "side": side,
        "down": row.get("down"),
        "distance": row.get("distance"),
        "yardline": row.get("yardline"),
        "quarter": row.get("quarter"),
        "situation_raw": row.get("situation_raw") or row.get("raw"),
        "coverage_hint": row.get("coverage_hint") or row.get("coverage_prediction"),
        "concept_hint": row.get("concept_hint") or row.get("concept_prediction"),
        "recommended_formation": None if _blank(recommended_formation) else str(recommended_formation),
        "recommended_play": None if _blank(recommended_play) else str(recommended_play),
        "recommended_macro": None if _blank(row.get("macro") or row.get("recommended_macro")) else str(
            row.get("macro") or row.get("recommended_macro")
        ),
        "our_call": None if _blank(row.get("our_call")) else str(row.get("our_call")),
        "executed_status": executed_status,
        "executed_verification": executed_verification,
        "executed_formation": None if _blank(executed_formation) else str(executed_formation),
        "executed_play": None if _blank(executed_play) else str(executed_play),
        "result": None if _blank(result) else str(result),
        "yards": yards_i,
        "success": success,
        "stop": stop,
        "label_available": bool(label_available),
        "behavior_propensity": behavior_propensity,
        "propensity_method": row.get("propensity_method") or "unknown",
        "coverage_seen": row.get("coverage_seen"),  # post-snap; never a pre-snap feature
        "concept_seen": row.get("concept_seen"),
        "source_path": row.get("source_path"),
        "trusted_vod": bool(row.get("trusted_vod")),
        "data_source": row.get("data_source") or provenance,
    }
    out["eligibility"] = classify_eligibility(out)
    out["supervised_eligible"] = out["eligibility"] in SUPERVISED_ELIGIBLE
    # Action features: only set for supervised-eligible rows. Never substitute a
    # recommendation when execution is unknown — that would mis-attribute the result.
    if out["eligibility"] == ELIGIBILITY_VERIFIED_EXECUTION and out.get("executed_play"):
        out["action_formation"] = out["executed_formation"]
        out["action_play"] = out["executed_play"]
    elif out["eligibility"] == ELIGIBILITY_TRUSTED_VOD:
        out["action_formation"] = (
            out.get("executed_formation")
            or out.get("recommended_formation")
        )
        out["action_play"] = out.get("executed_play") or out.get("recommended_play")
    else:
        out["action_formation"] = None
        out["action_play"] = None
    return out


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Report schema gaps, duplicates, label coverage, and leakage risks."""
    missing_fields: dict[str, int] = {}
    labeled = 0
    unlabeled = 0
    verified_exec = 0
    recommendation_only = 0
    duplicates = 0
    seen: set[tuple[Any, ...]] = set()
    leakage_hits = 0
    provenance: dict[str, int] = {}
    eligibility: dict[str, int] = {}
    games: set[str] = set()
    cpu_n = 0
    human_n = 0
    offense_n = 0
    defense_n = 0
    supervised_n = 0
    coverage_labeled = 0

    required_presence = (
        "snap_id",
        "side",
        "recommended_play",
        "success",
        "executed_status",
    )

    for row in rows:
        prov = str(row.get("provenance") or "unknown")
        provenance[prov] = provenance.get(prov, 0) + 1
        elig = str(row.get("eligibility") or classify_eligibility(row))
        eligibility[elig] = eligibility.get(elig, 0) + 1
        if row.get("supervised_eligible") or elig in SUPERVISED_ELIGIBLE:
            supervised_n += 1
        if row.get("game_id"):
            games.add(str(row["game_id"]))
        ot = str(row.get("opponent_type") or "").lower()
        if ot == "cpu":
            cpu_n += 1
        elif ot == "human":
            human_n += 1
        if str(row.get("side") or "").startswith("d"):
            defense_n += 1
        else:
            offense_n += 1
        if row.get("coverage_seen") or row.get("coverage_hint"):
            coverage_labeled += 1
        for key in required_presence:
            if _blank(row.get(key)):
                missing_fields[key] = missing_fields.get(key, 0) + 1
        if row.get("label_available"):
            labeled += 1
        else:
            unlabeled += 1
        if str(row.get("executed_status") or "") == ExecutedStatus.IDENTIFIED.value and (
            str(row.get("executed_verification") or "") == Verification.VERIFIED.value
        ):
            verified_exec += 1
        elif not _blank(row.get("recommended_play")):
            recommendation_only += 1
        key = (
            row.get("game_id"),
            row.get("snap_id"),
            row.get("recommended_formation"),
            row.get("recommended_play"),
            row.get("result"),
        )
        if key in seen and key[1] is not None:
            duplicates += 1
        else:
            seen.add(key)
        feature_map = row.get("features")
        if isinstance(feature_map, Mapping):
            leaked = LEAKAGE_FIELDS.intersection(feature_map)
            if leaked:
                leakage_hits += 1

    return {
        "n_rows": len(rows),
        "unique_games": len(games),
        "labeled": labeled,
        "unlabeled": unlabeled,
        "verified_executions": verified_exec,
        "verified_labeled_supervised": supervised_n,
        "recommendation_only": recommendation_only,
        "duplicates": duplicates,
        "cpu_rows": cpu_n,
        "human_rows": human_n,
        "offense_rows": offense_n,
        "defense_rows": defense_n,
        "coverage_label_rows": coverage_labeled,
        "eligibility": eligibility,
        "missing_fields": missing_fields,
        "leakage_feature_rows": leakage_hits,
        "provenance": provenance,
        "recommendation_keys": list(_RECOMMENDATION_KEYS),
        "execution_keys": list(_EXECUTION_KEYS),
        "label_keys": list(_LABEL_KEYS),
        "ambiguous_matches": sum(
            len(r.get("ambiguous_matches") or []) for r in rows
        ),
    }


def canonical_snap_key(row: Mapping[str, Any]) -> str | None:
    """Stable identity for one logical snap across tables.

    Prefer ``ml_snap_id`` / ML ``snap_id``. Legacy rows fall back to
    ``session_id:play_id`` or ``game_id:legacy:<id>``. Returns None when no
    durable identity exists (caller should not invent one).
    """
    ml = row.get("ml_snap_id") or row.get("snap_id")
    if ml is not None and str(ml).strip() and not str(ml).isdigit():
        # Numeric-only snap_id is usually a legacy snaps.id, not an ML id.
        text = str(ml).strip()
        if "-" in text or text.startswith("g") or len(text) >= 8:
            return f"ml:{text}"
    session = row.get("session_id") or row.get("game_id")
    play_id = row.get("play_id")
    if session and play_id:
        return f"legacy:{session}:{play_id}"
    if session and row.get("id") is not None:
        return f"legacy:{session}:row:{row.get('id')}"
    if ml is not None and str(ml).strip():
        return f"legacy:id:{ml}"
    return None


def build_rows(
    *,
    db: Any = None,
    extra_paths: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Build feature rows from a Madden database and optional existing files.

    Joins ``snaps`` with ``ml_decisions`` / ``ml_outcomes`` on ``ml_snap_id``.
    Dedupes by :func:`canonical_snap_key` so one logical snap yields one active
    training row. Ambiguous historical matches are reported on the row and in
    :func:`validate_rows` / :func:`quality_report`.
    Does not fabricate rows when ``db`` is empty.
    """
    rows: list[dict[str, Any]] = []
    by_canon: dict[str, dict[str, Any]] = {}
    ambiguous: list[dict[str, Any]] = []

    def _merge_or_add(row: dict[str, Any], *, provenance: str) -> None:
        row = dict(row)
        row.setdefault("provenance", provenance)
        key = canonical_snap_key(row)
        if key is None:
            rows.append(row)
            return
        existing = by_canon.get(key)
        if existing is None:
            by_canon[key] = row
            rows.append(row)
            return
        # Same logical snap seen again: keep the richer / later outcome, flag ambiguity.
        existing_sources = list(existing.get("_source_tables") or [existing.get("provenance")])
        existing_sources.append(provenance)
        existing["_source_tables"] = existing_sources
        # Prefer verified execution + labeled outcome over recommendation-only.
        def _richness(r: Mapping[str, Any]) -> tuple[int, int, int]:
            return (
                1 if r.get("supervised_eligible") else 0,
                1 if r.get("label_available") else 0,
                1 if r.get("executed_play") else 0,
            )
        if _richness(row) > _richness(existing):
            # Replace in-place so list identity stays one active row.
            existing.clear()
            existing.update(row)
            existing["_source_tables"] = existing_sources
            existing["canonical_key"] = key
        else:
            # Do not invent a merge of conflicting fields — report ambiguity.
            if (
                (row.get("result") and existing.get("result") and row.get("result") != existing.get("result"))
                or (
                    row.get("executed_play")
                    and existing.get("executed_play")
                    and row.get("executed_play") != existing.get("executed_play")
                )
            ):
                note = {
                    "canonical_key": key,
                    "kept_provenance": existing.get("provenance"),
                    "skipped_provenance": provenance,
                    "kept_result": existing.get("result"),
                    "skipped_result": row.get("result"),
                }
                ambiguous.append(note)
                existing.setdefault("ambiguous_matches", []).append(note)
        existing["canonical_key"] = key

    if db is not None:
        decisions: dict[str, dict[str, Any]] = {}
        outcomes: dict[str, dict[str, Any]] = {}
        try:
            for d in db.conn.execute(
                "SELECT * FROM ml_decisions ORDER BY id ASC"
            ).fetchall():
                dd = dict(d)
                if dd.get("snap_id"):
                    decisions[str(dd["snap_id"])] = dd  # latest wins
        except Exception:  # noqa: BLE001
            pass
        try:
            for o in db.conn.execute(
                "SELECT * FROM ml_outcomes ORDER BY id ASC"
            ).fetchall():
                oo = dict(o)
                if oo.get("snap_id"):
                    outcomes[str(oo["snap_id"])] = oo  # latest wins (corrections)
        except Exception:  # noqa: BLE001
            pass
        try:
            snap_rows = db.conn.execute("SELECT * FROM snaps ORDER BY id ASC").fetchall()
        except Exception:  # noqa: BLE001
            snap_rows = []
        covered_sessions: set[str] = set()
        for snap in snap_rows:
            mapping = dict(snap)
            ml_id = mapping.get("ml_snap_id")
            if ml_id and str(ml_id) in decisions:
                dec = decisions[str(ml_id)]
                mapping.setdefault(
                    "recommended_formation",
                    dec.get("heuristic_formation") or dec.get("final_formation"),
                )
                mapping.setdefault(
                    "recommended_play",
                    dec.get("heuristic_play") or dec.get("final_play"),
                )
                mapping.setdefault("game_id", dec.get("game_id") or mapping.get("session_id"))
            if ml_id and str(ml_id) in outcomes:
                outc = outcomes[str(ml_id)]
                mapping["executed_status"] = outc.get("executed_status") or mapping.get(
                    "executed_status"
                )
                mapping["executed_formation"] = outc.get("executed_formation") or mapping.get(
                    "executed_formation"
                )
                mapping["executed_play"] = outc.get("executed_play") or mapping.get(
                    "executed_play"
                )
                mapping["executed_verification"] = outc.get("executed_verification") or mapping.get(
                    "executed_verification"
                )
                if outc.get("outcome_json"):
                    try:
                        payload = json.loads(outc["outcome_json"])
                        # Latest outcome replaces prior result for amended snaps.
                        if payload.get("result") is not None:
                            mapping["result"] = payload.get("result")
                        if payload.get("yards") is not None:
                            mapping["yards"] = payload.get("yards")
                    except (TypeError, json.JSONDecodeError):
                        pass
            mapping["source_path"] = str(getattr(db, "path", "coach.db"))
            mapping["snap_id"] = ml_id or mapping.get("id")
            mapping["ml_snap_id"] = ml_id
            mapping["game_id"] = mapping.get("session_id") or mapping.get("game_id")
            if mapping.get("session_id"):
                covered_sessions.add(str(mapping["session_id"]))
            normalized = _normalize_row(mapping, provenance="madden_db.snaps")
            _merge_or_add(normalized, provenance="madden_db.snaps")

        # play_records only when not already covered by an ML snap identity in that session.
        try:
            play_rows = db.conn.execute("SELECT * FROM play_records ORDER BY id ASC").fetchall()
        except Exception:  # noqa: BLE001
            play_rows = []
        for play in play_rows:
            mapping = dict(play)
            sid = mapping.get("session_id")
            # Prefer ML snaps for sessions that already exported via snaps.ml_snap_id.
            if sid and str(sid) in covered_sessions and mapping.get("ml_snap_id"):
                # Will merge on canonical ml key if present.
                pass
            elif sid and str(sid) in covered_sessions and not mapping.get("ml_snap_id"):
                # Session already has snaps rows — skip legacy play_records without ML id
                # to avoid double-counting the same live game under a different key.
                continue
            payload = {}
            raw_payload = mapping.get("payload_json")
            if raw_payload:
                try:
                    payload = json.loads(raw_payload)
                except (TypeError, json.JSONDecodeError):
                    payload = {}
            mapping.setdefault("formation", mapping.get("formation") or payload.get("formation"))
            mapping.setdefault("play", payload.get("play") or mapping.get("our_call"))
            mapping.setdefault("result", mapping.get("result_type") or payload.get("result"))
            mapping.setdefault("situation_raw", payload.get("situation_raw"))
            mapping["source_path"] = str(getattr(db, "path", "coach.db"))
            if mapping.get("ml_snap_id"):
                mapping["snap_id"] = mapping["ml_snap_id"]
            else:
                mapping["snap_id"] = mapping.get("play_id") or mapping.get("id")
            mapping["game_id"] = mapping.get("session_id") or mapping.get("game_id")
            normalized = _normalize_row(mapping, provenance="madden_db.play_records")
            _merge_or_add(normalized, provenance="madden_db.play_records")

    for path in extra_paths:
        for row in import_path(path):
            _merge_or_add(row, provenance=str(row.get("provenance") or f"file:{path}"))

    if ambiguous:
        # Stash on a sentinel so quality_report can surface them without inventing snaps.
        for row in rows:
            if row.get("ambiguous_matches"):
                break
        else:
            pass
    return rows


def supervised_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rows eligible for play-specific supervised training."""
    return [dict(r) for r in rows if r.get("supervised_eligible") or r.get("eligibility") in SUPERVISED_ELIGIBLE]


def export_jsonl(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    """Write rows as JSONL."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), sort_keys=True, default=str))
            fh.write("\n")


_SANITIZE_DROP = frozenset(
    {
        "source_path",
        "our_call",
        "notes",
        "behavior_propensity",
    }
)
_SANITIZE_REDACT = frozenset({"opponent_id", "session_id", "game_id", "snap_id"})


def export_sanitized(rows: Sequence[Mapping[str, Any]], path: str) -> None:
    """Write a privacy-scrubbed JSONL for debugging (no personal DB paths).

    Opponent / session / snap ids are replaced with stable hashes. Local paths
    and free-text notes are dropped. Does not invent rows.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            scrubbed: dict[str, Any] = {}
            for key, value in dict(row).items():
                if key in _SANITIZE_DROP:
                    continue
                if key in _SANITIZE_REDACT and value is not None:
                    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:12]
                    scrubbed[key] = f"anon:{digest}"
                else:
                    scrubbed[key] = value
            scrubbed["sanitized"] = True
            fh.write(json.dumps(scrubbed, sort_keys=True, default=str))
            fh.write("\n")


def quality_report(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Human-facing data-quality summary for Franchise workflows."""
    base = validate_rows(rows)
    supervised = supervised_rows(rows)
    outcome_only = sum(
        1
        for r in rows
        if str(r.get("eligibility")) == ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC
    )
    unknown_exec = sum(
        1
        for r in rows
        if str(r.get("executed_status") or "") != ExecutedStatus.IDENTIFIED.value
    )
    return {
        **base,
        "total_snaps": base["n_rows"],
        "unique_games": base["unique_games"],
        "verified_executions": base["verified_executions"],
        "verified_labeled_examples": base["verified_labeled_supervised"],
        "outcome_only_examples": outcome_only,
        "unknown_executions": unknown_exec,
        "cpu_vs_human": {"cpu": base["cpu_rows"], "human": base["human_rows"]},
        "offense_vs_defense": {
            "offense": base["offense_rows"],
            "defense": base["defense_rows"],
        },
        "coverage_label_quality": {
            "rows_with_coverage": base["coverage_label_rows"],
            "pct": round(
                100.0 * base["coverage_label_rows"] / base["n_rows"], 1
            )
            if base["n_rows"]
            else 0.0,
        },
        "duplicates": base["duplicates"],
        "invalid_or_excluded": base["eligibility"].get(ELIGIBILITY_EXCLUDED, 0),
        "supervised_training_rows": len(supervised),
    }


def import_path(path: str) -> list[dict[str, Any]]:
    """Read CSV, JSON, JSONL, or SQLite into row dicts.

    Older files with missing columns stay readable. Absent fields are unknown.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _import_csv(p)
    if suffix == ".jsonl":
        return _import_jsonl(p)
    if suffix == ".json":
        return _import_json(p)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return _import_sqlite(p)
    # Peek at content for extensionless dumps.
    text = p.read_text(encoding="utf-8", errors="replace")
    if text.lstrip().startswith("{"):
        return _import_json(p)
    if "\t" in text.splitlines()[0] if text else False:
        return _import_csv(p)
    return _import_jsonl(p)


def _import_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            cleaned = {
                k: (None if v == CSV_UNKNOWN or v == "" else v)
                for k, v in raw.items()
            }
            cleaned["source_path"] = str(path)
            rows.append(_normalize_row(cleaned, provenance=f"csv:{path.name}"))
    return rows


def _import_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                continue
            raw["source_path"] = str(path)
            rows.append(_normalize_row(raw, provenance=f"jsonl:{path.name}"))
    return rows


def _import_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "rows" in payload:
        payload = payload["rows"]
    if not isinstance(payload, list):
        raise ValueError(f"{path}: expected a list of rows or {{'rows': [...]}}")
    rows: list[dict[str, Any]] = []
    for raw in payload:
        if not isinstance(raw, dict):
            continue
        raw = dict(raw)
        raw["source_path"] = str(path)
        rows.append(_normalize_row(raw, provenance=f"json:{path.name}"))
    return rows


def _import_sqlite(path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        rows: list[dict[str, Any]] = []
        if "snaps" in names:
            for snap in conn.execute("SELECT * FROM snaps ORDER BY id ASC"):
                mapping = dict(snap)
                mapping["source_path"] = str(path)
                rows.append(_normalize_row(mapping, provenance=f"sqlite:{path.name}:snaps"))
        return rows
    finally:
        conn.close()


def row_content_hash(row: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(row), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
