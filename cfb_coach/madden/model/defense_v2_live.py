"""Opt-in defensive V2 live bridge for an isolated testing franchise.

This bridge intentionally has NO action recommendations: verified individual
defensive adjustment effects are not trained yet. Never infer Xbox controls.
The explicit installed book and per-game identity are hard requirements.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import defense_v2_pilot as v2

MODE_ENV = "CFB_COACH_DEFENSE_V2_TEST_FRANCHISE"
ARTIFACT_ENV = "CFB_COACH_DEFENSE_V2_ARTIFACT"
LATENCY_MS = 150.0


def requested(sit: Any) -> bool:
    extras = getattr(sit, "extras", None) or {}
    return bool(
        isinstance(extras, Mapping)
        and extras.get("defense_model") == "v2"
        and str(os.environ.get(MODE_ENV) or "").strip()
        and str(os.environ.get(ARTIFACT_ENV) or "").strip()
    )


def _identity(conn: Any, session: str, opponent: str, franchise: str) -> str:
    rows = conn.execute(
        "SELECT dynasty,started_ts FROM game_sessions "
        "WHERE session_id=? AND opponent_id=? AND dynasty=?",
        (session, opponent, franchise),
    ).fetchall()
    if len(rows) != 1 or not rows[0][1]:
        raise ValueError("V2: test franchise/session/opponent missing or ambiguous")
    return str(rows[0][1])


def _model(path: str, game_started: str, session: str) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema") != v2.SCHEMA or raw.get("policy") != "shadow_only":
        raise ValueError("V2 artifact is not a supported research artifact")
    if not raw.get("n_verified") or not raw.get("n_games"):
        raise ValueError("V2 artifact lacks verified defensive training evidence")
    if "trained_sessions" not in raw or "trained_through_started_ts" not in raw:
        raise ValueError("V2 artifact lacks a strict training cutoff")
    if session in raw["trained_sessions"] or game_started <= raw["trained_through_started_ts"]:
        raise ValueError("V2 artifact was trained on the current game or future data")
    for key in ("global", "contexts", "families", "matchups", "plays", "concept_counts"):
        if not isinstance(raw.get(key), dict):
            raise ValueError(f"V2 artifact missing {key}")
    return raw


def _confirmed_book(db: Any, book: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    from cfb_coach.madden import playbook
    from cfb_coach.madden.defense_select import call_family
    stored = (playbook.load_books(db).get("defense") or {})
    formations = stored.get("formations")
    if not isinstance(formations, dict) or not stored.get("locked_ts"):
        raise ValueError("V2 needs a confirmed installed defensive book")
    normalized = {str(k): list(v) for k, v in book.items()}
    if not normalized or len(normalized) > 15 or normalized != formations:
        raise ValueError("V2 requires exact installed defensive inventory and <=15 formations")
    calls = []
    for form, plays in normalized.items():
        if not form or not plays or len(plays) != len(set(plays)):
            raise ValueError("invalid installed formation/plays")
        for play in plays:
            if call_family(play):
                calls.append({"formation": form, "play": play, "installed": True,
                              "legal": True})  # Stock plays only; no adjustments.
    if not calls:
        raise ValueError("no recognizable defensive calls in confirmed book")
    return calls


def _structure(form: str, play: str) -> dict[str, Any]:
    from cfb_coach.madden.model.defense_semantics import explain
    result = explain(play, form)
    return {
        "resolution": result.get("resolution"),
        "known": result.get("known") or [],
        "hypotheses": result.get("hypotheses") or [],
        "unknown": result.get("unknown") or [],
        "verification_level": result.get("verification_level") or "UNKNOWN",
    }


def try_live(call: Any, sit: Any, opponent: str, db: Any,
             book: Mapping[str, Sequence[str]]) -> Any:
    """Return a V2-selected MaddenCall or raise; caller catches & preserves its fallback.

    This is a controlled experimental model, NOT an approved release.
    No side effect if ineligible; DB reads are time-fenced.
    """
    if not requested(sit):
        return call
    from cfb_coach.madden.model import defense_coordinator as old
    from cfb_coach.madden.playcaller import MaddenCall
    from cfb_coach.madden.data import user_job_for
    from cfb_coach.madden.defense_select import FAMILY_USER_JOB
    from cfb_coach.madden.model.schema import PropensityMethod

    started = time.perf_counter()
    extras = getattr(sit, "extras", None) or {}
    franchise = str(os.environ[MODE_ENV]).strip()
    session = str(extras.get("session_id") or "")
    seq = int(extras.get("snap_seq") or 0)
    if not session or seq <= 0:
        raise ValueError("V2 requires real session ID and positive next snap sequence")
    game_started = _identity(db.conn, session, opponent, franchise)
    model = _model(os.environ[ARTIFACT_ENV], game_started, session)
    calls = _confirmed_book(db, book)
    history = v2.earlier_opponent_observations(
        db.conn, franchise=franchise, session=session, opponent=opponent, next_seq=seq)
    belief = v2.concept_belief(
        model, history, session=session, opponent=opponent, next_seq=seq)
    context = v2.context_key(getattr(sit, "down", None), getattr(sit, "distance", None))
    ranked = v2.rank(model, context=context, calls=calls, belief=belief)["ranked"]
    if not ranked:
        raise ValueError("V2 model ranked no installed defensive calls")
    winner = ranked[0]
    if (time.perf_counter() - started) * 1000.0 > LATENCY_MS:
        raise TimeoutError("V2 exhausted 150ms budget")
    formation, play, family = winner["formation"], winner["play"], winner["family"]
    detail = _structure(formation, play)
    unknown = "assignments UNKNOWN" if detail["unknown"] else "role catalog available"
    summary = ", ".join(detail["known"][:2] or detail["hypotheses"][:2])
    note = (
        f"DEFENSE V2 EXPERIMENTAL [{model['n_verified']} verified snaps] "
        f"expected offense success {winner['predicted_offense_success']:.3f}; "
        f"opponent concept {max(belief,key=belief.get)} (belief, NOT observed pre-snap); "
        f"{summary or 'structure unverified'}; {unknown}. "
        "NO ADJUSTMENT: effect not learned or physically verified."
    )
    result = MaddenCall("defense", formation, play, "No adj",
                        user_job_for(play) or FAMILY_USER_JOB.get(family, "User hook"),
                        note, macro=None, macro_info=None, adjustment=None)
    elapsed = (time.perf_counter() - started) * 1000.0
    if elapsed > LATENCY_MS:
        raise TimeoutError("V2 total selection exceeded 150ms")
    decision = old._model_decision(
        book=book, formation=formation, play=play,
        method=PropensityMethod.DETERMINISTIC, propensity=1.0, reason=None,
        session_id=session, snap_seq=seq, elapsed_ms=elapsed,
        heuristic=old._book_candidate(book, getattr(call, "formation", None),
                                      getattr(call, "play", None)),
        snap_id=extras.get("snap_id"), game_id=extras.get("game_id"),
    )
    from dataclasses import replace
    decision = replace(decision, policy_version=v2.SCHEMA,
                       ml_status_detail="V2 test franchise only; empirical observational success model")
    old._stamp_decision(result, decision, db, extras)
    result.ml_defense_v2 = {
        "applied": True, "experimental": True, "model_version": v2.SCHEMA,
        "franchise": franchise, "session_id": session, "snap_seq": seq,
        "n_training_games": model["n_games"], "n_verified": model["n_verified"],
        "prediction": winner["predicted_offense_success"],
        "concept_belief": belief, "knowledge": detail,
        "adjustment": "NO_ADJUSTMENT",
        "latency_ms": round(elapsed, 3),
        "installed_inventory_fingerprint": hashlib.sha256(
            json.dumps({k:list(v) for k,v in book.items()},sort_keys=True).encode()
        ).hexdigest(),
    }
    return result
