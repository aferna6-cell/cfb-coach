"""Testing-franchise-only live bridge for a learned defensive V2 shadow artifact.

It never promotes an artifact, writes weights, changes a playbook, or activates
itself. The caller must explicitly supply the test DB and franchise on every
session. If any identity or time gate fails, preserve the existing caller.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import defense_v2_pilot as v2

LATENCY_BUDGET_MS = 150.0


def _test_identity(db: Any, *, test_db: str, franchise: str,
                   session: str, opponent: str) -> str:
    if db is None or not test_db or not franchise or not session or not opponent:
        raise ValueError("explicit test DB, franchise, session and opponent are required")
    path = Path(str(db.path)).expanduser().resolve()
    provided = Path(test_db).expanduser().resolve()
    if path != provided or not provided.is_file():
        raise ValueError("test DB does not match the active database")
    live_directory = (Path.home() / ".cfb-coach").resolve()
    if path == live_directory or live_directory in path.parents:
        raise ValueError("V2 testing is prohibited in the default live data directory")
    row = db.conn.execute(
        "SELECT dynasty, opponent_id, started_ts FROM game_sessions WHERE session_id=?",
        (session,),
    ).fetchone()
    if row is None or str(row[0]) != franchise or str(row[1]) != opponent:
        raise ValueError("test game session / opponent / franchise mismatch")
    if not row[2]:
        raise ValueError("missing session start cutoff")
    return str(row[2])


@lru_cache(maxsize=8)
def _cached_model(path: str, mtime_ns: int, size: int) -> Mapping[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != v2.SCHEMA or data.get("policy") != "shadow_only":
        raise ValueError("unsupported defensive V2 artifact")
    if not isinstance(data.get("trained_sessions"), list):
        raise ValueError("artifact has no explicit training-session provenance")
    if not data.get("trained_through_started_ts"):
        raise ValueError("artifact lacks a chronology cutoff")
    if data.get("target") != "offense_success" or not isinstance(data.get("global"), dict):
        raise ValueError("wrong model target or malformed coefficients")
    return data


def load_model(path: str | Path) -> Mapping[str, Any]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise ValueError("V2 JSON model file not found")
    stat = p.stat()
    return _cached_model(str(p), stat.st_mtime_ns, stat.st_size)


def _fallback(call: Any, why: str) -> Any:
    # An error preserves the original caller, not an uninstalled guessed call.
    call.ml_defense_fallback = {
        "fell_back": True, "fallback_source": "select_defense",
        "fallback_reason": "defense_v2_" + why,
    }
    call.defense_v2 = {"applied": False, "reason": why, "model": v2.SCHEMA}
    return call


def maybe_apply_v2(
    call: Any, sit: Any, opponent_id: str, db: Any,
    book: Mapping[str, Sequence[str]], *,
    model_path: str, test_db: str, franchise: str,
    action_confirmations: str | None = None,
) -> Any:
    """Replace the real displayed defensive call with V2's chosen installed play.

    The surrounding playcaller invokes this only for --defense-model v2. Fails
    closed to the original call on incomplete private data or any runtime error.
    """
    start = time.perf_counter()
    try:
        if getattr(sit, "side", None) != "defense":
            raise ValueError("not defense")
        extras = getattr(sit, "extras", None) or {}
        session = str(extras.get("session_id") or "")
        seq = int(extras.get("snap_seq") or 0)
        if seq <= 0:
            raise ValueError("positive pre-snap snap_seq is required")
        cutoff = _test_identity(db, test_db=test_db, franchise=franchise,
                                session=session, opponent=opponent_id)
        model = load_model(model_path)
        if model.get("franchise") != franchise:
            raise ValueError("artifact was not trained for this franchise")
        if session in model["trained_sessions"] or cutoff <= str(model["trained_through_started_ts"]):
            raise ValueError("artifact includes current or later game")
        # Fail closed when no verified learned outcomes exist.
        if not int(model.get("n_verified") or 0):
            raise ValueError("no verified defensive training data")
        installed_calls = [
            {"formation": form, "play": play, "installed": True, "legal": True}
            for form, plays in book.items() for play in plays
        ]
        if not installed_calls or len(book) > 15:
            raise ValueError("empty book or more than 15 installed formations")
        from cfb_coach.madden.defense_select import eligible_formations
        eligible = set(eligible_formations(sit, {f: list(p) for f, p in book.items()}))
        installed_calls = [r for r in installed_calls if r["formation"] in eligible]
        history = v2.earlier_opponent_observations(
            db.conn, franchise=franchise, session=session,
            opponent=opponent_id, next_seq=seq,
        )
        belief = v2.concept_belief(
            model, history, session=session, opponent=opponent_id, next_seq=seq,
        )
        context = v2.context_key(getattr(sit, "down", None), getattr(sit, "distance", None))
        ranking = v2.rank(model, context=context, calls=installed_calls, belief=belief)
        if not ranking["ranked"]:
            raise ValueError("no supported installed defensive calls")
        best = ranking["ranked"][0]
        from cfb_coach.madden.model import defense_v2_actions as actions
        # No physically verified controller options => NO ADJUSTMENT.
        # Learned exact-combination outcomes are saved in the model only after
        # confirmed execution, never taken from this current game.
        confirmations = actions.read_confirmed_options(action_confirmations)
        plan = actions.choose_adjustments(
            model, context=context, formation=best["formation"], play=best["play"],
            stock_offense_success=float(best["predicted_offense_success"]),
            confirmed=confirmations,
        )
        if (time.perf_counter() - start) * 1000 > LATENCY_BUDGET_MS:
            raise TimeoutError("150 ms model decision budget exceeded")
        from cfb_coach.madden.playcaller import MaddenCall
        from cfb_coach.madden.model import defense_semantics
        from cfb_coach.madden.model.defense_coordinator import _model_decision
        from cfb_coach.madden.model.schema import PropensityMethod

        resolution = defense_semantics.resolve(best["play"], best["formation"])
        assignment = ("known catalog label, defender-level jobs NOT verified"
                      if resolution.key else "coverage and defender assignments UNKNOWN")
        detail = {
            "applied": True, "model": v2.SCHEMA,
            "selected": {"formation": best["formation"], "play": best["play"]},
            "predicted_offense_success": best["predicted_offense_success"],
            "support": best["support"], "opponent_belief": belief,
            "earlier_opponent_concepts": len(history),
            "coverage_resolution": resolution.resolution,
            "defender_assignments": "UNKNOWN",
            "evidence": "verified observational defensive executions",
            "no_adjustment": not bool(plan["actions"]),
            "adjustment_id": plan["id"],
            "adjustment_evidence_snaps": plan["effect_support"],
            "latency_ms": round((time.perf_counter() - start) * 1000.0, 3),
        }
        out = MaddenCall(
            "defense", best["formation"], best["play"],
            "No adj" if not plan["actions"] else "Individual adjustments",
            "Check in-game player assignments",
            (f"DEFENSE V2 experimental ML; estimated offense success "
             f"{best['predicted_offense_success']:.3f}; {assignment}; "
             f"{len(history)} earlier offensive concepts; "
             + ("NO ADJUSTMENT" if not plan["actions"] else
                f"verified individual actions {plan['id']} (observational benefit)")),
        )
        if plan["actions"]:
            # The Xbox button sequences come only from an owner-confirmed file;
            # never invent per-player sub-menu options or unverified combos.
            out.adjustment = {
                "id": plan["id"],
                "label": " + ".join(a["label"] for a in plan["actions"]),
                "buttons": " ; let players get set ; ".join(a["buttons"] for a in plan["actions"]),
                "why": ("verified execution evidence; conservative learned estimate, "
                        f"n={plan['effect_support']} (not causal)"),
            }
        out.defense_v2 = detail
        # Reuse the existing decision logging schema and attribution path.
        from cfb_coach.madden.model.defense_coordinator import _stamp_decision
        decision = _model_decision(
            book=book, formation=best["formation"], play=best["play"],
            method=PropensityMethod.DETERMINISTIC, propensity=1.0, reason=None,
            session_id=session, snap_seq=seq,
            elapsed_ms=detail["latency_ms"],
            heuristic=None,
            snap_id=extras.get("snap_id"), game_id=extras.get("game_id"),
        )
        if decision is not None:
            decision = replace(decision, policy_version=v2.SCHEMA)
        _stamp_decision(out, decision, db, extras)
        return out
    except TimeoutError:
        return _fallback(call, "timeout")
    except (OSError, ValueError, TypeError, KeyError, IndexError, json.JSONDecodeError):
        return _fallback(call, "invalid_or_unavailable")
    except Exception:
        return _fallback(call, "exception")
