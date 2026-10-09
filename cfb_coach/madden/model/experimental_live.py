"""Opt-in experimental ML offense selection with a sealed rebuild path.

Disabled by default. When ``ml_mode=experimental``:

1. Heuristic (incl. VOD) still produces a candidate.
2. The model ranks only *situationally eligible* in-book offense plays
   (reuses heuristic pool: zone fit, short yardage, 3rd-and-long, two-minute).
3. If the model pick differs, reads / Custom Adjustments are rebuilt for that
   play — never a post-hoc display swap of the heuristic's reads.
4. On timeout, error, illegal candidate, or missing artifact → heuristic.

Decision logging is deferred until snap identity is sealed
(:func:`commit_experimental_decision`) so every row has a real game/snap id.

Defense is never selected by this mode. CPU remains offense-only.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cfb_coach.madden.model import experimental_model as exp_mod
from cfb_coach.madden.model.schema import (
    ML_LATENCY_BUDGET_MS,
    CoachingDecision,
    CoachingMode,
    MLStatus,
    PolicySource,
    PropensityMethod,
    ShadowScore,
    Tri,
    candidate_in_book,
)
from cfb_coach.opponents import is_cpu_opponent

META_ARTIFACT = "ml_experimental_artifact"
META_KNOWLEDGE = "ml_knowledge_version"
META_DATA_VERSION = "ml_data_version"
PENDING_ATTR = "_pending_ml_decision"


def resolve_artifact_path(db: Any = None) -> Path | None:
    if db is not None:
        try:
            raw = db.get_meta(META_ARTIFACT)
            if raw:
                p = Path(str(raw))
                if p.is_file():
                    return p
        except Exception:  # noqa: BLE001
            pass
    from cfb_coach.games import data_dir

    default = Path(data_dir()) / "madden_ml_experimental" / "offense.json"
    return default if default.is_file() else None


def set_artifact_path(db: Any, path: str | Path) -> None:
    db.set_meta(META_ARTIFACT, str(path))


def situational_offense_candidates(
    sit: Any,
    book: dict[str, list[str]],
    *,
    db: Any = None,
    opponent_id: str | None = None,
) -> tuple[list[tuple[str, str]], dict[tuple[str, str], float]]:
    """Football-eligible in-book candidates — not the full installed menu.

    Reuses heuristic ``_offense_pool`` (zone fit, short yardage, 3rd-long,
    two-minute) and soft anti-repeat penalties. ML may only override among
    these appropriately eligible plays.
    """
    from cfb_coach.madden.data import load_meta_baseline
    from cfb_coach.madden.playcaller import _offense_pool, situation_key

    og = load_meta_baseline()["offense_gameplan"]
    key = situation_key(sit)
    arch_pref = None
    pool, bonus = _offense_pool(sit, book, og, key, arch_pref)
    if not pool:
        pool = [(f, p) for f, ps in (book or {}).items() for p in (ps or [])]
        bonus = {}
    # Explicit situational guard: the heuristic's -0.20 run bonus was ignored
    # by model-primary inference, making 3rd-and-long dives score as normal.
    # Keep every available PASS play eligible and let ML rank them.
    from cfb_coach.madden.catalog import is_run

    try:
        down, distance = int(getattr(sit, "down")), int(getattr(sit, "distance"))
    except (ValueError, TypeError):
        down, distance = 0, 0
    if down in (3, 4) and distance >= 7 and not getattr(sit, "goal_line", False):
        passes = [(form, play) for form, play in pool if not is_run(play)]
        if passes:
            pool = passes
    # Soft anti-repeat: down-weight recently used plays (still eligible).
    if db is not None and opponent_id:
        try:
            from cfb_coach.gameplan import anti_repeat_penalty

            for form, play in list(pool):
                pen = float(anti_repeat_penalty(db, opponent_id, form, play, side="offense") or 0.0)
                if pen:
                    bonus[(form, play)] = round(bonus.get((form, play), 0.0) - 0.05 * pen, 3)
        except Exception:  # noqa: BLE001
            pass
    return pool, bonus


def rebuild_offense_attachments(
    *,
    formation: str,
    play: str,
    sit: Any,
    book: dict[str, list[str]],
    active: Sequence[str],
    db: Any,
    opponent_id: str,
    rationale: str,
    audibles: dict[str, list[str]] | None = None,
    model_action_policy: bool = False,
    play_prediction: dict[str, Any] | None = None,
) -> Any:
    """Build reads / macro / adjustment for an already-chosen in-book play."""
    from cfb_coach.madden.data import reads_for
    from cfb_coach.madden.playcaller import MaddenCall, coverage_class
    from cfb_coach.tendency import is_repeated_coverage

    cov = getattr(sit, "coverage_hint", None)
    src = getattr(sit, "coverage_source", None) or "none"
    cls = coverage_class(cov) if cov else None
    # Repeated confirmation only when live (or tendency engine) says so —
    # last-snap alone is not confirmation.
    repeated = False
    if cov and src == "live":
        repeated = is_repeated_coverage(db, opponent_id, cov, sit, threshold=2)
    zone = (
        "gl"
        if getattr(sit, "goal_line", False)
        else "rz"
        if getattr(sit, "red_zone", False)
        else "open"
    )
    score_phase = None
    try:
        from cfb_coach.game_score import classify

        ctx = classify(sit)
        score_phase = ctx.phase if ctx else None
    except Exception:  # noqa: BLE001
        score_phase = None

    try:
        from cfb_coach.madden.playcaller import _cooled_macros, _learned_macros

        weights = _learned_macros(db, opponent_id)
        cooled = _cooled_macros(db, sit)
    except Exception:  # noqa: BLE001
        weights = {}
        cooled = set()

    if model_action_policy:
        from cfb_coach.madden.model.offense_action_policy import choose_offense_action

        decision = choose_offense_action(
            formation=formation, play=play, sit=sit, book=book,
            active=active, weights=weights, cooled=cooled,
            score_phase=score_phase, audibles=audibles,
            prediction=play_prediction,
            allow_macros=(getattr(sit, "extras", None) or {}).get(
                "live_macros", True
            ) is not False,
            repeated=repeated,
            db=db,
            opponent_id=opponent_id,
        )
        chosen_macro = decision.get("macro")
        chosen_adj = decision.get("adjustment")
        call = MaddenCall(
            "offense", formation, play,
            chosen_adj["label"] if chosen_adj else "No adj",
            reads_for(play),
            rationale + f" | ACTION {decision['kind']}: {decision['reason']}",
            macro=chosen_macro["id"] if chosen_macro else None,
            macro_info=chosen_macro,
            adjustment=chosen_adj,
        )
        call.ml_offense_action = {
            key: value for key, value in decision.items()
            if key not in ("macro", "adjustment")
        }
        return call

    adj = "Hot ready" if cls == "pressure" and src == "live" else "No adj"
    adjustment = None
    macro = None
    info = None
    try:
        from cfb_coach.madden.offense_macros import suggest_for_snap

        info = suggest_for_snap(
            zone=zone,
            play=play,
            coverage=cov,
            coverage_source=src,
            active=list(active),
            down=getattr(sit, "down", None),
            repeated=repeated,
            book=book,
            weights=weights,
            score_phase=score_phase,
            cooled=cooled,
        )
    except Exception:  # noqa: BLE001
        info = None
    withheld = False
    if info and (getattr(sit, "extras", None) or {}).get("live_macros", True) is False:
        info = None
        withheld = True
    if info:
        macro = info["id"]
        adj = "No adj"
        rationale = f"{rationale} | MACRO {info['name']}: {info['why']}"
    elif not withheld:
        try:
            from cfb_coach.madden.adjustments import offense_adjustment

            adjustment = offense_adjustment(
                play=play,
                formation=formation,
                coverage_class=cls,
                coverage_source=src,
                repeated=repeated,
                audibles=audibles,
            )
        except Exception:  # noqa: BLE001
            adjustment = None
        if adjustment:
            adj = adjustment["label"]
            rationale = f"{rationale} | adj {adjustment['label']}: {adjustment['why']}"

    return MaddenCall(
        "offense",
        formation,
        play,
        adj,
        reads_for(play),
        rationale,
        macro=macro,
        macro_info=info,
        adjustment=adjustment,
    )


def apply_experimental_offense(
    *,
    heuristic_call: Any,
    sit: Any,
    opponent_id: str,
    db: Any,
    book: dict[str, list[str]],
    active: Sequence[str] | None = None,
    audibles: dict[str, list[str]] | None = None,
    game_id: str | None = None,
    snap_id: str | None = None,
    session_id: str | None = None,
    snap_seq: int | None = None,
    budget_ms: float | None = None,
    commit: bool | None = None,
) -> tuple[Any, CoachingDecision]:
    """Maybe replace the offense call with an ML pick.

    When ``commit`` is False (default if ``snap_id`` is missing), the decision
    is attached to the call for later :func:`commit_experimental_decision`
    after seal — never an anonymous DB row.
    """
    from cfb_coach.madden.model.schema import Possession

    decision_ts = datetime.now(timezone.utc).isoformat()
    budget = float(ML_LATENCY_BUDGET_MS if budget_ms is None else budget_ms)
    if commit is None:
        commit = bool(snap_id)
    heur_form = getattr(heuristic_call, "formation", None)
    heur_play = getattr(heuristic_call, "play", None)
    heur_cand = None
    try:
        if heur_form and heur_play:
            heur_cand = candidate_in_book(
                book, side=Possession.OFFENSE, formation=heur_form, play=heur_play
            )
    except ValueError:
        heur_cand = None

    def _attach(call: Any, dec: CoachingDecision, info: dict[str, Any] | None = None) -> None:
        try:
            setattr(call, PENDING_ATTR, dec)
        except Exception:  # noqa: BLE001
            pass
        if info is not None:
            try:
                call.ml_experimental = info  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass

    def _fallback(status: MLStatus, latency: float | None = None) -> tuple[Any, CoachingDecision]:
        dec = CoachingDecision(
            decision_ts=decision_ts,
            latency_ms=latency,
            latency_ms_model=latency,
            inference_budget_ms=ML_LATENCY_BUDGET_MS,
            mode=CoachingMode.EXPERIMENTAL,
            effective_mode=CoachingMode.HEURISTIC,
            mode_downgrade_reason=f"experimental_fallback:{status.value}",
            game_id=game_id,
            snap_id=snap_id,
            session_id=session_id or game_id,
            snap_seq=snap_seq,
            policy_source=PolicySource.HEURISTIC,
            heuristic_pick=heur_cand,
            final_pick=heur_cand,
            shadow_pick=None,
            shadow_status=status,
            fell_back=True,
            fallback_reason=status,
            propensity_method=PropensityMethod.UNKNOWN,
            accepted=Tri.UNKNOWN,
        )
        info = {
            "heuristic_formation": heur_form,
            "heuristic_play": heur_play,
            "ml_formation": None,
            "ml_play": None,
            "final_formation": heur_form,
            "final_play": heur_play,
            "evidence_quality": "fallback",
            "explanation": f"experimental fallback → heuristic ({status.value})",
            "fell_back": True,
        }
        _attach(heuristic_call, dec, info)
        if commit and db is not None and snap_id:
            try:
                db.log_ml_decision(dec, agree=None)
            except Exception:  # noqa: BLE001
                pass
        return heuristic_call, dec

    if str(getattr(heuristic_call, "side", "offense") or "offense").startswith("d"):
        return _fallback(MLStatus.CIRCUIT_OPEN)

    pairs, bonuses = situational_offense_candidates(
        sit, book, db=db, opponent_id=opponent_id
    )
    # Model-primary: heuristic is only the fallback, never an extra candidate.
    if not pairs:
        return _fallback(MLStatus.INVALID_OUTPUT)

    artifact_path = resolve_artifact_path(db)
    started = time.perf_counter()
    try:
        if artifact_path is None:
            from cfb_coach.madden.model import dataset as dataset_mod
            from cfb_coach.games import data_dir

            rows = dataset_mod.build_rows(db=db) if db is not None else []
            art = exp_mod.train_experimental(rows, side="offense")
            dest = Path(data_dir()) / "madden_ml_experimental" / "offense.json"
            exp_mod.save_artifact(art, dest)
            if db is not None:
                set_artifact_path(db, dest)
                db.set_meta(META_KNOWLEDGE, art.knowledge_version)
                db.set_meta(META_DATA_VERSION, art.data_version)
            artifact = art
        else:
            artifact = exp_mod.load_artifact(artifact_path)

        elapsed = (time.perf_counter() - started) * 1000.0
        if elapsed > budget:
            return _fallback(MLStatus.TIMEOUT, elapsed)

        # Pre-snap coverage handling: live = observed; last = soft prior only.
        cov_hint = getattr(sit, "coverage_hint", None)
        cov_src = getattr(sit, "coverage_source", None) or "none"
        if cov_src not in ("live", "last"):
            cov_hint = None
            cov_src = "none"

        opp_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
        ranked = exp_mod.rank_candidates(
            artifact,
            pairs,
            down=getattr(sit, "down", None),
            distance=getattr(sit, "distance", None),
            yardline=getattr(sit, "yardline", None),
            coverage_hint=cov_hint,
            coverage_source=cov_src,
            opponent_id=opponent_id,
            opponent_type=opp_type,
            heuristic=None,
            heuristic_bonuses={},
            near_tie_margin=0.0,
        )
        latency = (time.perf_counter() - started) * 1000.0
        if latency > budget:
            return _fallback(MLStatus.TIMEOUT, latency)
        if not ranked:
            return _fallback(MLStatus.INVALID_OUTPUT, latency)

        # Repeated calls are observable even when the executed play has not
        # been verified. Keep the model primary, but penalize repeated play /
        # screen exposure before choosing its next in-book candidate.
        from cfb_coach.madden.model.offense_selection_policy import select_from_database

        ranked, selection_audit = select_from_database(
            ranked, db=db, opponent_id=opponent_id,
            sit=sit, opponent_type=opp_type,
            session_id=session_id or game_id, snap_seq=snap_seq,
        )
        latency = (time.perf_counter() - started) * 1000.0
        if latency > budget:
            return _fallback(MLStatus.TIMEOUT, latency)
        top = ranked[0]
        ml_form, ml_play = top["formation"], top["play"]
        # Hard guard: ML pick must be in situational pool (or the heuristic).
        legal_set = set(pairs)
        if (ml_form, ml_play) not in legal_set:
            return _fallback(MLStatus.INVALID_OUTPUT, latency)
        try:
            ml_cand = candidate_in_book(
                book, side=Possession.OFFENSE, formation=ml_form, play=ml_play
            )
        except ValueError:
            return _fallback(MLStatus.INVALID_OUTPUT, latency)

        shadows = tuple(
            ShadowScore(
                candidate_key=f"{r['formation']}|{r['play']}|unknown",
                score=float(r["probability"]),
                p=float(r["probability"]),
            )
            for r in ranked[:12]
        )

        quality = str(top.get("evidence_quality") or artifact.evidence_quality)
        prior_tag = "prior-driven" if quality == "prior_driven" else quality
        unc = top.get("uncertainty")
        agree = int(ml_form == heur_form and ml_play == heur_play)
        explanation = (
            f"ML experimental [{prior_tag}] → {ml_form}/{ml_play} "
            f"(p={top['probability']:.3f}, concept={top.get('play_concept')}, "
            f"unc={unc})"
        )
        if heur_form and heur_play:
            explanation = f"heuristic={heur_form}/{heur_play} | {explanation}"
        if top.get("near_tie_kept_heuristic"):
            explanation += " | near-tie kept heuristic"
        if cov_src == "last" and cov_hint:
            explanation += " | last-snap coverage used as soft prior only"

        # Always build the model-selected call, even when the play name agrees
        # with the heuristic. Accessories must never be inherited by accident.
        call = rebuild_offense_attachments(
            formation=ml_form,
            play=ml_play,
            sit=sit,
            book=book,
            active=list(active or []),
            db=db,
            opponent_id=opponent_id,
            rationale=f"{explanation} | model-primary reconstructed call",
            audibles=audibles,
            model_action_policy=True,
            play_prediction=top,
        )
        latency = (time.perf_counter() - started) * 1000.0
        if latency > budget:
            return _fallback(MLStatus.TIMEOUT, latency)

        info = {
            "heuristic_formation": heur_form,
            "heuristic_play": heur_play,
            "ml_formation": ml_form,
            "ml_play": ml_play,
            "final_formation": call.formation,
            "final_play": call.play,
            "evidence_quality": quality,
            "explanation": explanation,
            "model_version": artifact.model_version,
            "knowledge_version": artifact.knowledge_version,
            "knowledge_origin": artifact.knowledge_origin,
            "data_version": artifact.data_version,
            "probability": top["probability"],
            "uncertainty": unc,
            "rankings": ranked[:8],
            "fell_back": False,
            "n_eligible_candidates": len(pairs),
            "selection_policy": "model_primary_contextual_variety.v2",
            "selection_audit": selection_audit,
            "offense_action": getattr(call, "ml_offense_action", None),
        }

        dec = CoachingDecision(
            decision_ts=decision_ts,
            latency_ms=latency,
            latency_ms_model=latency,
            inference_budget_ms=ML_LATENCY_BUDGET_MS,
            mode=CoachingMode.EXPERIMENTAL,
            effective_mode=CoachingMode.EXPERIMENTAL,
            game_id=game_id,
            snap_id=snap_id,
            session_id=session_id or game_id,
            snap_seq=snap_seq,
            model_version=artifact.model_version,
            policy_version=exp_mod.MODEL_KIND,
            policy_source=PolicySource.MODEL,
            candidates=tuple(c for c in (heur_cand, ml_cand) if c is not None),
            heuristic_pick=heur_cand,
            final_pick=ml_cand,
            shadow_pick=ml_cand,
            shadow_scores=shadows,
            shadow_status=MLStatus.OK,
            fell_back=False,
            propensity_method=PropensityMethod.DETERMINISTIC,
            accepted=Tri.UNKNOWN,
        )
        _attach(call, dec, info)
        if commit and db is not None and snap_id:
            try:
                db.log_ml_decision(dec, agree=agree)
            except Exception:  # noqa: BLE001
                pass
        return call, dec
    except Exception:  # noqa: BLE001
        traceback.format_exc()
        latency = (time.perf_counter() - started) * 1000.0
        return _fallback(MLStatus.EXCEPTION, latency)


def commit_experimental_decision(
    db: Any,
    call: Any,
    *,
    game_id: str,
    snap_id: str,
    snap_seq: int,
    session_id: str | None = None,
) -> int | None:
    """Persist the pending experimental decision with sealed snap identity.

    Idempotent on ``snap_id``. Returns ``ml_decisions.id`` or None.
    """
    dec = getattr(call, PENDING_ATTR, None)
    if dec is None or not isinstance(dec, CoachingDecision):
        return None
    if dec.mode is not CoachingMode.EXPERIMENTAL:
        return None
    stamped = replace(
        dec,
        game_id=game_id,
        snap_id=snap_id,
        snap_seq=snap_seq,
        session_id=session_id or game_id,
    )
    agree = None
    if stamped.heuristic_pick and stamped.shadow_pick:
        agree = int(
            stamped.heuristic_pick.formation == stamped.shadow_pick.formation
            and stamped.heuristic_pick.play == stamped.shadow_pick.play
        )
    elif stamped.fell_back:
        agree = None
    try:
        row_id = db.log_ml_decision(stamped, agree=agree)
    except Exception:  # noqa: BLE001
        return None

    # Preserve the model's decision and action provenance for postgame learning.
    # The output is research-based for actions, not a causal effect estimate.
    info = getattr(call, "ml_experimental", None)
    if isinstance(info, dict):
        try:
            raw = db.conn.execute(
                "SELECT decision_json FROM ml_decisions WHERE id=?", (row_id,)
            ).fetchone()
            payload = json.loads(raw["decision_json"] or "{}") if raw else {}
            payload["experimental_offense"] = {
                "selection_policy": info.get("selection_policy", "legacy_experimental"),
                "selection_audit": info.get("selection_audit"),
                "model_version": info.get("model_version"),
                "evidence_quality": info.get("evidence_quality"),
                "probability": info.get("probability"),
                "uncertainty": info.get("uncertainty"),
                "knowledge_version": info.get("knowledge_version"),
                "offense_action": info.get("offense_action"),
            }
            db.conn.execute(
                "UPDATE ml_decisions SET decision_json=? WHERE id=?",
                (json.dumps(payload, default=str), row_id),
            )
            db.conn.commit()
        except Exception:  # noqa: BLE001
            # Never jeopardize the original verified sealed decision.
            pass
    try:
        setattr(call, PENDING_ATTR, stamped)
        info = getattr(call, "ml_experimental", None)
        if isinstance(info, dict):
            info = dict(info)
            info["snap_id"] = snap_id
            info["game_id"] = game_id
            info["committed"] = True
            call.ml_experimental = info  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass
    return int(row_id) if row_id is not None else None


def maybe_apply_experimental(
    *,
    call: Any,
    sit: Any,
    opponent_id: str,
    db: Any,
    book: dict[str, list[str]] | None = None,
    active: Sequence[str] | None = None,
    audibles: dict[str, list[str]] | None = None,
    game_id: str | None = None,
    snap_id: str | None = None,
    session_id: str | None = None,
    snap_seq: int | None = None,
    allow_human_ml: bool = False,
) -> Any:
    """No-op unless mode is experimental. Never raises into the live path.

    Human-opponent offense requires an explicit per-session ``allow_human_ml``
    opt-in; the flag is not persisted in the database. CPU offense behavior
    is unchanged. Does not write the DB unless ``snap_id`` is already known.
    Live callers must commit after sealing.
    """
    try:
        from cfb_coach.madden.model import inference as inference_mod
        from cfb_coach.madden.playbook import active_books, eligible

        if inference_mod.resolve_mode(db) is not CoachingMode.EXPERIMENTAL:
            return call
        if str(getattr(call, "side", "") or "").startswith("d"):
            return call
        from cfb_coach.opponents import is_cpu_opponent

        # Experimental offense is CPU-ready; human play control is a separate,
        # deliberate game-session opt-in. No global/sticky "human mode".
        if not is_cpu_opponent(opponent_id) and not allow_human_ml:
            return call
        if book is None:
            raw = active_books(db, ("offense",))
            book = eligible(raw).get("offense") or {}
        if not book:
            return call
        # HTML live controller stamps the current session onto the situation,
        # but ordinary Madden make_call does not pass those fields explicitly.
        # Recover them here so 20-call memory and exploration are PER GAME,
        # not cross-game opponent history. No prediction ever uses outcomes.
        extras = getattr(sit, "extras", None) or {}
        if session_id is None and isinstance(extras, dict):
            session_id = extras.get("session_id") or None
        if game_id is None:
            game_id = session_id
        if snap_seq is None and session_id and db is not None:
            from cfb_coach.madden.model.identity import next_seq_from_db

            snap_seq = next_seq_from_db(db, session_id)
        new_call, _dec = apply_experimental_offense(
            heuristic_call=call,
            sit=sit,
            opponent_id=opponent_id,
            db=db,
            book=book,
            active=active,
            audibles=audibles,
            game_id=game_id,
            snap_id=snap_id,
            session_id=session_id,
            snap_seq=snap_seq,
            commit=bool(snap_id),
        )
        return new_call
    except Exception:  # noqa: BLE001
        return call


def postgame_experimental_compare(db: Any, *, game_id: str | None = None) -> dict[str, Any]:
    """Heuristic vs ML comparison for experimental sessions. No counterfactuals."""
    params: list[Any] = []
    sql = "SELECT * FROM ml_decisions WHERE mode = 'experimental'"
    if game_id:
        sql += " AND (game_id = ? OR session_id = ?)"
        params.extend([game_id, game_id])
    sql += " ORDER BY id ASC"
    rows = list(db.conn.execute(sql, params))
    # Drop anonymous rows (pre-fix bug) from attribution rates.
    identified = [r for r in rows if r["snap_id"]]
    outcomes: dict[str, Any] = {}
    try:
        for row in db.conn.execute("SELECT * FROM ml_outcomes"):
            outcomes[str(row["snap_id"])] = row
    except Exception:  # noqa: BLE001
        pass

    n = len(identified)
    ok = [r for r in identified if r["shadow_status"] == MLStatus.OK.value]
    fallback = [r for r in identified if r["shadow_status"] != MLStatus.OK.value]
    agree = [r for r in ok if r["agree"] == 1]
    disagree = [r for r in ok if r["agree"] == 0]
    verified = 0
    linked_outcomes = 0
    for r in identified:
        outc = outcomes.get(str(r["snap_id"] or ""))
        if outc is None:
            continue
        linked_outcomes += 1
        if outc["executed_status"] == "identified" and outc["executed_verification"] == "verified":
            verified += 1

    # Action usage is measured only from the chosen and recorded snap.
    # A checked play execution is not proof a proposed hot route or macro ran.
    action_recommended = 0
    action_confirmed = 0
    action_unconfirmed = 0
    action_by_kind: dict[str, int] = {}
    for r in identified:
        try:
            decision_payload = json.loads(r["decision_json"] or "{}")
            action = (decision_payload.get("experimental_offense") or {}).get("offense_action") or {}
        except (TypeError, ValueError, json.JSONDecodeError):
            action = {}
        kind = str(action.get("kind") or "none")
        action_id = action.get("id")
        if not action_id or kind not in ("macro", "adjustment"):
            continue
        action_recommended += 1
        action_by_kind[kind] = action_by_kind.get(kind, 0) + 1
        outcome_row = outcomes.get(str(r["snap_id"] or ""))
        confirmed = False
        if outcome_row is not None and outcome_row["executed_status"] == "identified" and outcome_row["executed_verification"] == "verified":
            try:
                outcome_payload = json.loads(outcome_row["outcome_json"] or "{}")
                confirmed = bool(outcome_payload.get("offense_action_explicitly_confirmed")) and (
                    (kind == "macro" and outcome_payload.get("executed_macro") == action_id)
                    or (kind == "adjustment" and outcome_payload.get("executed_adjustment_id") == action_id)
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        if confirmed:
            action_confirmed += 1
        else:
            action_unconfirmed += 1

    # Evaluate *what the coach displayed*, including unmatched outcomes.
    # Do not equate a recommended play with a confirmed executed play.
    from cfb_coach.madden.model.offense_selection_policy import summarize_call_variety

    snap_context: dict[str, dict[str, Any]] = {}
    try:
        for snap in db.conn.execute(
            "SELECT ml_snap_id, down, distance FROM snaps WHERE ml_snap_id IS NOT NULL"
        ):
            snap_context[str(snap["ml_snap_id"])] = {
                "down": snap["down"], "distance": snap["distance"],
            }
    except Exception:  # noqa: BLE001
        pass
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in identified:
        rec = dict(row)
        key = str(rec.get("game_id") or rec.get("session_id") or "unknown")
        groups.setdefault(key, []).append(rec)
    call_quality = {
        game: summarize_call_variety(entries, by_snap_situation=snap_context)
        for game, entries in groups.items()
    }

    orphan_anonymous = len(rows) - len(identified)
    return {
        "n_experimental_calls": n,
        "anonymous_unlinked_rows_ignored": orphan_anonymous,
        "model_ok": len(ok),
        "fallback_or_failure": len(fallback),
        "fallback_rate": round(len(fallback) / n, 3) if n else 0.0,
        "agree_with_heuristic": len(agree),
        "disagree_with_heuristic": len(disagree),
        "verified_executions_linked": verified,
        "outcomes_linked": linked_outcomes,
        "call_quality_by_game": call_quality,
        "offense_actions": {
            "recommended": action_recommended,
            "verified_applied": action_confirmed,
            "unconfirmed": action_unconfirmed,
            "by_kind": action_by_kind,
            "note": "Only explicitly confirmed action execution counted. No counterfactual or causal action lift inferred.",
        },
        "game_id_filter": game_id,
        "disagreements": [
            {
                "snap_id": r["snap_id"],
                "game_id": r["game_id"],
                "heuristic": f"{r['heuristic_formation']}/{r['heuristic_play']}",
                "ml": f"{r['shadow_formation']}/{r['shadow_play']}",
                "final_displayed": f"{r['final_formation']}/{r['final_play']}",
                "counterfactual_result": "unknown",
                "note": (
                    "Outcome belongs only to the play that was executed. "
                    "Do not claim yards/wins for the unchosen alternative."
                ),
            }
            for r in disagree[:40]
        ],
        "caveat": (
            "This report does not claim win-rate or yardage improvement from "
            "unobserved counterfactuals. Rows without snap_id are ignored."
        ),
    }
