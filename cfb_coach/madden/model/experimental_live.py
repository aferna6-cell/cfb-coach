"""Opt-in experimental ML offense selection with a sealed rebuild path.

Disabled by default. When ``ml_mode=experimental``:

1. Heuristic (incl. VOD) still produces a candidate.
2. The experimental model ranks legal in-book offense plays within 150 ms.
3. If the model pick differs, reads / Custom Adjustments are rebuilt for that
   play — never a post-hoc display swap of the heuristic's reads.
4. On timeout, error, illegal candidate, or missing artifact → heuristic.

Defense is never selected by this mode. CPU remains offense-only.
"""

from __future__ import annotations

import time
import traceback
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
) -> Any:
    """Build reads / macro / adjustment for an already-chosen in-book play."""
    from cfb_coach.madden.data import reads_for
    from cfb_coach.madden.playcaller import MaddenCall
    from cfb_coach.madden.playcaller import coverage_class
    from cfb_coach.tendency import is_repeated_coverage

    cov = getattr(sit, "coverage_hint", None)
    src = getattr(sit, "coverage_source", None) or "none"
    cls = coverage_class(cov) if cov else None
    repeated = bool(cov) and is_repeated_coverage(db, opponent_id, cov, sit, threshold=2)
    zone = "gl" if getattr(sit, "goal_line", False) else "rz" if getattr(sit, "red_zone", False) else "open"
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


def _legal_pairs(book: dict[str, list[str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for form, plays in (book or {}).items():
        for play in plays or []:
            out.append((form, play))
    return out


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
) -> tuple[Any, CoachingDecision]:
    """Maybe replace the offense call with an ML pick; always return a decision log."""
    from cfb_coach.madden.model.schema import Possession

    decision_ts = datetime.now(timezone.utc).isoformat()
    budget = float(ML_LATENCY_BUDGET_MS if budget_ms is None else budget_ms)
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
        if db is not None:
            try:
                db.log_ml_decision(dec, agree=None)
            except Exception:  # noqa: BLE001
                pass
        # Annotate heuristic call so HTML still shows fallback notice.
        try:
            heuristic_call.ml_experimental = {  # type: ignore[attr-defined]
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
        except Exception:  # noqa: BLE001
            pass
        return heuristic_call, dec

    # Defense / non-offense: never select.
    if str(getattr(heuristic_call, "side", "offense") or "offense").startswith("d"):
        return _fallback(MLStatus.CIRCUIT_OPEN)

    pairs = _legal_pairs(book)
    if not pairs:
        return _fallback(MLStatus.INVALID_OUTPUT)

    artifact_path = resolve_artifact_path(db)
    started = time.perf_counter()
    try:
        if artifact_path is None:
            # Train a prior-driven artifact on the fly from whatever trustworthy
            # evidence exists (may be empty → labeled prior_driven).
            from cfb_coach.madden.model import dataset as dataset_mod

            rows = dataset_mod.build_rows(db=db) if db is not None else []
            art = exp_mod.train_experimental(rows, side="offense")
            dest = Path(str(artifact_path or "")) if artifact_path else None
            if dest is None:
                from cfb_coach.games import data_dir

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

        # Pre-snap coverage hint only (live/last) — never invent post-snap coverage.
        cov_hint = None
        src = getattr(sit, "coverage_source", None) or ""
        if src in ("live", "last") and getattr(sit, "coverage_hint", None):
            cov_hint = sit.coverage_hint

        opp_type = "cpu" if is_cpu_opponent(opponent_id) else "human"
        ranked = exp_mod.rank_candidates(
            artifact,
            pairs,
            down=getattr(sit, "down", None),
            distance=getattr(sit, "distance", None),
            yardline=getattr(sit, "yardline", None),
            coverage_hint=cov_hint,
            opponent_id=opponent_id,
            opponent_type=opp_type,
            heuristic=(heur_form, heur_play) if heur_form and heur_play else None,
        )
        latency = (time.perf_counter() - started) * 1000.0
        if latency > budget:
            return _fallback(MLStatus.TIMEOUT, latency)
        if not ranked:
            return _fallback(MLStatus.INVALID_OUTPUT, latency)

        top = ranked[0]
        ml_form, ml_play = top["formation"], top["play"]
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
        agree = int(ml_form == heur_form and ml_play == heur_play)
        explanation = (
            f"ML experimental [{prior_tag}] → {ml_form}/{ml_play} "
            f"(p={top['probability']:.3f}, family={top.get('play_family')})"
        )
        if heur_form and heur_play:
            explanation = (
                f"heuristic={heur_form}/{heur_play} | {explanation}"
            )
        if top.get("near_tie_kept_heuristic"):
            explanation += " | near-tie kept heuristic"

        # Sealed rebuild when ML differs — never keep heuristic reads/macros.
        if agree:
            call = heuristic_call
            call.rationale = (getattr(call, "rationale", "") or "") + f" | {explanation}"
        else:
            base_rationale = (
                f"{explanation} | sealed rebuild of reads/macros for ML pick"
            )
            call = rebuild_offense_attachments(
                formation=ml_form,
                play=ml_play,
                sit=sit,
                book=book,
                active=list(active or []),
                db=db,
                opponent_id=opponent_id,
                rationale=base_rationale,
                audibles=audibles,
            )

        # Attach display metadata for HTML / CLI.
        if not isinstance(getattr(call, "extras", None), dict):
            try:
                call.extras = {}  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        try:
            call.ml_experimental = {  # type: ignore[attr-defined]
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
                "data_version": artifact.data_version,
                "probability": top["probability"],
                "rankings": ranked[:8],
                "fell_back": False,
            }
        except Exception:  # noqa: BLE001
            pass

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
        if db is not None:
            try:
                db.log_ml_decision(dec, agree=agree)
            except Exception:  # noqa: BLE001
                pass
        return call, dec
    except Exception:  # noqa: BLE001
        traceback.format_exc()
        latency = (time.perf_counter() - started) * 1000.0
        return _fallback(MLStatus.EXCEPTION, latency)


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
) -> Any:
    """No-op unless mode is experimental. Never raises into the live path."""
    try:
        from cfb_coach.madden.model import inference as inference_mod
        from cfb_coach.madden.playbook import active_books, eligible

        if inference_mod.resolve_mode(db) is not CoachingMode.EXPERIMENTAL:
            return call
        if str(getattr(call, "side", "") or "").startswith("d"):
            return call
        if book is None:
            raw = active_books(db, ("offense",))
            book = eligible(raw).get("offense") or {}
        if not book:
            return call
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
    outcomes: dict[str, Any] = {}
    try:
        for row in db.conn.execute("SELECT * FROM ml_outcomes"):
            outcomes[str(row["snap_id"])] = row
    except Exception:  # noqa: BLE001
        pass

    n = len(rows)
    ok = [r for r in rows if r["shadow_status"] == MLStatus.OK.value]
    fallback = [r for r in rows if r["shadow_status"] != MLStatus.OK.value]
    agree = [r for r in ok if r["agree"] == 1]
    disagree = [r for r in ok if r["agree"] == 0]
    verified = 0
    for r in rows:
        outc = outcomes.get(str(r["snap_id"] or ""))
        if outc is None:
            continue
        if outc["executed_status"] == "identified" and outc["executed_verification"] == "verified":
            verified += 1

    return {
        "n_experimental_calls": n,
        "model_ok": len(ok),
        "fallback_or_failure": len(fallback),
        "fallback_rate": round(len(fallback) / n, 3) if n else 0.0,
        "agree_with_heuristic": len(agree),
        "disagree_with_heuristic": len(disagree),
        "verified_executions_linked": verified,
        "disagreements": [
            {
                "snap_id": r["snap_id"],
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
            "unobserved counterfactuals."
        ),
    }
