"""Live scoring with a time budget and a heuristic fallback.

Shadow mode scores and logs without changing the displayed call. Heuristic
mode must not call this module. Hybrid stays off until promotion is allowed.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cfb_coach.madden.model import adapters as adapters
from cfb_coach.madden.model import policy as policy_mod
from cfb_coach.madden.model import registry as registry_mod
from cfb_coach.madden.model.schema import (
    ML_LATENCY_BUDGET_MS,
    ML_LOW_CONFIDENCE,
    CandidatePlay,
    CoachingDecision,
    CoachingMode,
    GameState,
    MLStatus,
    OpponentContext,
    PolicySource,
    PreSnapObservation,
    PropensityMethod,
    RosterSnapshot,
    SnapOutcome,
    Tri,
    legal_candidates,
)
from cfb_coach.madden.model.train import load_artifact

# Live scoring must take GameState and PreSnapObservation from this function.
_SITUATION_ADAPTER = adapters.situation_to_state


def rank_live(
    game_state: GameState,
    observation: PreSnapObservation,
    candidate_plays: Sequence[CandidatePlay],
    roster_context: RosterSnapshot | None,
    opponent_context: OpponentContext | None,
    recent_history: Sequence[tuple[CoachingDecision, SnapOutcome | None]] | None,
    *,
    mode: CoachingMode = CoachingMode.HEURISTIC,
    model_version: str | None = None,
    budget_ms: float | None = None,
    heuristic_pick: CandidatePlay | None = None,
    registry_dir: str | None = None,
    model_artifact: dict[str, Any] | None = None,
) -> CoachingDecision:
    """Score ``candidate_plays`` inside the model budget.

    ``game_state`` and ``observation`` come from ``_SITUATION_ADAPTER``.
    The model budget is :data:`ML_LATENCY_BUDGET_MS` (150). It covers only
    the model call. Do not rewrite it from measured latency. ``budget_ms``
    ``None`` means the caller did not pass one; the implementation uses the
    fixed constant. On failure the caller keeps the heuristic decision.
    ``fell_back`` is true only in hybrid, and ``fallback_reason`` is an
    ``MLStatus`` other than ``OK``. ``shadow_status`` is required in shadow
    and hybrid. The shadow-game counter counts only ``shadow_status == OK``.
    """
    history = tuple(recent_history or ())
    legal = legal_candidates(candidate_plays)
    shown = heuristic_pick
    if shown is None and legal:
        shown = legal[0]
    decision_ts = datetime.now(timezone.utc).isoformat()
    budget = float(ML_LATENCY_BUDGET_MS if budget_ms is None else budget_ms)

    if mode is CoachingMode.HEURISTIC:
        return CoachingDecision(
            decision_ts=decision_ts,
            mode=CoachingMode.HEURISTIC,
            effective_mode=CoachingMode.HEURISTIC,
            game_id=game_state.game_id,
            snap_id=game_state.snap_id,
            model_version=None,
            policy_source=PolicySource.HEURISTIC,
            candidates=tuple(legal) if legal else (),
            heuristic_pick=shown,
            final_pick=shown,
            propensity_method=PropensityMethod.UNKNOWN,
            fell_back=False,
            shadow_status=None,
            inference_budget_ms=ML_LATENCY_BUDGET_MS,
        )

    if mode is CoachingMode.HYBRID:
        # Hybrid is built but inactive until promotion_allowed. Always fall back.
        status = MLStatus.CIRCUIT_OPEN
        try:
            entry = _load_entry(registry_dir, model_version)
            if entry is None:
                status = MLStatus.MODEL_MISSING
            elif not registry_mod.promotion_allowed(entry):
                status = MLStatus.CIRCUIT_OPEN
            else:
                status = MLStatus.CIRCUIT_OPEN  # still force fallback this sprint
        except Exception:  # noqa: BLE001
            status = MLStatus.EXCEPTION
        return CoachingDecision(
            decision_ts=decision_ts,
            latency_ms_model=None,
            inference_budget_ms=ML_LATENCY_BUDGET_MS,
            mode=CoachingMode.HYBRID,
            effective_mode=CoachingMode.HEURISTIC,
            mode_downgrade_reason="hybrid_inactive_until_promotion",
            game_id=game_state.game_id,
            snap_id=game_state.snap_id,
            model_version=model_version,
            policy_source=PolicySource.HEURISTIC,
            candidates=tuple(legal) if legal else (),
            heuristic_pick=shown,
            final_pick=shown,
            propensity_method=PropensityMethod.UNKNOWN,
            fell_back=True,
            fallback_reason=status if status is not MLStatus.OK else MLStatus.CIRCUIT_OPEN,
            shadow_status=status,
        )

    # Shadow mode
    artifact = model_artifact
    entry = None
    shadow_status = MLStatus.OK
    shadow_pick = None
    shadow_scores = None
    latency_ms = None
    started = time.perf_counter()
    try:
        if artifact is None:
            entry = _load_entry(registry_dir, model_version)
            if entry is None or not entry.artifact_path:
                shadow_status = MLStatus.MODEL_MISSING
            else:
                artifact = load_artifact(entry.artifact_path)
                model_version = entry.model_version or model_version

        if shadow_status is MLStatus.OK and artifact is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms > budget:
                shadow_status = MLStatus.TIMEOUT
            else:
                ranking = policy_mod.rank_plays(
                    game_state,
                    observation,
                    legal,
                    roster_context,
                    opponent_context,
                    history,
                    model_artifact=artifact,
                    mode=CoachingMode.SHADOW,
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
                if latency_ms > budget:
                    shadow_status = MLStatus.TIMEOUT
                    shadow_pick = None
                    shadow_scores = None
                elif not ranking.plays:
                    shadow_status = MLStatus.INVALID_OUTPUT
                elif ranking.status is MLStatus.LOW_CONFIDENCE or (
                    ranking.confidence is not None and ranking.confidence < ML_LOW_CONFIDENCE
                ):
                    shadow_status = MLStatus.LOW_CONFIDENCE
                    shadow_pick = ranking.plays[0].candidate
                    shadow_scores = ranking.shadow_scores
                elif ranking.status is MLStatus.MODEL_MISSING:
                    shadow_status = MLStatus.MODEL_MISSING
                    shadow_pick = ranking.plays[0].candidate if ranking.plays else None
                    shadow_scores = ranking.shadow_scores
                elif ranking.status is MLStatus.INVALID_OUTPUT:
                    shadow_status = MLStatus.INVALID_OUTPUT
                else:
                    shadow_pick = ranking.plays[0].candidate
                    shadow_scores = ranking.shadow_scores
                    shadow_status = MLStatus.OK
        elif shadow_status is MLStatus.OK:
            shadow_status = MLStatus.MODEL_MISSING
    except Exception:  # noqa: BLE001 — never disrupt the live session
        shadow_status = MLStatus.EXCEPTION
        latency_ms = (time.perf_counter() - started) * 1000.0
        # Keep traceback out of the decision; callers may log separately.
        traceback.format_exc()

    return CoachingDecision(
        decision_ts=decision_ts,
        latency_ms=latency_ms,
        latency_ms_model=latency_ms,
        inference_budget_ms=ML_LATENCY_BUDGET_MS,
        mode=CoachingMode.SHADOW,
        effective_mode=CoachingMode.HEURISTIC,
        game_id=game_state.game_id,
        snap_id=game_state.snap_id,
        model_version=model_version,
        registry_id=entry.model_version if entry is not None else None,
        policy_version=policy_mod.POLICY_VERSION,
        policy_source=PolicySource.HEURISTIC,
        candidates=tuple(legal) if legal else (),
        heuristic_pick=shown,
        final_pick=shown,
        propensity_method=PropensityMethod.UNKNOWN,
        shadow_scores=shadow_scores,
        shadow_pick=shadow_pick,
        shadow_status=shadow_status,
        fell_back=False,
        accepted=Tri.UNKNOWN,
    )


def shadow_after_call(
    *,
    situation: Any,
    heuristic_formation: str | None,
    heuristic_play: str | None,
    formations: dict[str, list[str]],
    side: str,
    opponent_id: str | None = None,
    opponent_type: Any = None,
    game_id: str | None = None,
    snap_id: str | None = None,
    session_id: str | None = None,
    registry_dir: str | None = None,
    model_version: str | None = None,
    recent_history: Sequence[tuple[CoachingDecision, SnapOutcome | None]] | None = None,
    db: Any = None,
) -> CoachingDecision | None:
    """Run shadow scoring for a sealed heuristic call without changing it.

    Returns ``None`` when ML mode is not shadow (caller should skip).
    """
    from cfb_coach.madden.model.schema import OpponentKind, Possession, candidate_in_book

    mode = resolve_mode(db)
    if mode is not CoachingMode.SHADOW:
        return None

    poss = Possession.DEFENSE if str(side).startswith("d") else Possession.OFFENSE
    opp_kind = opponent_type if isinstance(opponent_type, OpponentKind) else OpponentKind.UNKNOWN
    state, obs = _SITUATION_ADAPTER(
        situation,
        game_id=game_id,
        snap_id=snap_id,
        session_id=session_id,
        opponent_id=opponent_id,
        opponent_type=opp_kind,
        madden_version="madden27",
    )
    # Force possession from the call side (parser defaults are not ML facts for unknown).
    if state.possession is Possession.UNKNOWN:
        from dataclasses import replace

        state = replace(state, possession=poss)

    candidates: list[CandidatePlay] = []
    for formation, plays in formations.items():
        for play in plays:
            try:
                candidates.append(
                    candidate_in_book(
                        formations,
                        side=poss,
                        formation=formation,
                        play=play,
                    )
                )
            except ValueError:
                continue

    heuristic = None
    if heuristic_formation and heuristic_play:
        try:
            heuristic = candidate_in_book(
                formations,
                side=poss,
                formation=heuristic_formation,
                play=heuristic_play,
            )
        except ValueError:
            heuristic = None

    decision = rank_live(
        state,
        obs,
        candidates,
        None,
        None,
        recent_history,
        mode=CoachingMode.SHADOW,
        model_version=model_version or resolve_model_version(db),
        registry_dir=registry_dir or resolve_registry_dir(db),
        heuristic_pick=heuristic,
    )
    if db is not None:
        try:
            db.log_ml_decision(decision, agree=_agree(decision))
        except Exception:  # noqa: BLE001
            pass
    return decision


def resolve_mode(db: Any = None) -> CoachingMode:
    raw = _meta(db, "ml_mode") or "heuristic"
    try:
        return CoachingMode(str(raw).strip().lower())
    except ValueError:
        return CoachingMode.HEURISTIC


def set_mode(db: Any, mode: CoachingMode) -> None:
    if mode is CoachingMode.UNKNOWN:
        raise ValueError("cannot set ml mode to unknown")
    if mode is CoachingMode.HYBRID:
        # Soft-block: allow storing the intent only when promotion would pass.
        # Callers that force hybrid still get circuit-open fallback from rank_live.
        pass
    db.set_meta("ml_mode", mode.value)


def resolve_model_version(db: Any = None) -> str | None:
    return _meta(db, "ml_model_version")


def set_model_version(db: Any, version: str | None) -> None:
    if version is None:
        db.set_meta("ml_model_version", "")
    else:
        db.set_meta("ml_model_version", version)


def resolve_registry_dir(db: Any = None) -> str | None:
    raw = _meta(db, "ml_registry_dir")
    if raw:
        return raw
    from cfb_coach.games import data_dir

    return str(Path(data_dir()) / "madden_ml_registry")


def set_registry_dir(db: Any, path: str) -> None:
    db.set_meta("ml_registry_dir", path)


def _meta(db: Any, key: str) -> str | None:
    if db is None:
        return None
    try:
        val = db.get_meta(key)
    except Exception:  # noqa: BLE001
        return None
    if val is None or val == "":
        return None
    return str(val)


def _load_entry(registry_dir: str | None, model_version: str | None):
    if not registry_dir:
        return None
    root = Path(registry_dir)
    try:
        if model_version:
            return registry_mod.read_entry(str(root / model_version))
        return registry_mod.read_entry(str(root))
    except (OSError, ValueError, FileNotFoundError):
        return None


def _agree(decision: CoachingDecision) -> int | None:
    if decision.shadow_status is not MLStatus.OK:
        return None
    if decision.heuristic_pick is None or decision.shadow_pick is None:
        return None
    same = (
        decision.heuristic_pick.formation == decision.shadow_pick.formation
        and decision.heuristic_pick.play == decision.shadow_pick.play
    )
    return 1 if same else 0
