"""Turn outcome estimates into one legal ranking. Owner: Strategist.

The ranker does not invent play names and does not treat the highest raw
success probability as the call. It only scores candidates that
:func:`cfb_coach.madden.model.schema.legal_candidates` would keep.

Contract revision (this sprint): ``observation`` is required so the ranker can
build decision-time features without reading post-snap labels. See
``docs/madden_ml/ARCHITECTURE.md``.

``recent_history`` is ``Sequence[tuple[CoachingDecision, SnapOutcome | None]]``
(provisional schema update on this branch).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model import features as feature_mod
from cfb_coach.madden.model.schema import (
    FEATURE_SCHEMA_VERSION,
    ML_LOW_CONFIDENCE,
    CandidatePlay,
    CoachingDecision,
    CoachingMode,
    GameState,
    MLStatus,
    OpponentContext,
    OutcomeEstimates,
    PlayRanking,
    PolicySource,
    Possession,
    PreSnapObservation,
    RankedPlay,
    RecentSnap,
    RosterSnapshot,
    ScorePart,
    ShadowScore,
    SnapOutcome,
    Tri,
    legal_candidates,
)
from cfb_coach.madden.model.train import POLICY_VERSION, load_artifact, predict_proba

__all__ = ["rank_plays", "set_default_artifact", "POLICY_VERSION", "load_artifact_from_path", "history_as_recent"]

_DEFAULT_ARTIFACT: dict[str, Any] | None = None

HistoryItem = tuple[CoachingDecision, SnapOutcome | None]


def set_default_artifact(artifact: Mapping[str, Any] | None) -> None:
    """Test/CLI helper: pin the artifact used when ``model_artifact`` is omitted."""
    global _DEFAULT_ARTIFACT
    _DEFAULT_ARTIFACT = dict(artifact) if artifact is not None else None


def history_as_recent(recent_history: Sequence[HistoryItem]) -> tuple[RecentSnap, ...]:
    """Project sealed decision/outcome pairs into feature-time RecentSnap rows."""
    out: list[RecentSnap] = []
    for decision, outcome in recent_history:
        pick = decision.final_pick or decision.heuristic_pick
        formation = pick.formation if pick is not None else None
        play = pick.play if pick is not None else None
        possession = Possession.UNKNOWN
        if pick is not None and pick.playbook is not None:
            possession = pick.playbook.side
        out.append(
            RecentSnap(
                snap_id=decision.snap_id,
                possession=possession,
                formation=formation,
                play=play,
                outcome=outcome,
            )
        )
    return tuple(out)


def rank_plays(
    game_state: GameState,
    observation: PreSnapObservation,
    candidate_plays: Sequence[CandidatePlay],
    roster_context: RosterSnapshot | None,
    opponent_context: OpponentContext | None,
    recent_history: Sequence[HistoryItem],
    *,
    model_artifact: Mapping[str, Any] | None = None,
    mode: CoachingMode = CoachingMode.HEURISTIC,
) -> PlayRanking:
    """Rank the legal pool for this snap.

    ``recent_history`` is its own argument. It is not a field on
    ``game_state``. Each pair is a decision that already ended and the outcome
    of that snap, or ``None`` when the outcome is still unknown. The outcome
    of the snap being ranked must not be included. Pass an empty sequence when
    no prior snap was supplied. Each kept candidate later gets a ``sampling_p``
    under ``policy_source``. Equal scores need a stable tie break on
    formation, play, then adjustment slot.
    """
    legal = legal_candidates(candidate_plays)
    artifact = model_artifact if model_artifact is not None else _DEFAULT_ARTIFACT
    model_version = None
    if artifact is not None:
        model_version = str(artifact.get("model_version") or artifact.get("side") or "artifact")

    recent = history_as_recent(recent_history)

    sit = game_state.situation
    down = sit.down if sit is not None else None
    distance = sit.distance if sit is not None else None
    yardline = sit.yardline if sit is not None else None
    score_diff = None
    if game_state.score is not None:
        score_diff = game_state.score.us - game_state.score.them
    clock = game_state.clock_seconds

    if not legal:
        return PlayRanking(
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            model_version=model_version,
            policy_version=POLICY_VERSION,
            policy_source=PolicySource.MODEL if artifact is not None else PolicySource.HEURISTIC,
            mode=mode,
            plays=(),
            status=MLStatus.INVALID_OUTPUT,
            status_detail="no legal candidates",
        )

    scored: list[tuple[float, RankedPlay, ShadowScore]] = []
    for cand in legal:
        vec = feature_mod.vector_for(
            game_state,
            observation,
            cand,
            roster_context,
            opponent_context,
            recent,
        )
        if artifact is not None:
            raw_p = predict_proba(artifact, vec)
        elif cand.heuristic_p is not None:
            raw_p = float(cand.heuristic_p)
        else:
            raw_p = 0.5

        context_score, parts, reasons = _context_adjust(
            game_state.possession,
            down=down,
            distance=distance,
            yardline=yardline,
            score_diff=score_diff,
            clock_seconds=clock,
            candidate=cand,
            raw_p=raw_p,
            recent=recent,
            opponent_context=opponent_context,
            heuristic_score=cand.heuristic_score,
        )
        estimates = _estimates_for(game_state.possession, raw_p)
        conf = _confidence(raw_p, opponent_context, artifact)
        ranked = RankedPlay(
            candidate=cand,
            estimates=estimates,
            confidence=conf,
            reasons=tuple(reasons),
            rank=None,
            sampling_p=None,
            sample_size=opponent_context.sample_size if opponent_context else None,
            shrinkage_weight=opponent_context.shrinkage_weight if opponent_context else None,
            score_breakdown=tuple(parts),
        )
        shadow = ShadowScore(
            candidate_key=f"{cand.formation}|{cand.play}|{cand.adjustment_slot}",
            score=context_score,
            p=raw_p,
        )
        scored.append((context_score, ranked, shadow))

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].candidate.formation or "",
            item[1].candidate.play or "",
            item[1].candidate.adjustment_slot
            if item[1].candidate.adjustment_slot is not None
            else -1,
            _idx_tie(item),
        )
    )

    plays: list[RankedPlay] = []
    shadows: list[ShadowScore] = []
    max_s = max(s for s, _, _ in scored)
    exps = [_math_exp(s - max_s) for s, _, _ in scored]
    total = sum(exps) or 1.0
    for rank_i, ((_score, ranked, shadow), weight) in enumerate(zip(scored, exps), start=1):
        plays.append(
            RankedPlay(
                candidate=ranked.candidate,
                estimates=ranked.estimates,
                confidence=ranked.confidence,
                reasons=ranked.reasons,
                rank=rank_i,
                sampling_p=weight / total,
                sample_size=ranked.sample_size,
                shrinkage_weight=ranked.shrinkage_weight,
                score_breakdown=ranked.score_breakdown,
            )
        )
        shadows.append(shadow)

    overall_conf = plays[0].confidence if plays else None
    low = Tri.UNKNOWN
    if overall_conf is not None:
        low = Tri.TRUE if overall_conf < ML_LOW_CONFIDENCE else Tri.FALSE

    if artifact is None:
        status = MLStatus.MODEL_MISSING
        detail = "heuristic prior only"
    elif overall_conf is not None and overall_conf < ML_LOW_CONFIDENCE:
        status = MLStatus.LOW_CONFIDENCE
        detail = f"confidence={overall_conf:.3f}"
    else:
        status = MLStatus.OK
        detail = None

    source = PolicySource.MODEL if artifact is not None else PolicySource.HEURISTIC
    return PlayRanking(
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        model_version=model_version,
        policy_version=POLICY_VERSION,
        policy_source=source,
        mode=mode,
        plays=tuple(plays),
        confidence=overall_conf,
        low_confidence=low,
        shadow_scores=tuple(shadows) if shadows else None,
        sample_size=opponent_context.n_snaps if opponent_context else None,
        shrinkage_weight=opponent_context.shrinkage_weight if opponent_context else None,
        status=status,
        status_detail=detail,
    )


def _idx_tie(item: tuple[float, RankedPlay, ShadowScore]) -> str:
    c = item[1].candidate
    return f"{c.formation}|{c.play}|{c.adjustment_slot}"


def _math_exp(x: float) -> float:
    return math.exp(max(-20.0, min(20.0, x)))


def load_artifact_from_path(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return load_artifact(str(p))


def _estimates_for(possession: Possession, raw_p: float) -> OutcomeEstimates:
    if possession is Possession.DEFENSE:
        return OutcomeEstimates(stop_probability=raw_p, n=None)
    if possession is Possession.OFFENSE:
        return OutcomeEstimates(success_probability=raw_p, n=None)
    return OutcomeEstimates(success_probability=raw_p, stop_probability=raw_p, n=None)


def _confidence(
    raw_p: float,
    opponent: OpponentContext | None,
    artifact: Mapping[str, Any] | None,
) -> float:
    base = abs(raw_p - 0.5) * 2.0
    n = 0.0
    if opponent is not None and opponent.n_snaps is not None:
        n = float(opponent.n_snaps)
    if artifact is not None:
        n = max(n, float(artifact.get("n_train") or 0))
    shrink = n / (n + 20.0) if n > 0 else 0.25
    return max(0.0, min(1.0, 0.35 + 0.5 * base + 0.15 * shrink))


def _context_adjust(
    possession: Possession,
    *,
    down: int | None,
    distance: int | None,
    yardline: int | None,
    score_diff: int | None,
    clock_seconds: int | None,
    candidate: CandidatePlay,
    raw_p: float,
    recent: Sequence[RecentSnap],
    opponent_context: OpponentContext | None,
    heuristic_score: float | None,
) -> tuple[float, list[ScorePart], list[str]]:
    """Context-dependent utility — not raw success probability alone."""
    parts = [ScorePart(name="model_or_prior", value=raw_p)]
    reasons = [f"base_p={raw_p:.3f}"]
    score = float(raw_p)

    if down is not None and distance is not None:
        if down >= 3 and distance >= 7:
            if raw_p > 0.55:
                score += 0.05
                parts.append(ScorePart(name="long_yardage_boost", value=0.05))
                reasons.append("long_yardage_needs_chunk")
            else:
                score -= 0.03
                parts.append(ScorePart(name="long_yardage_penalty", value=-0.03))
        if down == 4 or (down == 3 and distance <= 2):
            score += 0.08 * (raw_p - 0.5)
            parts.append(ScorePart(name="must_convert", value=0.08 * (raw_p - 0.5)))
            reasons.append("must_convert_situation")

    if yardline is not None and yardline >= 80:
        score += 0.02
        parts.append(ScorePart(name="red_zone", value=0.02))
        reasons.append("red_zone")

    if score_diff is not None and clock_seconds is not None and clock_seconds <= 120:
        if possession is Possession.OFFENSE and score_diff < 0:
            score += 0.04 * (raw_p - 0.4)
            parts.append(ScorePart(name="two_minute_trailing", value=0.04 * (raw_p - 0.4)))
            reasons.append("two_minute_trailing")
        if possession is Possession.OFFENSE and score_diff > 7:
            score -= 0.02
            parts.append(ScorePart(name="protect_lead", value=-0.02))
            reasons.append("protect_lead")

    same = sum(1 for s in recent if s.play and s.play == candidate.play)
    if same:
        pen = min(0.12, 0.04 * same)
        score -= pen
        parts.append(ScorePart(name="anti_repeat", value=-pen))
        reasons.append(f"anti_repeat_x{same}")

    if opponent_context is not None and opponent_context.tendencies:
        n = float(opponent_context.n_snaps or opponent_context.sample_size or 0)
        w = opponent_context.shrinkage_weight
        if w is None:
            w = n / (n + 20.0) if n > 0 else 0.0
        if opponent_context.seed_shrink_weight is not None:
            w = float(opponent_context.seed_shrink_weight)
        score += 0.03 * float(w) * (raw_p - 0.5)
        parts.append(ScorePart(name="tendency_shrink", value=0.03 * float(w) * (raw_p - 0.5)))
        reasons.append("tendency_shrinkage")

    if heuristic_score is not None:
        score += 0.15 * float(heuristic_score)
        parts.append(ScorePart(name="heuristic_blend", value=0.15 * float(heuristic_score)))
        reasons.append("heuristic_signal")

    if candidate.adjustment_id:
        score += 0.01
        parts.append(ScorePart(name="armed_macro", value=0.01))
        reasons.append(f"macro={candidate.adjustment_id}")

    return score, parts, reasons
