"""Live scoring with a time budget and a heuristic fallback.

Shadow mode scores and logs without changing the displayed call. Heuristic
mode must not call this module. Hybrid stays off until promotion is allowed.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

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
    from_dict,
    is_live_selectable,
    PreSnapObservation,
    PropensityMethod,
    RosterSnapshot,
    SnapOutcome,
    Tri,
    legal_candidates,
    to_dict,
)
from cfb_coach.madden.model.train import load_artifact

logger = logging.getLogger(__name__)

# Live scoring must take GameState and PreSnapObservation from this function.
_SITUATION_ADAPTER = adapters.situation_to_state


def read_scored_on(model: Any) -> str:
    """Read the training-report tag ``scored_on``.

    ``reviewed`` and ``unreviewed`` are the only stored values. A missing
    tag, a missing report, or any other text counts as ``unknown``.
    """
    report = None
    if isinstance(model, dict):
        report = model.get("training_report")
        if report is None:
            report = model.get("training-report")
    else:
        report = getattr(model, "training_report", None)
        if report is None:
            report = getattr(model, "training-report", None)
    if not isinstance(report, dict):
        return "unknown"
    tags = report.get("tags")
    source = tags if isinstance(tags, dict) else report
    value = source.get("scored_on")
    if value in ("reviewed", "unreviewed"):
        return value
    return "unknown"


def _hybrid_refused(scored: str) -> str:
    return f"hybrid refused: scored_on is {scored}"


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
    session_id: str | None = None,
    snap_seq: int | None = None,
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
            session_id=session_id,
            snap_seq=snap_seq,
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
        # Hybrid may use a model only when its training report says reviewed.
        # Anything else falls back to the heuristic with a clear message.
        # A reviewed model stays inactive until promotion_allowed.
        scored = read_scored_on(model_artifact)
        status = MLStatus.CIRCUIT_OPEN
        refused = scored != "reviewed"
        try:
            entry = _load_entry(registry_dir, model_version)
            if model_artifact is None and entry is not None and entry.artifact_path:
                scored = read_scored_on(load_artifact(entry.artifact_path))
                refused = scored != "reviewed"
            if refused:
                status = MLStatus.CIRCUIT_OPEN
            elif entry is None and model_artifact is None:
                status = MLStatus.MODEL_MISSING
            elif entry is not None and not registry_mod.promotion_allowed(entry):
                status = MLStatus.CIRCUIT_OPEN
            else:
                status = MLStatus.CIRCUIT_OPEN  # still force fallback this sprint
        except Exception:  # noqa: BLE001
            if model_artifact is None:
                scored = "unknown"
                refused = True
            status = MLStatus.EXCEPTION if not refused else MLStatus.CIRCUIT_OPEN
        return CoachingDecision(
            decision_ts=decision_ts,
            latency_ms_model=None,
            inference_budget_ms=ML_LATENCY_BUDGET_MS,
            mode=CoachingMode.HYBRID,
            effective_mode=CoachingMode.HEURISTIC,
            mode_downgrade_reason=(
                _hybrid_refused(scored) if refused else "hybrid_inactive_until_promotion"
            ),
            game_id=game_state.game_id,
            snap_id=game_state.snap_id,
            session_id=session_id,
            snap_seq=snap_seq,
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
        session_id=session_id,
        snap_seq=snap_seq,
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


def evaluate_live_shadow(
    *,
    db: Any,
    situation: Any,
    call: Any,
    opponent_id: str,
    game_id: str,
    snap_id: str,
    snap_seq: int,
    session_id: str | None = None,
    formations: dict[str, list[str]] | None = None,
    run: bool = True,
) -> tuple[CoachingDecision | None, int | None]:
    """Shared HTML / terminal / one-shot shadow entry point.

    Never changes the displayed call. Safe no-op when mode is not shadow or
    ``run`` is False (duplicate seal). Exceptions never propagate.
    Returns ``(decision, ml_decisions.id)``.
    """
    try:
        from cfb_coach.madden.model.schema import CoachingMode, OpponentKind
        from cfb_coach.madden.playbook import eligible, load_books
        from cfb_coach.opponents import is_cpu_opponent

        if not run:
            return None, None
        if resolve_mode(db) is not CoachingMode.SHADOW:
            return None, None
        if formations is None:
            books = load_books(db)
            formations = eligible(books).get(getattr(call, "side", "offense")) or {}
        if not formations:
            return None, None
        decision = shadow_after_call(
            situation=situation,
            heuristic_formation=getattr(call, "formation", None),
            heuristic_play=getattr(call, "play", None),
            formations=formations,
            side=getattr(call, "side", "offense"),
            opponent_id=opponent_id,
            opponent_type=OpponentKind.CPU if is_cpu_opponent(opponent_id) else OpponentKind.HUMAN,
            game_id=game_id,
            snap_id=snap_id,
            session_id=session_id or game_id,
            snap_seq=snap_seq,
            db=db,
        )
        row_id = None
        if decision is not None and db is not None:
            # row id already written inside shadow_after_call; look up by snap_id
            try:
                row = db.conn.execute(
                    "SELECT id FROM ml_decisions WHERE snap_id = ? ORDER BY id DESC LIMIT 1",
                    (snap_id,),
                ).fetchone()
                row_id = int(row["id"] if hasattr(row, "keys") else row[0]) if row else None
            except Exception:  # noqa: BLE001
                row_id = None
        return decision, row_id
    except Exception:  # noqa: BLE001
        return None, None


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
    snap_seq: int | None = None,
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

    from dataclasses import replace as dc_replace

    # Ensure game/snap identity is on the GameState for features/logging.
    if game_id and state.game_id != game_id:
        state = dc_replace(state, game_id=game_id)
    if snap_id and state.snap_id != snap_id:
        state = dc_replace(state, snap_id=snap_id)

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
        session_id=session_id or game_id,
        snap_seq=snap_seq,
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


def resolve_live_db_path(explicit: str | Path | None = None) -> Path:
    """Madden database for live rows.

    Decisions, outcomes, snap logs, opponent context, and the postgame
    report use this file. An explicit ``--db`` path wins. Otherwise the
    path is :func:`cfb_coach.games.madden_db_path` (``CFB_COACH_MADDEN_DB``,
    else ``madden27.db``). It is not ``coach.db``.
    """
    if explicit is not None and str(explicit).strip():
        path = Path(explicit).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    from cfb_coach.games import madden_db_path

    return madden_db_path()


_OPPONENT_CONTEXT_META = "ml_opponent_context.v1"


def _opponent_context_key(opponent_id: str) -> str:
    return f"{_OPPONENT_CONTEXT_META}:{opponent_id}"


def save_opponent_context(
    context: OpponentContext,
    *,
    db: Any = None,
    db_path: str | Path | None = None,
) -> Path:
    """Store one opponent-context row in the live database."""
    from cfb_coach.db import CoachDB

    owns = db is None
    if owns:
        path = resolve_live_db_path(db_path)
        db = CoachDB(path)
    else:
        path = Path(db.path)
    try:
        db.set_meta(
            _opponent_context_key(context.opponent_id or ""),
            json.dumps(to_dict(context)),
        )
    finally:
        if owns:
            db.close()
    return path


def load_opponent_context(
    opponent_id: str,
    *,
    db: Any = None,
    db_path: str | Path | None = None,
) -> OpponentContext | None:
    """Read one opponent-context row from the live database."""
    from cfb_coach.db import CoachDB

    owns = db is None
    if owns:
        path = resolve_live_db_path(db_path)
        if not path.is_file():
            return None
        db = CoachDB(path)
    try:
        raw = db.get_meta(_opponent_context_key(opponent_id))
    finally:
        if owns:
            db.close()
    if not raw:
        return None
    loaded = from_dict(json.loads(raw))
    if not isinstance(loaded, OpponentContext):
        return None
    return loaded


def resolve_registry_dir(db: Any = None) -> str | None:
    raw = _meta(db, "ml_registry_dir")
    if raw:
        return raw
    return str(resolve_live_db_path().parent / "madden_ml_registry")


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


# ---------------------------------------------------------------------------
# Live snap presentation. Heuristic make_call output is not rewritten here.
# ---------------------------------------------------------------------------

_ARROW = " → "
_EM = "—"
_OFFENSE_LINE = re.compile(r"^.+ — .+ \| .+ \| .+ → .+$")
_DEFENSE_LINE = re.compile(r"^.+ — .+ \| .+ \| User .+$")

_DEFENSE_MODELS: dict[str, Any] = {}
_PARSED_MODELS: dict[tuple, Any] = {}


class StageClock:
    """Named stage durations for one live offense call."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self._name: str | None = None
        self._started = 0.0

    def begin(self, name: str) -> None:
        self.end()
        self._name = name
        self._started = time.perf_counter()

    def end(self) -> None:
        if self._name is None:
            return
        elapsed = (time.perf_counter() - self._started) * 1000.0
        self.stages[self._name] = self.stages.get(self._name, 0.0) + elapsed
        self._name = None

    def slowest(self) -> str | None:
        if not self.stages:
            return None
        best = max(self.stages.values())
        for name, value in self.stages.items():
            if value == best:
                return name
        return None


def log_offense_inference(
    *,
    latency_ms: float | None,
    timed_out: bool,
    slowest_stage: str | None,
) -> None:
    """One line per live offense call: latency, timeout, and the slowest stage."""
    latency = "none" if latency_ms is None else f"{float(latency_ms):.3f}"
    logger.info(
        "latency_ms=%s timed_out=%s slowest_stage=%s",
        latency,
        "true" if timed_out else "false",
        slowest_stage or "none",
    )


def file_revision(*paths: str | Path) -> tuple | None:
    """Identity of on-disk model files. Missing required files return None."""
    parts: list[tuple[str, int, int]] = []
    for raw in paths:
        path = Path(raw)
        try:
            st = path.stat()
        except OSError:
            return None
        parts.append((str(path), int(st.st_mtime_ns), int(st.st_size)))
    return tuple(parts)


def cached_parsed_model(revision: tuple, loader: Callable[[], Any]) -> Any:
    """Return a previously parsed model when ``revision`` is unchanged.

    ``None`` is a cached result. A missing file is not a revision, so the
    caller loads it every time and a mocked path still runs the loader.
    """
    if revision in _PARSED_MODELS:
        return _PARSED_MODELS[revision]
    loaded = loader()
    if len(_PARSED_MODELS) >= 8:
        _PARSED_MODELS.clear()
    _PARSED_MODELS[revision] = loaded
    return loaded


def clear_parsed_model_cache() -> None:
    _PARSED_MODELS.clear()


class _UnsetOppLook:
    """Look argument omitted. Distinct from an explicit missing look."""


UNSET_OPP_LOOK = _UnsetOppLook()


def normalize_opp_look(value: Any) -> str | None:
    """Blank input is a missing look. ``empty`` is empty backfield.

    ``None``, ``\"\"``, and whitespace-only text are missing and return
    ``None``. Any other text is kept, including the exact token ``empty``.
    The result is what gets passed to ``Situation`` and stored. It is never
    ``\"\"``.
    """
    if value is None or value is UNSET_OPP_LOOK:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def bind_opp_look(sit: Any, look: Any = UNSET_OPP_LOOK, *, raw: str | None = None) -> Any:
    """Attach one normalized opponent look to ``sit`` before a live call.

    An omitted ``look`` reads the exact token ``empty`` from ``raw`` (or
    ``sit.raw``). An explicit blank, including whitespace, is missing and
    does not fall through to that token. Blank coverage or concept hints
    are cleared so they are not passed on as ``\"\"``.
    """
    if look is UNSET_OPP_LOOK:
        text = raw if raw is not None else (getattr(sit, "raw", "") or "")
        value = "empty" if re.search(r"(?<!\S)empty(?!\S)", text) else None
    else:
        value = normalize_opp_look(look)
    extras = getattr(sit, "extras", None)
    if not isinstance(extras, dict):
        extras = {}
        sit.extras = extras
    extras["opp_look"] = value
    for name in ("coverage_hint", "concept_hint"):
        current = getattr(sit, name, None)
        if isinstance(current, str) and not current.strip():
            setattr(sit, name, None)
    return sit


class OppLookSession:
    """Pre-snap opponent look for one live session.

    ``None`` is a missing look. A blank string is the same fact and is stored
    as ``None``. The token ``\"empty\"`` is empty backfield and is kept.
    ``finish_snap`` clears the look so the next snap cannot reuse it.
    """

    def __init__(self) -> None:
        self.current: str | None = None

    def note(self, value: Any) -> None:
        self.current = normalize_opp_look(value)

    def finish_snap(self) -> None:
        self.current = None


@dataclass(frozen=True)
class LiveSnap:
    """One displayed live line and the look that belonged to that snap only."""

    line: str
    opp_look: str | None
    effective_mode: CoachingMode
    fell_back: bool


def live_display_mode(mode: CoachingMode | str | None) -> CoachingMode:
    """Map a requested mode to the mode whose line is shown.

    ``experimental`` is shadow: the heuristic line stays on screen. It is
    never hybrid.
    """
    if isinstance(mode, str):
        try:
            mode = CoachingMode(mode.strip().lower())
        except ValueError:
            return CoachingMode.HEURISTIC
    if mode is CoachingMode.EXPERIMENTAL:
        return CoachingMode.SHADOW
    if mode is None:
        return CoachingMode.HEURISTIC
    return mode


def _user_job(raw: str | None) -> str:
    job = (raw or "").strip() or "hook"
    job = re.split(r"\s*→\s*|\s*>\s*|\s*/\s*", job)[0].strip() or "hook"
    if not job.lower().startswith("user"):
        job = f"User {job}"
    return job


def _adjustment_slot(
    adjustment: str | None,
    macro_name: str | None,
    macro_state: Any,
) -> str:
    """Live slot. A macro below armed is omitted, never printed."""
    if macro_name and str(macro_name).strip() and is_live_selectable(macro_state):
        return str(macro_name).strip()
    text = (adjustment or "").strip()
    return text or "No adj"


def exact_offense_line(
    formation: str,
    play: str,
    adjustment: str | None,
    read1: str | None,
    read2: str | None,
    *,
    macro_name: str | None = None,
    macro_state: Any = None,
) -> str:
    """``Formation — Play | Adjustment | Read 1 → Read 2``.

    A missing second read is an em dash. This formatter does not invent
    the word Checkdown.
    """
    first = (read1 or "").strip() or _EM
    second = (read2 or "").strip() or _EM
    adj = _adjustment_slot(adjustment, macro_name, macro_state)
    return f"{formation} — {play} | {adj} | {first}{_ARROW}{second}"


def exact_defense_line(
    formation: str,
    play: str,
    adjustment: str | None,
    user_job: str | None,
    *,
    macro_name: str | None = None,
    macro_state: Any = None,
) -> str:
    """``Formation — Play | Adjustment/Macro | User responsibility``."""
    adj = _adjustment_slot(adjustment, macro_name, macro_state)
    return f"{formation} — {play} | {adj} | {_user_job(user_job)}"


def line_matches(line: str, side: str) -> bool:
    """True when ``line`` is one exact live line and nothing else."""
    if not isinstance(line, str) or "\n" in line or not line.strip():
        return False
    if side == "defense":
        return _DEFENSE_LINE.fullmatch(line) is not None
    return _OFFENSE_LINE.fullmatch(line) is not None


# Slack between the join and the recorded budget. The budget constant stays
# ML_LATENCY_BUDGET_MS; this only stops the wait early enough that the
# fallback itself returns inside that budget.
_CUTOFF_MARGIN_MS = 15.0


class BudgetCut(Exception):
    """An injected clock stepped past the armed cutoff."""


class LiveClock:
    """Clock used by the live cutoff.

    A realtime clock follows ``time.perf_counter``. A manual clock moves only
    when ``advance`` is called, so the cutoff does not depend on a wall-clock race.
    """

    def realtime(self) -> bool:
        return True

    def now_ms(self) -> float:
        return time.perf_counter() * 1000.0

    def arm(self, deadline_ms: float) -> None:
        del deadline_ms

    @property
    def cut(self) -> bool:
        return False


class ManualClock(LiveClock):
    """Clock that starts at 0 ms and moves only through ``advance``."""

    def __init__(self) -> None:
        self._t_ms = 0.0
        self._deadline_ms: float | None = None
        self._cut = False

    def realtime(self) -> bool:
        return False

    def now_ms(self) -> float:
        return self._t_ms

    def arm(self, deadline_ms: float) -> None:
        self._deadline_ms = float(deadline_ms)
        self._cut = False

    def advance(self, seconds: float) -> None:
        """Move forward by ``seconds``.

        A step that would pass the armed deadline snaps to that deadline and
        raises :class:`BudgetCut`. The reported time is the cutoff, which is
        inside the budget.
        """
        target = self._t_ms + float(seconds) * 1000.0
        if self._deadline_ms is not None and target > self._deadline_ms:
            self._t_ms = self._deadline_ms
            self._cut = True
            raise BudgetCut()
        self._t_ms = target

    @property
    def cut(self) -> bool:
        return self._cut


def run_within_budget(
    model: Callable[[], str],
    fallback: str,
    budget_ms: float | None = None,
    *,
    clock: LiveClock | None = None,
) -> tuple[str, float, bool]:
    """Run ``model`` until the cutoff, then return ``fallback``.

    The recorded budget stays :data:`ML_LATENCY_BUDGET_MS`. Waiting stops
    ``_CUTOFF_MARGIN_MS`` early so the fallback is inside that budget. The
    model's thread is not joined past that cutoff, and a late result is
    discarded.

    ``clock`` defaults to a realtime clock. Pass a :class:`ManualClock` to
    test the cutoff from simulated time: any step past the cutoff returns
    the fallback with elapsed time before the budget.
    """
    budget = float(ML_LATENCY_BUDGET_MS if budget_ms is None else budget_ms)
    wait_ms = max(0.0, budget - _CUTOFF_MARGIN_MS)
    tick = clock if clock is not None else LiveClock()
    if not tick.realtime():
        return _run_on_manual_clock(model, fallback, tick, wait_ms)
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = model()
        except Exception as exc:  # noqa: BLE001 — a model error is a fallback
            box["error"] = exc

    worker = threading.Thread(target=_target, name="cfb-live-model", daemon=True)
    started = time.perf_counter()
    worker.start()
    worker.join(wait_ms / 1000.0)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if worker.is_alive() or "error" in box:
        return fallback, elapsed_ms, True
    value = box.get("value")
    if not isinstance(value, str):
        return fallback, elapsed_ms, True
    return value, elapsed_ms, False


def _run_on_manual_clock(
    model: Callable[[], str],
    fallback: str,
    clock: LiveClock,
    wait_ms: float,
) -> tuple[str, float, bool]:
    clock.arm(wait_ms)
    started = clock.now_ms()
    value: Any = None
    failed = False
    try:
        value = model()
    except BudgetCut:
        failed = True
    except Exception:  # noqa: BLE001 — a model error is a fallback
        failed = True
    elapsed_ms = max(0.0, clock.now_ms() - started)
    if elapsed_ms > wait_ms:
        elapsed_ms = wait_ms
    if failed or clock.cut or not isinstance(value, str):
        return fallback, elapsed_ms, True
    return value, elapsed_ms, False


def present_snap(
    *,
    side: str = "offense",
    formation: str = "Gun Bunch",
    play: str = "Mesh",
    adjustment: str | None = None,
    read1: str | None = "Primary",
    read2: str | None = None,
    user_job: str | None = "hook",
    macro_name: str | None = None,
    macro_state: Any = None,
    opp_look: Any = None,
    session: OppLookSession | None = None,
    mode: CoachingMode | str = CoachingMode.HEURISTIC,
    model_line: str | None = None,
    model_fn: Callable[[], str] | None = None,
    fallback: str | None = None,
) -> LiveSnap:
    """Build the one live line for this snap.

    Does not call the VOD prior, training, or research fetch. Opponent look
    is recorded for this snap and then cleared on ``session``.
    """
    look_session = session if session is not None else OppLookSession()
    look_session.note(opp_look)
    used = look_session.current
    effective = live_display_mode(mode)
    if side == "defense":
        candidate = exact_defense_line(
            formation, play, adjustment, user_job,
            macro_name=macro_name, macro_state=macro_state,
        )
    else:
        candidate = exact_offense_line(
            formation, play, adjustment, read1, read2,
            macro_name=macro_name, macro_state=macro_state,
        )
    safe = fallback if isinstance(fallback, str) and line_matches(fallback, side) else candidate
    fell_back = False
    shown = candidate
    # Shadow, including experimental, keeps the heuristic line.
    if effective is not CoachingMode.SHADOW and effective is not CoachingMode.HEURISTIC:
        if model_fn is not None:
            produced, _elapsed, timed_out = run_within_budget(model_fn, safe)
            if timed_out or not line_matches(produced, side):
                shown = safe
                fell_back = True
            else:
                shown = produced
        elif model_line is not None:
            shown = model_line
    if not line_matches(shown, side) or (
        macro_name
        and not is_live_selectable(macro_state)
        and str(macro_name) in shown
    ):
        shown = safe if line_matches(safe, side) and (
            not macro_name or is_live_selectable(macro_state) or str(macro_name) not in safe
        ) else candidate
        fell_back = shown != candidate
    if not line_matches(shown, side):
        shown = candidate
    look_session.finish_snap()
    return LiveSnap(
        line=shown,
        opp_look=used,
        effective_mode=effective,
        fell_back=fell_back or shown != candidate,
    )


def display_or_fallback(candidate: str, fallback: str, side: str) -> str:
    """Return ``fallback`` when ``candidate`` fails the exact-line checks.

    The failing text is never the returned line.
    """
    if line_matches(candidate, side):
        return candidate
    return fallback


def finish_live_snap(sit: Any, db: Any) -> None:
    """Clear the opponent look after one snap.

    Returns nothing and does not touch the call. It does not load a model
    or call VOD, so it adds no model-path time. ``db`` is accepted so the
    call site can pass the open database; the look lives on ``sit``.
    """
    del db
    extras = getattr(sit, "extras", None)
    if not isinstance(extras, dict):
        return
    session = extras.get("opp_look_session")
    if isinstance(session, OppLookSession):
        session.finish_snap()
    extras["opp_look"] = None


def cached_defense_model(
    session_id: str,
    loader: Callable[[], Any],
    *,
    path: str | None = None,
) -> Any:
    """Load the defensive model once for ``session_id``.

    A later snap in the same session reuses the object. A different session
    loads again.
    """
    key = f"{session_id}\0{path or ''}"
    if key not in _DEFENSE_MODELS:
        _DEFENSE_MODELS[key] = loader()
    return _DEFENSE_MODELS[key]


def clear_defense_model_cache() -> None:
    _DEFENSE_MODELS.clear()


def _decision_payload(row: Any) -> dict[str, Any]:
    raw = row["decision_json"] if hasattr(row, "keys") else None
    if raw is None and isinstance(row, dict):
        raw = row.get("decision_json")
    try:
        payload = json.loads(raw or "{}")
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    return payload


def _vod_override(payload: dict[str, Any]) -> bool:
    """A hard VOD override is deterministic with behavior propensity 1.0."""
    method = str(payload.get("propensity_method") or "").strip().lower()
    if method != "deterministic":
        return False
    raw = payload.get("behavior_propensity")
    try:
        return raw is not None and float(raw) == 1.0
    except (TypeError, ValueError):
        return False


def _off_policy_eligible(payload: dict[str, Any]) -> bool:
    """True when the snap is a behavior-policy draw that off-policy eval can use.

    VOD overrides are not eligible. Unknown and empty methods are not draws.
    """
    if _vod_override(payload):
        return False
    method = str(payload.get("propensity_method") or "").strip().lower()
    if method == "override" or payload.get("propensity_override") is True:
        return True
    if method == "exact":
        try:
            propensity = float(payload.get("behavior_propensity"))
        except (TypeError, ValueError):
            return False
        return 0.0 < propensity <= 1.0
    return False


def override_rate(decisions: Sequence[Any]) -> dict[str, Any]:
    """Override rate over decisions eligible for off-policy evaluation.

    The denominator is that eligible set. A VOD override
    (``propensity_method=deterministic`` and ``behavior_propensity`` 1.0)
    is left out of both the numerator and the denominator and counted on
    its own line. Unknown rows are not eligible. The rate is not the
    verified-execution count.
    """
    overrides = 0
    denominator = 0
    vod_overrides = 0
    for row in decisions:
        payload = _decision_payload(row)
        if _vod_override(payload):
            vod_overrides += 1
            continue
        if not _off_policy_eligible(payload):
            continue
        denominator += 1
        method = str(payload.get("propensity_method") or "").strip().lower()
        if method == "override" or payload.get("propensity_override") is True:
            overrides += 1
    rate = None if denominator == 0 else overrides / denominator
    override_rate_line = f"Off-policy override rate: {overrides}/{denominator} eligible decisions"
    vod_override_line = f"VOD overrides excluded: {vod_overrides}"
    return {
        "overrides": overrides,
        "denominator": denominator,
        "rate": rate,
        "vod_overrides": vod_overrides,
        "override_rate_line": override_rate_line,
        "vod_override_line": vod_override_line,
    }
