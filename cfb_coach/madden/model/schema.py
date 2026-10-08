"""Shared Madden ML contracts.

Stable records use ``madden-ml.schema.1``. :class:`PlayRanking` and
:class:`OpponentContext` are provisional (``madden-ml.provisional.1``) because
the Strategist will add fields. Additive changes to those two types bump only
the provisional version.

Unknown values are ``None`` in Python, JSON ``null``, SQLite ``NULL``, and the
literal ``__UNKNOWN__`` in CSV (:data:`CSV_UNKNOWN`). Every enum has an
``UNKNOWN`` member. ``None`` is a missing scalar. The enum member is a recorded
unknown category. Do not write ``0``, ``False``, or a guessed play name in
place of either.

These types are stdlib-only.

A candidate that will be ranked must be built with :func:`candidate_in_book`
so the formation and play are keys in the applied book, not free text.
Custom Adjustment slots are the existing eight-per-side loadout
(``LOADOUT_N``). Slot ``None`` means no slot was chosen or the slot is
unknown. It is not slot 0.

:class:`PreSnapObservation` is the decision-time boundary. Yards, labels,
verified coverage, and the executed play live on :class:`SnapOutcome` and
:class:`ExecutedPlay` and must not be copied onto the observation.

The live objects this package reuses instead of redefining:

* :class:`cfb_coach.situation.Situation` — parsed down, distance, field, hints
* :class:`cfb_coach.game_score.GameScore` — us-them score when one was supplied
* :class:`cfb_coach.outcome.ParsedOutcome` — the existing outcome-string parser
* :data:`cfb_coach.madden.macros.LOADOUT_N` — eight Custom Adjustment slots per side

``Situation`` still uses its own parser defaults (for example side
``offense`` when the line did not say). Those defaults are not ML facts.
Training code should read :class:`GameState` fields, which stay unknown until
a caller sets them.
"""

from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, Mapping, Sequence, Union, get_args, get_origin, get_type_hints

from cfb_coach.game_score import GameScore
from cfb_coach.madden.macros import LOADOUT_N
from cfb_coach.outcome import ParsedOutcome
from cfb_coach.situation import Situation

CONTRACT_VERSION = "madden-ml.schema.1"
PROVISIONAL_SCHEMA_VERSION = "madden-ml.provisional.1"
FEATURE_SCHEMA_VERSION = "madden-ml.features.1"
DECISION_SCHEMA_VERSION = "madden-ml.decision.1"

# Missing CSV cell. Distinct from the enum value "unknown".
CSV_UNKNOWN = "__UNKNOWN__"

# Snap yards. Expected yards use the same closed interval when a number is set.
YARDS_MIN = -99
YARDS_MAX = 99

# Fixed product settings. Change them only by editing this module.
# Nothing at runtime may rewrite the budget from measured latency.
ML_LOW_CONFIDENCE = 0.6
ML_LATENCY_BUDGET_MS = 150

# Names that are outcomes or post-snap labels. None of them may be fields on
# PreSnapObservation. Decision features are built only from that type.
LEAKAGE_FIELDS = frozenset(
    {
        "yards",
        "result",
        "touchdown",
        "sack",
        "interception",
        "fumble",
        "explosive",
        "first_down",
        "stop",
        "incompletion",
        "drive_result",
        "observed_coverage",
        "observed_concept",
        "executed_play",
        "executed",
        "success",
        "success_probability",
        "parsed",
    }
)


class OpponentKind(str, Enum):
    CPU = "cpu"
    HUMAN = "human"
    UNKNOWN = "unknown"


class Possession(str, Enum):
    OFFENSE = "offense"
    DEFENSE = "defense"
    UNKNOWN = "unknown"


class CoachingMode(str, Enum):
    """How the live coach uses a model. Existing installs stay heuristic.

    ``UNKNOWN`` is for a record that does not say. It is not a mode to run.
    """

    HEURISTIC = "heuristic"
    SHADOW = "shadow"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


class MLStatus(str, Enum):
    """How a model attempt finished. JSON values are these lowercase strings.

    ``OK`` is a successful model result. It is a legal ``shadow_status`` and
    is never a fallback reason. The shadow-game counter counts only rows
    whose ``shadow_status`` is ``OK``.
    """

    OK = "ok"
    MODEL_MISSING = "model_missing"
    WORKER_BUSY = "worker_busy"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    INVALID_OUTPUT = "invalid_output"
    CIRCUIT_OPEN = "circuit_open"
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN = "unknown"


# Statuses rank_plays may put on PlayRanking. Inference-only failures
# (timeout, exception, worker_busy, circuit_open) stay on CoachingDecision.
RANKING_STATUSES = frozenset(
    {
        MLStatus.OK,
        MLStatus.LOW_CONFIDENCE,
        MLStatus.MODEL_MISSING,
        MLStatus.INVALID_OUTPUT,
    }
)


class PolicySource(str, Enum):
    """Which policy's sampling distribution produced the shown call.

    ``behavior_propensity`` is that policy's end-to-end probability of the
    shown play, after anti-repeat, pivot, the book guard, and the VOD
    override. Model probabilities are shadow scores, not this source.
    """

    HEURISTIC = "heuristic"
    MODEL = "model"
    FALLBACK = "fallback"
    UNKNOWN = "unknown"


class PropensityMethod(str, Enum):
    """How ``behavior_propensity`` was produced. Default is ``UNKNOWN``.

    ``EXACT`` means the probability was carried through every guard.
    ``DETERMINISTIC`` means a later step forced the shown play, so the logged
    value is a point mass. A VOD override is logged as ``DETERMINISTIC`` with
    ``behavior_propensity`` 1.0. Those snaps are excluded from off-policy
    evaluation: the 1.0 is the hard swap, not a draw from the behavior policy.
    ``OVERRIDE`` means a step could not be expressed as a probability.
    ``propensity_override`` is true only for this member.
    """

    EXACT = "exact"
    DETERMINISTIC = "deterministic"
    OVERRIDE = "override"
    UNKNOWN = "unknown"


class PropensityOverrideReason(str, Enum):
    """Why ``propensity_override`` is true.

    The field on :class:`CoachingDecision` is ``None`` unless
    ``propensity_override`` is true. ``UNKNOWN`` means the override happened
    and the kind of step was not classified.
    """

    CAP = "cap"
    UNEXPRESSIBLE = "unexpressible"
    OTHER = "other"
    UNKNOWN = "unknown"


class ExecutedStatus(str, Enum):
    """Whether anyone confirmed the play that was actually run.

    ``UNKNOWN`` is the default. A displayed call is not confirmation.
    ``OTHER`` means the user said they ran something else. ``IDENTIFIED``
    is the only status that may carry an executed play, and that play must
    be verified.
    """

    UNKNOWN = "unknown"
    OTHER = "other"
    IDENTIFIED = "identified"


class PenaltyCall(str, Enum):
    """How a penalty on the snap was handled. ``UNKNOWN`` is not "no penalty"."""

    NO_PENALTY = "no_penalty"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    UNKNOWN = "unknown"


class HintSource(str, Enum):
    """Where a pre-snap coverage or concept hint came from.

    ``NONE`` means the situation parser recorded no hint. ``UNKNOWN`` means
    this record does not say. ``SCOUTED`` is a prior from logs, not the snap.
    """

    LIVE = "live"
    LAST = "last"
    SCOUTED = "scouted"
    NONE = "none"
    UNKNOWN = "unknown"


class Tri(str, Enum):
    """Three-way fact. ``FALSE`` is an observation, not the stand-in for missing."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


class Verification(str, Enum):
    VERIFIED = "verified"
    INFERRED = "inferred"
    UNKNOWN = "unknown"


class GateResult(str, Enum):
    NOT_RUN = "not_run"
    INSUFFICIENT = "insufficient"
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"


class DecisionStage(str, Enum):
    """One step that can change a call after the first candidate is chosen.

    The live caller still applies several of these after :class:`cfb_coach.madden.playcaller.MaddenCall`
    exists. Later integration records them here and then seals the call once.
    """

    HEURISTIC_SAMPLE = "heuristic_sample"
    COVERAGE_LEAN = "coverage_lean"
    ANTI_REPEAT = "anti_repeat"
    PIVOT = "pivot"
    BOOK_GUARD = "book_guard"
    SOFT_SHELL = "soft_shell"
    MACRO = "macro"
    ADJUSTMENT = "adjustment"
    SITUATION_MACRO = "situation_macro"
    FIT_BOOK = "fit_book"
    LABEL_GATE = "label_gate"
    VOD_PRIOR = "vod_prior"
    MODEL = "model"
    POLICY = "policy"
    FINAL = "final"
    UNKNOWN = "unknown"


def _reject_bool_int(name: str, value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int or unknown")


def _check_finite(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number or unknown")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite or unknown")


def _check_unit(name: str, value: float | None) -> None:
    _check_finite(name, value)
    if value is None:
        return
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1 or unknown")


def _check_open_unit(name: str, value: float | None) -> None:
    """Chosen-play propensity: in (0, 1] when known. Zero is not a shown play."""
    _check_finite(name, value)
    if value is None:
        return
    if not 0.0 < float(value) <= 1.0:
        raise ValueError(f"{name} must be in (0, 1] or unknown")


def _check_yards(name: str, value: float | int | None) -> None:
    if isinstance(value, bool) or (value is not None and not isinstance(value, (int, float))):
        raise TypeError(f"{name} must be a number or unknown")
    if value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite or unknown")
    if not YARDS_MIN <= float(value) <= YARDS_MAX:
        raise ValueError(f"{name} must be in [{YARDS_MIN}, {YARDS_MAX}] or unknown")


def _check_slot(slot_index: int | None) -> None:
    _reject_bool_int("adjustment slot", slot_index)
    if slot_index is None:
        return
    if not 0 <= slot_index < LOADOUT_N:
        raise ValueError(
            f"Custom Adjustment slot must be 0..{LOADOUT_N - 1} or unknown; got {slot_index}"
        )


def _check_nonneg(name: str, value: int | None) -> None:
    _reject_bool_int(name, value)
    if value is not None and value < 0:
        raise ValueError(f"{name} must be >= 0 or unknown")


def _check_propensity_pair(
    method: PropensityMethod,
    override: bool,
    reason: PropensityOverrideReason | None,
) -> None:
    """``propensity_override`` is true exactly when ``method`` is ``OVERRIDE``.

    ``reason`` is ``None`` unless the flag is true. ``to_dict`` and
    ``from_dict`` both call this, and so does :class:`CoachingDecision`.
    """
    if override != (method is PropensityMethod.OVERRIDE):
        raise ValueError("propensity_override must equal (propensity_method == OVERRIDE)")
    if override:
        if not isinstance(reason, PropensityOverrideReason):
            raise ValueError("propensity_override requires propensity_override_reason")
    elif reason is not None:
        raise ValueError("propensity_override_reason is None unless propensity_override is true")


@dataclass(frozen=True)
class AppliedPlaybookRef:
    """The locked book a candidate was resolved against.

    ``revision`` is the applied record's ``rev``. ``None`` means the revision
    was not recorded. ``mode`` is ``stock``, ``custom``, or unknown.
    """

    side: Possession = Possession.UNKNOWN
    revision: int | None = None
    source_book: str | None = None
    mode: str | None = None

    def __post_init__(self) -> None:
        _reject_bool_int("playbook revision", self.revision)
        if self.revision is not None and self.revision < 1:
            raise ValueError("playbook revision must be >= 1 or unknown")
        if self.mode is not None and self.mode not in ("stock", "custom"):
            raise ValueError("playbook mode must be stock, custom, or unknown")


@dataclass(frozen=True)
class AdjustmentRef:
    """One configured Custom Adjustment slot, or an unknown slot.

    ``macro_id`` is a catalog id from the offense or defense macro pool, not
    an explanation sentence. ``None`` means no id was recorded.
    """

    side: Possession = Possession.UNKNOWN
    slot_index: int | None = None
    macro_id: str | None = None

    def __post_init__(self) -> None:
        _check_slot(self.slot_index)


@dataclass(frozen=True)
class CandidatePlay:
    """A play the ranker may score.

    ``formation`` and ``play`` are keys in the applied book's formation map.
    ``in_applied_book`` is ``TRUE`` only after :func:`candidate_in_book`
    finds that pair. A direct constructor leaves it ``UNKNOWN``, which
    :func:`legal_candidates` will not rank.
    """

    playbook: AppliedPlaybookRef = field(default_factory=AppliedPlaybookRef)
    formation: str | None = None
    play: str | None = None
    adjustment_slot: int | None = None
    adjustment_id: str | None = None
    audible: Tri = Tri.UNKNOWN
    in_applied_book: Tri = Tri.UNKNOWN
    heuristic_score: float | None = None
    heuristic_p: float | None = None

    def __post_init__(self) -> None:
        _check_slot(self.adjustment_slot)
        _check_finite("heuristic_score", self.heuristic_score)
        _check_unit("heuristic_p", self.heuristic_p)
        if self.in_applied_book is Tri.TRUE:
            if not self.formation or not self.play:
                raise ValueError("a confirmed book play needs a formation and a play key")
            if self.playbook.side is Possession.UNKNOWN:
                raise ValueError("a confirmed book play needs an offense or defense book")


def candidate_in_book(
    formations: Mapping[str, Sequence[str]],
    *,
    side: Possession,
    formation: str,
    play: str,
    revision: int | None = None,
    source_book: str | None = None,
    mode: str | None = None,
    adjustment_slot: int | None = None,
    adjustment_id: str | None = None,
    audible: Tri = Tri.UNKNOWN,
    heuristic_score: float | None = None,
    heuristic_p: float | None = None,
) -> CandidatePlay:
    """Build a candidate that is a member of ``formations``.

    ``formations`` is the applied book's ``{formation: [play, ...]}`` map
    (the same shape ``playbook.eligible`` returns for one side). A name that
    is not in that map raises ``ValueError``.
    """
    if side not in (Possession.OFFENSE, Possession.DEFENSE):
        raise ValueError("a book candidate needs offense or defense")
    plays = formations.get(formation)
    if plays is None or play not in list(plays):
        raise ValueError(f"{formation!r} / {play!r} is not in the applied {side.value} book")
    return CandidatePlay(
        playbook=AppliedPlaybookRef(
            side=side,
            revision=revision,
            source_book=source_book,
            mode=mode,
        ),
        formation=formation,
        play=play,
        adjustment_slot=adjustment_slot,
        adjustment_id=adjustment_id,
        audible=audible,
        in_applied_book=Tri.TRUE,
        heuristic_score=heuristic_score,
        heuristic_p=heuristic_p,
    )


def legal_candidates(plays: Sequence[CandidatePlay]) -> tuple[CandidatePlay, ...]:
    """Candidates a ranker is allowed to score. Unconfirmed names are dropped."""
    return tuple(play for play in plays if play.in_applied_book is Tri.TRUE)


@dataclass(frozen=True)
class GameState:
    """One snap's context. Unset fields stay unknown.

    ``situation`` is the existing parsed line when the live parser ran.
    ``goal_to_go`` is separate because ``Situation.goal_line`` defaults to
    false when the phrase was absent, and that default is not a confirmation.
    """

    game_id: str | None = None
    snap_id: str | None = None
    madden_version: str | None = None
    profile: str | None = None
    season: int | None = None
    week: int | None = None
    our_team: str | None = None
    opponent_id: str | None = None
    opponent_team: str | None = None
    opponent_type: OpponentKind = OpponentKind.UNKNOWN
    possession: Possession = Possession.UNKNOWN
    situation: Situation | None = None
    score: GameScore | None = None
    quarter: int | None = None
    clock_seconds: int | None = None
    timeouts_us: int | None = None
    timeouts_them: int | None = None
    goal_to_go: Tri = Tri.UNKNOWN

    def __post_init__(self) -> None:
        _reject_bool_int("season", self.season)
        _reject_bool_int("week", self.week)
        _check_nonneg("clock_seconds", self.clock_seconds)
        _check_nonneg("timeouts_us", self.timeouts_us)
        _check_nonneg("timeouts_them", self.timeouts_them)
        if self.quarter is not None:
            _reject_bool_int("quarter", self.quarter)
            if not 1 <= self.quarter <= 5:
                raise ValueError("quarter must be 1..5 or unknown")
        if self.season is not None and self.season < 1:
            raise ValueError("season must be >= 1 or unknown")
        if self.week is not None and self.week < 1:
            raise ValueError("week must be >= 1 or unknown")


@dataclass(frozen=True)
class PreSnapObservation:
    """What was known before the ball was snapped.

    Coverage here is a prediction. Verified coverage after the play belongs
    on :class:`ObservedLook`. Pressure ``None`` means the indicator was not
    observed. It does not mean a clean pocket.
    """

    observation_id: str | None = None
    captured_at: str | None = None
    source: str | None = None
    confidence: float | None = None
    offensive_formation: str | None = None
    offensive_personnel: str | None = None
    defensive_formation: str | None = None
    defensive_front: str | None = None
    shell: str | None = None
    pressure: str | None = None
    motion: str | None = None
    coverage_prediction: str | None = None
    coverage_prediction_confidence: float | None = None
    coverage_prediction_source: HintSource = HintSource.UNKNOWN
    concept_prediction: str | None = None
    concept_prediction_source: HintSource = HintSource.UNKNOWN
    tendency_refs: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        _check_unit("confidence", self.confidence)
        _check_unit("coverage_prediction_confidence", self.coverage_prediction_confidence)
        leaked = LEAKAGE_FIELDS.intersection(pre_snap_field_names())
        if leaked:
            raise RuntimeError(f"pre-snap observation grew outcome fields: {sorted(leaked)}")


def pre_snap_field_names() -> frozenset[str]:
    return frozenset(item.name for item in fields(PreSnapObservation))


@dataclass(frozen=True)
class ObservedLook:
    """Coverage or concept identified after the snap, or from a later label."""

    coverage: str | None = None
    concept: str | None = None
    confidence: float | None = None
    source: str | None = None
    verification: Verification = Verification.UNKNOWN

    def __post_init__(self) -> None:
        _check_unit("confidence", self.confidence)


@dataclass(frozen=True)
class SnapOutcome:
    """Labels for the play that was actually graded.

    Each event is :class:`Tri` so a missing flag is not stored as "did not
    happen". ``yards`` ``None`` is unknown, including when the result string
    did not carry a number. Known yards lie in ``[YARDS_MIN, YARDS_MAX]``.
    ``parsed`` is the existing string parser when one was run; it is not
    copied into the other fields automatically.

    Offensive ``success`` uses the same rule as ``learning.SUCCESS_NEED``:
    40% of yards to go on 1st, 60% on 2nd, 100% on 3rd and 4th. A turnover
    (interception or lost fumble) is a failure. ``stop`` is the defensive
    side of that rule: the offense failed it, or turned the ball over.
    A sack is the sack flag, negative yards when yards are known, and an
    offensive failure. This type does not grade a snap by itself.

    An accepted penalty leaves ``success`` unknown and the snap is not a
    training label. A declined penalty keeps the play's label. A no-play is
    excluded: ``no_play`` true forces ``success`` to stay unknown.
    """

    snap_id: str | None = None
    yards: int | None = None
    success: Tri = Tri.UNKNOWN
    first_down: Tri = Tri.UNKNOWN
    stop: Tri = Tri.UNKNOWN
    touchdown: Tri = Tri.UNKNOWN
    sack: Tri = Tri.UNKNOWN
    incompletion: Tri = Tri.UNKNOWN
    interception: Tri = Tri.UNKNOWN
    fumble: Tri = Tri.UNKNOWN
    explosive: Tri = Tri.UNKNOWN
    penalty: PenaltyCall = PenaltyCall.UNKNOWN
    no_play: Tri = Tri.UNKNOWN
    drive_result: str | None = None
    confidence: float | None = None
    source: str | None = None
    verification: Verification = Verification.UNKNOWN
    look: ObservedLook | None = None
    parsed: ParsedOutcome | None = None

    def __post_init__(self) -> None:
        _check_yards("yards", self.yards)
        _check_unit("confidence", self.confidence)
        if self.penalty is PenaltyCall.ACCEPTED and self.success is not Tri.UNKNOWN:
            raise ValueError("an accepted penalty leaves the success label unknown")
        if self.no_play is Tri.TRUE and self.success is not Tri.UNKNOWN:
            raise ValueError("a no-play is excluded; the success label stays unknown")
        if self.sack is Tri.TRUE and self.yards is not None and self.yards >= 0:
            raise ValueError("a sack with known yards has negative yards")
        if self.sack is Tri.TRUE and self.success is Tri.TRUE:
            raise ValueError("a sack is an offensive failure")


@dataclass(frozen=True)
class ExecutedPlay:
    """The formation and play that were verified to have been run.

    This is not the recommendation. Leave it unset until something other than
    the on-screen call confirms it.
    """

    formation: str | None = None
    play: str | None = None
    adjustment_id: str | None = None
    adjustment_slot: int | None = None
    verification: Verification = Verification.UNKNOWN
    source: str | None = None

    def __post_init__(self) -> None:
        _check_slot(self.adjustment_slot)


@dataclass(frozen=True)
class RosterFeature:
    """One roster input. ``missing`` with both values empty is an explicit gap.

    A number and ``missing=True`` together is rejected. So is a present
    feature with no value. Absence from :class:`RosterSnapshot.features` means
    the import never mentioned the key.
    """

    key: str
    value_number: float | None = None
    value_text: str | None = None
    missing: bool = True

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("roster feature key is required")
        _check_finite(self.key, self.value_number)
        has_value = self.value_number is not None or self.value_text is not None
        if self.missing and has_value:
            raise ValueError(f"missing roster feature {self.key!r} cannot carry a value")
        if not self.missing and not has_value:
            raise ValueError(f"roster feature {self.key!r} needs a value or missing=True")


@dataclass(frozen=True)
class RosterSnapshot:
    """Versioned roster slice. ``completeness`` ``FALSE`` means known partial.

    ``UNKNOWN`` means the import did not say whether the slice is complete.
    An empty ``features`` tuple is not a complete roster.
    """

    roster_version: str | None = None
    profile: str | None = None
    team: str | None = None
    captured_at: str | None = None
    completeness: Tri = Tri.UNKNOWN
    features: tuple[RosterFeature, ...] = ()


@dataclass(frozen=True)
class TendencyCount:
    """One stored opponent bucket.

    ``count`` is the stored counter. ``sample_size`` is the n the Strategist
    used for that tendency. Neither is copied into the other. ``shrinkage_weight``
    stays unknown until the Strategist sets it.
    """

    bucket: str | None = None
    key: str | None = None
    count: int | None = None
    success: int | None = None
    sample_size: int | None = None
    shrinkage_weight: float | None = None

    def __post_init__(self) -> None:
        _check_nonneg("count", self.count)
        _check_nonneg("success", self.success)
        _check_nonneg("sample_size", self.sample_size)
        _check_unit("shrinkage_weight", self.shrinkage_weight)


@dataclass(frozen=True)
class OpponentContextKey:
    """Identity of one opponent-context row.

    The key is opponent id + patch id + roster-snapshot id. ``table_version``
    is the storage schema of that table. A missing part stays ``None``.
    """

    opponent_id: str | None = None
    patch_id: str | None = None
    roster_snapshot_id: str | None = None
    table_version: str | None = None


@dataclass(frozen=True)
class OpponentContext:
    """PROVISIONAL. Per-opponent context keyed like :class:`OpponentContextKey`.

    The Strategist owns the math and will add fields. This type's wire version
    is :data:`PROVISIONAL_SCHEMA_VERSION`, not :data:`CONTRACT_VERSION`, so
    those additions do not bump stable records.

    ``tendencies_known`` ``FALSE`` means a lookup ran and stored nothing.
    ``UNKNOWN`` means the lookup did not run. ``n_snaps`` ``None`` is not zero
    snaps. Per-tendency sample sizes live on :class:`TendencyCount`.
    ``sample_size`` and ``shrinkage_weight`` on this object are the optional
    context-level slots; leave them unknown until the Strategist fills them.

    There is no ``stale`` field. A new patch or roster is a new key.
    ``seeded_from_key`` and ``seed_shrink_weight`` are both unset or both set.
    The weight is in ``[0, 1]``. The seed key has this row's opponent id and
    is not this row's own key.
    """

    provisional_schema_version: str = PROVISIONAL_SCHEMA_VERSION
    opponent_id: str | None = None
    patch_id: str | None = None
    roster_snapshot_id: str | None = None
    table_version: str | None = None
    opponent_type: OpponentKind = OpponentKind.UNKNOWN
    team: str | None = None
    built_from_snap_id: str | None = None
    n_snaps: int | None = None
    built_ts: str | None = None
    last_reset_ts: str | None = None
    last_reset_reason: str | None = None
    tendencies_known: Tri = Tri.UNKNOWN
    tendencies: tuple[TendencyCount, ...] = ()
    global_prior_id: str | None = None
    sample_size: int | None = None
    shrinkage_weight: float | None = None
    seeded_from_key: OpponentContextKey | None = None
    seed_shrink_weight: float | None = None

    def __post_init__(self) -> None:
        if self.provisional_schema_version != PROVISIONAL_SCHEMA_VERSION:
            raise ValueError(
                f"OpponentContext version is fixed at {PROVISIONAL_SCHEMA_VERSION}"
            )
        _check_nonneg("n_snaps", self.n_snaps)
        _check_nonneg("sample_size", self.sample_size)
        _check_unit("shrinkage_weight", self.shrinkage_weight)
        self.validate()

    def validate(self) -> None:
        """Reject a seed key and weight that do not describe one prior row."""
        seeded = self.seeded_from_key is not None
        weighted = self.seed_shrink_weight is not None
        if seeded != weighted:
            raise ValueError("seeded_from_key and seed_shrink_weight are both set or both unset")
        seed = self.seeded_from_key
        if seed is None:
            return
        _check_unit("seed_shrink_weight", self.seed_shrink_weight)
        if seed.opponent_id != self.opponent_id:
            raise ValueError("seeded_from_key must have the same opponent id as this row")
        if seed == self.key():
            raise ValueError("seeded_from_key must not equal this row's key")

    def key(self) -> OpponentContextKey:
        return OpponentContextKey(
            opponent_id=self.opponent_id,
            patch_id=self.patch_id,
            roster_snapshot_id=self.roster_snapshot_id,
            table_version=self.table_version,
        )


@dataclass(frozen=True)
class RecentSnap:
    """A snap that already ended before the decision being made.

    Do not put the current snap's outcome here. Historical outcomes are legal
    context. The current snap's outcome is not.
    """

    snap_id: str | None = None
    possession: Possession = Possession.UNKNOWN
    formation: str | None = None
    play: str | None = None
    outcome: SnapOutcome | None = None


@dataclass(frozen=True)
class OutcomeEstimates:
    """Model outputs for one candidate. Unset estimates stay unknown.

    Offense uses success, expected yards, explosive, sack, and turnover.
    Defense uses stop probability and expected yards allowed. A model that
    cannot support a head leaves that head unknown instead of copying another.
    ``n`` is the sample behind the estimate, not a fill-in of zero.
    """

    success_probability: float | None = None
    expected_yards: float | None = None
    explosive_probability: float | None = None
    sack_risk: float | None = None
    turnover_risk: float | None = None
    stop_probability: float | None = None
    expected_yards_allowed: float | None = None
    n: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "success_probability",
            "explosive_probability",
            "sack_risk",
            "turnover_risk",
            "stop_probability",
        ):
            _check_unit(name, getattr(self, name))
        _check_yards("expected_yards", self.expected_yards)
        _check_yards("expected_yards_allowed", self.expected_yards_allowed)
        _check_nonneg("n", self.n)


@dataclass(frozen=True)
class CandidateScore:
    """One score attached to ``candidates[candidate_index]`` when the index is known."""

    candidate_index: int | None = None
    source: str | None = None
    total: float | None = None
    learned: float | None = None
    meta: float | None = None

    def __post_init__(self) -> None:
        _check_nonneg("candidate_index", self.candidate_index)
        for name in ("total", "learned", "meta"):
            _check_finite(name, getattr(self, name))


@dataclass(frozen=True)
class StagePick:
    """What a decision stage held after it ran. ``changed`` unknown if not compared."""

    stage: DecisionStage
    formation: str | None = None
    play: str | None = None
    macro_id: str | None = None
    changed: Tri = Tri.UNKNOWN


@dataclass(frozen=True)
class ScorePart:
    """One named term in a ranking score. ``value`` stays unknown until set."""

    name: str
    value: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("score part name is required")
        _check_finite(self.name, self.value)


@dataclass(frozen=True)
class ShadowScore:
    """A model score for one candidate. This is not a behavior propensity.

    ``p`` is the model's probability for that candidate. The policy's sampling
    probability is :attr:`RankedPlay.sampling_p` and, for the shown play,
    :attr:`CoachingDecision.behavior_propensity`.
    """

    candidate_key: str | None = None
    score: float | None = None
    p: float | None = None

    def __post_init__(self) -> None:
        _check_finite("score", self.score)
        _check_unit("p", self.p)


@dataclass(frozen=True)
class RankedPlay:
    """One ranked candidate.

    ``sampling_p`` is this candidate's weight in the policy distribution named
    by the parent :class:`PlayRanking`. It may be zero. It is not the model's
    success probability. ``score_breakdown``, ``sample_size``, and
    ``shrinkage_weight`` are optional room for the Strategist.
    """

    candidate: CandidatePlay
    estimates: OutcomeEstimates
    confidence: float | None = None
    reasons: tuple[str, ...] = ()
    rank: int | None = None
    sampling_p: float | None = None
    sample_size: int | None = None
    shrinkage_weight: float | None = None
    score_breakdown: tuple[ScorePart, ...] | None = None

    def __post_init__(self) -> None:
        _check_unit("confidence", self.confidence)
        _check_unit("sampling_p", self.sampling_p)
        _check_nonneg("sample_size", self.sample_size)
        _check_unit("shrinkage_weight", self.shrinkage_weight)
        if self.rank is not None:
            _reject_bool_int("rank", self.rank)
            if self.rank < 1:
                raise ValueError("rank must be >= 1 or unknown")


@dataclass(frozen=True)
class PlayRanking:
    """PROVISIONAL return type of ``rank_plays``.

    Wire version :data:`PROVISIONAL_SCHEMA_VERSION`. An empty ``plays`` tuple
    is not a ranking of the book. ``policy_source`` names the distribution in
    each play's ``sampling_p``. ``shadow_scores`` are model outputs and stay
    off the propensity. ``sample_size`` and ``shrinkage_weight`` are optional
    ranking-level slots for the Strategist.

    ``status`` is the ranker's own :class:`MLStatus`. It may only be ``ok``,
    ``low_confidence``, ``model_missing``, or ``invalid_output``. Timeouts and
    other inference failures stay on :class:`CoachingDecision`. This type has
    no ``fallback_reason``. ``status_detail`` is free text for that status.
    A legacy payload that omits ``status`` loads as ``ok``.
    """

    provisional_schema_version: str = PROVISIONAL_SCHEMA_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    model_version: str | None = None
    policy_version: str | None = None
    policy_source: PolicySource = PolicySource.UNKNOWN
    mode: CoachingMode = CoachingMode.HEURISTIC
    plays: tuple[RankedPlay, ...] = ()
    confidence: float | None = None
    low_confidence: Tri = Tri.UNKNOWN
    shadow_scores: tuple[ShadowScore, ...] | None = None
    sample_size: int | None = None
    shrinkage_weight: float | None = None
    status: MLStatus = MLStatus.OK
    status_detail: str | None = None

    def __post_init__(self) -> None:
        if self.provisional_schema_version != PROVISIONAL_SCHEMA_VERSION:
            raise ValueError(f"PlayRanking version is fixed at {PROVISIONAL_SCHEMA_VERSION}")
        _check_unit("confidence", self.confidence)
        _check_nonneg("sample_size", self.sample_size)
        _check_unit("shrinkage_weight", self.shrinkage_weight)
        self.validate()

    def validate(self) -> None:
        """Reject a ranking status that inference, not rank_plays, would set."""
        if self.status not in RANKING_STATUSES:
            raise ValueError(
                "PlayRanking.status must be ok, low_confidence, model_missing, or invalid_output"
            )


@dataclass(frozen=True)
class CoachingDecision:
    """One sealed recommendation, kept apart from what was later verified.

    ``mode`` is the coaching mode (heuristic, shadow, hybrid, or unknown). It
    defaults to heuristic, which is the product default, including for legacy
    payloads that omit the field. ``effective_mode`` stays unknown until a
    session resolves downgrades. ``heuristic_pick`` is the deterministic
    caller's choice. ``final_pick`` is what this mode showed.
    ``policy_source`` names the policy that made the shown call. A heuristic
    or shadow decision, and any hybrid decision that fell back, uses the
    heuristic policy. The field defaults to heuristic so a blank record is a
    valid heuristic decision.
    ``behavior_propensity`` is that policy's real end-to-end sampling
    probability of the shown play after anti-repeat, pivot, the book guard,
    and the VOD override. It is in ``(0, 1]`` when known.
    ``propensity_override`` is true exactly when ``propensity_method`` is
    ``OVERRIDE``. ``propensity_override_reason`` is ``None`` unless that flag
    is true. A VOD override is logged as ``propensity_method=DETERMINISTIC``
    and ``behavior_propensity=1.0``. Those snaps are excluded from off-policy
    evaluation. Model scores go only in ``shadow_scores`` and ``shadow_pick``.

    ``accepted`` stays unknown until the user confirms. Displaying a call does
    not accept it. ``executed_status`` stays unknown until the user says
    otherwise. ``executed`` is set only when that status is ``identified`` and
    the play is verified.

    ``fell_back`` is the fallback flag. Heuristic and shadow decisions do not
    fall back: the flag is false, ``fallback_reason`` is unset, and
    ``policy_source`` is heuristic. ``fallback_reason`` is an :class:`MLStatus`
    and is never ``OK``. It is set only when ``fell_back`` is true, which is
    legal only in hybrid mode, and that hybrid fallback still has
    ``policy_source`` heuristic. ``shadow_status`` is ``None`` in heuristic
    mode and required in shadow and hybrid. The shadow-game counter counts
    only rows with ``shadow_status == OK``. A timeout or other non-OK status
    is a logged model result, not a counted shadow game.

    ``ml_seed`` is the optional seed for that decision. ``None`` means the run
    was unseeded. ``candidates`` ``None`` means the pool was not recorded. An
    empty tuple means the pool was recorded and empty.

    ``latency_ms`` is the single latency figure when one number was stored.
    The heuristic and model timings are the split fields. The model budget,
    when recorded, is the fixed :data:`ML_LATENCY_BUDGET_MS`.
    """

    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    decision_schema_version: str = DECISION_SCHEMA_VERSION
    decision_ts: str | None = None
    latency_ms: float | None = None
    latency_ms_heuristic: float | None = None
    latency_ms_model: float | None = None
    inference_budget_ms: int | None = None
    mode: CoachingMode = CoachingMode.HEURISTIC
    effective_mode: CoachingMode = CoachingMode.UNKNOWN
    mode_downgrade_reason: str | None = None
    game_id: str | None = None
    snap_id: str | None = None
    session_id: str | None = None
    snap_seq: int | None = None
    model_version: str | None = None
    registry_id: str | None = None
    policy_version: str | None = None
    policy_source: PolicySource = PolicySource.HEURISTIC
    ml_seed: int | None = None
    candidates: tuple[CandidatePlay, ...] | None = None
    scores: tuple[CandidateScore, ...] | None = None
    heuristic_pick: CandidatePlay | None = None
    final_pick: CandidatePlay | None = None
    macro: AdjustmentRef | None = None
    accepted: Tri = Tri.UNKNOWN
    executed_status: ExecutedStatus = ExecutedStatus.UNKNOWN
    executed: ExecutedPlay | None = None
    behavior_propensity: float | None = None
    propensity_method: PropensityMethod = PropensityMethod.UNKNOWN
    propensity_override: bool = False
    propensity_override_reason: PropensityOverrideReason | None = None
    heuristic_sample_p: float | None = None
    heuristic_dist: tuple[tuple[str, float], ...] | None = None
    propensity_ms: float | None = None
    heuristic_trace_hash: str | None = None
    ml_status_detail: str | None = None
    shadow_scores: tuple[ShadowScore, ...] | None = None
    shadow_pick: CandidatePlay | None = None
    shadow_status: MLStatus | None = None
    fell_back: bool | None = False
    fallback_reason: MLStatus | None = None
    opponent_context_key: OpponentContextKey | None = None
    roster_snapshot_id: str | None = None
    config_source: tuple[tuple[str, str], ...] | None = None
    displayed_text: str | None = None
    trace: tuple[StagePick, ...] | None = None

    def __post_init__(self) -> None:
        for name in ("latency_ms", "latency_ms_heuristic", "latency_ms_model"):
            value = getattr(self, name)
            _check_finite(name, value)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or unknown")
        if self.inference_budget_ms is not None and self.inference_budget_ms != ML_LATENCY_BUDGET_MS:
            raise ValueError(
                f"inference_budget_ms must be {ML_LATENCY_BUDGET_MS} or unknown; it is not self-tuned"
            )
        _reject_bool_int("ml_seed", self.ml_seed)
        _check_nonneg("snap_seq", self.snap_seq)
        _check_open_unit("behavior_propensity", self.behavior_propensity)
        _check_unit("heuristic_sample_p", self.heuristic_sample_p)
        if self.propensity_ms is not None:
            _check_finite("propensity_ms", self.propensity_ms)
            if self.propensity_ms < 0:
                raise ValueError("propensity_ms must be >= 0 or unknown")
        if self.heuristic_dist is not None:
            for key, value in self.heuristic_dist:
                if not key:
                    raise ValueError("heuristic_dist keys are candidate keys")
                _check_unit(f"heuristic_dist[{key}]", value)
        if self.executed_status is ExecutedStatus.IDENTIFIED:
            if self.executed is None or self.executed.verification is not Verification.VERIFIED:
                raise ValueError("an identified execution requires a verified ExecutedPlay")
            if not self.executed.formation or not self.executed.play:
                raise ValueError("an identified execution needs a formation and a play")
        elif self.executed is not None:
            raise ValueError("executed play is set only when executed_status is identified")
        self.validate()

    def validate(self) -> None:
        """Reject mode, fallback, and propensity combinations the log cannot mean.

        The shadow-game counter counts only rows with ``shadow_status == OK``.
        """

        _check_propensity_pair(
            self.propensity_method,
            self.propensity_override,
            self.propensity_override_reason,
        )
        if self.fallback_reason is MLStatus.OK:
            raise ValueError("fallback_reason must never be OK")
        if self.fell_back is True:
            if self.mode is not CoachingMode.HYBRID:
                raise ValueError("fallback is only allowed in hybrid mode")
            if self.fallback_reason is None:
                raise ValueError("fallback requires a fallback_reason")
            if self.policy_source is not PolicySource.HEURISTIC:
                raise ValueError("a hybrid fallback has policy_source heuristic")
        elif self.fallback_reason is not None:
            raise ValueError("fallback_reason is set only when fallback is true")

        if self.mode in (CoachingMode.HEURISTIC, CoachingMode.SHADOW):
            if self.fell_back is not False:
                raise ValueError("heuristic and shadow mode do not fall back")
            if self.policy_source is not PolicySource.HEURISTIC:
                raise ValueError("heuristic and shadow mode use policy_source heuristic")
        if self.mode is CoachingMode.HEURISTIC and self.shadow_status is not None:
            raise ValueError("shadow_status is None in heuristic mode")
        if self.mode in (CoachingMode.SHADOW, CoachingMode.HYBRID) and self.shadow_status is None:
            raise ValueError("shadow_status is required in shadow and hybrid mode")


@dataclass(frozen=True)
class FeatureValue:
    """One decision-time feature. Missing is explicit so ``0`` can stay a real value."""

    name: str
    value: float | None = None
    missing: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("feature name is required")
        _check_finite(self.name, self.value)
        if self.missing and self.value is not None:
            raise ValueError(f"missing feature {self.name!r} cannot carry a value")
        if not self.missing and self.value is None:
            raise ValueError(f"feature {self.name!r} needs a value or missing=True")


@dataclass(frozen=True)
class FeatureVector:
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    items: tuple[FeatureValue, ...] = ()


@dataclass(frozen=True)
class DataHash:
    """Hash of one training input. ``sha256`` stays unknown until the bytes were hashed."""

    label: str
    sha256: str | None = None
    uri: str | None = None

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("data hash label is required")


@dataclass(frozen=True)
class MetricRecord:
    """One held-out metric. ``value`` ``None`` means it was not computed."""

    name: str
    value: float | None = None
    split: str | None = None
    n: int | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name is required")
        _check_finite(self.name, self.value)
        _check_nonneg("n", self.n)


@dataclass(frozen=True)
class ModelRegistryEntry:
    """Provenance for one trained artifact. The gate starts at ``not_run``.

    Nothing in this package promotes an entry. ``metrics`` may be empty.
    Empty metrics are not a passing gate.
    """

    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    model_version: str | None = None
    code_version: str | None = None
    game_version: str | None = None
    data_hashes: tuple[DataHash, ...] = ()
    metrics: tuple[MetricRecord, ...] = ()
    gate: GateResult = GateResult.NOT_RUN
    gate_passed: bool | None = None
    gate_report_path: str | None = None
    seed: int | None = None
    trained_at: str | None = None
    created_ts: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    data_hash: str | None = None
    side: Possession = Possession.UNKNOWN
    opponent_type_scope: OpponentKind = OpponentKind.UNKNOWN
    title_update: str | None = None

    def __post_init__(self) -> None:
        _reject_bool_int("seed", self.seed)
        if self.gate_passed is True and self.gate is not GateResult.PASSED:
            raise ValueError("gate_passed is true only when gate is passed")
        if self.gate_passed is False and self.gate is GateResult.PASSED:
            raise ValueError("a passed gate cannot have gate_passed false")


@dataclass(frozen=True)
class MLConfig:
    """Resolved ML settings for one session. These are defaults, not measurements.

    ``mode`` is heuristic unless a caller selects another runnable mode.
    ``low_confidence`` and ``latency_budget_ms`` are the fixed constants.
    Passing any other number raises. ``seed`` ``None`` leaves the heuristic
    unseeded. A set seed is logged on each decision as ``ml_seed``.
    """

    mode: CoachingMode = CoachingMode.HEURISTIC
    disabled: bool = False
    model_version: str | None = None
    low_confidence: float = ML_LOW_CONFIDENCE
    latency_budget_ms: int = ML_LATENCY_BUDGET_MS
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode is CoachingMode.UNKNOWN:
            raise ValueError("ml.mode cannot be unknown; the default is heuristic")
        if self.low_confidence != ML_LOW_CONFIDENCE:
            raise ValueError(f"ml.low_confidence is fixed at {ML_LOW_CONFIDENCE}")
        if self.latency_budget_ms != ML_LATENCY_BUDGET_MS:
            raise ValueError(
                f"ml.latency_budget_ms is fixed at {ML_LATENCY_BUDGET_MS} and is not self-tuned"
            )
        _reject_bool_int("ml.seed", self.seed)


SCHEMA_RECORD_TYPES: tuple[type, ...] = (
    AppliedPlaybookRef,
    AdjustmentRef,
    CandidatePlay,
    GameState,
    PreSnapObservation,
    ObservedLook,
    SnapOutcome,
    ExecutedPlay,
    RosterFeature,
    RosterSnapshot,
    TendencyCount,
    OpponentContextKey,
    OpponentContext,
    RecentSnap,
    OutcomeEstimates,
    CandidateScore,
    StagePick,
    ScorePart,
    ShadowScore,
    RankedPlay,
    PlayRanking,
    CoachingDecision,
    FeatureValue,
    FeatureVector,
    DataHash,
    MetricRecord,
    ModelRegistryEntry,
    MLConfig,
    Situation,
    GameScore,
    ParsedOutcome,
)

# Strategist-owned records. Their wire version is PROVISIONAL_SCHEMA_VERSION.
PROVISIONAL_TYPES: tuple[type, ...] = (
    PlayRanking,
    OpponentContext,
)

_TYPE_BY_NAME: dict[str, type] = {cls.__name__: cls for cls in SCHEMA_RECORD_TYPES}


def schema_version_for(cls: type) -> str:
    """Wire version for ``cls``. Provisional types do not use the stable version."""
    if cls in PROVISIONAL_TYPES:
        return PROVISIONAL_SCHEMA_VERSION
    return CONTRACT_VERSION


def to_dict(obj: Any) -> Any:
    """JSON-ready data. Dataclasses carry ``__type__`` and ``__schema__``.

    ``None`` becomes JSON ``null``. Provisional records use
    :data:`PROVISIONAL_SCHEMA_VERSION`. A coaching decision is checked so
    ``propensity_override`` still matches ``propensity_method``.
    """
    if isinstance(obj, CoachingDecision):
        _check_propensity_pair(
            obj.propensity_method,
            obj.propensity_override,
            obj.propensity_override_reason,
        )
    if is_dataclass(obj) and not isinstance(obj, type):
        payload = {item.name: to_dict(getattr(obj, item.name)) for item in fields(obj)}
        payload["__type__"] = type(obj).__name__
        payload["__schema__"] = schema_version_for(type(obj))
        return payload
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, tuple):
        return [to_dict(item) for item in obj]
    if isinstance(obj, list):
        return [to_dict(item) for item in obj]
    if isinstance(obj, dict):
        return {str(key): to_dict(value) for key, value in obj.items()}
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    raise TypeError(f"cannot serialize {type(obj).__name__}")


def from_dict(payload: Any) -> Any:
    """Inverse of :func:`to_dict` for a schema record.

    A field missing from a legacy payload uses the dataclass default.
    For :class:`CoachingDecision` that is heuristic mode, heuristic
    ``policy_source``, ``fell_back=False``, ``fallback_reason=None``,
    ``shadow_status=None``, ``propensity_method=UNKNOWN``, and
    ``propensity_override_reason=None``. :meth:`CoachingDecision.validate`
    runs again after the object is built.
    """
    if not isinstance(payload, dict) or "__type__" not in payload:
        raise TypeError("serialized record is missing __type__")
    loaded = _load_typed(payload)
    if isinstance(loaded, CoachingDecision):
        loaded.validate()
    return loaded


def csv_cell(value: Any) -> str:
    """CSV encoding. ``None`` is ``__UNKNOWN__``. Enums use their value."""
    if value is None:
        return CSV_UNKNOWN
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_csv_cell(text: str) -> str | None:
    """Inverse of :func:`csv_cell` for a missing cell. Other text is returned as-is."""
    if text == CSV_UNKNOWN:
        return None
    return text


def _load_typed(payload: Mapping[str, Any]) -> Any:
    name = payload.get("__type__")
    cls = _TYPE_BY_NAME.get(str(name))
    if cls is None:
        raise TypeError(f"unknown schema type {name!r}")
    version = payload.get("__schema__")
    expected = schema_version_for(cls)
    if version not in (None, expected):
        raise ValueError(f"{cls.__name__} schema {version!r} does not match {expected}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in payload:
            kwargs[item.name] = _field_default(cls, item)
            continue
        kwargs[item.name] = _decode(hints[item.name], payload[item.name])
    return cls(**kwargs)


def _field_default(cls: type, item: Any) -> Any:
    """Default for a key a legacy payload left out. Required fields still raise."""
    if item.default is not MISSING:
        return item.default
    if item.default_factory is not MISSING:
        return item.default_factory()
    raise ValueError(f"{cls.__name__} payload missing {item.name}")


def _decode(hint: Any, data: Any) -> Any:
    if isinstance(data, dict) and "__type__" in data:
        loaded = _load_typed(data)
        if hint is Any:
            return loaded
        origin = get_origin(hint)
        if origin in (Union, UnionType):
            allowed = [arg for arg in get_args(hint) if arg is not type(None)]
            if any(_is_instance(loaded, arg) for arg in allowed):
                return loaded
            raise TypeError(f"{type(loaded).__name__} does not match {hint!r}")
        if _is_instance(loaded, hint):
            return loaded
        raise TypeError(f"{type(loaded).__name__} does not match {hint!r}")

    if hint is Any:
        if isinstance(data, list):
            return [_decode(Any, item) for item in data]
        if isinstance(data, dict):
            return {str(key): _decode(Any, value) for key, value in data.items()}
        return data

    origin = get_origin(hint)
    if origin in (Union, UnionType):
        args = get_args(hint)
        if data is None and type(None) in args:
            return None
        options = [arg for arg in args if arg is not type(None)]
        if len(options) == 1:
            return _decode(options[0], data)
        errors: list[str] = []
        for option in options:
            try:
                return _decode(option, data)
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        raise TypeError("; ".join(errors) or f"cannot decode {data!r} as {hint!r}")

    if origin is tuple:
        if not isinstance(data, list):
            raise TypeError("expected a list for a tuple field")
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], item) for item in data)
        if len(args) == len(data):
            return tuple(_decode(arg, item) for arg, item in zip(args, data))
        raise TypeError("tuple length does not match its annotation")

    if origin is dict or hint is dict:
        if not isinstance(data, dict):
            raise TypeError("expected an object")
        args = get_args(hint)
        value_type = args[1] if len(args) == 2 else Any
        return {str(key): _decode(value_type, value) for key, value in data.items()}

    if origin is list:
        if not isinstance(data, list):
            raise TypeError("expected a list")
        args = get_args(hint)
        item_type = args[0] if args else Any
        return [_decode(item_type, item) for item in data]

    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint(data)

    if hint is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise TypeError("expected a number")
        number = float(data)
        if not math.isfinite(number):
            raise ValueError("non-finite number")
        return number

    if hint is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise TypeError("expected an int")
        return data

    if hint is str:
        if not isinstance(data, str):
            raise TypeError("expected a string")
        return data

    if hint is bool:
        if not isinstance(data, bool):
            raise TypeError("expected a bool")
        return data

    if hint is type(None):
        if data is not None:
            raise TypeError("expected null")
        return None

    if isinstance(hint, type) and is_dataclass(hint):
        raise TypeError(f"{hint.__name__} payload missing __type__")

    raise TypeError(f"unsupported annotation {hint!r}")


def _is_instance(loaded: Any, hint: Any) -> bool:
    origin = get_origin(hint)
    if origin is not None:
        return True
    return isinstance(hint, type) and isinstance(loaded, hint)


__all__ = [
    "CONTRACT_VERSION",
    "CSV_UNKNOWN",
    "DECISION_SCHEMA_VERSION",
    "FEATURE_SCHEMA_VERSION",
    "LEAKAGE_FIELDS",
    "LOADOUT_N",
    "ML_LATENCY_BUDGET_MS",
    "ML_LOW_CONFIDENCE",
    "PROVISIONAL_SCHEMA_VERSION",
    "PROVISIONAL_TYPES",
    "RANKING_STATUSES",
    "SCHEMA_RECORD_TYPES",
    "YARDS_MAX",
    "YARDS_MIN",
    "AdjustmentRef",
    "AppliedPlaybookRef",
    "CandidatePlay",
    "CandidateScore",
    "CoachingDecision",
    "CoachingMode",
    "MLStatus",
    "DataHash",
    "DecisionStage",
    "ExecutedPlay",
    "ExecutedStatus",
    "FeatureValue",
    "FeatureVector",
    "GameScore",
    "GameState",
    "GateResult",
    "HintSource",
    "MLConfig",
    "MetricRecord",
    "ModelRegistryEntry",
    "ObservedLook",
    "OpponentContext",
    "OpponentContextKey",
    "OpponentKind",
    "OutcomeEstimates",
    "ParsedOutcome",
    "PenaltyCall",
    "PlayRanking",
    "PolicySource",
    "PropensityMethod",
    "PropensityOverrideReason",
    "Possession",
    "PreSnapObservation",
    "RankedPlay",
    "RecentSnap",
    "RosterFeature",
    "RosterSnapshot",
    "ScorePart",
    "ShadowScore",
    "Situation",
    "SnapOutcome",
    "StagePick",
    "TendencyCount",
    "Tri",
    "Verification",
    "candidate_in_book",
    "csv_cell",
    "from_dict",
    "legal_candidates",
    "parse_csv_cell",
    "pre_snap_field_names",
    "schema_version_for",
    "to_dict",
]
