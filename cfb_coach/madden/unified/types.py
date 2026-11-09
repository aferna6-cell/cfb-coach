"""Versioned, typed football objects shared by the offensive and defensive coordinators.

Rules this module enforces by construction:
* A recommendation is never an execution. ``SnapObservation.executed`` is ``None``
  and ``executed_status`` is ``unknown`` until someone records it.
* An opponent's revealed call lives in ``opponent_action`` and is never copied
  into ``executed``.
* Unknown is ``None`` / ``"UNKNOWN"``, never a guess.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

SCHEMA_VERSION = "unified.v1"
OFF, DEF = "offense", "defense"
SIDES = (OFF, DEF)
UNKNOWN = "UNKNOWN"
# executed_status values
EXEC_UNKNOWN, EXEC_CONFIRMED, EXEC_MODIFIED, EXEC_NOT_RUN = "unknown", "confirmed", "modified", "not_executed"
# verification levels for observations, strongest first
VERIFICATION = ("owner_confirmed", "typed_exact", "visual_inferred", "unverified")


def other_side(side: str) -> str:
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return DEF if side == OFF else OFF


def stable_hash(obj: Any, n: int = 12) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:n]


@dataclass(frozen=True)
class GameState:
    """One snap's pre-snap context, identical for offense and defense.

    ``our_side`` is the side WE play on this snap. Yard line is yards from our
    own goal when we have the ball (1..99) and yards from the opponent's goal
    when they do — i.e. always "yards the ball is from the goal the offense is
    defending", so ``to_goal = 100 - yardline`` for both sides.
    """

    game_id: str
    seq: int
    our_side: str
    opponent_id: str
    down: int | None = None
    distance: int | None = None
    yardline: int | None = None
    franchise_id: str | None = None
    session_id: str | None = None
    opponent_class: str = UNKNOWN  # cpu | human | UNKNOWN
    quarter: int | None = None
    score_us: int | None = None
    score_opp: int | None = None
    clock_seconds: int | None = None  # optional; never required
    personnel: str | None = None
    playbook_fingerprint: str | None = None
    info_available_ts: str | None = None
    info_confidence: float = 1.0
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.our_side not in SIDES:
            raise ValueError(f"our_side must be one of {SIDES}")
        if not isinstance(self.seq, int) or isinstance(self.seq, bool) or self.seq < 1:
            raise ValueError("seq must be an int >= 1")
        if self.down is not None and self.down not in (1, 2, 3, 4):
            raise ValueError("down must be 1-4 or None")
        if self.yardline is not None and not 1 <= self.yardline <= 99:
            raise ValueError("yardline must be 1..99 or None")

    @property
    def to_goal(self) -> int | None:
        return None if self.yardline is None else 100 - self.yardline

    @property
    def red_zone(self) -> bool:
        tg = self.to_goal
        return tg is not None and tg <= 20

    @property
    def goal_to_go(self) -> bool:
        tg = self.to_goal
        return tg is not None and self.distance is not None and tg <= self.distance

    @property
    def snap_id(self) -> str:
        return f"{self.game_id}-{self.seq:04d}"

    def dist_bucket(self) -> str:
        d = self.distance
        if d is None:
            return "d?"
        return "short" if d <= 3 else "med" if d <= 7 else "long" if d <= 12 else "xlong"

    def situation_key(self) -> str:
        """Coarse situation used for smoothing; shared by both sides."""
        dn = f"{self.down}" if self.down else "?"
        zone = "gl" if self.goal_to_go else "rz" if self.red_zone else "own" if (self.yardline or 50) < 25 else "mid"
        return f"{dn}|{self.dist_bucket()}|{zone}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GameState":
        if d.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise ValueError("unsupported GameState schema_version")
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass(frozen=True)
class Assignment:
    slot: str
    role: str  # e.g. route family, coverage duty. UNKNOWN allowed.
    detail: str | None = None
    verified: bool = False
    source: str | None = None


@dataclass(frozen=True)
class FootballPlay:
    """A stock play with concept hierarchy, provenance and explicit unknowns."""

    side: str
    book: str | None
    formation: str
    play: str
    personnel: str | None = None
    # offense: (play_type, concept) ; defense: (coverage_type, coverage_family)
    concept_path: tuple[str, ...] = ()
    # opponent-facing signature: coarse identity the OTHER side's beliefs/models use
    signature: tuple[str, ...] = ()
    variant: tuple[str, ...] = ()  # exact variant tokens (hole, robber, sky, palms ...)
    features: Mapping[str, Any] = field(default_factory=dict)
    assignments: tuple[Assignment, ...] = ()
    confidence: float = 0.35
    provenance: tuple[str, ...] = ()
    unknown_fields: tuple[str, ...] = ()

    @property
    def play_id(self) -> str:
        return f"{self.side[:3]}:{self.formation}|{self.play}"

    @property
    def concept(self) -> str:
        return self.concept_path[-1] if self.concept_path else UNKNOWN

    @property
    def play_type(self) -> str:
        return self.concept_path[0] if self.concept_path else UNKNOWN

    @property
    def verified_assignments(self) -> int:
        return sum(1 for a in self.assignments if a.verified)


@dataclass(frozen=True)
class Adjustment:
    action_id: str  # executable_actions.json id
    label: str
    params: Mapping[str, str] = field(default_factory=dict)
    support: str = "weak_prior"  # structural_verified | learned | weak_prior
    exec_cost_s: float = 2.0

    @property
    def key(self) -> str:
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.action_id}[{p}]"


@dataclass(frozen=True)
class FootballAction:
    """Stock play + zero or more compatible adjustments (NO ADJUSTMENT = empty)."""

    play: FootballPlay
    adjustments: tuple[Adjustment, ...] = ()
    modified_features: Mapping[str, Any] = field(default_factory=dict)
    legal: bool = True
    illegal_reason: str | None = None
    executable: bool = True

    @property
    def key(self) -> str:
        adj = "+".join(a.key for a in self.adjustments) or "NO_ADJ"
        return f"{self.play.play_id}#{adj}"

    @property
    def no_adjustment(self) -> bool:
        return not self.adjustments

    @property
    def exec_cost_s(self) -> float:
        return sum(a.exec_cost_s for a in self.adjustments)

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(self.modified_features.get("signature", self.play.signature))

    def effective(self, name: str, default: Any = None) -> Any:
        if name in self.modified_features:
            return self.modified_features[name]
        return self.play.features.get(name, default)


@dataclass(frozen=True)
class Outcome:
    yards: float | None = None
    first_down: bool | None = None
    touchdown: bool | None = None
    turnover: bool | None = None
    sack: bool | None = None
    explosive: bool | None = None
    result_text: str | None = None

    @property
    def known(self) -> bool:
        return self.yards is not None


@dataclass(frozen=True)
class OpponentAction:
    """What the other side was observed/revealed to run. Never our execution."""

    side: str
    formation: str | None = None
    play: str | None = None
    signature: tuple[str, ...] = ()
    verification: str = "unverified"
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.verification not in VERIFICATION:
            raise ValueError(f"verification must be one of {VERIFICATION}")


@dataclass(frozen=True)
class SnapObservation:
    state: GameState
    recommended: FootballAction | None = None
    executed: FootballAction | None = None
    executed_status: str = EXEC_UNKNOWN
    actual_adjustments: tuple[Adjustment, ...] | None = None  # None == unknown
    opponent_action: OpponentAction | None = None
    outcome: Outcome | None = None
    data_source: str = "live"
    verification: str = "unverified"
    closed: bool = False
    closed_ts: str | None = None
    cutoff_seq: int | None = None  # only seq < cutoff may use this row for learning
    decision_id: str | None = None

    @property
    def snap_id(self) -> str:
        return self.state.snap_id

    def with_execution(self, executed: FootballAction | None, status: str,
                       actual_adjustments: tuple[Adjustment, ...] | None = None) -> "SnapObservation":
        if status not in (EXEC_UNKNOWN, EXEC_CONFIRMED, EXEC_MODIFIED, EXEC_NOT_RUN):
            raise ValueError("bad executed_status")
        if status in (EXEC_CONFIRMED, EXEC_MODIFIED) and executed is None:
            raise ValueError("a confirmed/modified execution needs the executed action")
        return replace(self, executed=executed, executed_status=status, actual_adjustments=actual_adjustments)

    def with_opponent(self, action: OpponentAction) -> "SnapObservation":
        # intentionally touches ONLY opponent_action
        return replace(self, opponent_action=action)

    @property
    def labelled_for_own_learning(self) -> bool:
        """Own-execution learning needs a verified execution, not a recommendation."""
        return self.executed is not None and self.executed_status in (EXEC_CONFIRMED, EXEC_MODIFIED) \
            and self.outcome is not None and self.outcome.known
