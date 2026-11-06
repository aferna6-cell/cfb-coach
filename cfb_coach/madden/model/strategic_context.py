"""One typed pre-snap picture of the football situation, for either side of the ball.

Built from the existing ``Situation``, an opponent profile from
``opponent_tendencies`` and the earlier snaps of the current game. Every
value carries a status (known, derived, inferred, missing) and a source.
Nothing missing is filled in, and nothing about the opponent is taken from
our own call.

Yard-line convention (as logged): on offense ``yardline`` is yards from our
own goal; on defense it is yards from the goal we defend. ``yards_to_goal``
is always the offense's distance to the end zone.

Only information available before the snap belongs here. Results and the
concept or coverage seen on *this* snap are post-snap (``POST_SNAP_FIELDS``);
from earlier snaps they are fair history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

CONTEXT_VERSION = "strategic_context.v1"

POST_SNAP_FIELDS = frozenset({
    "result", "yards", "success", "outcome", "label_available",
    "coverage_seen", "concept_seen",
    "executed_formation", "executed_play", "executed_status", "executed_verification",
    "executed_adjustment_id", "executed_macro", "action_formation", "action_play",
})

RED_ZONE_YARDS = 20
GOAL_LINE_YARDS = 3
BACKED_UP_YARDS = 88
TWO_MINUTE_SECONDS = 120
QUARTER_SECONDS = 900


class PostSnapLeak(ValueError):
    """A field only known after the snap was offered as pre-snap input."""


@dataclass(frozen=True)
class ContextValue:
    value: Any
    status: str  # known | derived | inferred | missing
    source: str
    note: str = ""

    @property
    def known(self) -> bool:
        return self.status in ("known", "derived")


def _missing(note: str) -> ContextValue:
    return ContextValue(None, "missing", "none", note)


@dataclass(frozen=True)
class OpponentContext:
    opponent_id: str | None
    kind: str  # cpu | human | unknown
    run_share: ContextValue
    concept_shares: Mapping[str, float]
    own_plays: int
    pooled_plays: int
    scope: str
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GameHistory:
    """Earlier snaps of this game. Our calls and the opponent's observed plays stay apart."""

    our_previous_calls: tuple[str, ...] = ()
    opponent_observed_concepts: tuple[str, ...] = ()
    opponent_observed_coverages: tuple[str, ...] = ()
    previous_results: tuple[str, ...] = ()

    @property
    def opponent_run_pass(self) -> tuple[int, ...]:
        """1 = pass, 0 = run, for each earlier observed concept that classifies."""
        from cfb_coach.madden.model.heldout_eval import run_or_pass

        out = []
        for c in self.opponent_observed_concepts:
            fam = run_or_pass(c)
            if fam == "run":
                out.append(0)
            elif fam in ("pass", "screen"):
                out.append(1)
        return tuple(out)


@dataclass(frozen=True)
class PreSnapContext:
    version: str
    side: str
    down: ContextValue
    distance: ContextValue
    yards_to_goal: ContextValue
    quarter: ContextValue
    clock_seconds: ContextValue
    score_diff: ContextValue
    timeouts_us: ContextValue
    goal_to_go: ContextValue
    red_zone: ContextValue
    goal_line: ContextValue
    backed_up: ContextValue
    two_minute: ContextValue
    short_yardage: ContextValue
    long_yardage: ContextValue
    offense_personnel: ContextValue
    defense_personnel: ContextValue
    live_look: ContextValue
    legal_candidates: ContextValue
    opponent: OpponentContext
    history: GameHistory
    missing: tuple[str, ...]
    inconsistencies: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def as_features(self) -> dict[str, float]:
        """Flat numbers for models. Every optional value has a ``*_missing`` flag."""
        f: dict[str, float] = {}
        down = self.down.value if self.down.known else None
        for d in (1, 2, 3, 4):
            f[f"down_{d}"] = float(down == d)
        f["down_missing"] = float(down is None)
        for name in ("distance", "yards_to_goal", "quarter", "clock_seconds", "score_diff"):
            cv = getattr(self, name)
            f[name] = float(cv.value) if cv.known else 0.0
            f[f"{name}_missing"] = float(not cv.known)
        for name in ("goal_to_go", "red_zone", "goal_line", "backed_up", "two_minute",
                     "short_yardage", "long_yardage"):
            cv = getattr(self, name)
            f[name] = float(bool(cv.value)) if cv.known else 0.0
            f[f"{name}_missing"] = float(not cv.known)
        rs = self.opponent.run_share
        f["opp_run_share"] = float(rs.value) if rs.known or rs.status == "inferred" else 0.5
        f["opp_run_share_missing"] = float(rs.value is None)
        f["opp_own_plays"] = float(self.opponent.own_plays)
        f["opp_pooled_plays"] = float(self.opponent.pooled_plays)
        f["opp_is_cpu"] = float(self.opponent.kind == "cpu")
        rp = self.history.opponent_run_pass
        f["game_opp_pass_share"] = (sum(rp) / len(rp)) if rp else 0.5
        f["game_opp_plays"] = float(len(rp))
        f["last_opp_pass"] = float(rp[-1]) if rp else 0.5
        f["last_opp_missing"] = float(not rp)
        return f


def assert_pre_snap(fields: Mapping[str, Any] | Iterable[str]) -> None:
    """Raise if any post-snap field of the *current* snap is offered as an input."""
    names = set(fields.keys() if isinstance(fields, Mapping) else fields)
    leaked = sorted(names & POST_SNAP_FIELDS)
    if leaked:
        raise PostSnapLeak(f"post-snap fields offered as pre-snap input: {leaked}")


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get(sit: Any, name: str) -> Any:
    v = getattr(sit, name, None)
    if v is None:
        extras = getattr(sit, "extras", None) or {}
        if isinstance(extras, Mapping):
            v = extras.get(name)
    return v


def history_from_snaps(earlier: Sequence[Mapping[str, Any]], *, side: str) -> GameHistory:
    """Game history from snaps logged before this one (same game, any order given)."""
    ours, concepts, coverages, results = [], [], [], []
    for r in earlier:
        r_side = str(r.get("side") or "")
        if r.get("formation") or r.get("play"):
            if r_side == side:
                ours.append(f"{r.get('formation') or ''}|{r.get('play') or ''}")
        seen_c = str(r.get("concept_seen") or "").strip()
        if seen_c and r_side.startswith("d"):
            concepts.append(seen_c)
        seen_v = str(r.get("coverage_seen") or "").strip()
        if seen_v and not r_side.startswith("d"):
            coverages.append(seen_v)
        if r_side == side and r.get("result") not in (None, ""):
            results.append(str(r.get("result")))
    return GameHistory(tuple(ours), tuple(concepts), tuple(coverages), tuple(results))


def opponent_context(opponent_id: str | None, profile: Mapping[str, Any] | None) -> OpponentContext:
    from cfb_coach.opponents import is_cpu_opponent

    oid = str(opponent_id or "").strip() or None
    kind = "unknown" if oid is None else ("cpu" if is_cpu_opponent(oid) else "human")
    p = dict(profile or {})
    n = int(p.get("n") or 0)
    scope = str(p.get("scope") or "")
    rs = p.get("run_share")
    if scope == "pooled human opponents":
        own, pooled = 0, n
    else:
        own, pooled = n, 0
    if rs is None or n <= 0:
        run_share = _missing("no observed opponent plays before this snap")
    elif own == 0:
        run_share = ContextValue(float(rs), "inferred", "league_prior", "other human opponents, not this one")
    elif "league prior" in scope:
        run_share = ContextValue(float(rs), "inferred", "own_plays_blended_with_league_prior", scope)
    else:
        run_share = ContextValue(float(rs), "derived", "observed_opponent_plays", scope)
    return OpponentContext(
        opponent_id=oid, kind=kind, run_share=run_share,
        concept_shares=dict(p.get("concepts") or {}), own_plays=own, pooled_plays=pooled,
        scope=scope, evidence=dict(p.get("evidence") or {}),
    )


SNAP_PRE_SNAP_COLUMNS = ("side", "down", "distance", "yardline", "quarter", "situation_raw",
                         "score_for", "score_against", "opponent_id")


def context_from_snap(
    row: Mapping[str, Any],
    *,
    earlier: Sequence[Mapping[str, Any]] = (),
    opponent_profile: Mapping[str, Any] | None = None,
    legal_candidates: Sequence[Any] | None = None,
) -> PreSnapContext:
    """Context for a logged snap, read only from its pre-snap columns.

    ``score_for``/``score_against`` are stored before the play (the score moves on
    the row after a touchdown). ``earlier`` must be this game's preceding snaps.
    """
    from cfb_coach.madden.situation import parse_madden_situation

    side = str(row.get("side") or "offense")
    sit = parse_madden_situation(str(row.get("situation_raw") or ""), default_side=side)
    sit.side = side
    for name in ("down", "distance", "yardline"):
        if row.get(name) is not None:
            setattr(sit, name, row.get(name))
    if row.get("score_for") is not None and row.get("score_against") is not None:
        sit.score_us, sit.score_them = row["score_for"], row["score_against"]
    extras = dict(getattr(sit, "extras", None) or {})
    if row.get("quarter") is not None:
        extras["quarter"] = row["quarter"]
    sit.extras = extras
    return build_pre_snap_context(
        sit, opponent_id=row.get("opponent_id"), opponent_profile=opponent_profile,
        history=history_from_snaps(earlier, side=side), legal_candidates=legal_candidates,
    )


def build_pre_snap_context(
    sit: Any,
    *,
    opponent_id: str | None = None,
    opponent_profile: Mapping[str, Any] | None = None,
    history: GameHistory | None = None,
    legal_candidates: Sequence[Any] | None = None,
    personnel: Mapping[str, Any] | None = None,
) -> PreSnapContext:
    side = "defense" if str(getattr(sit, "side", "offense") or "offense").startswith("d") else "offense"
    problems: list[str] = []

    down_v = _int(_get(sit, "down"))
    if down_v is not None and down_v not in (1, 2, 3, 4):
        problems.append(f"down_out_of_range:{down_v}")
        down_v = None
    down = ContextValue(down_v, "known", "situation") if down_v else _missing("down not logged")

    dist_v = _int(_get(sit, "distance"))
    if dist_v is not None and not 0 < dist_v < 100:
        problems.append(f"distance_out_of_range:{dist_v}")
        dist_v = None
    distance = ContextValue(dist_v, "known", "situation") if dist_v else _missing("distance not logged")

    explicit_ytg = _int(_get(sit, "yards_to_goal"))
    yl = _int(_get(sit, "yardline"))
    ytg_v: int | None
    if explicit_ytg is not None:
        ytg_v, src = explicit_ytg, "explicit"
    elif yl is not None:
        ytg_v, src = (yl if side == "defense" else 100 - yl), "yardline"
    else:
        ytg_v, src = None, "none"
    if ytg_v is not None and not 0 < ytg_v < 100:
        problems.append(f"yards_to_goal_out_of_range:{ytg_v}")
        ytg_v = None
    ytg = ContextValue(ytg_v, "derived" if src == "yardline" else "known", src) if ytg_v else _missing("field position not logged")

    q_v = _int(_get(sit, "quarter"))
    if q_v is not None and q_v not in (1, 2, 3, 4, 5):
        problems.append(f"quarter_out_of_range:{q_v}")
        q_v = None
    quarter = ContextValue(q_v, "known", "situation") if q_v else _missing("quarter not logged")

    clock_v = _int(_get(sit, "clock_seconds"))
    if clock_v is not None and not 0 <= clock_v <= QUARTER_SECONDS:
        problems.append(f"clock_out_of_range:{clock_v}")
        clock_v = None
    clock = ContextValue(clock_v, "known", "situation") if clock_v is not None else _missing("game clock not logged")

    s_us, s_them = _int(_get(sit, "score_us")), _int(_get(sit, "score_them"))
    if (s_us is not None and s_us < 0) or (s_them is not None and s_them < 0):
        problems.append("negative_score")
        s_us = s_them = None
    score_diff = (ContextValue(s_us - s_them, "known", "situation", "our points minus theirs")
                  if s_us is not None and s_them is not None else _missing("both scores are required"))
    to_v = _int(_get(sit, "timeouts_us"))
    timeouts = ContextValue(to_v, "known", "situation") if to_v is not None and 0 <= to_v <= 3 else _missing("timeouts not logged")

    if ytg_v is not None and dist_v is not None and dist_v > ytg_v:
        problems.append(f"distance_{dist_v}_exceeds_yards_to_goal_{ytg_v}")

    def flag(value: bool | None, note: str) -> ContextValue:
        return ContextValue(value, "derived", "yards_to_goal", note) if value is not None else _missing(note)

    if ytg_v is not None:
        red_zone = flag(ytg_v <= RED_ZONE_YARDS, f"inside the {RED_ZONE_YARDS}")
        goal_line = flag(ytg_v <= GOAL_LINE_YARDS, f"inside the {GOAL_LINE_YARDS}")
        backed_up = flag(ytg_v >= BACKED_UP_YARDS, f"offense inside its own {100 - BACKED_UP_YARDS}")
        goal_to_go = flag(dist_v is not None and dist_v >= ytg_v, "distance reaches the goal line") \
            if dist_v is not None else _missing("distance not logged")
        if bool(getattr(sit, "red_zone", False)) and ytg_v > RED_ZONE_YARDS:
            problems.append("red_zone_flag_contradicts_field_position")
    else:
        rz_flag = bool(getattr(sit, "red_zone", False))
        gl_flag = bool(getattr(sit, "goal_line", False))
        red_zone = ContextValue(True, "known", "explicit_flag") if rz_flag or gl_flag else _missing("no field position or red-zone flag")
        goal_line = ContextValue(True, "known", "explicit_flag") if gl_flag else _missing("no field position or goal-line flag")
        backed_up = _missing("no field position")
        goal_to_go = _missing("no field position")

    if q_v is not None and clock_v is not None:
        two_minute = ContextValue(q_v in (2, 4) and clock_v <= TWO_MINUTE_SECONDS, "derived", "quarter_and_clock")
    elif bool(getattr(sit, "two_minute", False)):
        two_minute = ContextValue(True, "known", "explicit_flag")
    else:
        two_minute = _missing("no game clock or two-minute flag")

    if down_v is not None and dist_v is not None:
        short = ContextValue(dist_v <= 2 and down_v in (2, 3, 4), "derived", "down_and_distance")
        long_ = ContextValue(down_v in (3, 4) and dist_v >= 7, "derived", "down_and_distance")
    else:
        short = _missing("down or distance not logged")
        long_ = _missing("down or distance not logged")

    pers = dict(personnel or {})
    off_p = ContextValue(pers["offense"], "known", "operator") if pers.get("offense") else _missing("offensive personnel not logged")
    def_p = ContextValue(pers["defense"], "known", "operator") if pers.get("defense") else _missing("defensive personnel not logged")

    look_attr = "concept_hint" if side == "defense" else "coverage_hint"
    src_attr = "concept_source" if side == "defense" else "coverage_source"
    hint, hint_src = getattr(sit, look_attr, None), getattr(sit, src_attr, None) or "none"
    if hint and hint_src == "live":
        live_look = ContextValue(hint, "known", "pre_snap_look", "shown before this snap")
    elif hint and hint_src == "last":
        live_look = ContextValue(hint, "inferred", "previous_snap", "previous snap only; not this snap's look")
    else:
        live_look = _missing("no pre-snap look")

    cands = ContextValue(len(legal_candidates), "known", "installed_book") if legal_candidates is not None \
        else _missing("legal candidates not supplied")

    values = dict(down=down, distance=distance, yards_to_goal=ytg, quarter=quarter, clock_seconds=clock,
                  score_diff=score_diff, timeouts_us=timeouts, goal_to_go=goal_to_go, red_zone=red_zone,
                  goal_line=goal_line, backed_up=backed_up, two_minute=two_minute, short_yardage=short,
                  long_yardage=long_, offense_personnel=off_p, defense_personnel=def_p, live_look=live_look,
                  legal_candidates=cands)
    opp = opponent_context(opponent_id, opponent_profile)
    missing = tuple(k for k, v in values.items() if v.status == "missing")
    if opp.run_share.status == "missing":
        missing += ("opponent_run_share",)
    return PreSnapContext(version=CONTEXT_VERSION, side=side, opponent=opp, history=history or GameHistory(),
                          missing=missing, inconsistencies=tuple(problems), **values)
