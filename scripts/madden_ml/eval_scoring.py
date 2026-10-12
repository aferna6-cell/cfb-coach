"""Offline decision-quality scoring for the offense and defense coordinators.

This module does not change a live call. It compares the new coordinator
decision to the old caller on a fixed situation set and writes a JSON report
plus Markdown. The numbers in that report are counts from the rows it scored.

These are decision-quality checks only. They are not effectiveness claims.
Nothing here estimates yards, win probability, or whether a call worked.

The live comparison itself is `scripts/madden_ml/eval_harness.py::run_situation`.
That module may land on another branch. This file calls it through
`adapt_run_situation` only, so a rebase stays in one function. Tests pass a
stub runner and do not import the harness.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cfb_coach.madden.catalog import is_deep, is_run
from cfb_coach.madden.model.schema import MLStatus, is_live_selectable
from cfb_coach.zones import field_zone

DISCLAIMER = (
    "These are decision-quality checks only. They are not effectiveness claims."
)

# Any single formation or call above this share of decisions on a side is flagged.
SHARE_CAP = 0.25

BUDGET_MS = 150

FIXTURE_DIR = _ROOT / "tests" / "madden_ml" / "fixtures" / "item3"
SYNTHETIC_FIXTURE = FIXTURE_DIR / "synthetic_grid.json"
OBSERVED_CSV = FIXTURE_DIR / "presnap_observed_cpu.csv"

SIDES = ("offense", "defense")

# Distance buckets from the quality plan: short ≤2, medium 3-6, long 7+.
DISTANCE_BUCKETS = ("short", "medium", "long")
FIELD_ZONES = (
    "own_deep",
    "midfield",
    "plus_territory",
    "red_zone",
    "goal_line",
)
CLOCK_BUCKETS = ("normal", "2-minute", "4-min protect")
SCORE_BUCKETS = ("trailing_big", "close", "leading_big")

# Yards from our goal. Own-deep and midfield stay in the open-field code zone.
# Plus territory is past midfield and outside the red zone. Red zone is inside
# the opponent 20 but outside the goal line. Goal line is inside the opponent 5.
_ZONE_YARDLINE = {
    "own_deep": 12,
    "midfield": 50,
    "plus_territory": 70,
    "red_zone": 88,
    "goal_line": 97,
}
_SCORE_PAIR = {
    "trailing_big": (3, 24),  # margin -21
    "close": (17, 20),  # margin -3, inside one score
    "leading_big": (31, 10),  # margin +21
}
_CLOCK = {
    "normal": {"quarter": 2, "clock_seconds": 600, "two_minute": False},
    "2-minute": {"quarter": 4, "clock_seconds": 90, "two_minute": True},
    "4-min protect": {"quarter": 4, "clock_seconds": 240, "two_minute": False},
}

# Recorded packages from the data layer. NULL is missing. "empty" is its own look.
_REAL_LOOKS = (
    ("11", "nickel"),
    ("12", "base"),
    ("21", "dime"),
    ("10", "dollar"),
    ("22", "quarter"),
    ("13", "goal_line"),
)
_SYNTHETIC_LOOKS = (
    {"look_id": "null", "opp_personnel": None, "opp_set": None},
    {"look_id": "empty", "opp_personnel": "empty", "opp_set": "empty"},
    {"look_id": "11-nickel", "opp_personnel": "11", "opp_set": "nickel"},
    {"look_id": "12-base", "opp_personnel": "12", "opp_set": "base"},
)

_NO_ADJUSTMENT = frozenset(
    {"", "none", "no adj", "no_adj", "no adjustment", "no_adjustment"}
)
_CLOCK_KILL = re.compile(r"kneel|victory|spike|take a knee|qb kneel", re.I)
_PREVENT = re.compile(r"prevent|hail\s*mary", re.I)
_DIME = re.compile(r"^dime\b", re.I)
_UNSET = object()

_ML_STATUS_VALUES = frozenset(item.value for item in MLStatus)

Runner = Callable[..., Mapping[str, Any]]


def distance_bucket(distance: int) -> str:
    """short ≤2, medium 3-6, long 7+."""
    if distance <= 2:
        return "short"
    if distance <= 6:
        return "medium"
    return "long"


def score_bucket(us: int, them: int) -> str:
    """trailing_big margin ≤ -14, leading_big margin ≥ 14, otherwise close."""
    margin = int(us) - int(them)
    if margin <= -14:
        return "trailing_big"
    if margin >= 14:
        return "leading_big"
    return "close"


def stable_seed(situation_id: str) -> int:
    """Same situation id always yields the same seed. Not a measured latency."""
    digest = hashlib.sha256(situation_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def normalize_opp_look(look: Any) -> Any:
    """Blank becomes NULL. The literal string ``empty`` stays ``empty``.

    A mapping normalizes ``opp_personnel`` and ``opp_set`` the same way.
    """
    if look is None:
        return None
    if isinstance(look, str):
        if look.strip() == "":
            return None
        return look
    if isinstance(look, Mapping):
        return {
            "opp_personnel": _blank_to_none(look.get("opp_personnel")),
            "opp_set": _blank_to_none(look.get("opp_set")),
        }
    return look


def _blank_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    return value


def _is_no_adjustment(value: Any) -> bool:
    if value is None:
        return True
    return str(value).strip().casefold() in _NO_ADJUSTMENT


# ---------------------------------------------------------------------------
# Situation grid
# ---------------------------------------------------------------------------


def build_synthetic_identities() -> list[dict[str, Any]]:
    """72 grid rows: 36 offense and 36 defense.

    The full down × distance × zone × clock × score cube is 540 cells per
    side. This keeps every level of each factor and a balanced side split,
    dropping every fifth cell of the distance × zone × clock cube (9 of 45)
    so the grid stays near the planned ~70.
    """
    rows: list[dict[str, Any]] = []
    for side_i, side in enumerate(SIDES):
        seen = 0
        kept = 0
        for dist_name in DISTANCE_BUCKETS:
            for zone in FIELD_ZONES:
                for clock_name in CLOCK_BUCKETS:
                    skip = (seen + side_i) % 5 == 4
                    seen += 1
                    if skip:
                        continue
                    kept += 1
                    score_name = SCORE_BUCKETS[kept % 3]
                    down = (1, 2, 3, 4)[kept % 4]
                    if dist_name == "short" and down == 1:
                        down = 3
                    distance = _distance_value(dist_name, zone)
                    rows.append(
                        _identity(
                            situation_id=f"syn-{side}-{kept:03d}",
                            side=side,
                            tags=["synthetic"],
                            source="synthetic",
                            down=down,
                            distance=distance,
                            yardline=_ZONE_YARDLINE[zone],
                            field_zone=zone,
                            clock_name=clock_name,
                            score_name=score_name,
                            game_id=None,
                            default_filled=[],
                            looks=[dict(item) for item in _SYNTHETIC_LOOKS],
                            look_observed=False,
                        )
                    )
        if kept != 36:
            raise RuntimeError(f"{side} synthetic grid kept {kept}, expected 36")
    if len(rows) != 72:
        raise RuntimeError(f"synthetic grid has {len(rows)} rows, expected 72")
    return rows


def _distance_value(bucket: str, zone: str) -> int:
    if bucket == "short":
        return 1 if zone == "goal_line" else 2
    if bucket == "medium":
        return 4
    return 10


def _identity(
    *,
    situation_id: str,
    side: str,
    tags: list[str],
    source: str,
    down: int,
    distance: int,
    yardline: int,
    field_zone_name: str | None = None,
    field_zone: str | None = None,
    clock_name: str,
    score_name: str,
    game_id: str | None,
    default_filled: list[str],
    looks: list[dict[str, Any]],
    look_observed: bool,
    quarter: int | None = _UNSET,
    clock_seconds: int | None = _UNSET,
    two_minute: bool | None = _UNSET,
    score_us: int | None = _UNSET,
    score_them: int | None = _UNSET,
) -> dict[str, Any]:
    clock = _CLOCK[clock_name]
    us, them = _SCORE_PAIR[score_name]
    q = clock["quarter"] if quarter is _UNSET else quarter
    seconds = clock["clock_seconds"] if clock_seconds is _UNSET else clock_seconds
    two = bool(clock["two_minute"] if two_minute is _UNSET else two_minute)
    sus = us if score_us is _UNSET else score_us
    sthem = them if score_them is _UNSET else score_them
    zone_name = field_zone if field_zone is not None else field_zone_name
    raw = _raw_line(
        side=side,
        down=down,
        distance=distance,
        yardline=yardline,
        quarter=q,
        clock_bucket=clock_name,
        clock_seconds=seconds,
        score_us=sus,
        score_them=sthem,
        include_score="score" not in default_filled,
        include_clock="clock" not in default_filled and seconds is not None,
    )
    return {
        "situation_id": situation_id,
        "side": side,
        "tags": list(tags),
        "source": source,
        "down": int(down),
        "distance": int(distance),
        "distance_bucket": distance_bucket(int(distance)),
        "yardline": int(yardline),
        "field_zone": zone_name,
        "quarter": int(q) if q is not None else None,
        "clock_bucket": clock_name,
        "clock_seconds": int(seconds) if seconds is not None else None,
        "two_minute": two,
        "score_bucket": score_name,
        "score_us": int(sus) if sus is not None else None,
        "score_them": int(sthem) if sthem is not None else None,
        "default_filled": list(default_filled),
        "game_id": game_id,
        "raw": raw,
        "looks": looks,
        "look_observed": look_observed,
        "seed": stable_seed(situation_id),
    }


def _raw_line(
    *,
    side: str,
    down: int,
    distance: int,
    yardline: int,
    quarter: int | None,
    clock_bucket: str,
    clock_seconds: int | None,
    score_us: int | None,
    score_them: int | None,
    include_score: bool,
    include_clock: bool,
) -> str:
    prefix = "d " if side == "defense" else ""
    if yardline <= 50:
        spot = f"my {yardline}"
    else:
        spot = f"opp {100 - yardline}"
    parts = [f"{prefix}{down}&{distance} {spot}".strip()]
    if quarter is not None:
        parts.append(f"q{quarter}")
    if include_clock and clock_seconds is not None:
        minutes, seconds = divmod(int(clock_seconds), 60)
        parts.append(f"clock {minutes}:{seconds:02d}")
    if include_score and score_us is not None and score_them is not None:
        parts.append(f"score {score_us}-{score_them}")
    if clock_bucket == "2-minute":
        parts.append("two minute")
    elif clock_bucket == "4-min protect":
        parts.append("4-min protect")
    return " ".join(parts)


def assigned_observed_looks(index: int) -> list[dict[str, Any]]:
    """Three assigned looks. None of them is an observed opponent look.

    NULL and the literal ``empty`` are both included so they stay distinct.
    The third look rotates through recorded personnel/set pairs.
    """
    personnel, package = _REAL_LOOKS[index % len(_REAL_LOOKS)]
    return [
        {"look_id": "null", "opp_personnel": None, "opp_set": None},
        {"look_id": "empty", "opp_personnel": "empty", "opp_set": "empty"},
        {
            "look_id": f"{personnel}-{package}",
            "opp_personnel": personnel,
            "opp_set": package,
        },
    ]


def load_observed_csv(path: Path) -> list[dict[str, Any]]:
    """Read presnap rows. Down, distance, yardline, and quarter are kept as written.

    Score and clock are neutral defaults (tied, clock bucket ``normal``) and
    listed in ``default_filled``. Opponent looks are not read from this file.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        missing = [name for name in ("side", "down", "distance", "yardline", "quarter") if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        game_key = next((key for key in ("game_id", "game") if key in fieldnames), None)
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(reader, start=1):
            side_raw = (raw.get("side") or "").strip()
            side = _normalize_side(side_raw)
            down = _exact_int(raw.get("down"), "down")
            distance = _exact_int(raw.get("distance"), "distance")
            yardline = _exact_int(raw.get("yardline"), "yardline")
            quarter = _exact_int(raw.get("quarter"), "quarter")
            game_id = (raw.get(game_key) or "").strip() if game_key else ""
            if game_id and game_id not in {"g1", "g2", "g3", "g4"}:
                raise ValueError(f"row {index} game {game_id!r} is not g1-g4")
            zone = observed_field_zone(yardline, distance)
            rows.append(
                _identity(
                    situation_id=f"obs-{index:04d}",
                    side=side,
                    tags=["observed"],
                    source="observed",
                    down=down,
                    distance=distance,
                    yardline=yardline,
                    field_zone=zone,
                    clock_name="normal",
                    score_name="close",
                    game_id=game_id or None,
                    default_filled=["score", "clock"],
                    looks=assigned_observed_looks(index - 1),
                    look_observed=False,
                    quarter=quarter,
                    clock_seconds=None,
                    two_minute=False,
                    score_us=0,
                    score_them=0,
                )
            )
    return rows


def observed_field_zone(yardline: int, distance: int) -> str | None:
    """Map a stored yardline with the repo's own-goal convention.

    Synthetic rows name the zone themselves. Observed rows only have the
    integer from the CSV, so this is the zone used for context rules.
    Yardlines outside 0-100 are left unset.
    """
    if yardline < 0 or yardline > 100:
        return None
    code = field_zone(yardline, distance)
    if code == "gl":
        return "goal_line"
    if code == "rz":
        return "red_zone"
    if yardline <= 20:
        return "own_deep"
    if 40 <= yardline <= 60:
        return "midfield"
    if 60 < yardline < 80:
        return "plus_territory"
    return None


def _normalize_side(value: str) -> str:
    key = value.strip().casefold()
    if key in {"offense", "o", "off"}:
        return "offense"
    if key in {"defense", "d", "def"}:
        return "defense"
    raise ValueError(f"side must be offense or defense, got {value!r}")


def _exact_int(value: Any, name: str) -> int:
    text = "" if value is None else str(value).strip()
    if not text or not re.fullmatch(r"-?\d+", text):
        raise ValueError(f"{name} must be an integer token, got {value!r}")
    return int(text)


def expand_decision_rows(identities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per situation × look. Observed looks are tagged assigned, not observed."""
    expanded: list[dict[str, Any]] = []
    for identity in identities:
        source = identity["source"]
        for look in identity["looks"]:
            row = dict(identity)
            row.pop("looks", None)
            if source == "observed":
                row["tags"] = ["observed_situation", "assigned_look"]
            else:
                row["tags"] = list(identity["tags"])
            row["look_id"] = look["look_id"]
            row["opp_personnel"] = look["opp_personnel"]
            row["opp_set"] = look["opp_set"]
            row["opp_look"] = normalize_opp_look(
                {
                    "opp_personnel": look["opp_personnel"],
                    "opp_set": look["opp_set"],
                }
            )
            row["look_observed"] = False
            row["row_id"] = f"{identity['situation_id']}__{look['look_id']}"
            expanded.append(row)
    return expanded


def load_situation_identities() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Synthetic identities from the builder, plus the CSV when that file is present."""
    synthetic = build_synthetic_identities()
    note: dict[str, Any] = {
        "observed_csv": str(OBSERVED_CSV),
        "observed_csv_present": OBSERVED_CSV.is_file(),
    }
    if not OBSERVED_CSV.is_file():
        note["observed_csv_reason"] = (
            "presnap_observed_cpu.csv was not in the workspace, so no observed "
            "rows were invented"
        )
        note["fixture_counts"] = fixture_counts(synthetic)
        return synthetic, note
    observed = load_observed_csv(OBSERVED_CSV)
    note["observed_source_rows"] = len(observed)
    identities = synthetic + observed
    note["fixture_counts"] = fixture_counts(identities)
    return identities, note


def _empty_side_counts() -> dict[str, int]:
    return {"offense": 0, "defense": 0}


def fixture_counts(identities: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts of the situation identities and the expanded decision rows.

    Identity tags and decision-row tags are split by offense and defense.
    A side with no rows stays 0.
    """
    by_source_side: dict[str, dict[str, int]] = {}
    tag_counts: dict[str, int] = {}
    identity_tags_by_side: dict[str, dict[str, int]] = {}
    for row in identities:
        by_source_side.setdefault(row["source"], _empty_side_counts())
        by_source_side[row["source"]][row["side"]] += 1
        for tag in row["tags"]:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
            bucket = identity_tags_by_side.setdefault(tag, _empty_side_counts())
            bucket[row["side"]] += 1
    expanded = expand_decision_rows(identities)
    expanded_tags: dict[str, int] = {}
    decision_tags_by_side: dict[str, dict[str, int]] = {}
    for row in expanded:
        key = ",".join(row["tags"])
        expanded_tags[key] = expanded_tags.get(key, 0) + 1
        bucket = decision_tags_by_side.setdefault(key, _empty_side_counts())
        bucket[row["side"]] += 1
    return {
        "identities": len(identities),
        "by_source_side": by_source_side,
        "identity_tags": tag_counts,
        "identity_tags_by_side": identity_tags_by_side,
        "decision_rows": len(expanded),
        "decision_tags": expanded_tags,
        "decision_tags_by_side": decision_tags_by_side,
    }


# ---------------------------------------------------------------------------
# Installed book
# ---------------------------------------------------------------------------


def load_installed_book(db_path: Path) -> dict[str, dict[str, list[str]]]:
    """Install the trimmed book into ``active_playbook_json`` and read it back.

    The path must already be a temp database. This function does not resolve
    the user home directory and does not open ``~/.cfb-coach``.
    """
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import playbook
    from cfb_coach.madden.adjustments.synthesis import install_trimmed_book

    db = CoachDB(db_path)
    try:
        install_trimmed_book(db)
        applied = playbook.load_books(db)
        book: dict[str, dict[str, list[str]]] = {}
        for side in SIDES:
            formations = (applied.get(side) or {}).get("formations") or {}
            book[side] = {
                str(name): [str(play) for play in plays]
                for name, plays in formations.items()
            }
        if len(book["offense"]) != 5 or len(book["defense"]) != 5:
            raise RuntimeError(
                "active_playbook_json does not have 5 offense and 5 defense formations"
            )
        return book
    finally:
        db.close()


def decision_legal(
    side: str,
    formation: str | None,
    play: str | None,
    adjustment_id: Any,
    macro_state: Any,
    book: Mapping[str, Mapping[str, Sequence[str]]],
) -> tuple[bool, list[str]]:
    """In-book play, and an armed-or-later macro or no adjustment.

    Armed-or-later is ``is_live_selectable`` (armed, eligible, evaluated).
    A pick outside the book or an unarmed macro is a hard fail.
    """
    reasons: list[str] = []
    plays = (book.get(side) or {}).get(formation or "")
    if not formation or not play or plays is None or play not in list(plays):
        reasons.append("outside_book")
    if not _is_no_adjustment(adjustment_id):
        try:
            armed = is_live_selectable(macro_state)
        except (TypeError, ValueError):
            armed = False
        if not armed:
            reasons.append("unarmed_macro")
    return (not reasons), reasons


# ---------------------------------------------------------------------------
# Context rules
# ---------------------------------------------------------------------------


def scoring_rules() -> list[dict[str, str]]:
    """The written rule list. Each rule is a decision-quality check."""
    return [
        {"id": rule_id, "side": side, "text": text}
        for rule_id, side, text, _check in _RULES
    ]


_RULES: tuple[tuple[str, str, str, Callable[..., dict[str, Any]]], ...] = (
    (
        "no_deep_shot_on_short_or_goal_line",
        "offense",
        "On offense, a 3rd-or-4th-and-short (distance ≤ 2) or any goal-line "
        "short-yardage snap must not be a deep shot. Deep means "
        "cfb_coach.madden.catalog.is_deep (vert, flood, shot, seam, fade, "
        "and the other tokens that classifier already uses). "
        "Example: no deep shot on 3rd-and-1 or goal-line short yardage.",
        lambda row, play, formation: _rule_no_deep(row, play),
    ),
    (
        "two_minute_trailing_not_clock_kill_run",
        "offense",
        "On offense, a two-minute snap while trailing big (margin ≤ -14) must "
        "be clock-aware: the call fails when catalog.is_run is true or the "
        "play name is a kneel, victory, spike, or take-a-knee. A non-run "
        "passes. The rule does not run when score or clock was default-filled.",
        lambda row, play, formation: _rule_two_minute(row, play),
    ),
    (
        "no_kneel_or_run_clock_when_trailing_late",
        "offense",
        "On offense, while trailing big inside the two-minute window or the "
        "4-min protect window, the call fails when the play name is a kneel, "
        "victory, spike, or take-a-knee. A normal run is not this rule; the "
        "two-minute rule covers clock-killing runs. Default-filled score or "
        "clock does not trigger it.",
        lambda row, play, formation: _rule_no_kneel(row, play),
    ),
    (
        "goal_line_short_not_dime",
        "defense",
        "On defense, goal-line short yardage (distance ≤ 2) must not be a Dime "
        "formation. The installed heavier packages are 4-3 Over and the Nickel "
        "formations. Dime 3-2 Odd fails. This follows the existing note that "
        "goal line and true short yardage keep the heavy package.",
        lambda row, play, formation: _rule_not_dime(row, formation),
    ),
    (
        "trailing_late_no_prevent",
        "defense",
        "On defense, while trailing big in the two-minute or 4-min protect "
        "window, a play named prevent or hail mary fails. Cover 4 Quarters is "
        "not this rule. Default-filled score or clock does not trigger it. "
        "The installed book has no prevent or hail-mary play name, so an "
        "in-book call passes and an invented name fails.",
        lambda row, play, formation: _rule_no_prevent(row, play),
    ),
)


def _filled(row: Mapping[str, Any], field: str) -> bool:
    return field in set(row.get("default_filled") or [])


def _trailing_late_window(row: Mapping[str, Any]) -> bool:
    if row.get("score_bucket") != "trailing_big" or _filled(row, "score"):
        return False
    if _filled(row, "clock"):
        return False
    return row.get("clock_bucket") in {"2-minute", "4-min protect"}


def _rule_no_deep(row: Mapping[str, Any], play: str | None) -> dict[str, Any]:
    short = row.get("distance_bucket") == "short"
    late_down = row.get("down") in {3, 4}
    goal = row.get("field_zone") == "goal_line"
    applies = row.get("side") == "offense" and short and (late_down or goal)
    if not applies:
        return {"applies": False, "passed": None, "detail": ""}
    failed = bool(play) and is_deep(play or "")
    return {
        "applies": True,
        "passed": not failed,
        "detail": "catalog.is_deep" if failed else "not a deep shot",
    }


def _rule_two_minute(row: Mapping[str, Any], play: str | None) -> dict[str, Any]:
    applies = (
        row.get("side") == "offense"
        and row.get("clock_bucket") == "2-minute"
        and row.get("score_bucket") == "trailing_big"
        and not _filled(row, "clock")
        and not _filled(row, "score")
    )
    if not applies:
        return {"applies": False, "passed": None, "detail": ""}
    text = play or ""
    failed = is_run(text) or bool(_CLOCK_KILL.search(text))
    return {
        "applies": True,
        "passed": not failed,
        "detail": "clock-kill run" if failed else "not a clock-kill run",
    }


def _rule_no_kneel(row: Mapping[str, Any], play: str | None) -> dict[str, Any]:
    applies = row.get("side") == "offense" and _trailing_late_window(row)
    if not applies:
        return {"applies": False, "passed": None, "detail": ""}
    failed = bool(_CLOCK_KILL.search(play or ""))
    return {
        "applies": True,
        "passed": not failed,
        "detail": "kneel or spike" if failed else "not a kneel or spike",
    }


def _rule_not_dime(row: Mapping[str, Any], formation: str | None) -> dict[str, Any]:
    applies = (
        row.get("side") == "defense"
        and row.get("field_zone") == "goal_line"
        and row.get("distance_bucket") == "short"
    )
    if not applies:
        return {"applies": False, "passed": None, "detail": ""}
    failed = bool(_DIME.search(formation or ""))
    return {
        "applies": True,
        "passed": not failed,
        "detail": "Dime on goal-line short" if failed else "not a Dime package",
    }


def _rule_no_prevent(row: Mapping[str, Any], play: str | None) -> dict[str, Any]:
    applies = row.get("side") == "defense" and _trailing_late_window(row)
    if not applies:
        return {"applies": False, "passed": None, "detail": ""}
    failed = bool(_PREVENT.search(play or ""))
    return {
        "applies": True,
        "passed": not failed,
        "detail": "prevent or hail mary" if failed else "not prevent",
    }


def apply_context_rules(
    row: Mapping[str, Any], formation: str | None, play: str | None
) -> list[dict[str, Any]]:
    results = []
    for rule_id, side, _text, check in _RULES:
        outcome = check(row, play, formation)
        results.append(
            {
                "id": rule_id,
                "side": side,
                "applies": outcome["applies"],
                "passed": outcome["passed"],
                "detail": outcome["detail"],
            }
        )
    return results


# ---------------------------------------------------------------------------
# Harness adapter and row scoring
# ---------------------------------------------------------------------------


def _situation_line(sit: Any) -> Any:
    """Situation text for ``run_situation``.

    The scorer builds a dict whose ``raw`` field is the line. The harness
    parses a string (or a ``Situation``). Passing the dict would parse
    ``str(dict)`` and would not be that line.
    """
    if isinstance(sit, Mapping):
        raw = sit.get("raw")
        if isinstance(raw, str) and raw.strip():
            return raw
        for key in ("situation", "sit"):
            value = sit.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError("scoring row has no situation line for run_situation")
    return sit


def _look_token(look: Any) -> Any:
    """One harness look. NULL and the literal ``empty`` stay distinct.

    The scorer's look is a personnel/set pair. ``run_situation`` stores one
    string: inference ``normalize_opp_look`` stringifies a mapping, so a
    pair would not be stored as NULL or ``empty``. Both fields missing is
    NULL. The token ``empty`` on the set, with personnel missing or
    ``empty``, is ``empty``. Any other pair passes the set token when it
    is present, otherwise the personnel token. The requested pair stays on
    the scoring row as ``opp_look_sent``.
    """
    normalized = normalize_opp_look(look)
    if not isinstance(normalized, Mapping):
        return normalized
    personnel = normalized.get("opp_personnel")
    opp_set = normalized.get("opp_set")
    if personnel is None and opp_set is None:
        return None
    if opp_set == "empty" and personnel in (None, "empty"):
        return "empty"
    if personnel == "empty" and opp_set is None:
        return "empty"
    if isinstance(opp_set, str) and opp_set.strip():
        return opp_set
    if isinstance(personnel, str) and personnel.strip():
        return personnel
    return None


_HARNESS_RUN: Callable[..., Mapping[str, Any]] | None = None


def _load_run_situation() -> Callable[..., Mapping[str, Any]]:
    """Load ``run_situation`` once. Re-executing the harness does not change a call."""
    global _HARNESS_RUN
    if _HARNESS_RUN is not None:
        return _HARNESS_RUN
    path = Path(__file__).with_name("eval_harness.py")
    if not path.is_file():
        raise FileNotFoundError(
            "scripts/madden_ml/eval_harness.py is not on this branch; "
            "pass a stub runner to evaluate()"
        )
    import importlib.util

    spec = importlib.util.spec_from_file_location("madden_ml_eval_harness", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run_situation", None)
    if run is None:
        raise AttributeError("eval_harness.py has no run_situation")
    _HARNESS_RUN = run
    return run


def adapt_run_situation(
    sit: Mapping[str, Any],
    *,
    mode: str,
    opp_look: Any,
    side: str,
    seed: int,
) -> dict[str, Any]:
    """The only call into ``eval_harness.run_situation``.

    ``sit`` is the JSON situation dict this scorer builds. Blank opponent
    looks are NULL before the harness sees them.
    """
    run = _load_run_situation()
    return dict(
        run(
            _situation_line(sit),
            mode=mode,
            opp_look=_look_token(opp_look),
            side=side,
            seed=seed,
        )
    )


def evaluate(
    identities: Sequence[Mapping[str, Any]],
    runner: Runner,
    book: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    mode: str = "heuristic",
    inventory_note: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run each situation × look twice with the same seed, then score.

    The second call is only the seed check. Latency, fallbacks, and agreement
    use the first call so the rerun is not double-counted.
    """
    prepared: list[dict[str, Any]] = []
    for request in expand_decision_rows(identities):
        look = normalize_opp_look(request["opp_look"])
        kwargs = {
            "mode": mode,
            "opp_look": look,
            "side": request["side"],
            "seed": int(request["seed"]),
        }
        first = dict(runner(request, **kwargs))
        second = dict(runner(request, **kwargs))
        prepared.append({"request": request, "first": first, "second": second})
    return build_report(prepared, book, inventory_note=inventory_note)


def build_report(
    prepared: Sequence[Mapping[str, Any]],
    book: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    inventory_note: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    scored = [_score_prepared(item, book) for item in prepared]
    summary = summarize(scored, book)
    return {
        "disclaimer": DISCLAIMER,
        "share_cap": SHARE_CAP,
        "budget_ms": BUDGET_MS,
        "rules": scoring_rules(),
        "inventory_note": dict(inventory_note or {}),
        "summary": summary,
        "decisions": scored,
    }


def _score_prepared(
    item: Mapping[str, Any], book: Mapping[str, Mapping[str, Sequence[str]]]
) -> dict[str, Any]:
    request = item["request"]
    first = _normalize_harness(item["first"])
    second = _normalize_harness(item["second"])
    new = first["new"]
    old = first["old"]
    legal, reasons = decision_legal(
        request["side"],
        new.get("formation"),
        new.get("play"),
        new.get("adjustment_id"),
        new.get("macro_state"),
        book,
    )
    whole, model = first["latency_whole"], first["latency_model"]
    over = _over_budget(first["over_budget"], whole, model)
    seed_match = _signature(new) == _signature(second["new"])
    context = apply_context_rules(request, new.get("formation"), new.get("play"))
    context_failed = [rule["id"] for rule in context if rule["applies"] and rule["passed"] is False]
    return {
        "row_id": request["row_id"],
        "situation_id": request["situation_id"],
        "side": request["side"],
        "source": request["source"],
        "tags": list(request["tags"]),
        "game_id": request.get("game_id"),
        "down": request["down"],
        "distance": request["distance"],
        "distance_bucket": request["distance_bucket"],
        "yardline": request["yardline"],
        "field_zone": request.get("field_zone"),
        "quarter": request.get("quarter"),
        "clock_bucket": request.get("clock_bucket"),
        "score_bucket": request.get("score_bucket"),
        "default_filled": list(request.get("default_filled") or []),
        "look_observed": False,
        "opp_look_sent": request["opp_look"],
        "opp_look": first["opp_look"],
        "seed": first["seed"] if first["seed"] is not None else request["seed"],
        "seed_match": seed_match,
        "new": new,
        "old": old,
        "formation_agree": new.get("formation") == old.get("formation"),
        "play_agree": new.get("play") == old.get("play"),
        "adjustment_differs": new.get("adjustment_id") != old.get("adjustment_id"),
        "legal_our": legal,
        "legal_reasons": reasons,
        "legal_harness": first["legal"],
        "legal_disagreement": (
            None if first["legal"] is None else bool(first["legal"]) != legal
        ),
        "hard_fail": not legal,
        "fallback_reason": first["fallback_reason"],
        "model_version": first["model_version"],
        "scored_on": first["scored_on"],
        "latency_ms": {"whole": whole, "model": model},
        "budget_ms": first["budget_ms"],
        "over_budget": over,
        "harness_over_budget": first["over_budget"],
        "propensity": first["propensity"],
        "context": context,
        "context_failed": context_failed,
    }


def _signature(decision: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        decision.get("formation"),
        decision.get("play"),
        decision.get("adjustment_id"),
    )


def _over_budget(flag: Any, whole: float | None, model: float | None) -> bool:
    measured = False
    if whole is not None and whole > BUDGET_MS:
        measured = True
    if model is not None and model > BUDGET_MS:
        measured = True
    return bool(flag) or measured


def _normalize_harness(raw: Mapping[str, Any]) -> dict[str, Any]:
    new = _normalize_decision(raw.get("new") or {})
    old = _normalize_decision(raw.get("old") or {})
    whole, model = _latency(raw)
    fallback = _status(raw.get("fallback_reason"))
    budget = raw.get("budget_ms", BUDGET_MS)
    try:
        budget_ms = int(budget) if budget is not None else BUDGET_MS
    except (TypeError, ValueError):
        budget_ms = BUDGET_MS
    legal = raw.get("legal")
    if legal is not None:
        legal = bool(legal)
    seed = raw.get("seed")
    return {
        "new": new,
        "old": old,
        "fallback_reason": fallback,
        "model_version": raw.get("model_version"),
        "scored_on": raw.get("scored_on"),
        "latency_whole": whole,
        "latency_model": model,
        "budget_ms": budget_ms,
        "over_budget": raw.get("over_budget"),
        "legal": legal,
        "propensity": raw.get("propensity"),
        "opp_look": normalize_opp_look(raw.get("opp_look")),
        "seed": int(seed) if isinstance(seed, int) or (isinstance(seed, str) and str(seed).isdigit()) else seed,
    }


def _normalize_decision(raw: Mapping[str, Any]) -> dict[str, Any]:
    adjustment = _first(
        raw,
        "adjustment_id",
        "macro_id",
        "adjustment",
        "adj_or_macro",
    )
    if _is_no_adjustment(adjustment):
        adjustment = None
    return {
        "formation": _text(raw.get("formation")),
        "play": _text(raw.get("play")),
        "adjustment_id": adjustment,
        "macro_state": _text(raw.get("macro_state", raw.get("state"))),
        "display_line": _text(raw.get("display_line", raw.get("display", raw.get("line")))),
        "mode": _text(raw.get("mode")),
    }


def _first(raw: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key not in raw:
            continue
        value = raw.get(key)
        if _is_no_adjustment(value):
            continue
        return value
    return None


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _status(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, MLStatus):
        return value.value
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _latency(raw: Mapping[str, Any]) -> tuple[float | None, float | None]:
    whole = model = None
    latency = raw.get("latency_ms")
    if isinstance(latency, Mapping):
        whole = latency.get("whole", latency.get("whole_call", latency.get("call")))
        model = latency.get("model", latency.get("model_only"))
    if whole is None:
        whole = raw.get("latency_ms_whole")
    if model is None:
        model = raw.get("latency_ms_model")
    return _number(whole), _number(model)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def summarize(
    rows: Sequence[Mapping[str, Any]],
    book: Mapping[str, Mapping[str, Sequence[str]]],
) -> dict[str, Any]:
    slices: dict[str, Any] = {}
    for side in SIDES:
        slices[side] = {}
        for source in ("synthetic", "observed"):
            chosen = [row for row in rows if row["side"] == side and row["source"] == source]
            slices[side][source] = _slice_summary(chosen, book, side)
        side_rows = [row for row in rows if row["side"] == side]
        slices[side]["all"] = _slice_summary(side_rows, book, side)
    return {
        "decision_rows": len(rows),
        "by_side": slices,
        "adaptation": _adaptation(rows),
        "latency": _latency_summary(rows),
    }


def _slice_summary(
    rows: Sequence[Mapping[str, Any]],
    book: Mapping[str, Mapping[str, Sequence[str]]],
    side: str,
) -> dict[str, Any]:
    n = len(rows)
    formation_agree = sum(1 for row in rows if row["formation_agree"])
    play_agree = sum(1 for row in rows if row["play_agree"])
    hard_fail = sum(1 for row in rows if row["hard_fail"])
    in_book = sum(1 for row in rows if "outside_book" not in row["legal_reasons"])
    adjustment_legal = sum(1 for row in rows if "unarmed_macro" not in row["legal_reasons"])
    context_fail_rows = sum(1 for row in rows if row["context_failed"])
    rule_counts = []
    for rule in scoring_rules():
        applies = passed = failed = 0
        for row in rows:
            match = next(item for item in row["context"] if item["id"] == rule["id"])
            if not match["applies"]:
                continue
            applies += 1
            if match["passed"]:
                passed += 1
            else:
                failed += 1
        rule_counts.append(
            {"id": rule["id"], "applies": applies, "passed": passed, "failed": failed}
        )
    diversity = _diversity(rows, book.get(side) or {})
    return {
        "n": n,
        "formation_agree": formation_agree,
        "play_agree": play_agree,
        "formation_agree_rate": _rate(formation_agree, n),
        "play_agree_rate": _rate(play_agree, n),
        "in_book": in_book,
        "adjustment_legal": adjustment_legal,
        "hard_fail": hard_fail,
        "adjustment_differs": sum(1 for row in rows if row["adjustment_differs"]),
        "context_fail_rows": context_fail_rows,
        "rules": rule_counts,
        "diversity": diversity,
    }


def _rate(count: int, n: int) -> str | None:
    if n == 0:
        return None
    return f"{count}/{n}"


def _diversity(
    rows: Sequence[Mapping[str, Any]], book_side: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    eligible_calls = [
        (formation, play)
        for formation, plays in book_side.items()
        for play in plays
    ]
    eligible_formations = list(book_side)
    call_counts: dict[tuple[str, str], int] = {}
    formation_counts: dict[str, int] = {}
    used_calls = 0
    for row in rows:
        if row["hard_fail"] and "outside_book" in row["legal_reasons"]:
            continue
        formation = row["new"].get("formation")
        play = row["new"].get("play")
        if not formation or not play:
            continue
        if play not in list(book_side.get(formation) or []):
            continue
        call_counts[(formation, play)] = call_counts.get((formation, play), 0) + 1
        formation_counts[formation] = formation_counts.get(formation, 0) + 1
        used_calls += 1
    call_share = _top_share(call_counts)
    formation_share = _top_share(formation_counts)
    flagged = bool(
        (call_share["top_share"] is not None and call_share["top_share"] > SHARE_CAP)
        or (
            formation_share["top_share"] is not None
            and formation_share["top_share"] > SHARE_CAP
        )
    )
    return {
        "eligible_calls": len(eligible_calls),
        "eligible_formations": len(eligible_formations),
        "in_book_decisions": used_calls,
        "calls_used": len(call_counts),
        "formations_used": len(formation_counts),
        "call_share_of_book": _rate(len(call_counts), len(eligible_calls)),
        "formation_share_of_book": _rate(len(formation_counts), len(eligible_formations)),
        "call_entropy_bits": shannon_bits(list(call_counts.values())),
        "formation_entropy_bits": shannon_bits(list(formation_counts.values())),
        "top_call_share": call_share["top_share"],
        "top_call": call_share["top"],
        "top_formation_share": formation_share["top_share"],
        "top_formation": formation_share["top"],
        "share_cap": SHARE_CAP,
        "top_share_flag": flagged,
    }


def _top_share(counts: Mapping[Any, int]) -> dict[str, Any]:
    total = sum(counts.values())
    if total <= 0 or not counts:
        return {"top_share": None, "top": None}
    top = max(counts.items(), key=lambda item: (item[1], str(item[0])))
    return {"top_share": top[1] / total, "top": top[0]}


def shannon_bits(counts: Sequence[int]) -> float | None:
    """Shannon entropy in bits. Empty input is null, not a made-up zero."""
    total = sum(counts)
    if total <= 0:
        return None
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def nearest_rank(samples: Sequence[float], pct: float) -> float | None:
    """Nearest-rank percentile. Empty input is null.

    Index is ``ceil(pct/100 * n) - 1``, the same rank the coordinator demo uses.
    """
    if not samples:
        return None
    ordered = sorted(samples)
    index = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[index]


def _adaptation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["situation_id"], []).append(row)
    out: dict[str, Any] = {}
    for side in SIDES:
        for source in ("synthetic", "observed"):
            key = f"{side}:{source}"
            ids = [
                situation_id
                for situation_id, group in grouped.items()
                if group and group[0]["side"] == side and group[0]["source"] == source
            ]
            adapted = seed_noise = unchanged = insufficient = 0
            for situation_id in ids:
                group = grouped[situation_id]
                looks = {row["row_id"] for row in group}
                if len(looks) < 3:
                    insufficient += 1
                    continue
                if any(not row["seed_match"] for row in group):
                    seed_noise += 1
                    continue
                signatures = {_signature(row["new"]) for row in group}
                if len(signatures) > 1:
                    adapted += 1
                else:
                    unchanged += 1
            out[key] = {
                "situations": len(ids),
                "adapted": adapted,
                "unchanged_across_looks": unchanged,
                "seed_noise": seed_noise,
                "insufficient_looks": insufficient,
            }
    return out


def _latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    whole = [row["latency_ms"]["whole"] for row in rows if row["latency_ms"]["whole"] is not None]
    model = [row["latency_ms"]["model"] for row in rows if row["latency_ms"]["model"] is not None]
    fallbacks: dict[str, int] = {}
    unrecognized = 0
    for row in rows:
        status = row["fallback_reason"]
        if status is None or status == MLStatus.OK.value:
            continue
        fallbacks[status] = fallbacks.get(status, 0) + 1
        if status not in _ML_STATUS_VALUES:
            unrecognized += 1
    return {
        "n": len(rows),
        "budget_ms": BUDGET_MS,
        "whole": {
            "n": len(whole),
            "p50": nearest_rank(whole, 50),
            "p95": nearest_rank(whole, 95),
            "max": max(whole) if whole else None,
        },
        "model": {
            "n": len(model),
            "p50": nearest_rank(model, 50),
            "p95": nearest_rank(model, 95),
            "max": max(model) if model else None,
        },
        "fallback_counts": fallbacks,
        "unrecognized_fallback_status": unrecognized,
        "over_budget": sum(1 for row in rows if row["over_budget"]),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Coordinator decision-quality evaluation",
        "",
        report["disclaimer"],
        "",
        "Display lines are copied from the harness so a reader can see them. Agreement is formation and play, not formatting.",
        "Context rules are applied to the new decision. The old decision is the paired caller, not a second effectiveness score.",
        "",
        f"Share cap: {report['share_cap']}",
        "",
        f"Latency budget: {report['budget_ms']} ms",
        "",
        "## Scoring rules",
        "",
    ]
    for rule in report["rules"]:
        lines.append(f"- `{rule['id']}` ({rule['side']}): {rule['text']}")
    lines.extend(["", "## Fixture inventory", ""])
    note = report.get("inventory_note") or {}
    if note:
        for key, value in note.items():
            if isinstance(value, dict):
                lines.append(f"- {key}:")
                lines.append("")
                lines.append("```json")
                lines.append(json.dumps(value, indent=2, sort_keys=True))
                lines.append("```")
                lines.append("")
            else:
                lines.append(f"- {key}: {value}")
    summary = report["summary"]
    lines.extend(["", "## Summary counts", ""])
    lines.append(f"Decision rows scored: {summary['decision_rows']}")
    lines.append("")
    for side, sources in summary["by_side"].items():
        for source, block in sources.items():
            lines.append(f"### {side} / {source}")
            lines.append("")
            lines.append(f"- n: {block['n']}")
            lines.append(f"- formation agree: {block['formation_agree_rate']}")
            lines.append(f"- play agree: {block['play_agree_rate']}")
            lines.append(f"- in book: {block['in_book']}")
            lines.append(f"- adjustment legal (armed macro or no adjustment): {block['adjustment_legal']}")
            lines.append(f"- hard fail: {block['hard_fail']}")
            lines.append(f"- adjustment id differs: {block['adjustment_differs']}")
            lines.append(f"- context fail rows: {block['context_fail_rows']}")
            diversity = block["diversity"]
            lines.append(
                f"- book used: calls {diversity['call_share_of_book']}, "
                f"formations {diversity['formation_share_of_book']}"
            )
            lines.append(f"- call entropy bits: {diversity['call_entropy_bits']}")
            lines.append(f"- formation entropy bits: {diversity['formation_entropy_bits']}")
            lines.append(f"- top call share: {diversity['top_call_share']}")
            lines.append(f"- top formation share: {diversity['top_formation_share']}")
            lines.append(f"- top share flag (cap {diversity['share_cap']}): {diversity['top_share_flag']}")
            for rule in block["rules"]:
                lines.append(
                    f"- rule {rule['id']}: applies {rule['applies']}, "
                    f"passed {rule['passed']}, failed {rule['failed']}"
                )
            lines.append("")
    lines.append("## Opponent adaptation")
    lines.append("")
    lines.append(
        "A situation is adapted when it has at least three looks, every look "
        "matches on a same-seed rerun, and the new formation/play/adjustment "
        "is not the same on every look. A same-seed mismatch is seed noise "
        "and is not counted as adaptation."
    )
    lines.append("")
    for key, block in summary["adaptation"].items():
        lines.append(
            f"- {key}: situations {block['situations']}, adapted {block['adapted']}, "
            f"unchanged {block['unchanged_across_looks']}, seed noise {block['seed_noise']}, "
            f"fewer than 3 looks {block['insufficient_looks']}"
        )
    latency = summary["latency"]
    lines.extend(["", "## Latency", ""])
    lines.append(f"- rows: {latency['n']}")
    lines.append(
        f"- whole p50/p95/max: {latency['whole']['p50']} / {latency['whole']['p95']} / {latency['whole']['max']}"
    )
    lines.append(
        f"- model p50/p95/max: {latency['model']['p50']} / {latency['model']['p95']} / {latency['model']['max']}"
    )
    lines.append(f"- over budget: {latency['over_budget']}")
    lines.append(f"- fallback counts: {json.dumps(latency['fallback_counts'], sort_keys=True)}")
    lines.extend(["", "## Paired decisions", ""])
    lines.append(
        "| row | side | tags | look sent | new formation | new play | new adjustment | new state | old formation | old play | old adjustment | formation agree | play agree | our legal | hard fail |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["decisions"]:
        cells = [
            row["row_id"],
            row["side"],
            ",".join(row["tags"]),
            json.dumps(row["opp_look_sent"], sort_keys=True),
            row["new"].get("formation"),
            row["new"].get("play"),
            row["new"].get("adjustment_id"),
            row["new"].get("macro_state"),
            row["old"].get("formation"),
            row["old"].get("play"),
            row["old"].get("adjustment_id"),
            row["formation_agree"],
            row["play_agree"],
            row["legal_our"],
            row["hard_fail"],
        ]
        lines.append("| " + " | ".join(_cell(value) for value in cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "/")


def write_report(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_scoring.json"
    md_path = out_dir / "eval_scoring.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score coordinator decisions against the old caller.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True, help="Temp CFB_COACH_MADDEN_DB path")
    parser.add_argument("--mode", default="heuristic")
    args = parser.parse_args(list(argv) if argv is not None else None)
    identities, note = load_situation_identities()
    book = load_installed_book(args.db)
    report = evaluate(
        identities,
        adapt_run_situation,
        book,
        mode=args.mode,
        inventory_note={**note, **{"fixture_counts": fixture_counts(identities)}},
    )
    write_report(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
