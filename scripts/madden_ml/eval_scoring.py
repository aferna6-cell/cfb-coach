"""Offline decision-quality scoring for the offense and defense coordinators.

This module does not change a live call. It compares the new coordinator
decision to the old caller on a fixed situation set and writes a JSON report
plus Markdown. The numbers in that report are counts from the rows it scored.

These are decision-quality checks only. They are not effectiveness claims.
Nothing here estimates yards, win probability, or whether a call worked.

The live comparison is reached only through ``adapt_run_situation``, which
calls ``eval_harness.run_situation``. The offline model is reached only
through ``adapt_score_offline_model``, which calls
``eval_harness.score_offline_model``. Both require the harness only from
inside those two functions.

A call that passes here passed its defined contextual rules, which cover a
limited set of decisions.
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
from cfb_coach.vod_success import shared_features as _sf
from cfb_coach.vod_success.shared_features import look_family as _look_family
from cfb_coach.vod_success.shared_features import norm_key as _norm_key
from cfb_coach.zones import field_zone


def _norm(value: Any) -> str:
    return _norm_key(value) or ""


DISCLAIMER = (
    "These are decision-quality checks only. They are not effectiveness claims."
)

CONTEXT_RULES_NOTE = (
    "A call that passes here passed its defined contextual rules, which cover a limited set of decisions."
)

BUDGET_MS = 150
# Shadow reruns use seeds 0 .. SEED_COUNT-1. The default is seeds 0-9.
# model_pick is the seed-0 play. The situation-id seed is not added.
SEED_COUNT = 10
_WILSON_Z = 1.959963984540054
DEFENSE_PENDING_REASON = "defense path pending (call keyed by offense play)"
# Highest seed count wins. A tie uses exact, case-sensitive string order.
HEURISTIC_TIE_BREAK = (
    "heuristic_majority_pick breaks ties by lexicographic play name "
    "(exact string order, case-sensitive)."
)
# Adjacent seed pairs whose plays differ, divided by (seed count - 1).
HEURISTIC_SEED_CHANGE = (
    "heuristic_seed_change_rate is the fraction of adjacent seeds whose "
    "plays differ. One seed has rate 0."
)
# First key present on the seed-0 heuristic (then any later seed) wins.
_HEURISTIC_WEIGHT_KEYS = ("candidate_probabilities", "sampling_weights", "weights")
# Look text and family follow the shared feature module (recipe f7c17d16 O4,
# live-shaped): coverage_seen, then def_overlay. def_call has no live source and
# is never a look or family source; it is still recorded as a present column.
_COVERAGE_FIELDS = ("def_call", "coverage_seen", "def_overlay")
_LOOK_FIELDS = ("coverage_seen", "def_overlay")
# These are personnel packages. They are not trained look or look_family keys.
_PACKAGE_TOKENS = frozenset(
    {
        "nickel",
        "dime",
        "base",
        "empty",
        "dollar",
        "quarter",
        "goal_line",
        "11",
        "12",
        "21",
        "10",
        "22",
        "13",
    }
)
# Fallback family labels are not trained-key hits.
_FALLBACK_LOOK_KEYS = frozenset({"other", "none"})
_SNAP_ID_FIELDS = ("video_id", "vod_id", "game_id", "youtube_id", "id")
LABEL_LOOK_CHECK = "training rows, key-construction check, not held-out"

FIXTURE_DIR = _ROOT / "tests" / "madden_ml" / "fixtures" / "item3"
SYNTHETIC_FIXTURE = FIXTURE_DIR / "synthetic_grid.json"
OBSERVED_CSV = FIXTURE_DIR / "presnap_observed_cpu.csv"
OBSERVED_DEFENSE_CSV = FIXTURE_DIR / "presnap_observed_defense.csv"

# Strata the offline diagnostic is allowed to score. cfb27 is never used.
_STRATUM_RE = re.compile(r"^madden27/(cpu|human)/(offense|defense)$")
# Checked-in presnap rows use generated ids such as syn-game-01.
_SYNTHETIC_GAME_ID = re.compile(r"^syn-game-\d{2}$")

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
            if game_id and _SYNTHETIC_GAME_ID.fullmatch(game_id) is None:
                raise ValueError(f"row {index} game id is not a synthetic presnap id")
            zone = observed_field_zone(yardline, distance)
            looks = []
            coverage = _csv_coverage(fieldnames, raw)
            for item in assigned_observed_looks(index - 1):
                look = dict(item)
                look.update(coverage)
                looks.append(look)
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
                    looks=looks,
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


def _csv_coverage(fieldnames: Sequence[str], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Coverage columns that are actually on this file. Absent columns stay absent."""
    return {key: raw.get(key) for key in _COVERAGE_FIELDS if key in fieldnames}


def expand_decision_rows(identities: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per situation × look. Observed looks are tagged assigned, not observed."""
    expanded: list[dict[str, Any]] = []
    for identity in identities:
        source = identity["source"]
        for look in identity["looks"]:
            row = dict(identity)
            row.pop("looks", None)
            if source == "observed" and identity.get("looks_from_file"):
                tag = identity.get("source_tag") or "observed"
                row["tags"] = ["observed_situation", tag]
                row["look_observed"] = True
            elif source == "observed":
                row["tags"] = ["observed_situation", "assigned_look"]
                row["look_observed"] = False
            else:
                row["tags"] = list(identity["tags"])
                row["look_observed"] = False
            row["look_id"] = look["look_id"]
            row["opp_personnel"] = look["opp_personnel"]
            row["opp_set"] = look["opp_set"]
            for key in _COVERAGE_FIELDS:
                if key in look:
                    row[key] = look[key]
                elif key in identity:
                    row[key] = identity[key]
            if "defense_look" in look:
                row["defense_look"] = look["defense_look"]
            elif "defense_look" in identity:
                row["defense_look"] = identity["defense_look"]
            row["opp_look"] = normalize_opp_look(
                {
                    "opp_personnel": look["opp_personnel"],
                    "opp_set": look["opp_set"],
                }
            )
            row["row_id"] = f"{identity['situation_id']}__{look['look_id']}"
            row["offensive_personnel"] = offensive_personnel_of(identity)
            row["fold_id"] = identity.get("fold_id")
            row["source_tag"] = identity.get("source_tag")
            expanded.append(row)
    return expanded


def offensive_personnel_of(row: Mapping[str, Any]) -> Any:
    """Personnel only when the row itself carries it.

    A formation name is not a personnel value. Missing stays None, and the
    literal ``empty`` is left as ``empty``.
    """
    if "offensive_personnel" in row:
        return _blank_to_none(row.get("offensive_personnel"))
    if "off_personnel" in row:
        return _blank_to_none(row.get("off_personnel"))
    return None


def load_observed_defense_csv(path: Path) -> list[dict[str, Any]]:
    """Optional defense snaps. The file is not required.

    Each row keeps its own ``fold_id``. Opponent personnel and set are taken
    from their columns. Formation is not read, and personnel is not inferred
    from a formation name. Score and clock are neutral defaults flagged
    ``default_filled`` because this file does not carry them.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        required = ("side", "down", "distance", "opp_personnel", "opp_set", "source_tag", "fold_id")
        missing = [name for name in required if name not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
        if "yardline" not in fieldnames and "field_zone" not in fieldnames and "yardline/field_zone" not in fieldnames:
            raise ValueError(f"{path} needs yardline or field_zone")
        rows: list[dict[str, Any]] = []
        for index, raw in enumerate(reader, start=1):
            side = _normalize_side(raw.get("side") or "")
            down = _exact_int(raw.get("down"), "down")
            distance = _exact_int(raw.get("distance"), "distance")
            yardline, zone = _yardline_or_zone(raw)
            personnel = _blank_to_none(raw.get("opp_personnel"))
            opp_set = _blank_to_none(raw.get("opp_set"))
            source_tag = (raw.get("source_tag") or "").strip() or "observed"
            fold_id = (raw.get("fold_id") or "").strip()
            if not fold_id:
                raise ValueError(f"row {index} is missing fold_id")
            look = {
                "look_id": f"file-{index}",
                "opp_personnel": personnel,
                "opp_set": opp_set,
            }
            look.update(_csv_coverage(fieldnames, raw))
            identity = _identity(
                situation_id=f"obs-def-{index:04d}",
                side=side,
                tags=["observed"],
                source="observed",
                down=down,
                distance=distance,
                yardline=yardline if yardline is not None else 0,
                field_zone=zone,
                clock_name="normal",
                score_name="close",
                game_id=None,
                default_filled=["score", "clock"],
                looks=[look],
                look_observed=True,
                quarter=None,
                clock_seconds=None,
                two_minute=False,
                score_us=0,
                score_them=0,
            )
            if yardline is None:
                prefix = "d " if side == "defense" else ""
                identity["yardline"] = None
                identity["raw"] = f"{prefix}{down}&{distance}".strip()
            identity["looks_from_file"] = True
            identity["source_tag"] = source_tag
            identity["fold_id"] = fold_id
            identity["offensive_personnel"] = None
            identity["formation"] = None
            rows.append(identity)
    return rows


def _yardline_or_zone(raw: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """Read yardline and field zone from their own columns.

    A combined ``yardline/field_zone`` header is accepted. Digits are a
    yardline. Any other token is a field-zone label. Neither is invented,
    and neither is treated as personnel.
    """
    yardline = None
    zone = None
    if "yardline" in raw and str(raw.get("yardline") or "").strip():
        yardline = _exact_int(raw.get("yardline"), "yardline")
    if "field_zone" in raw and str(raw.get("field_zone") or "").strip():
        zone = str(raw.get("field_zone")).strip()
    combined = raw.get("yardline/field_zone")
    if combined is not None and str(combined).strip() and yardline is None and zone is None:
        text = str(combined).strip()
        if re.fullmatch(r"-?\d+", text):
            yardline = int(text)
        else:
            zone = text
    if zone is None and yardline is not None:
        zone = observed_field_zone(yardline, _exact_int(raw.get("distance"), "distance"))
    return yardline, zone


def _attach_optional_defense(
    identities: list[dict[str, Any]],
    note: dict[str, Any],
    defense_csv: Path | None,
) -> list[dict[str, Any]]:
    defense_path = OBSERVED_DEFENSE_CSV if defense_csv is None else Path(defense_csv)
    note["observed_defense_csv"] = str(defense_path)
    explicit = defense_csv is not None
    if explicit and not defense_path.is_file():
        raise FileNotFoundError(defense_path)
    note["observed_defense_csv_present"] = defense_path.is_file()
    if defense_path.is_file():
        defense_rows = load_observed_defense_csv(defense_path)
        note["observed_defense_source_rows"] = len(defense_rows)
        return identities + defense_rows
    note["observed_defense_source_rows"] = 0
    return identities


def load_situation_identities(
    defense_csv: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Synthetic identities from the builder, plus a CSV when that file is present."""
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
        identities = _attach_optional_defense(synthetic, note, defense_csv)
        note["fixture_counts"] = fixture_counts(identities)
        return identities, note
    observed = load_observed_csv(OBSERVED_CSV)
    note["observed_source_rows"] = len(observed)
    identities = _attach_optional_defense(synthetic + observed, note, defense_csv)
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


_HARNESS: Any = _UNSET
_SHADOW_RESOURCES: Any = _UNSET
_BUNDLES: dict[str, dict[str, Any]] = {}


def _harness_module() -> Any:
    """Load the sibling harness module once. Missing file returns None."""
    global _HARNESS
    if _HARNESS is not _UNSET:
        return _HARNESS
    path = Path(__file__).with_name("eval_harness.py")
    if not path.is_file():
        _HARNESS = None
        return None
    import importlib.util

    spec = importlib.util.spec_from_file_location("madden_ml_eval_harness", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _HARNESS = module
    return module


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
    module = _harness_module()
    if module is None:
        raise FileNotFoundError(
            "scripts/madden_ml/eval_harness.py is not on this branch; "
            "pass a stub runner to evaluate()"
        )
    run = getattr(module, "run_situation", None)
    if run is None:
        raise AttributeError("eval_harness.py has no run_situation")
    return dict(run(sit, mode=mode, opp_look=normalize_opp_look(opp_look), side=side, seed=seed))


def limit_strata(strata: Any, side: str) -> list[str]:
    """Keep ``madden27/cpu/*`` and ``madden27/human/*`` for this side only."""
    if isinstance(strata, str):
        items = [strata]
    else:
        items = list(strata or [])
    kept: list[str] = []
    for item in items:
        text = str(item)
        if _STRATUM_RE.fullmatch(text) is None:
            raise ValueError(
                f"stratum {text!r} is outside madden27/cpu/* and madden27/human/*"
            )
        if not text.endswith("/" + side):
            raise ValueError(f"stratum {text!r} does not match side {side}")
        if text not in kept:
            kept.append(text)
    if not kept:
        raise ValueError("no madden27 strata for this side")
    return kept


def strata_for_side(side: str) -> list[str]:
    return [f"madden27/cpu/{side}", f"madden27/human/{side}"]


def _coerce_csv_number(value: Any) -> Any:
    """CSV text that is an integer becomes an int. Blank text becomes None."""
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        if re.fullmatch(r"-?\d+", text):
            return int(text)
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _trainer_text(value: Any) -> str | None:
    """Trainer normalization. Structured values are not look text."""
    if value is None or isinstance(value, (Mapping, list, tuple, bool)):
        return None
    text = _norm(value)
    return text or None


def _raw_coverage_nonblank(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _coverage_containers(sit: Any, opp_look: Any) -> list[Mapping[str, Any]]:
    containers: list[Mapping[str, Any]] = []
    for source in (sit, opp_look):
        if not isinstance(source, Mapping):
            continue
        defense = source.get("defense_look")
        if isinstance(defense, Mapping):
            containers.append(defense)
        containers.append(source)
    return containers


def _raise_if_coverage_dropped(*, columns_present: bool, nonblank: bool, look: str | None) -> None:
    """Hard-fail only when coverage columns hold a value and the look is empty."""
    if columns_present and nonblank and look is None:
        raise RuntimeError(
            "source has coverage columns but the look came out empty"
        )


def build_prep_look(sit: Any, opp_look: Any = None) -> dict[str, Any]:
    """Look from coverage_seen, then def_overlay. def_call is never read.

    ``look_family`` is ``shared_features.look_family`` of that same look.
    A source with none of those columns is no look (``None``), not a failure.
    A personnel package is not a coverage column.
    """
    raws: dict[str, Any] = {}
    columns_present = False
    for container in _coverage_containers(sit, opp_look):
        for field in _COVERAGE_FIELDS:
            if field in container and field not in raws:
                raws[field] = container[field]
                columns_present = True
    if isinstance(opp_look, str) and "coverage_seen" not in raws:
        text = _trainer_text(opp_look)
        if text and text not in _PACKAGE_TOKENS:
            raws["coverage_seen"] = opp_look
            columns_present = True
    nonblank = any(
        _raw_coverage_nonblank(raws[field]) for field in _LOOK_FIELDS if field in raws
    )
    look = _trainer_text(raws.get("coverage_seen")) or _trainer_text(raws.get("def_overlay"))
    _raise_if_coverage_dropped(columns_present=columns_present, nonblank=nonblank, look=look)
    family = _look_family(look or "")
    if not columns_present:
        status = "no_coverage_in_source"
    elif look is None:
        status = "blank_coverage"
    else:
        status = "coverage"
    return {
        "look": look,
        "family": family,
        "columns_present": columns_present,
        "nonblank": nonblank,
        "status": status,
    }


_TRAINER_ZONE_NOTE = (
    "Trainer zone buckets are own_deep (yardline <= 20), own (< 50), "
    "opp (50-79), and red_zone (>= 80). There is no midfield or goal_line label. "
    "The eval field_zone is the report label. The yardline passed to the hook is unchanged."
)


def _prepare_hook_call(sit: Any, opp_look: Any, side: str) -> dict[str, Any]:
    """Coerce CSV numbers, then record the look value the hook will receive."""
    types = {"down": "missing", "distance": "missing", "yardline": "missing"}
    coerced = {"down": None, "distance": None, "yardline": None}
    hook_sit: Any = sit
    if isinstance(sit, Mapping):
        from cfb_coach.madden.situation import parse_madden_situation

        present = {}
        for key in ("down", "distance", "yardline"):
            if key not in sit:
                continue
            raw_value = sit.get(key)
            types[key] = type(raw_value).__name__
            coerced[key] = _coerce_csv_number(raw_value)
            present[key] = coerced[key]
        raw = sit.get("raw")
        if isinstance(raw, str) and raw.strip():
            parsed = parse_madden_situation(raw, default_side=side)
            for key, value in present.items():
                setattr(parsed, key, value)
            hook_sit = parsed
        else:
            hook_sit = {key: sit.get(key) for key in sit}
            hook_sit.update(present)
    prep = build_prep_look(sit, opp_look)
    passed_look = prep["look"]
    assigned = normalize_opp_look(opp_look)
    return {
        "hook_sit": hook_sit,
        "opp_look": passed_look,
        "diagnostics": {
            "opp_look": passed_look,
            "opp_look_type": type(passed_look).__name__,
            "opp_look_raw": repr(passed_look),
            "opp_look_assigned": assigned,
            "prep_look": prep["look"],
            "prep_look_family": prep["family"],
            "no_coverage_in_source": prep["status"] == "no_coverage_in_source",
            "coverage_status": prep["status"],
            "down_type": types["down"],
            "distance_type": types["distance"],
            "yardline_type": types["yardline"],
            "down": coerced["down"],
            "distance": coerced["distance"],
            "yardline": coerced["yardline"],
        },
    }


def adapt_score_offline_model(
    sit: Any,
    *,
    side: str,
    model_dir: Any,
    strata: Any,
    opp_look: Any = None,
    seed: int = 0,
    offline_ok: bool = False,
) -> dict[str, Any]:
    """The only call into ``eval_harness.score_offline_model``.

    Refuses unless ``offline_ok`` is true. The harness ranks the installed
    book. This adapter does not rank on its own.
    """
    if offline_ok is not True:
        raise RuntimeError("score_offline_model refuses to run unless offline_ok=True")
    checked = limit_strata(strata, side)
    module = _harness_module()
    if module is None:
        raise FileNotFoundError(
            "scripts/madden_ml/eval_harness.py is not on this branch"
        )
    hook = getattr(module, "score_offline_model", None)
    if hook is None:
        raise AttributeError("eval_harness.py has no score_offline_model")
    prepared = _prepare_hook_call(sit, opp_look, side)
    result = dict(
        hook(
            prepared["hook_sit"],
            side=side,
            model_dir=model_dir,
            strata=checked,
            opp_look=prepared["opp_look"],
            seed=int(seed),
            offline_ok=True,
        )
    )
    result["input_diagnostics"] = prepared["diagnostics"]
    return result


def wilson_interval(successes: int, n: int) -> dict[str, Any]:
    """Wilson score interval at 95%. ``n`` of 0 has no rate and no bounds."""
    if n <= 0:
        return {"k": successes, "n": 0, "rate": None, "low": None, "high": None}
    z = _WILSON_Z
    phat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n)))
    return {"k": successes, "n": n, "rate": phat, "low": center - margin, "high": center + margin}


def is_vod_derived(row: Mapping[str, Any]) -> bool:
    """True when a row carries a VOD id. Synthetic presnap game ids are not VODs."""
    for key in ("vod", "video_id", "vod_id"):
        if row.get(key):
            return True
    blob = " ".join(str(row.get(key) or "") for key in ("game_id", "source_tag"))
    return re.search(r"\bvod[:_-]|video_id", blob, re.I) is not None


def load_model_bundle(model_dir: Any) -> dict[str, Any]:
    """Read ``manifest.json`` and ``model_params.json`` from a directory."""
    directory = Path(model_dir)
    if directory.is_file():
        directory = directory.parent
    key = str(directory.resolve())
    cached = _BUNDLES.get(key)
    if cached is not None:
        return cached
    manifest_path = directory / "manifest.json"
    params_path = directory / "model_params.json"
    if not manifest_path.is_file() or not params_path.is_file():
        raise FileNotFoundError(f"{directory} needs manifest.json and model_params.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    params = json.loads(params_path.read_text(encoding="utf-8"))
    version = manifest.get("label_version")
    if not version or params.get("label_version") != version:
        raise ValueError(
            f"{directory} manifest version {version!r} does not match model_params "
            f"{params.get('label_version')!r}"
        )
    bundle = {
        "dir": directory,
        "manifest": manifest,
        "params": params,
        "model_version": str(version),
        "fold_id": manifest.get("fold_id"),
    }
    _BUNDLES[key] = bundle
    return bundle


def resolve_fold_dir(model_dir: Path, fold_id: str) -> Path | None:
    """Per-fold folder that is passed through as ``model_dir``. No match is None."""
    root = Path(model_dir)
    for candidate in (root / "folds" / fold_id, root / fold_id):
        if (candidate / "manifest.json").is_file() and (candidate / "model_params.json").is_file():
            return candidate
    return None


class _ShadowResources:
    """Trimmed book on one Madden file, with a different coach file left unused."""

    def __init__(self, madden: Path, coach: Path) -> None:
        if madden.resolve() == coach.resolve():
            raise ValueError("CFB_COACH_DB and CFB_COACH_MADDEN_DB must be different files")
        madden.parent.mkdir(parents=True, exist_ok=True)
        coach.parent.mkdir(parents=True, exist_ok=True)
        from cfb_coach.db import CoachDB

        self.book = load_installed_book(madden)
        held = CoachDB(coach)
        held.close()


def set_shadow_databases(madden: Path, coach: Path) -> _ShadowResources:
    """Install the trimmed book on ``madden`` and keep ``coach`` a different file."""
    global _SHADOW_RESOURCES
    resources = _ShadowResources(Path(madden), Path(coach))
    _SHADOW_RESOURCES = resources
    return resources


def _norm_token(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def _tokens_agree(left: Any, right: Any) -> bool:
    return _norm_token(left) == _norm_token(right)


def _payload_signature(payload: Mapping[str, Any], which: str) -> tuple[Any, Any, Any]:
    block = payload if which == "model" else (payload.get("heuristic") or {})
    return (
        _norm_token(block.get("formation")),
        _norm_token(block.get("play")),
        _norm_token(block.get("adjustment_id")),
    )


def _is_model_refusal(payload: Mapping[str, Any]) -> str | None:
    reason = payload.get("fallback_reason")
    text = "" if reason is None else str(reason).strip().casefold()
    if text and text != "ok":
        return str(reason)
    if not _norm_token(payload.get("play")):
        return "no_model_play"
    return None


def _seed_values(seed: int, seed_count: int) -> list[int]:
    """Seeds ``0 .. seed_count-1``. ``seed`` is not added.

    The default report compares seeds 0-9. ``model_pick`` is the seed-0 play.
    """
    _ = int(seed)
    count = max(1, int(seed_count))
    return list(range(count))


def _is_defense_path(side: Any, stratum: Any) -> bool:
    if side == "defense":
        return True
    return isinstance(stratum, str) and stratum.endswith("/defense")


def _ordered_runs(runs: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(runs, key=lambda run: int(run.get("seed") or 0))


def _run_at_seed(runs: Sequence[Mapping[str, Any]], seed: int) -> Mapping[str, Any]:
    for run in runs:
        if int(run.get("seed") or 0) == seed:
            return run
    return runs[0] if runs else {}


def _carried_features(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """Copy the hook ``features`` dict. Keys are not rebuilt here."""
    raw = payload.get("features")
    if not isinstance(raw, Mapping):
        return None
    return json.loads(json.dumps(raw))


def _heuristic_weight_table(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Sampling weights or probabilities exposed by the heuristic, if any."""
    ordered = _ordered_runs(runs)
    preferred = []
    seed0 = _run_at_seed(ordered, 0) if ordered else None
    if seed0 is not None:
        preferred.append(seed0)
    preferred.extend(run for run in ordered if run is not seed0)
    for run in preferred:
        heuristic = run.get("heuristic") if isinstance(run.get("heuristic"), Mapping) else {}
        for source in (heuristic, run):
            if not isinstance(source, Mapping):
                continue
            for key in _HEURISTIC_WEIGHT_KEYS:
                table = source.get(key)
                if isinstance(table, Mapping) and table:
                    return dict(table)
    return None


def _weight_for_pick(table: Mapping[str, Any], model_pick: Any) -> float:
    target = _norm_token(model_pick)
    for key, value in table.items():
        if _norm_token(key) == target:
            return float(value)
    return 0.0


def _majority_play(plays: Sequence[Any]) -> tuple[Any, float]:
    """Modal play. Ties break by lexicographic play name. Share is count/n."""
    counts: dict[str, int] = {}
    labels: dict[str, Any] = {}
    for play in plays:
        key = "" if play is None else str(play)
        counts[key] = counts.get(key, 0) + 1
        labels.setdefault(key, play)
    if not counts:
        return None, 0.0
    best = max(counts.values())
    winner = sorted(key for key, count in counts.items() if count == best)[0]
    return labels[winner], best / float(len(plays))


def _heuristic_seed_change_rate(plays: Sequence[Any]) -> float:
    """Fraction of adjacent seeds whose plays differ."""
    if len(plays) <= 1:
        return 0.0
    changes = 0
    for left, right in zip(plays, plays[1:]):
        if _norm_token(left) != _norm_token(right):
            changes += 1
    return changes / float(len(plays) - 1)


def _model_pick_probability(
    model_pick: Any,
    heuristic_plays: Sequence[Any],
    runs: Sequence[Mapping[str, Any]],
) -> tuple[Any, str | None]:
    """Weight of ``model_pick`` when the heuristic exposes one, else its seed frequency.

    A zero frequency is ``'<0.1'`` (no hit in the seed sample). Weights are
    used as given; they are not renormalized.
    """
    table = _heuristic_weight_table(runs)
    if table is not None:
        return _weight_for_pick(table, model_pick), "heuristic_weights"
    if not heuristic_plays:
        return None, None
    count = sum(1 for play in heuristic_plays if _tokens_agree(play, model_pick))
    if count == 0:
        return "<0.1", "seed_frequency_estimate"
    return count / float(len(heuristic_plays)), "seed_frequency_estimate"


def _paired_columns(
    request: Mapping[str, Any],
    stratum: str | None,
    runs: Sequence[Mapping[str, Any]],
    *,
    refusal: str | None,
) -> dict[str, Any]:
    """Per-situation paired columns. Defense stays in the file and out of the metrics."""
    ordered = _ordered_runs(runs)
    seed0 = _run_at_seed(ordered, 0) if ordered else {}
    model_pick = seed0.get("play") if ordered else None
    model_plays = [_norm_token(run.get("play")) for run in ordered]
    model_seed_stable = bool(ordered) and len(set(model_plays)) <= 1
    heuristic_picks = [
        {"seed": int(run.get("seed") or 0), "play": (run.get("heuristic") or {}).get("play")}
        for run in ordered
    ]
    plays = [item["play"] for item in heuristic_picks]
    majority, share = _majority_play(plays)
    stable = bool(plays) and share >= 0.5
    if not stable:
        agreement = "no_stable_heuristic_pick"
        agree_with_majority = None
    else:
        agree_with_majority = _tokens_agree(model_pick, majority)
        agreement = "agree" if agree_with_majority else "disagree"
    if ordered:
        prob, prob_source = _model_pick_probability(model_pick, plays, ordered)
        change_rate: float | None = _heuristic_seed_change_rate(plays)
    else:
        prob, prob_source = None, None
        change_rate = None
        agreement = "no_stable_heuristic_pick"
        agree_with_majority = None
        stable = False
        majority = None
        share = None
    if _is_defense_path(request.get("side"), stratum):
        evaluable = False
        reason = DEFENSE_PENDING_REASON
    elif refusal:
        evaluable = False
        reason = refusal
    else:
        evaluable = True
        reason = None
    features = _carried_features(seed0) if ordered else None
    groups = features.get("groups") if isinstance(features, Mapping) else None
    zone_group = groups.get("zone") if isinstance(groups, Mapping) else None
    trainer_zone = zone_group.get("key") if isinstance(zone_group, Mapping) else None
    field_zone = request.get("field_zone")
    zone_note = None
    if trainer_zone is not None and trainer_zone != field_zone:
        zone_note = (
            f"eval field_zone {field_zone!r} at yardline {request.get('yardline')}; "
            f"trainer zone key {trainer_zone!r}. {_TRAINER_ZONE_NOTE}"
        )
    hook_look = seed0.get("opp_look") if isinstance(seed0, Mapping) and "opp_look" in seed0 else None
    return {
        "look_id": request.get("look_id"),
        "model_pick": model_pick,
        "model_seed_stable": model_seed_stable if ordered else None,
        "heuristic_picks": heuristic_picks,
        "heuristic_majority_pick": majority,
        "heuristic_majority_share": share,
        "heuristic_seed_change_rate": change_rate,
        "stable_heuristic": stable,
        "agreement": agreement,
        "agree_with_majority": agree_with_majority,
        "model_pick_heuristic_prob": prob,
        "prob_source": prob_source,
        "features": features,
        "hook_look": hook_look,
        "trainer_zone": trainer_zone,
        "zone_note": zone_note,
        "evaluable": evaluable,
        "evaluable_reason": reason,
        "heuristic_tie_break": HEURISTIC_TIE_BREAK,
    }


def executed_call_label(side: Any, call: Any, *, defense_book: Any = None) -> dict[str, Any]:
    """Label a held-out row's executed call with the shared feature rules.

    Defense: family and pressure through ``shared_features.defense_family``,
    the same book-then-pattern lookup training and live use (addendum 3). Offense: ``shared_features.call_class``. Held-out CSVs carry
    no confidence columns, so no confidence filter applies here. A blank call
    is not a scored row (addendum 2) and comes back with ``scored=False``.
    """
    text = _norm_key(call)
    side_n = _sf._side(side)
    if side_n == "defense":
        family, pressure, path = _sf.defense_family_path(text, defense_book)
        return {
            "side": side_n,
            "call": text,
            "scored": text is not None,
            "family": family,
            "pressure": bool(pressure),
            "is_pressure": _sf.is_pressure(family, pressure),
            "family_path": path,
        }
    return {
        "side": side_n,
        "call": text,
        "scored": text is not None,
        "call_class": _sf.call_class(text),
        "kneel": bool(text and _sf.is_kneel(text)),
    }


def model_fallback(
    side: Any,
    opp_look: Any,
    scored: Any,
    *,
    candidate: Any = None,
) -> dict[str, Any]:
    """Fallback reason for one ranked row: deduped top-two margin, side-aware blank look.

    ``scored`` is ``(play, score)`` pairs or ``{play: score}`` (higher is
    better). The same play in two formations counts once. Same function as
    live: ``inference.vod_fallback_reason``.
    """
    from cfb_coach.madden.model.inference import vod_fallback_reason

    margin = _sf.top_two_margin(scored)
    return {
        "top_two_margin": margin,
        "vod_fallback": vod_fallback_reason(opp_look, margin, side=_sf._side(side), candidate=candidate),
    }


def _prob_number(value: Any) -> float | None:
    """``'<0.1'`` sorts as 0, below every positive seed frequency."""
    if value == "<0.1":
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _median_prob(values: Sequence[Any]) -> float | None:
    nums = [number for number in (_prob_number(value) for value in values) if number is not None]
    if not nums:
        return None
    nums.sort()
    mid = len(nums) // 2
    if len(nums) % 2:
        return nums[mid]
    return (nums[mid - 1] + nums[mid]) / 2.0


def _coef_keys(bundle: Mapping[str, Any], stratum: str, group: str) -> set[str]:
    strata = (bundle.get("params") or {}).get("strata") or {}
    block = strata.get(stratum) or {}
    coef = ((block.get("model") or {}).get("coef") or {})
    raw = coef.get(group) or {}
    if not isinstance(raw, Mapping):
        return set()
    return {_norm(key) for key in raw}


def _is_trained_match(key: Any, trained: set[str], *, exclude_packages: bool) -> bool:
    """True when ``key`` is a trained key and not a fallback or package token."""
    text = _trainer_text(key)
    if text is None or text in _FALLBACK_LOOK_KEYS:
        return False
    if exclude_packages and text in _PACKAGE_TOKENS:
        return False
    return text in trained


def _feature_real_hit(features: Any, group_name: str, *, exclude_packages: bool) -> bool:
    groups = features.get("groups") if isinstance(features, Mapping) else None
    group = groups.get(group_name) if isinstance(groups, Mapping) else None
    if not isinstance(group, Mapping) or group.get("matched") is not True:
        return False
    return _is_trained_match(group.get("key"), {_norm(str(group.get("key")))}, exclude_packages=exclude_packages)


def _look_match_flags(
    request: Mapping[str, Any],
    features: Any,
    look_trained: set[str],
    family_trained: set[str],
) -> dict[str, Any]:
    prep = build_prep_look(request, request.get("opp_look"))
    look_hit = _is_trained_match(prep["look"], look_trained, exclude_packages=True) or _feature_real_hit(
        features, "look", exclude_packages=True
    )
    family_hit = _is_trained_match(prep["family"], family_trained, exclude_packages=False) or _feature_real_hit(
        features, "look_family", exclude_packages=False
    )
    return {
        "prep_look": prep["look"],
        "prep_look_family": prep["family"],
        "no_coverage_in_source": prep["status"] == "no_coverage_in_source",
        "look_key_matched": look_hit,
        "look_family_key_matched": family_hit,
    }


def _look_flip_count(rows: Sequence[Mapping[str, Any]]) -> int:
    """Situations whose model_pick changes across looks that hit a trained key.

    A fallback key (``other`` or ``none``) or a package token does not count.
    """
    grouped: dict[Any, dict[Any, Any]] = {}
    for row in rows:
        if row.get("look_key_matched") is not True and row.get("look_family_key_matched") is not True:
            continue
        looks = grouped.setdefault(row.get("situation_id"), {})
        looks[row.get("look_id")] = row.get("model_pick")
    count = 0
    for looks in grouped.values():
        picks = {_norm_token(play) for play in looks.values()}
        if len(looks) >= 2 and len(picks) > 1:
            count += 1
    return count


def _situation_changes_within_look(rows: Sequence[Mapping[str, Any]]) -> int:
    """Looks whose model_pick is not the same on every situation."""
    grouped: dict[Any, dict[Any, Any]] = {}
    for row in rows:
        situations = grouped.setdefault(row.get("look_id"), {})
        situations[row.get("situation_id")] = row.get("model_pick")
    count = 0
    for situations in grouped.values():
        picks = {_norm_token(play) for play in situations.values()}
        if len(situations) >= 2 and len(picks) > 1:
            count += 1
    return count


def _look_match_shares(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Share of non-refused rows whose look or look_family hits a trained key."""
    counted = [row for row in rows if row.get("outcome") != "refused"]
    n = len(counted)
    look_hits = sum(1 for row in counted if row.get("look_key_matched") is True)
    family_hits = sum(1 for row in counted if row.get("look_family_key_matched") is True)
    return {
        "look_match_n": n,
        "look_key_matches": look_hits,
        "look_key_match_share": (look_hits / n) if n else None,
        "look_family_key_matches": family_hits,
        "look_family_key_match_share": (family_hits / n) if n else None,
        "look_match_note": (
            "Fallback keys other and none, and package tokens, do not count. "
            "The share uses non-refused rows."
        ),
    }


def _paired_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stratum metrics. Defense rows stay out of the agreement metrics."""
    shares = _look_match_shares(rows)
    if rows and all(_is_defense_path(row.get("side"), row.get("stratum")) for row in rows):
        return {
            "evaluable": False,
            "reason": DEFENSE_PENDING_REASON,
            "n_rows_excluded": len(rows),
            **shares,
        }
    evaluable = [row for row in rows if row.get("evaluable") is True]
    stable = [row for row in evaluable if row.get("stable_heuristic") is True]
    agreed = sum(1 for row in stable if row.get("agree_with_majority") is True)
    rates = [
        float(row["heuristic_seed_change_rate"])
        for row in evaluable
        if isinstance(row.get("heuristic_seed_change_rate"), (int, float))
        and not isinstance(row.get("heuristic_seed_change_rate"), bool)
    ]
    probs = [row.get("model_pick_heuristic_prob") for row in evaluable]
    return {
        "evaluable": True,
        "n_situations": len(evaluable),
        "n_stable": len(stable),
        "agreement_vs_majority": wilson_interval(agreed, len(stable)),
        "mean_heuristic_seed_change_rate": (sum(rates) / len(rates)) if rates else None,
        "median_model_pick_heuristic_prob": _median_prob(probs),
        "look_flip_count": _look_flip_count(evaluable),
        "situation_changes_pick_within_look": _situation_changes_within_look(evaluable),
        "heuristic_tie_break": HEURISTIC_TIE_BREAK,
        "heuristic_seed_change_rate_definition": HEURISTIC_SEED_CHANGE,
        **shares,
    }


def _refused_row(request: Mapping[str, Any], reason: str, *, model_version: str | None, stratum: str | None, model_dir: Any) -> dict[str, Any]:
    tags = list(request.get("tags") or [])
    if request.get("side") == "defense" and request.get("source") == "synthetic":
        for tag in ("synthetic_only", "exploratory"):
            if tag not in tags:
                tags.append(tag)
    return {
        "row_id": request.get("row_id"),
        "situation_id": request.get("situation_id"),
        "side": request.get("side"),
        "source": request.get("source"),
        "summary_bucket": "refused",
        "tags": tags,
        "stratum": stratum,
        "model_version": model_version,
        "manifest_model_version": model_version,
        "returned_model_version": None,
        "version_mismatch": False,
        "fold_id": request.get("fold_id"),
        "returned_fold_id": None,
        "fold_mismatch": False,
        "indicative": False,
        "outcome": "refused",
        "refusal_reason": reason,
        "play_agree": None,
        "formation_agree": None,
        "adjustment_agree": None,
        "gate_bypassed": "offline_eval",
        "scored_on": "unreviewed",
        "offensive_personnel": offensive_personnel_of(request),
        "opp_personnel": (request.get("opp_look") or {}).get("opp_personnel") if isinstance(request.get("opp_look"), Mapping) else None,
        "opp_set": (request.get("opp_look") or {}).get("opp_set") if isinstance(request.get("opp_look"), Mapping) else None,
        "formation": None,
        "model_formation": None,
        "heuristic_formation": None,
        "model_play": None,
        "heuristic_play": None,
        "model_adjustment": None,
        "heuristic_adjustment": None,
        "down": request.get("down"),
        "distance": request.get("distance"),
        "distance_bucket": request.get("distance_bucket"),
        "down_and_distance": f"{request.get('down')}|{request.get('distance_bucket')}",
        "field_zone": request.get("field_zone"),
        "yardline": request.get("yardline"),
        "live_eligible": False,
        "seed": request.get("seed"),
        "seed_runs": [],
        "model_dir": str(model_dir) if model_dir is not None else None,
        "context_rules_note": CONTEXT_RULES_NOTE,
        **_paired_columns(request, stratum, [], refusal=reason),
        **_look_match_flags(request, None, set(), set()),
    }


def _paired_from_runs(
    request: Mapping[str, Any],
    stratum: str,
    bundle: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    book: Mapping[str, Mapping[str, Sequence[str]]],
    model_dir: Path,
) -> dict[str, Any]:
    primary = dict(runs[0])
    manifest_version = str(bundle["model_version"])
    returned_version = primary.get("model_version")
    returned_fold = primary.get("fold_id")
    our_fold = request.get("fold_id")
    version_mismatch = returned_version is not None and str(returned_version) != manifest_version
    fold_mismatch = False
    if our_fold and returned_fold not in (None, "", our_fold):
        fold_mismatch = True
    if bundle.get("fold_id") and our_fold and str(bundle.get("fold_id")) != str(our_fold):
        fold_mismatch = True
    indicative = bool(version_mismatch or fold_mismatch)
    refusal = _is_model_refusal(primary)
    heuristic = primary.get("heuristic") or {}
    model_play = primary.get("play")
    model_formation = primary.get("formation")
    model_adjustment = primary.get("adjustment_id")
    if refusal:
        outcome = "refused"
        play_agree = formation_agree = adjustment_agree = None
    else:
        play_agree = _tokens_agree(model_play, heuristic.get("play"))
        formation_agree = _tokens_agree(model_formation, heuristic.get("formation"))
        adjustment_agree = _tokens_agree(model_adjustment, heuristic.get("adjustment_id"))
        outcome = "same" if play_agree and formation_agree and adjustment_agree else "disagree"
    tags = list(request.get("tags") or [])
    if request.get("side") == "defense" and request.get("source") == "synthetic" and not indicative:
        for tag in ("synthetic_only", "exploratory"):
            if tag not in tags:
                tags.append(tag)
    if indicative and "indicative" not in tags:
        tags.append("indicative")
    if indicative:
        summary_bucket = "indicative"
    elif refusal:
        summary_bucket = "refused"
    else:
        summary_bucket = "synthetic" if request.get("source") == "synthetic" else "observed"
    legal, reasons = decision_legal(
        str(request.get("side")),
        model_formation,
        model_play,
        model_adjustment,
        primary.get("macro_state"),
        book,
    )
    model_context = apply_context_rules(request, model_formation, model_play) if model_play else []
    heuristic_context = apply_context_rules(
        request, heuristic.get("formation"), heuristic.get("play")
    ) if heuristic.get("play") else []
    seed_runs = []
    for payload in runs:
        seed_runs.append(
            {
                "seed": payload.get("seed"),
                "model": list(_payload_signature(payload, "model")),
                "heuristic": list(_payload_signature(payload, "heuristic")),
                "refused": _is_model_refusal(payload) is not None,
            }
        )
    look = request.get("opp_look") if isinstance(request.get("opp_look"), Mapping) else {}
    # The summary key is the label_version of the manifest that was loaded
    # for this row (root or per-fold). A hook that returns a different
    # version flags the row indicative, but does not move it into that
    # other version's counts.
    group_version = manifest_version
    return {
        "row_id": f"{request.get('row_id')}__{stratum}",
        "situation_id": request.get("situation_id"),
        "side": request.get("side"),
        "source": request.get("source"),
        "summary_bucket": summary_bucket,
        "tags": tags,
        "stratum": stratum,
        "model_version": group_version,
        "manifest_model_version": manifest_version,
        "returned_model_version": None if returned_version is None else str(returned_version),
        "version_mismatch": version_mismatch,
        "fold_id": our_fold,
        "returned_fold_id": returned_fold,
        "fold_mismatch": fold_mismatch,
        "indicative": indicative,
        "outcome": outcome,
        "refusal_reason": refusal,
        "play_agree": play_agree,
        "formation_agree": formation_agree,
        "adjustment_agree": adjustment_agree,
        "gate_bypassed": primary.get("gate_bypassed") or "offline_eval",
        "scored_on": "unreviewed",
        "offensive_personnel": offensive_personnel_of(request),
        "opp_personnel": look.get("opp_personnel") if isinstance(look, Mapping) else None,
        "opp_set": look.get("opp_set") if isinstance(look, Mapping) else None,
        "formation": model_formation,
        "model_formation": model_formation,
        "heuristic_formation": heuristic.get("formation"),
        "model_play": model_play,
        "heuristic_play": heuristic.get("play"),
        "model_adjustment": model_adjustment,
        "heuristic_adjustment": heuristic.get("adjustment_id"),
        "down": request.get("down"),
        "distance": request.get("distance"),
        "distance_bucket": request.get("distance_bucket"),
        "down_and_distance": f"{request.get('down')}|{request.get('distance_bucket')}",
        "field_zone": request.get("field_zone"),
        "yardline": request.get("yardline"),
        "input_diagnostics": dict(primary.get("input_diagnostics") or {}),
        "live_eligible": (
            bool(primary.get("live_eligible"))
            if "live_eligible" in primary
            else bool(primary.get("legal")) and bool(model_play) and bool(model_formation)
        ),
        "legal_our": legal,
        "legal_reasons": reasons,
        "model_context": model_context,
        "heuristic_context": heuristic_context,
        "seed": request.get("seed"),
        "seed_runs": seed_runs,
        "latency_ms": primary.get("latency_ms"),
        "over_budget": primary.get("over_budget"),
        "model_dir": str(model_dir),
        "context_rules_note": CONTEXT_RULES_NOTE,
        **_paired_columns(request, stratum, runs, refusal=refusal),
        **_look_match_flags(
            request,
            _carried_features(primary),
            _coef_keys(bundle, stratum, "look"),
            _coef_keys(bundle, stratum, "look_family"),
        ),
    }


def evaluate_shadow(
    identities: Sequence[Mapping[str, Any]],
    runner: Callable[..., Mapping[str, Any]],
    book: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    model_dir: Path,
    requested_version: str | None = None,
    seed_count: int = SEED_COUNT,
    inventory_note: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score each situation with the offline hook. Versions are not pooled."""
    root = load_model_bundle(model_dir)
    if requested_version is not None and root["model_version"] != requested_version:
        raise ValueError(
            f"manifest label_version is {root['model_version']!r}, not {requested_version!r}"
        )
    decisions: list[dict[str, Any]] = []
    excluded: list[str] = []
    for request in expand_decision_rows(identities):
        if is_vod_derived(request):
            excluded.append(str(request.get("situation_id")))
            continue
        side = str(request["side"])
        fold_id = request.get("fold_id")
        if fold_id:
            fold_dir = resolve_fold_dir(root["dir"], str(fold_id))
            if fold_dir is None:
                decisions.append(
                    _refused_row(
                        request,
                        "no_matching_fold",
                        model_version=None,
                        stratum=None,
                        model_dir=root["dir"],
                    )
                )
                continue
            try:
                bundle = load_model_bundle(fold_dir)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                decisions.append(
                    _refused_row(
                        request,
                        f"no_matching_fold: {exc}",
                        model_version=None,
                        stratum=None,
                        model_dir=fold_dir,
                    )
                )
                continue
            if bundle.get("fold_id") and str(bundle["fold_id"]) != str(fold_id):
                decisions.append(
                    _refused_row(
                        request,
                        "no_matching_fold",
                        model_version=bundle["model_version"],
                        stratum=None,
                        model_dir=fold_dir,
                    )
                )
                continue
            directory = fold_dir
        else:
            bundle = root
            directory = root["dir"]
        for stratum in strata_for_side(side):
            if stratum not in (bundle["params"].get("strata") or {}):
                decisions.append(
                    _refused_row(
                        request,
                        "no_matching_fold" if fold_id else "stratum_missing",
                        model_version=bundle["model_version"],
                        stratum=stratum,
                        model_dir=directory,
                    )
                )
                continue
            runs = []
            hook_sit = {
                "raw": request.get("raw"),
                "down": request.get("down"),
                "distance": request.get("distance"),
                "yardline": request.get("yardline"),
                "look_id": request.get("look_id"),
            }
            for key in (*_COVERAGE_FIELDS, "defense_look"):
                if key in request:
                    hook_sit[key] = request[key]
            for seed in _seed_values(int(request["seed"]), seed_count):
                payload = dict(
                    runner(
                        hook_sit,
                        side=side,
                        model_dir=directory,
                        strata=[stratum],
                        opp_look=request.get("opp_look"),
                        seed=seed,
                        offline_ok=True,
                    )
                )
                payload.setdefault("seed", seed)
                runs.append(payload)
            decisions.append(_paired_from_runs(request, stratum, bundle, runs, book, directory))
    return {
        "disclaimer": DISCLAIMER,
        "context_rules_note": CONTEXT_RULES_NOTE,
        "budget_ms": BUDGET_MS,
        "rules": scoring_rules(),
        "inventory_note": dict(inventory_note or {}),
        "requested_model_version": requested_version or root["model_version"],
        "summary_grouped_by": "loaded_manifest_label_version",
        "seed_count": int(seed_count),
        "seeds": _seed_values(0, seed_count),
        "heuristic_tie_break": HEURISTIC_TIE_BREAK,
        "no_coverage_in_source": _no_coverage_in_source(decisions),
        "vod_derived_excluded": excluded,
        "vod_derived_in_sample": [],
        "by_model_version": _shadow_by_version(decisions),
        "decisions": decisions,
    }


def _no_coverage_in_source(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Rows whose source file has no def_call, coverage_seen, or def_overlay."""
    rows = [row for row in decisions if row.get("no_coverage_in_source") is True]
    return {
        "n": len(rows),
        "row_ids": [row.get("row_id") for row in rows],
        "note": (
            "These rows come from a source with no def_call, coverage_seen, or "
            "def_overlay. With that field order every one of them is no look. "
            "They are not a failure."
        ),
    }


def _shadow_by_version(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    versions: dict[str, list[Mapping[str, Any]]] = {}
    unversioned_refused = [row for row in decisions if row.get("model_version") in (None, "")]
    for row in decisions:
        version = row.get("model_version")
        if version in (None, ""):
            continue
        versions.setdefault(str(version), []).append(row)
    out: dict[str, Any] = {}
    for version, rows in versions.items():
        out[version] = {
            "model_version": version,
            "grouped_by": "loaded_manifest_label_version",
            "by_side": {
                side: _shadow_side(side, [row for row in rows if row.get("side") == side])
                for side in SIDES
            },
        }
    out["refused_without_version"] = {
        "n": len(unversioned_refused),
        "by_reason": _count_reasons(unversioned_refused),
    }
    return out


def _shadow_side(side: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    blocks = {}
    for bucket in ("synthetic", "observed", "indicative", "refused"):
        chosen = [row for row in rows if row.get("summary_bucket") == bucket]
        blocks[bucket] = _shadow_block(chosen)
    if side == "defense":
        blocks["synthetic_diff_tags"] = ["synthetic_only", "exploratory"]
        blocks["indicative_kept_separate"] = True
    return blocks


def _shadow_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("outcome") in {"same", "disagree"}]
    refused = [row for row in rows if row.get("outcome") == "refused"]
    by_stratum = {}
    strata = sorted({str(row.get("stratum")) for row in rows if row.get("stratum")})
    for stratum in strata:
        stratum_rows = [row for row in rows if row.get("stratum") == stratum]
        by_stratum[stratum] = _agreement_block(stratum_rows)
    return {
        "n": len(rows),
        "refused": _agreement_block(refused)["refused"] if refused else {"n": 0, "by_reason": {}},
        "scored_n": len(scored),
        "by_stratum": by_stratum,
        "pooled_across_strata": False,
    }


def _agreement_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    refused = [row for row in rows if row.get("outcome") == "refused"]
    same = [row for row in rows if row.get("outcome") == "same"]
    disagree = [row for row in rows if row.get("outcome") == "disagree"]
    scored = same + disagree
    play_k = sum(1 for row in scored if row.get("play_agree") is False)
    formation_k = sum(1 for row in scored if row.get("formation_agree") is False)
    adjustment_k = sum(1 for row in scored if row.get("adjustment_agree") is False)
    joint = wilson_interval(len(disagree), len(scored))
    return {
        "n": len(rows),
        "same": len(same),
        "disagree": len(disagree),
        "refused": {"n": len(refused), "by_reason": _count_reasons(refused)},
        "play_disagree": play_k,
        "formation_disagree": formation_k,
        "adjustment_disagree": adjustment_k,
        "disagreement": joint,
        "play_disagreement": wilson_interval(play_k, len(scored)),
        "by_down_and_distance": _split_disagreement(scored, "down_and_distance"),
        "by_field_zone": _split_disagreement(scored, "field_zone"),
        "by_opp_personnel": _bucket_quality(scored, "opp_personnel"),
        "by_offensive_personnel": _bucket_quality(scored, "offensive_personnel"),
        "by_down_and_distance_quality": _bucket_quality(scored, "down_and_distance"),
        "by_field_zone_quality": _bucket_quality(scored, "field_zone"),
        "seed_noise": _seed_noise(rows),
        "penalty": None,
        "paired_summary": _paired_summary(rows),
    }


def _count_reasons(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("refusal_reason") or "refused")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _split_disagreement(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(key), []).append(row)
    out = []
    for value, group in sorted(groups.items(), key=lambda item: str(item[0])):
        disagree = sum(1 for row in group if row.get("outcome") == "disagree")
        out.append({"value": value, "disagreement": wilson_interval(disagree, len(group))})
    return out


def _bucket_quality(rows: Sequence[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(key), []).append(row)
    out = []
    for value, group in sorted(groups.items(), key=lambda item: str(item[0])):
        out.append(
            {
                "value": value,
                "n": len(group),
                "heuristic": _caller_quality(group, "heuristic"),
                "model": _caller_quality(group, "model"),
                "penalty": None,
            }
        )
    return out


def _caller_quality(rows: Sequence[Mapping[str, Any]], caller: str) -> dict[str, Any]:
    formation_key = "heuristic_formation" if caller == "heuristic" else "model_formation"
    context_key = "heuristic_context" if caller == "heuristic" else "model_context"
    play_key = "heuristic_play" if caller == "heuristic" else "model_play"
    usable = [row for row in rows if row.get(play_key)]
    counts: dict[Any, int] = {}
    for row in usable:
        counts[row.get(formation_key)] = counts.get(row.get(formation_key), 0) + 1
    return {
        "n": len(usable),
        "formation_usage": _usage_shares(counts),
        "rules": _rule_pass_rates(usable, context_key),
    }


def _rule_pass_rates(rows: Sequence[Mapping[str, Any]], context_key: str) -> list[dict[str, Any]]:
    rates = []
    for rule in scoring_rules():
        applies = passed = 0
        for row in rows:
            match = next((item for item in row.get(context_key) or [] if item.get("id") == rule["id"]), None)
            if match is None or not match.get("applies"):
                continue
            applies += 1
            if match.get("passed"):
                passed += 1
        rates.append(
            {
                "id": rule["id"],
                "applies": applies,
                "passed": passed,
                "pass_rate": _rate(passed, applies),
            }
        )
    return rates


def _seed_noise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """How often a pick changes across seeds, and how often it changes across situations."""
    out = {}
    for caller in ("model", "heuristic"):
        comparable = []
        seed_shift = 0
        primary_sigs = []
        for row in rows:
            runs = [item for item in row.get("seed_runs") or [] if not item.get("refused")]
            if len(runs) < 1:
                continue
            signatures = [tuple(item.get(caller) or []) for item in runs]
            comparable.append(row)
            if len(set(signatures)) > 1:
                seed_shift += 1
            primary_sigs.append(signatures[0])
        n = len(comparable)
        distinct = len(set(primary_sigs))
        modal = 0
        if primary_sigs:
            modal = max(primary_sigs.count(sig) for sig in set(primary_sigs))
        situation_shift = (n - modal) / n if n else None
        out[caller] = {
            "n": n,
            "seeds_compared": max((len(row.get("seed_runs") or []) for row in comparable), default=0),
            "seed_shift_n": seed_shift,
            "seed_shift_rate": (seed_shift / n) if n else None,
            "distinct_situation_picks": distinct,
            "situation_shift_rate": situation_shift,
        }
    return out


def _render_shadow_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Offline shadow decision-quality",
        "",
        report["disclaimer"],
        "",
        report.get("context_rules_note") or CONTEXT_RULES_NOTE,
        "",
        "Formation usage shares and contextual-rule pass rates are reported side by side.",
        "No share penalty is applied.",
        "A refused or fallback row is its own bucket and is not counted as the same as the heuristic.",
        "Every summary is grouped by the model_version on the manifest that was loaded for those rows.",
        "Observed-defense rows scored with a per-fold fit are reported under that fold manifest's model_version and are not pooled with any other model_version.",
        "Indicative rows, where the returned version or fold does not match, stay out of the synthetic exploratory counts.",
        HEURISTIC_TIE_BREAK,
        HEURISTIC_SEED_CHANGE,
        "model_pick is the seed-0 play. model_seed_stable records whether that play is the same on every seed (0-9 by default).",
        f"Defense rows are still emitted and marked not evaluable: {DEFENSE_PENDING_REASON} They are excluded from paired summary agreement metrics. Look-key match share is still reported for them.",
        "Rows whose source has no def_call, coverage_seen, or def_overlay are no look. They are reported as no_coverage_in_source and are not a failure.",
        "A row is a hard failure only when its source has those coverage columns and a non-blank value still produces an empty look.",
        _TRAINER_ZONE_NOTE,
        "",
        f"Rows with no coverage in source: {(report.get('no_coverage_in_source') or {}).get('n')}",
        "",
        f"Requested model version: {report.get('requested_model_version')}",
        f"VOD-derived rows excluded: {len(report.get('vod_derived_excluded') or [])}",
        "",
    ]
    versions = report.get("by_model_version") or {}
    for version, block in versions.items():
        if version == "refused_without_version":
            lines.append(f"## Refused before a model version was loaded: {block.get('n')}")
            lines.append("")
            lines.append(json.dumps(block.get("by_reason") or {}, sort_keys=True))
            lines.append("")
            continue
        lines.append(f"## Model version {version}")
        lines.append("")
        for side, side_block in (block.get("by_side") or {}).items():
            lines.append(f"### {side}")
            lines.append("")
            for bucket in ("synthetic", "observed", "indicative", "refused"):
                part = side_block.get(bucket) or {}
                lines.append(f"#### {bucket}")
                lines.append("")
                lines.append(f"- rows: {part.get('n')}")
                lines.append(f"- scored: {part.get('scored_n')}")
                for stratum, stats in (part.get("by_stratum") or {}).items():
                    disagreement = stats.get("disagreement") or {}
                    lines.append(
                        f"- {stratum}: same {stats.get('same')} disagree {stats.get('disagree')} "
                        f"refused {stats.get('refused', {}).get('n')} "
                        f"disagreement {disagreement.get('k')}/{disagreement.get('n')} "
                        f"95% CI {disagreement.get('low')} .. {disagreement.get('high')}"
                    )
                    noise = stats.get("seed_noise") or {}
                    for caller, caller_noise in noise.items():
                        lines.append(
                            f"  - {caller} seed shift {caller_noise.get('seed_shift_n')}/{caller_noise.get('n')} "
                            f"situation shift {caller_noise.get('situation_shift_rate')}"
                        )
                    paired = stats.get("paired_summary") or {}
                    if paired.get("evaluable") is False:
                        lines.append(
                            f"  - paired summary excluded: {paired.get('reason')} "
                            f"({paired.get('n_rows_excluded')} rows)"
                        )
                        lines.append(
                            f"  - look key match {paired.get('look_key_matches')}/{paired.get('look_match_n')} "
                            f"look family match {paired.get('look_family_key_matches')}/{paired.get('look_match_n')}"
                        )
                    elif paired:
                        agree = paired.get("agreement_vs_majority") or {}
                        lines.append(
                            f"  - paired situations {paired.get('n_situations')} "
                            f"stable {paired.get('n_stable')} "
                            f"agree {agree.get('k')}/{agree.get('n')} "
                            f"95% CI {agree.get('low')} .. {agree.get('high')} "
                            f"mean heuristic seed change {paired.get('mean_heuristic_seed_change_rate')} "
                            f"median model_pick_heuristic_prob {paired.get('median_model_pick_heuristic_prob')} "
                            f"look flips {paired.get('look_flip_count')} "
                            f"situation_changes_pick_within_look {paired.get('situation_changes_pick_within_look')} "
                            f"look key match {paired.get('look_key_matches')}/{paired.get('look_match_n')} "
                            f"look family match {paired.get('look_family_key_matches')}/{paired.get('look_match_n')}"
                        )
                lines.append("")
    lines.extend(_label_look_lines(report))
    lines.append("Paired rows are in `paired_rows.jsonl`.")
    lines.append("")
    return "\n".join(lines)


def _label_look_lines(report: Mapping[str, Any]) -> list[str]:
    check = report.get("label_look_key_check")
    if not isinstance(check, Mapping):
        return []
    lines = [
        f"## {check.get('label')}",
        "",
        f"Excluded snaps: {check.get('excluded_n')}",
        "",
    ]
    for stratum, block in (check.get("by_stratum") or {}).items():
        lines.append(
            f"- {stratum} ({block.get('side')}): "
            f"n {block.get('n')} "
            f"look key {block.get('look_key_matches')}/{block.get('n')} "
            f"share {block.get('look_key_match_share')} "
            f"look family {block.get('look_family_key_matches')}/{block.get('n')} "
            f"share {block.get('look_family_key_match_share')}"
        )
    lines.append("")
    return lines


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
        "context_rules_note": CONTEXT_RULES_NOTE,
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
        "gate_bypassed": first.get("gate_bypassed"),
        "offensive_personnel": offensive_personnel_of(request),
        "stratum": request.get("stratum"),
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
        "gate_bypassed": raw.get("gate_bypassed"),
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
        "formation_usage": _usage_shares(formation_counts),
    }


def _usage_shares(counts: Mapping[Any, int]) -> list[dict[str, Any]]:
    """Formation or call shares. No threshold and no penalty."""
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
    return [
        {
            "value": key,
            "n": count,
            "share": (count / total) if total else None,
        }
        for key, count in ranked
    ]


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
        entropy -= p * math.log(p, 2)
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
    if "by_model_version" in report:
        return _render_shadow_markdown(report)
    lines = [
        "# Coordinator decision-quality evaluation",
        "",
        report["disclaimer"],
        "",
        "Display lines are copied from the harness so a reader can see them. Agreement is formation and play, not formatting.",
        "Context rules are applied to the new decision. The old decision is the paired caller, not a second effectiveness score.",
        "",
        report.get("context_rules_note") or CONTEXT_RULES_NOTE,
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
            lines.append(f"- formation usage: {json.dumps(diversity['formation_usage'], sort_keys=True)}")
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


def write_report(report: Mapping[str, Any], out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_scoring.json"
    md_path = out_dir / "eval_scoring.md"
    jsonl_path = out_dir / "paired_rows.jsonl"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in report.get("decisions") or []:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return json_path, md_path, jsonl_path


def load_label_snaps(path: Any) -> list[dict[str, Any]]:
    """Read a JSON list of snaps. Each snap's ``defense_look`` carries the coverage fields."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and isinstance(payload.get("snaps"), list):
        payload = payload["snaps"]
    if not isinstance(payload, list):
        raise ValueError("label snaps must be a JSON list")
    return [dict(item) for item in payload]


def _snap_id_values(snap: Mapping[str, Any]) -> set[str]:
    found: set[str] = set()
    sources: list[Mapping[str, Any]] = [snap]
    defense = snap.get("defense_look")
    if isinstance(defense, Mapping):
        sources.append(defense)
    for source in sources:
        for key in _SNAP_ID_FIELDS:
            value = source.get(key)
            if value not in (None, ""):
                found.add(str(value))
    return found


def _snap_stratum(snap: Mapping[str, Any]) -> tuple[str, str]:
    side = str(snap.get("side") or snap.get("streamer_side") or "").strip().casefold()
    stratum = snap.get("stratum")
    if stratum:
        text = str(stratum)
        if text.endswith("/defense"):
            side = "defense"
        elif text.endswith("/offense"):
            side = "offense"
        elif side not in SIDES:
            side = "offense"
        return text, side
    if side not in SIDES:
        side = "offense"
    opponent = str(snap.get("opponent_type") or "cpu").strip().casefold()
    if opponent not in {"cpu", "human"}:
        opponent = "cpu"
    return f"madden27/{opponent}/{side}", side


def label_look_key_check(
    snaps: Sequence[Mapping[str, Any]],
    model_dir: Any,
    exclude_ids: Sequence[str],
) -> dict[str, Any]:
    """Match share of constructed look keys on label snaps.

    ``exclude_ids`` is the caller's list. This function does not supply ids
    of its own. The result is a key-construction check on training rows.
    """
    bundle = load_model_bundle(model_dir)
    excluded = {str(item) for item in exclude_ids}
    grouped: dict[str, dict[str, Any]] = {}
    excluded_n = 0
    for snap in snaps:
        if _snap_id_values(snap) & excluded:
            excluded_n += 1
            continue
        stratum, side = _snap_stratum(snap)
        prep = build_prep_look(snap, snap.get("defense_look"))
        look_hit = _is_trained_match(
            prep["look"], _coef_keys(bundle, stratum, "look"), exclude_packages=True
        )
        family_hit = _is_trained_match(
            prep["family"], _coef_keys(bundle, stratum, "look_family"), exclude_packages=False
        )
        block = grouped.setdefault(stratum, {"side": side, "rows": []})
        block["rows"].append({"look": look_hit, "family": family_hit})
    by_stratum: dict[str, Any] = {}
    for stratum, block in sorted(grouped.items()):
        rows = block["rows"]
        n = len(rows)
        look_hits = sum(1 for row in rows if row["look"])
        family_hits = sum(1 for row in rows if row["family"])
        by_stratum[stratum] = {
            "side": block["side"],
            "n": n,
            "look_key_matches": look_hits,
            "look_key_match_share": (look_hits / n) if n else None,
            "look_family_key_matches": family_hits,
            "look_family_key_match_share": (family_hits / n) if n else None,
        }
    return {
        "label": LABEL_LOOK_CHECK,
        "excluded_n": excluded_n,
        "by_stratum": by_stratum,
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Score coordinator decisions against the old caller.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True, help="Temp CFB_COACH_MADDEN_DB path")
    parser.add_argument("--coach-db", type=Path, default=None, help="A different temp CFB_COACH_DB path")
    parser.add_argument("--mode", default="heuristic")
    parser.add_argument("--shadow-model", default=None, help="Manifest label_version to score offline")
    parser.add_argument(
        "--model-params",
        type=Path,
        default=None,
        help="Model directory (manifest.json and model_params.json), such as /workspace/vod_data/models/v0.21",
    )
    parser.add_argument("--observed-defense", type=Path, default=None)
    parser.add_argument(
        "--seed-count",
        type=int,
        default=SEED_COUNT,
        help="Seeds 0 .. seed-count-1. The default 10 is seeds 0-9.",
    )
    parser.add_argument(
        "--label-snaps",
        type=Path,
        default=None,
        help="Optional JSON snaps. defense_look supplies def_call, coverage_seen, and def_overlay.",
    )
    parser.add_argument(
        "--exclude-id",
        action="append",
        default=None,
        help="Id to leave out of the label look-key check. Repeat for each id.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    exclude_ids = [str(item) for item in (args.exclude_id or [])]
    label_check = None
    if args.label_snaps is not None:
        if args.model_params is None:
            raise SystemExit("--model-params is required with --label-snaps")
        label_check = label_look_key_check(
            load_label_snaps(args.label_snaps),
            args.model_params,
            exclude_ids,
        )
    coach = args.coach_db
    if coach is None:
        coach = args.db.parent / "coach.db"
    if coach.resolve() == args.db.resolve():
        coach = args.db.with_name("coach-session.db")
    identities, note = load_situation_identities(defense_csv=args.observed_defense)
    if args.shadow_model:
        if args.model_params is None:
            raise SystemExit("--model-params is required with --shadow-model")
        resources = set_shadow_databases(args.db, coach)
        report = evaluate_shadow(
            identities,
            adapt_score_offline_model,
            resources.book,
            model_dir=args.model_params,
            requested_version=args.shadow_model,
            seed_count=args.seed_count,
            inventory_note={**note, **{"fixture_counts": fixture_counts(identities)}},
        )
    else:
        book = load_installed_book(args.db)
        report = evaluate(
            identities,
            adapt_run_situation,
            book,
            mode=args.mode,
            inventory_note={**note, **{"fixture_counts": fixture_counts(identities)}},
        )
    if label_check is not None:
        report["label_look_key_check"] = label_check
    write_report(report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
