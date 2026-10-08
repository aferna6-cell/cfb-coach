"""Snap fields the v0.7 trainer reads.

Strata are ``{game}/{opponent_type}`` and are never pooled. A blank opponent
type stays ``unknown``. Distance, zone, and look families use the same labels
as the box model (``medium``, ``own_deep``, ``cover_3``, and so on).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from cfb_coach.learning import evaluate_snap

CONF_MIN_NAMES = 0.5

SITUATION_GROUPS = ("dist", "down", "down_x_dist", "score", "side", "time", "zone")
ENTITY_GROUPS = ("call", "call_class", "look", "look_family", "off_adj", "def_adj", "pressure")
INTERACTION_GROUPS = ("call_x_family", "class_x_family")
ALL_GROUPS = SITUATION_GROUPS + ENTITY_GROUPS + INTERACTION_GROUPS

_RUN_WORDS = (
    "inside zone", "outside zone", "mid zone", "zone toss", "hb base", "counter",
    "duo", "dive", "stretch", "power", "draw", "slam", "wham", "trap", "toss",
    "sweep", "blast", "iso", "gut", "lead",
)
_PASS_WORDS = (
    "mesh", "spot", "flood", "dig", "corner", "post", "seam", "wheel", "slant",
    "hitch", "curl", "drag", "screen", "vertical", "sail", "levels", "dagger",
    "ohio", "bench", "stick", "cross", "drive", "out", "comeback", "fade",
    "smash", "spacing", "y-trail", "h-trail",
)
_SPECIAL_RE = re.compile(
    r"\b(kick\s*off|kickoff|punt|field\s*goal|\bfg\b|extra\s*point|\bpat\b|onside)\b",
    re.I,
)
_CLOCK_RE = re.compile(r"\bq\s*([1-4])\b[^0-9]{0,6}(\d{1,2}):(\d{2})", re.I)
_QUARTER_RE = re.compile(r"\bq\s*([1-4])\b", re.I)
_SCORE_NOTE = re.compile(r"score_us\s*=\s*(-?\d+).{0,40}score_them\s*=\s*(-?\d+)", re.I)


@dataclass
class Snap:
    game: str
    opponent_type: str
    type_source: str
    vod: str
    side: str
    call: str
    call_key: str
    call_class: str
    look: str
    look_key: str
    look_family: str
    down: str
    dist: str
    zone: str
    score: str
    time: str
    pressure: bool
    off_adj: bool
    def_adj: bool
    scrimmage: bool
    success: bool | None
    success_source: str
    conf_play: float | None
    conf_success: float | None

    @property
    def stratum(self) -> str:
        return f"{self.game}/{self.opponent_type}"

    @property
    def known(self) -> bool:
        return self.scrimmage and self.success is not None

    @property
    def clean(self) -> bool:
        return bool(self.call_key) and self.look_family != "none"


def snaps_from_rows(rows: list[dict[str, Any]]) -> list[Snap]:
    return [snap_from_row(row) for row in rows]


def snap_from_row(row: dict[str, Any]) -> Snap:
    game = _game(row.get("game"))
    opponent_type, type_source = _opponent_type(row)
    side = _side(row.get("streamer_side") or row.get("side"))
    conf_play, conf_success = _confidence(row)
    call_text = _call_text(row, side)
    if conf_play is not None and conf_play < CONF_MIN_NAMES:
        call_text = ""
    look_text = _look_text(row)
    look_conf = _float(row.get("conf_look") or row.get("conf_coverage"))
    if look_conf is not None and look_conf < CONF_MIN_NAMES:
        look_text = ""
    family = _look_family(look_text)
    success, success_source = _success(row, side, call_text)
    down_n = _int(row.get("down"))
    dist_n = _int(row.get("distance"))
    return Snap(
        game=game,
        opponent_type=opponent_type,
        type_source=type_source,
        vod=_vod(row),
        side=side,
        call=call_text,
        call_key=_norm(call_text),
        call_class=_call_class(call_text),
        look=look_text,
        look_key=_norm(look_text),
        look_family=family,
        down=str(down_n) if down_n in (1, 2, 3, 4) else "",
        dist=_dist(row, dist_n),
        zone=_zone(row),
        score=_score(row),
        time=_time(row),
        pressure=_pressure(look_text, family),
        off_adj=_flag(row, "off_adjustment", "off_adj") or _side_flag(row, side, "offense"),
        def_adj=_flag(row, "def_adjustment", "def_adj") or _side_flag(row, side, "defense"),
        scrimmage=_scrimmage(row, down_n, call_text),
        success=success,
        success_source=success_source,
        conf_play=conf_play,
        conf_success=conf_success,
    )


def feature_map(snap: Snap, *, situation_only: bool = False) -> dict[str, str]:
    """One level per active group. Missing names are omitted, not filled in."""
    feats: dict[str, str] = {
        "score": snap.score or "unk",
        "side": snap.side or "unknown",
        "time": snap.time or "unk",
        "zone": snap.zone or "unk",
    }
    if snap.dist:
        feats["dist"] = snap.dist
    if snap.down:
        feats["down"] = snap.down
    if snap.dist and snap.down:
        feats["down_x_dist"] = f"{snap.down}|{snap.dist}"
    if situation_only:
        return feats
    if snap.call_key:
        feats["call"] = snap.call_key
    feats["call_class"] = snap.call_class
    if snap.look_key:
        feats["look"] = snap.look_key
    feats["look_family"] = snap.look_family or "none"
    if snap.call_key:
        feats["call_x_family"] = f"{snap.call_key}|{snap.look_family or 'none'}"
    feats["class_x_family"] = f"{snap.call_class}|{snap.look_family or 'none'}"
    if snap.off_adj:
        feats["off_adj"] = "yes"
    if snap.def_adj:
        feats["def_adj"] = "yes"
    if snap.pressure:
        feats["pressure"] = "yes"
    return feats


def penalty_for(group: str, hp: tuple[float, float, float]) -> float:
    situation, entity, interaction = hp
    if group in SITUATION_GROUPS:
        return situation
    if group in INTERACTION_GROUPS:
        return interaction
    return entity


def _game(value: Any) -> str:
    text = _norm(value)
    if "cfb" in text or "college" in text:
        return "cfb27"
    if text in {"", "madden", "madden27", "m27"}:
        return "madden27"
    return text.replace(" ", "") or "madden27"


def _opponent_type(row: dict[str, Any]) -> tuple[str, str]:
    explicit = _text(row.get("opponent_type_source"))
    raw_source = _text(row.get("_type_source")) or "column"
    if explicit:
        source = explicit
    elif raw_source == "manifest":
        source = "scout_manifest"
    elif raw_source == "missing":
        source = "missing_scout_tag"
    elif raw_source == "column":
        source = "column"
    else:
        source = raw_source
    raw = _norm(row.get("opponent_type"))
    if raw in {"cpu", "ai", "computer"} or raw.startswith("cpu"):
        kind = "cpu"
    elif raw in {"human", "h2h", "user", "person"} or raw.startswith("human"):
        kind = "human"
    else:
        kind = "unknown"
        if not explicit and raw_source == "missing":
            source = "missing_scout_tag"
    return kind, source


def _success(row: dict[str, Any], side: str, call: str) -> tuple[bool | None, str]:
    if "success" in row:
        parsed = _parse_success_value(row.get("success"))
        return parsed, "label"
    graded = evaluate_snap(
        {
            "side": "offense" if side == "unknown" else side,
            "play": call or "?",
            "formation": row.get("formation") or "",
            "result": _result_text(row),
            "down": _int(row.get("down")),
            "distance": _int(row.get("distance")),
            "yardline": _int(row.get("yardline")),
            "coverage_seen": row.get("coverage_seen") or "",
            "concept_seen": row.get("concept_seen") or "",
            "notes": row.get("notes") or "",
            "quarter": row.get("quarter"),
        }
    )
    if graded is None:
        return None, "coach_rule"
    if side == "unknown":
        return None, "coach_rule"
    return bool(graded.success), "coach_rule"


def _parse_success_value(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "success", "y"}:
        return True
    if text in {"0", "false", "no", "fail", "failure", "n"}:
        return False
    return None


def _result_text(row: dict[str, Any]) -> str:
    result = _text(row.get("result"))
    if result:
        return result
    yards = _int(row.get("yards"))
    if yards is None:
        return ""
    if yards > 0:
        return f"+{yards}"
    if yards < 0:
        return str(yards)
    return "+0"


def _call_text(row: dict[str, Any], side: str) -> str:
    for key in ("play", "call", "our_call"):
        text = _text(row.get(key))
        if text and text not in {"?", "-"}:
            return text
    if side == "defense":
        return _text(row.get("coverage_seen"))
    return ""


def _look_text(row: dict[str, Any]) -> str:
    return _text(row.get("coverage_seen") or row.get("look") or row.get("coverage"))


def _call_class(call: str) -> str:
    if not call:
        return "none"
    text = call.casefold()
    if any(word in text for word in _RUN_WORDS):
        return "run"
    if any(word in text for word in _PASS_WORDS):
        return "pass"
    return "unknown"


def _look_family(look: str) -> str:
    c = _norm(look)
    if not c:
        return "none"
    if "cover 2 man" in c or "c2 man" in c or c == "2 man":
        return "cover_2_man"
    if "tampa" in c:
        return "tampa_2"
    if "cover 0" in c or "cover zero" in c:
        return "cover_0"
    if "cover 1" in c or c == "man" or c.endswith(" man"):
        if "cover 1" in c:
            return "cover_1"
        return "man_other"
    if "cover 2" in c:
        return "cover_2"
    if "cover 3" in c:
        return "cover_3"
    if "cover 4" in c or "quarters" in c:
        return "cover_4"
    if "cover 6" in c:
        return "cover_6"
    if "cover 9" in c:
        return "cover_9"
    if any(word in c for word in ("blitz", "fire", "pressure", " sim", "mug", "stunt")):
        return "pressure_other"
    return "other"


def _pressure(look: str, family: str) -> bool:
    if family == "pressure_other":
        return True
    c = _norm(look)
    return any(word in c for word in ("blitz", "fire", "pressure", "sim", "mug"))


def _dist(row: dict[str, Any], distance: int | None) -> str:
    named = _norm(row.get("dist") or row.get("distance_band"))
    if named in {"short", "medium", "long", "xlong"}:
        return named
    if distance is None:
        return ""
    if distance <= 3:
        return "short"
    if distance <= 6:
        return "medium"
    if distance <= 10:
        return "long"
    return "xlong"


def _zone(row: dict[str, Any]) -> str:
    named = _norm(row.get("zone") or row.get("field_zone"))
    aliases = {
        "opp": "opp",
        "own": "own",
        "own_deep": "own_deep",
        "deep": "own_deep",
        "red_zone": "red_zone",
        "redzone": "red_zone",
        "rz": "red_zone",
        "unk": "unk",
        "unknown": "unk",
    }
    if named in aliases:
        return aliases[named]
    yardline = _int(row.get("yardline"))
    if yardline is None or yardline < 0 or yardline > 100:
        return "unk"
    if yardline <= 20:
        return "own_deep"
    if yardline < 50:
        return "own"
    if yardline >= 80:
        return "red_zone"
    return "opp"


def _score(row: dict[str, Any]) -> str:
    named = _norm(row.get("score_state") or row.get("score"))
    if named in {"close", "not_close", "unk"}:
        return named
    us = _int(row.get("score_us"))
    them = _int(row.get("score_them"))
    if us is None or them is None:
        note = _text(row.get("notes"))
        match = _SCORE_NOTE.search(note)
        if not match:
            return "unk"
        us, them = int(match.group(1)), int(match.group(2))
    return "close" if abs(us - them) <= 8 else "not_close"


def _time(row: dict[str, Any]) -> str:
    named = _norm(row.get("time") or row.get("time_state"))
    if named in {"late_half", "normal", "unk"}:
        return named
    blob = " ".join(
        _text(part)
        for part in (row.get("situation_raw"), row.get("clock"), f"q{row.get('quarter') or ''}")
    )
    match = _CLOCK_RE.search(blob)
    if match:
        quarter = int(match.group(1))
        seconds = int(match.group(2)) * 60 + int(match.group(3))
        if quarter in (2, 4) and seconds <= 120:
            return "late_half"
        return "normal"
    quarter_match = _QUARTER_RE.search(blob)
    if quarter_match and int(quarter_match.group(1)) in (1, 3):
        return "normal"
    return "unk"


def _scrimmage(row: dict[str, Any], down: int | None, call: str) -> bool:
    flag = row.get("scrimmage")
    if isinstance(flag, bool):
        return flag
    if _text(flag).lower() in {"0", "false", "no"}:
        return False
    if _text(flag).lower() in {"1", "true", "yes"}:
        return True
    if _SPECIAL_RE.search(call):
        return False
    return down in (1, 2, 3, 4)


def _vod(row: dict[str, Any]) -> str:
    raw = _text(row.get("video_id") or row.get("vod") or row.get("session_id") or row.get("game_id"))
    if not raw:
        raw = _text(row.get("_source")) or "unknown"
    if raw.startswith("vod:"):
        raw = raw.split(":")[-1]
    if raw.isdigit():
        return "v" + raw
    return raw


def _confidence(row: dict[str, Any]) -> tuple[float | None, float | None]:
    play = _float(row.get("conf_play"))
    success = _float(row.get("conf_success"))
    blob = row.get("confidence")
    if isinstance(blob, dict):
        if play is None:
            play = _float(blob.get("play"))
        if success is None:
            success = _float(blob.get("success"))
            if success is None:
                success = _float(blob.get("result"))
    return play, success


def _flag(row: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        text = _text(row.get(key))
        if text and text.lower() not in {"0", "false", "no", "none"}:
            return True
    return False


def _side_flag(row: dict[str, Any], side: str, want: str) -> bool:
    if side != want:
        return False
    return _flag(row, "adjustment", "audible")


def _side(value: Any) -> str:
    text = _norm(value)
    if text.startswith("off"):
        return "offense"
    if text.startswith("def"):
        return "defense"
    if text in {"o", "offense"}:
        return "offense"
    if text in {"d", "defense"}:
        return "defense"
    if not text:
        return "unknown"
    if text[0] == "o":
        return "offense"
    if text[0] == "d":
        return "defense"
    return "unknown"


def _norm(value: Any) -> str:
    return " ".join(_text(value).casefold().split())


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
