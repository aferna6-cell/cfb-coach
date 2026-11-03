"""Team and score from the scorebug.

Do not drop a token because it contains ``(``. The record is painted in
front of the abbreviation, ``(0-0) ATL``, and that whole token is the team.
A MUT abbreviation is not on the NFL list; it still counts when the record
is attached or when the bar has exactly two abbreviations and two scores.
A third bare number (yard line, play clock) is not a score unless it sits
next to a team. Two scores are required. One number is not assigned to a side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vod_pilot.digits import normalize_digits, safe_int
from vod_pilot.tokens import Token, as_tok

# 32 clubs plus the abbreviations Madden still paints.
NFL_ABBREV = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB",
    "TEN", "WAS",
    "JAC", "WSH", "ARZ", "GBP", "KAN", "TAM", "NOR", "NWE", "SFO", "OAK",
    "SD",
}

# Short words that share the shape of an abbreviation and are not clubs.
_STOP = {
    "AND", "THE", "QTR", "YDS", "YRD", "OWN", "OPP", "BALL", "DOWN", "GOAL",
    "INCH", "PLAY", "CALL", "HOT", "DEF", "OFF", "NFL", "MUT", "FOR", "NOT",
    "OUT", "ALL", "TOP", "NEW", "RED", "BIG", "YARD", "TIME", "OVER", "UNDER",
    "ODD", "EVEN", "ZONE", "MAN", "GUN", "YDS", "SEC", "MIN", "MAX", "TD",
    "ST", "ND", "RD", "TH", "VS", "AT",
}

_RECORD_TEAM_SCORE = re.compile(
    r"^\(\s*(\d{1,2})\s*-\s*(\d{1,2})(?:\s*-\s*\d{1,2})?\s*\)\s*([A-Z]{2,4})\s+(\d{1,2})$"
)
_SCORE_RECORD_TEAM = re.compile(
    r"^(\d{1,2})\s+\(\s*(\d{1,2})\s*-\s*(\d{1,2})(?:\s*-\s*\d{1,2})?\s*\)\s*([A-Z]{2,4})$"
)
_RECORD_TEAM = re.compile(
    r"^\(\s*(\d{1,2})\s*-\s*(\d{1,2})(?:\s*-\s*\d{1,2})?\s*\)\s*([A-Z]{2,4})$"
)
_TEAM_SCORE = re.compile(r"^([A-Z]{2,4})\s+(\d{1,2})$")
_SCORE_TEAM = re.compile(r"^(\d{1,2})\s+([A-Z]{2,4})$")
_DOWN = re.compile(r"^[1-4](?:ST|ND|RD|TH)\d", re.I)


@dataclass(frozen=True)
class ScoreRead:
    team_left: str = ""
    team_right: str = ""
    score_left: int | None = None
    score_right: int | None = None
    record_left: str = ""
    record_right: str = ""
    confidence: float = 0.0

    @property
    def complete(self) -> bool:
        return bool(self.team_left and self.team_right and self.score_left is not None and self.score_right is not None)


@dataclass
class _Piece:
    kind: str  # team | score | both
    team: str = ""
    record: str = ""
    score: int | None = None
    x: float | None = None
    nfl: bool = False


def _record(a: str, b: str) -> str:
    left, right = safe_int(a), safe_int(b)
    if left is None or right is None:
        return ""
    if left > 20 or right > 20:
        return ""
    return f"{left}-{right}"


def _ok_team(name: str, *, from_record: bool) -> bool:
    if name in _STOP or len(name) < 2:
        return False
    if from_record or name in NFL_ABBREV:
        return True
    return name.isalpha() and 2 <= len(name) <= 4


def _piece_from_text(text: str, x: float | None) -> _Piece | None:
    """One scorebug piece. A leading ``(record)`` is the team, not noise."""
    raw = " ".join(normalize_digits(text).upper().replace("—", "-").split())
    if not raw or ":" in raw or _DOWN.match(raw.replace(" ", "")):
        return None
    match = _RECORD_TEAM_SCORE.match(raw)
    if match:
        rec = _record(match.group(1), match.group(2))
        team = match.group(3)
        score = safe_int(match.group(4))
        if rec and score is not None and _ok_team(team, from_record=True) and score <= 99:
            return _Piece("both", team, rec, score, x, team in NFL_ABBREV)
        return None
    match = _SCORE_RECORD_TEAM.match(raw)
    if match:
        score = safe_int(match.group(1))
        rec = _record(match.group(2), match.group(3))
        team = match.group(4)
        if rec and score is not None and _ok_team(team, from_record=True) and score <= 99:
            return _Piece("both", team, rec, score, x, team in NFL_ABBREV)
        return None
    match = _RECORD_TEAM.match(raw)
    if match:
        rec = _record(match.group(1), match.group(2))
        team = match.group(3)
        if rec and _ok_team(team, from_record=True):
            return _Piece("team", team, rec, None, x, team in NFL_ABBREV)
        return None
    match = _TEAM_SCORE.match(raw)
    if match and _ok_team(match.group(1), from_record=False) and match.group(1) in NFL_ABBREV:
        points = safe_int(match.group(2))
        if points is None:
            return None
        return _Piece("both", match.group(1), "", points, x, True)
    match = _SCORE_TEAM.match(raw)
    if match and _ok_team(match.group(2), from_record=False) and match.group(2) in NFL_ABBREV:
        points = safe_int(match.group(1))
        if points is None:
            return None
        return _Piece("both", match.group(2), "", points, x, True)
    if re.fullmatch(r"[A-Z]{2,4}", raw) and _ok_team(raw, from_record=False):
        # Bare MUT letters wait until the bar has a shape. NFL letters are teams.
        if raw in NFL_ABBREV:
            return _Piece("team", raw, "", None, x, True)
        return _Piece("maybe", raw, "", None, x, False)
    if re.fullmatch(r"\d{1,2}", raw):
        val = safe_int(raw)
        if val is not None and val <= 99:
            return _Piece("score", "", "", val, x, False)
    return None


def _adjacent_scores(pieces: list[_Piece]) -> list[_Piece]:
    """Bare numbers that touch a team. A yard line between them does not."""
    kept: list[_Piece] = []
    for i, piece in enumerate(pieces):
        if piece.kind != "score":
            continue
        prev_team = i > 0 and pieces[i - 1].kind in {"team", "both"}
        next_team = i + 1 < len(pieces) and pieces[i + 1].kind in {"team", "both"}
        if prev_team or next_team:
            kept.append(piece)
    return kept


def parse_scorebug(tokens: list[Token]) -> ScoreRead:
    pieces: list[_Piece] = []
    for item in tokens:
        tok = as_tok(item)
        piece = _piece_from_text(tok.text, tok.x)
        if piece is not None:
            pieces.append(piece)
    if not pieces:
        return ScoreRead()

    both = [p for p in pieces if p.kind == "both"]
    # The score is already on the team token. A later bare number is the spot.
    if len(both) == 2:
        ordered = _order_pair(both)
        return _finish(ordered[0], ordered[1], ordered[0], ordered[1])

    teams = [p for p in pieces if p.kind == "team"]
    maybes = [p for p in pieces if p.kind == "maybe"]
    if len(teams) < 2 and maybes:
        # Promote MUT letters only when that yields exactly two clubs.
        need = 2 - len(teams)
        if len(maybes) == need:
            for piece in maybes:
                piece.kind = "team"
            teams = [p for p in pieces if p.kind == "team"]
    if len(teams) != 2:
        return ScoreRead()

    scores = _adjacent_scores(pieces)
    if len(scores) != 2:
        return ScoreRead()
    if all(p.x is not None for p in teams + scores):
        teams = sorted(teams, key=lambda p: p.x or 0)
        scores = sorted(scores, key=lambda p: p.x or 0)
    return _finish(teams[0], teams[1], scores[0], scores[1])


def _order_pair(pieces: list[_Piece]) -> list[_Piece]:
    if all(p.x is not None for p in pieces):
        return sorted(pieces, key=lambda p: p.x or 0)
    return pieces


def _finish(left_team: _Piece, right_team: _Piece, left_score: _Piece, right_score: _Piece) -> ScoreRead:
    if left_team.team == right_team.team:
        return ScoreRead()
    if left_score.score is None or right_score.score is None:
        return ScoreRead()
    nfl = left_team.nfl and right_team.nfl
    recorded = bool(left_team.record or right_team.record)
    if nfl:
        conf = 0.86
    elif recorded:
        conf = 0.78
    else:
        conf = 0.62
    return ScoreRead(
        team_left=left_team.team,
        team_right=right_team.team,
        score_left=left_score.score,
        score_right=right_score.score,
        record_left=left_team.record,
        record_right=right_team.record,
        confidence=conf,
    )
