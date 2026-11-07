"""Smarter retrain v2 — situation-aware, leverage-weighted, capped, W/L-aware.

Why v2 (Aidan's 5 Ohio State games vs CPU, Sep 2026): the old retrain treated every
snap the same, ignored win/loss, counted an INT like an incompletion, and let
weights grow without bound (run_first 11.7, cluster_changeup 8.75). The live
caller also never read the learned weights. v2 fixes all of that:

* Every snap is bucketed by field zone (``cfb_coach.zones``): open / rz / gl.
  Zone-specific stores (family, per-play, play-vs-coverage) sit next to the
  general store. A play that fails at the 5 drops in rz+gl but barely moves
  the open-field numbers (see ``ZONE_SPILL``).
* Leverage: 3rd/4th down and goal-to-go snaps count 2-3x; turnovers 3.5x a normal
  failure, sacks 1.75x; a drive-ending red-zone failure another 1.5x.
* Success is situational (40% of the distance on 1st, 60% on 2nd, 100% on
  3rd/4th), so "+1 on 3rd & 5" is a failure, not a win.
* Weights are a shrunken, squashed mean — ``cap * tanh(mean)`` with pseudo-count
  ``PRIOR_OBS`` — so nothing snowballs and ordering is preserved.
* Game over uses result_wl + score margin (small, capped adjustments).
* A live us-them score logged on the snap (`score_us=` / `score_them=` in notes)
  tags the snap. Late, two-score snaps get a small leverage bump (capped).
  Snaps with no live score are unchanged. The final game score stays winner-first.
* ``rebuild_all`` recomputes every weight from scratch from all logged snaps +
  game results. It is deterministic and idempotent, so postgame simply rebuilds.

ALL tunables live in the "Constants" block below.
"""

from __future__ import annotations

import math
import re
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from cfb_coach.outcome import outcome_success, parse_outcome
from cfb_coach.zones import GOAL_LINE, OPEN, RED_ZONE, ZONE_LABELS, field_zone, is_goal_to_go, yards_to_goal

# ===========================================================================
# Constants — every learning tunable in ONE place (documented in README)
# ===========================================================================

RULES_VERSION = "v2-2026-09-27"
META_RULES_KEY = "learn_rules_version"
META_REBUILT_AT_KEY = "learn_rules_rebuilt_at"
META_BACKUP_KEY = "learn_rules_backup_path"

# --- Leverage: observation weight of a snap by situation ---
LEVERAGE_BY_DOWN = {1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0}
LEVERAGE_GOAL_TO_GO_FLOOR = 2.0  # any goal-to-go snap counts >= 2x
LEVERAGE_GOAL_TO_GO_MULT = 1.25  # 3rd & goal = 2.5x, 4th & goal = 3x (capped)
LEVERAGE_MAX = 3.0
# Late snaps that were logged with a live us-them score and a two-score margin.
SCORE_SENSITIVE_LEV = 1.1

# --- Outcome scores (offense-centric: normal success +1, normal failure -1) ---
SCORE_SUCCESS = 1.0
SCORE_EXPLOSIVE = 1.5  # gain >= EXPLOSIVE_YARDS
SCORE_TD = 2.0
SCORE_NEAR_MISS = -0.4  # 1st/2nd down positive gain, short of the success target
SCORE_FAIL = -1.0  # incomplete / stuffed / short on 3rd-4th
SCORE_SACK = -1.75  # moderate negative
SCORE_TURNOVER = -3.5  # INT / fumble lost: ~3.5x a normal failure
DRIVE_END_RZ_FAIL_MULT = 1.5  # drive-ending failure inside the 20: heavier still
OUTCOME_CLIP = 5.0  # |score| cap per snap before leverage
EXPLOSIVE_YARDS = 15

# Success target = share of the distance needed (standard success-rate rule)
SUCCESS_NEED = {1: 0.4, 2: 0.6, 3: 1.0, 4: 1.0}

# --- Zone spill: how much a snap counts in each store (by the snap's zone) ---
ZONE_SPILL: dict[str, dict[str, float]] = {
    OPEN: {"general": 1.0, OPEN: 1.0},
    RED_ZONE: {"general": 0.5, RED_ZONE: 1.0, GOAL_LINE: 0.25},
    GOAL_LINE: {"general": 0.35, GOAL_LINE: 1.0, RED_ZONE: 0.6},
}

# --- Game over: win/loss awareness (modest; margin factor 1.0-1.5) ---
LOSS_STALL_PENALTY = -0.75  # added to the play that ended a stalled drive in a loss
LOSS_STALL_RZ_MULT = 1.5  # ... more when that drive died inside the 20
WIN_SCORING_BOOST = 0.25  # added to every play of a TD drive in a win
MARGIN_CAP = 14
MARGIN_EXTRA = 0.5  # factor = 1 + MARGIN_EXTRA * min(margin, MARGIN_CAP) / MARGIN_CAP

# --- Shrinkage + saturation (diminishing returns; no snowballing) ---
PRIOR_OBS = 5.0  # pseudo-observations at 0 in every mean (2 lucky snaps != proven)
SQUASH_GAIN = 0.7  # weight = cap * tanh(SQUASH_GAIN * shrunken_mean)
CAP_FAMILY = 4.0
CAP_PLAY = 2.5
CAP_VS_LOOK = 1.5
CAP_ANTI_REPEAT = 1.5
ANTI_REPEAT_STEP = 0.02  # per over-repeat call (4th+ call of one play in a game)
ANTI_REPEAT_FREE_CALLS = 3
GLOBAL_SCALE = 0.35  # global bucket = 0.35 x weights over all opponents

# --- Recommendation blend (zone-specific + general) ---
REC_BLEND: dict[str, dict[str, float]] = {
    OPEN: {OPEN: 0.5, "general": 0.5},
    RED_ZONE: {RED_ZONE: 0.65, "general": 0.35},
    GOAL_LINE: {GOAL_LINE: 0.6, RED_ZONE: 0.25, "general": 0.15},
}
REC_PLAY_SHARE = 0.7  # per-play vs family inside one store
REC_FAMILY_SHARE = 0.3
REC_COVERAGE_LIVE = 0.4  # vs-look term when the look is live (pre-snap)
REC_COVERAGE_LAST = 0.15  # ... when it is only the previous snap's look
REC_UNTESTED_PENALTY = -0.08  # book play never logged in this zone: tiny caution
REC_META_PRIOR_OBS = 6.0  # meta prior fades as Aidan's own zone sample grows
REC_TEMPERATURE = 0.25  # softmax temperature for sampling a call (lower = follow the ranking harder)
REC_EXPLORE = 0.1  # share of probability spread uniformly (keeps testing the menu)

STORES = ("general", OPEN, RED_ZONE, GOAL_LINE)


def constants_table() -> list[tuple[str, str]]:
    """Human-readable constants (README / prep page)."""
    return [
        ("rules version", RULES_VERSION),
        ("leverage by down", "1st 1x, 2nd 1x, 3rd 2x, 4th 3x"),
        ("goal-to-go leverage", f">= {LEVERAGE_GOAL_TO_GO_FLOOR}x, x{LEVERAGE_GOAL_TO_GO_MULT} on 3rd/4th, cap {LEVERAGE_MAX}x"),
        ("success target", "40% of distance on 1st, 60% on 2nd, 100% on 3rd/4th"),
        ("outcome scores", f"success +{SCORE_SUCCESS}, explosive +{SCORE_EXPLOSIVE}, TD +{SCORE_TD}, near-miss {SCORE_NEAR_MISS}, fail {SCORE_FAIL}, sack {SCORE_SACK}, turnover {SCORE_TURNOVER}"),
        ("drive-ending red-zone failure", f"x{DRIVE_END_RZ_FAIL_MULT}"),
        ("zone spill", "open snap: general 1.0 | rz snap: rz 1.0, gl 0.25, general 0.5 | gl snap: gl 1.0, rz 0.6, general 0.35"),
        ("loss adjustment", f"{LOSS_STALL_PENALTY} on the play that ended a stalled drive (x{LOSS_STALL_RZ_MULT} in the red zone), x margin factor"),
        ("win adjustment", f"+{WIN_SCORING_BOOST} on each play of a TD drive, x margin factor"),
        ("margin factor", f"1 + {MARGIN_EXTRA} x min(margin, {MARGIN_CAP})/{MARGIN_CAP}"),
        ("caps", f"family ±{CAP_FAMILY}, play ±{CAP_PLAY}, vs-coverage ±{CAP_VS_LOOK}, anti-repeat ±{CAP_ANTI_REPEAT}"),
        ("diminishing returns", f"weight = cap x tanh({SQUASH_GAIN} x sum(w*score) / (sum(w) + {PRIOR_OBS}))"),
        ("recommendation blend", "open: 0.5 open + 0.5 general | red zone: 0.65 rz + 0.35 general | goal line: 0.6 gl + 0.25 rz + 0.15 general"),
        ("play vs family", f"{REC_PLAY_SHARE} play + {REC_FAMILY_SHARE} family; coverage term x{REC_COVERAGE_LIVE} live / x{REC_COVERAGE_LAST} last snap"),
        ("meta prior", f"fades as n_obs grows: prior x {REC_META_PRIOR_OBS}/({REC_META_PRIOR_OBS}+n_obs); untested play {REC_UNTESTED_PENALTY}"),
        ("call sampling", f"softmax T={REC_TEMPERATURE}, {REC_EXPLORE:.0%} spread evenly (keeps testing the menu)"),
        ("live score", f"notes score_us/score_them tag the snap; Q4+ and margin ≥ 8 → leverage x{SCORE_SENSITIVE_LEV} (capped)"),
    ]


# ===========================================================================
# Snap evaluation
# ===========================================================================


def _get(s: Any, key: str, default: Any = None) -> Any:
    if isinstance(s, dict):
        return s.get(key, default)
    try:
        return s[key]
    except (KeyError, IndexError, TypeError):
        return getattr(s, key, default)


def _int_or_none(v: Any) -> int | None:
    try:
        return None if v is None or v == "" else int(v)
    except (TypeError, ValueError):
        return None


_RUN_WORDS = (
    "inside zone", "hb base", "counter", "duo", "dive", "outside zone", "stretch",
    "power", "draw", "slam", "wham", "trap", "zone split", "toss", "qb sweep",
    "qb blast", "iso", "zone wk",
)


def play_family(formation: str | None, play: str | None) -> str | None:
    """v2 family mapper (keys compatible with baseline emphasis_keys)."""
    if not play:
        return None
    p = play.lower()
    f = (formation or "").lower()
    if "cluster" in f:
        return "cluster_changeup"
    if "empty" in f or "quads" in f:
        return "empty_stress"
    if "deuce" in f:
        return "deuce_bully"
    if "rpo" in p:
        return "rpo_alert"
    if any(w in p for w in _RUN_WORDS):
        return "run_first"
    if "whip" in p:
        return "whip_hot"
    if "mesh" in p or "drive" in p:
        return "mesh_family"
    if "spot" in p:
        return "spot_family"
    if "flood" in p or "cross post" in p or "vertical" in p or "dig" in p:
        return "flood_family"
    return None


@dataclass
class SnapEval:
    snap_id: int
    session_id: str | None
    opponent_id: str
    side: str
    down: int | None
    distance: int | None
    yardline: int | None
    ytg: int | None
    zone: str
    goal_to_go: bool
    formation: str
    play: str
    look: str
    family: str | None
    kind: str
    yards: int | None
    success: bool
    base_score: float
    leverage: float
    score: float = 0.0  # final (after drive/W-L adjustments, clipped)
    tags: set[str] = field(default_factory=set)
    drive_id: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        dd = f"{self.down}&{self.distance}" if self.down and self.distance is not None else "?&?"
        spot = "?" if self.ytg is None else (f"opp {self.ytg}" if self.ytg <= 50 else f"own {100 - self.ytg}")
        return f"#{self.snap_id} {dd} @{spot} {self.formation} — {self.play}: {self.kind}" + (
            f" {self.yards:+d}" if self.yards is not None and self.kind in ("gain", "loss") else ""
        )


def leverage_for(down: int | None, goal_to_go: bool) -> float:
    lev = LEVERAGE_BY_DOWN.get(down or 1, 1.0)
    if goal_to_go:
        lev = max(LEVERAGE_GOAL_TO_GO_FLOOR, lev * LEVERAGE_GOAL_TO_GO_MULT)
    return min(LEVERAGE_MAX, lev)


def _success_target(down: int | None, distance: int | None, ytg: int | None) -> float | None:
    if distance is None or down is None:
        return None
    need = SUCCESS_NEED.get(down, 1.0) * float(distance)
    if ytg is not None:
        need = min(need, float(ytg))
    return max(need, 0.5)


def evaluate_snap(s: Any) -> SnapEval | None:
    """Grade one logged snap. Returns None when the result is unknown."""
    side = (_get(s, "side") or "offense").strip().lower()
    side = "defense" if side.startswith("d") else "offense"
    result = _get(s, "result")
    parsed = parse_outcome(result)
    down = _int_or_none(_get(s, "down"))
    distance = _int_or_none(_get(s, "distance"))
    yardline = _int_or_none(_get(s, "yardline"))
    ytg = yards_to_goal(yardline)
    gtg = is_goal_to_go(yardline, distance)
    zone = field_zone(yardline, distance)
    form = (_get(s, "formation") or "?").strip() or "?"
    play = (_get(s, "play") or "?").strip() or "?"
    look = (
        (_get(s, "coverage_seen") or "").strip()
        or (_get(s, "concept_seen") or "").strip()
        or "unknown"
    )
    kind = parsed.kind
    yards = parsed.yards
    tags: set[str] = set()

    if kind == "unknown":
        ok = outcome_success(result, "offense")
        if ok is None:
            return None
        success = bool(ok)
        base = SCORE_SUCCESS if success else SCORE_FAIL
    elif kind == "td":
        success, base = True, SCORE_TD
        tags.add("td")
    elif kind in ("int", "fumble"):
        success, base = False, SCORE_TURNOVER
        tags.add("turnover")
    elif kind == "sack":
        success, base = False, SCORE_SACK
        tags.add("sack")
    elif kind in ("incomplete", "stop"):
        success, base = False, SCORE_FAIL
    elif kind == "convert":
        success, base = True, SCORE_SUCCESS
    else:  # gain / loss with yards
        y = yards or 0
        need = _success_target(down, distance, ytg)
        if need is None:
            success = y > 0
            base = SCORE_SUCCESS if success else SCORE_FAIL
        elif y >= need:
            success = True
            base = SCORE_EXPLOSIVE if y >= EXPLOSIVE_YARDS else SCORE_SUCCESS
        else:
            success = False
            if (down or 1) <= 2 and y > 0 and y >= 0.5 * need:
                base = SCORE_NEAR_MISS
            else:
                base = SCORE_FAIL
        if y >= EXPLOSIVE_YARDS:
            tags.add("explosive")

    if side == "defense":
        # Our D: offense-centric success flips; forced turnovers/sacks are big wins
        success = not success
        base = -base
        if "turnover" in tags:
            base = 2.5
        elif "sack" in tags:
            base = 1.5

    ev = SnapEval(
        snap_id=int(_get(s, "id") or 0),
        session_id=_get(s, "session_id"),
        opponent_id=str(_get(s, "opponent_id") or ""),
        side=side,
        down=down,
        distance=distance,
        yardline=yardline,
        ytg=ytg,
        zone=zone,
        goal_to_go=gtg,
        formation=form,
        play=play,
        look=look,
        family=play_family(form, play),
        kind=kind,
        yards=yards,
        success=success,
        base_score=base,
        leverage=leverage_for(down, gtg),
        tags=tags,
    )
    _attach_logged_score(ev, s)
    return ev


def _attach_logged_score(ev: SnapEval, snap: Any) -> None:
    """Tag snaps that were logged with a live us-them score. Numeric grade stays put
    except a small capped leverage bump on late two-score snaps."""
    from cfb_coach.game_score import parse_snap_score_note

    parsed = parse_snap_score_note(_get(snap, "notes"))
    if parsed is None:
        return
    us, them = parsed
    ev.tags.add("live_score")
    ev.notes.append(f"live score {us}-{them}")
    q = _int_or_none(_get(snap, "quarter"))
    if q is not None and q >= 4 and abs(us - them) >= 8:
        ev.tags.add("score_sensitive")
        ev.leverage = min(LEVERAGE_MAX, round(ev.leverage * SCORE_SENSITIVE_LEV, 3))


# ===========================================================================
# Drives + game results
# ===========================================================================


def _made_first_down(e: SnapEval) -> bool:
    if e.kind in ("convert", "td"):
        return True
    if e.yards is not None and e.distance is not None:
        return e.yards >= e.distance
    return False


def _new_drive(prev: SnapEval, cur: SnapEval) -> bool:
    if prev.side != cur.side:
        return True
    if prev.kind in ("td", "int", "fumble"):
        return True
    if prev.yardline is not None and cur.yardline is not None:
        expected = prev.yardline + (prev.yards or 0)
        if cur.yardline < expected - 15:
            return True
    if prev.down is None or cur.down is None:
        return False
    if cur.down == 1:
        if prev.down == 4 and not _made_first_down(prev):
            return True  # turnover on downs
        return not _made_first_down(prev)  # punt / FG / unlogged turnover
    if cur.down > prev.down:
        return False
    # Same or lower down number without a first down in between -> new drive
    return not _made_first_down(prev)


def assign_drives(evals: list[SnapEval]) -> list[dict[str, Any]]:
    """Group snaps into drives (per session, id order) and tag drive ends.

    Tags added: drive_end, stalled_end, drive_end_rz_fail, scoring_drive.
    Returns drive summaries.
    """
    drives: list[dict[str, Any]] = []
    by_session: dict[str, list[SnapEval]] = defaultdict(list)
    for e in evals:
        by_session[f"{e.opponent_id}|{e.session_id or '-'}"].append(e)
    did = 0
    for key in sorted(by_session, key=lambda k: min(x.snap_id for x in by_session[k])):
        seq = sorted(by_session[key], key=lambda x: x.snap_id)
        cur: list[SnapEval] = []
        groups: list[list[SnapEval]] = []
        for e in seq:
            if cur and _new_drive(cur[-1], e):
                groups.append(cur)
                cur = []
            cur.append(e)
        if cur:
            groups.append(cur)
        for gi, g in enumerate(groups):
            did += 1
            last = g[-1]
            td = last.kind == "td" or any(x.kind == "td" for x in g)
            reached_rz = any(x.zone in (RED_ZONE, GOAL_LINE) for x in g)
            session_end = gi == len(groups) - 1
            for x in g:
                x.drive_id = did
                if td:
                    x.tags.add("scoring_drive")
            last.tags.add("drive_end")
            stalled = (not td) and (not last.success)
            if stalled:
                last.tags.add("stalled_end")
                if last.zone in (RED_ZONE, GOAL_LINE):
                    last.tags.add("drive_end_rz_fail")
            drives.append(
                {
                    "drive_id": did,
                    "session_id": last.session_id,
                    "opponent_id": last.opponent_id,
                    "side": last.side,
                    "snaps": [x.snap_id for x in g],
                    "result": "td" if td else ("turnover" if last.kind in ("int", "fumble") else ("stalled" if stalled else ("clock/end" if session_end else "ended"))),
                    "reached_rz": reached_rz,
                    "end_snap": last.snap_id,
                    "end_zone": last.zone,
                }
            )
    return drives


def parse_score(score: str | None, result_wl: str | None) -> dict[str, Any] | None:
    """Parse a stored score. Aidan enters it winner-first ("14-7" in a loss means
    he scored 7). We only rely on result_wl + |a-b|, so either order works:
    our points = max in a win, min in a loss."""
    if not score:
        return None
    nums = [int(x) for x in re.findall(r"\d+", str(score))[:2]]
    if len(nums) < 2:
        return None
    a, b = nums
    wl = (result_wl or "").strip().lower()
    if wl in ("w", "win"):
        ours, theirs = max(a, b), min(a, b)
    elif wl in ("l", "loss"):
        ours, theirs = min(a, b), max(a, b)
    else:
        ours, theirs = a, b
    return {"ours": ours, "theirs": theirs, "margin": abs(a - b), "raw": str(score)}


def margin_factor(margin: int | None) -> float:
    m = max(0, min(int(margin or 0), MARGIN_CAP))
    return 1.0 + MARGIN_EXTRA * m / MARGIN_CAP


def load_game_results(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(game_sessions)").fetchall()}
        if "result_wl" not in cols:
            return out
        for r in conn.execute(
            "SELECT session_id, opponent_id, result_wl, score, started_ts FROM game_sessions"
        ).fetchall():
            wl = (r[2] or "").strip().lower()
            if wl in ("w",):
                wl = "win"
            if wl in ("l",):
                wl = "loss"
            out[str(r[0])] = {
                "session_id": str(r[0]),
                "opponent_id": r[1],
                "result_wl": wl or None,
                "score": r[3],
                "started_ts": r[4],
                "parsed": parse_score(r[3], wl),
            }
    except sqlite3.Error:
        pass
    return out


def apply_drive_and_result_adjustments(
    evals: list[SnapEval], results: dict[str, dict[str, Any]]
) -> None:
    """Final per-snap score = base (+drive-end RZ mult) (+W/L adjust), clipped."""
    for e in evals:
        score = e.base_score
        if "drive_end_rz_fail" in e.tags and e.side == "offense":
            score *= DRIVE_END_RZ_FAIL_MULT
            e.notes.append(f"drive-ending red-zone failure x{DRIVE_END_RZ_FAIL_MULT}")
        res = results.get(str(e.session_id)) if e.session_id else None
        if res and e.side == "offense":
            wl = res.get("result_wl")
            mf = margin_factor((res.get("parsed") or {}).get("margin"))
            if wl == "loss" and "stalled_end" in e.tags:
                adj = LOSS_STALL_PENALTY * mf
                if e.zone in (RED_ZONE, GOAL_LINE):
                    adj *= LOSS_STALL_RZ_MULT
                score += adj
                e.tags.add("loss_adjusted")
                e.notes.append(f"loss: stalled-drive end {adj:+.2f}")
            elif wl == "win" and "scoring_drive" in e.tags:
                adj = WIN_SCORING_BOOST * mf
                score += adj
                e.tags.add("win_adjusted")
                e.notes.append(f"win: scoring drive {adj:+.2f}")
        e.score = max(-OUTCOME_CLIP, min(OUTCOME_CLIP, score))


# ===========================================================================
# Aggregation -> weights
# ===========================================================================


def _k_family(fam: str) -> str:
    return fam


def _k_play(form: str, play: str) -> str:
    return f"play::{form}::{play}"


def _k_vs(form: str, play: str, look: str) -> str:
    return f"vs_look::{form}::{play}::{look}"


def zone_key(zone: str, base_key: str) -> str:
    """zone::rz::run_first | zone::gl::play::F::P | zone::rz::vs_look::F::P::C"""
    if "::" not in base_key:
        return f"zone::{zone}::fam::{base_key}"
    return f"zone::{zone}::{base_key}"


def cap_for_key(key: str) -> float:
    k = key.split("zone::", 1)[-1]
    if k.startswith(("open::", "rz::", "gl::")):
        k = k.split("::", 1)[1]
    if k.startswith("fam::"):
        return CAP_FAMILY
    if k.startswith("play::"):
        return CAP_PLAY
    if k.startswith("vs_look::"):
        return CAP_VS_LOOK
    if k == "anti_repeat_penalty":
        return CAP_ANTI_REPEAT
    return CAP_FAMILY


def squash(mean: float, cap: float) -> float:
    return cap * math.tanh(SQUASH_GAIN * mean)


@dataclass
class _Acc:
    num: float = 0.0
    den: float = 0.0
    n: int = 0

    def add(self, w: float, score: float) -> None:
        self.num += w * score
        self.den += w
        self.n += 1

    @property
    def mean(self) -> float:
        return self.num / (self.den + PRIOR_OBS)


def _snap_keys(e: SnapEval) -> list[tuple[str, str, float]]:
    """(store, base_key, spill) for one snap."""
    out: list[tuple[str, str, float]] = []
    spill = ZONE_SPILL.get(e.zone, ZONE_SPILL[OPEN])
    bases: list[str] = []
    if e.family:
        bases.append(_k_family(e.family))
    if e.play and e.play != "?":
        bases.append(_k_play(e.formation, e.play))
        bases.append(_k_vs(e.formation, e.play, e.look))
    for store, sp in spill.items():
        for b in bases:
            out.append((store, b, sp))
    return out


def aggregate(evals: Iterable[SnapEval]) -> dict[tuple[str, str], _Acc]:
    """(side, stored_key) -> accumulator. stored_key already zone-prefixed."""
    acc: dict[tuple[str, str], _Acc] = defaultdict(_Acc)
    for e in evals:
        for store, base, sp in _snap_keys(e):
            key = base if store == "general" else zone_key(store, base)
            acc[(e.side, key)].add(e.leverage * sp, e.score)
    return acc


def anti_repeat_raw(evals: Iterable[SnapEval]) -> float:
    per_game: dict[tuple[str, str], int] = defaultdict(int)
    for e in evals:
        if e.side != "offense":
            continue
        per_game[(f"{e.opponent_id}|{e.session_id or '-'}", f"{e.formation}::{e.play}")] += 1
    excess = sum(max(0, n - ANTI_REPEAT_FREE_CALLS) for n in per_game.values())
    return -ANTI_REPEAT_STEP * excess


def weights_from_acc(acc: dict[tuple[str, str], _Acc], *, scale: float = 1.0) -> dict[tuple[str, str], dict[str, float]]:
    out: dict[tuple[str, str], dict[str, float]] = {}
    for (side, key), a in acc.items():
        w = squash(a.mean, cap_for_key(key)) * scale
        out[(side, key)] = {"weight": w, "mean": a.mean, "n_obs": a.den, "n": a.n}
    return out


# ===========================================================================
# Rebuild (from scratch) + persistence
# ===========================================================================


def _ensure_stats_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learn_stats (
            opponent_id TEXT NOT NULL,
            side TEXT NOT NULL,
            key TEXT NOT NULL,
            n INTEGER NOT NULL DEFAULT 0,
            n_obs REAL NOT NULL DEFAULT 0,
            mean REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (opponent_id, side, key)
        )
        """
    )


def load_evals(conn: sqlite3.Connection, *, opponent_id: str | None = None) -> list[SnapEval]:
    conn.row_factory = sqlite3.Row
    cols = {r[1] for r in conn.execute("PRAGMA table_info(snaps)").fetchall()}
    sess = "session_id" if "session_id" in cols else "NULL AS session_id"
    notes = "notes" if "notes" in cols else "NULL AS notes"
    quarter = "quarter" if "quarter" in cols else "NULL AS quarter"
    sql = (
        f"SELECT id, opponent_id, side, down, distance, yardline, formation, play, macro, "
        f"result, coverage_seen, concept_seen, {sess}, {notes}, {quarter} FROM snaps"
    )
    params: tuple = ()
    if opponent_id:
        sql += " WHERE opponent_id = ?"
        params = (opponent_id,)
    sql += " ORDER BY id ASC"
    evals: list[SnapEval] = []
    for r in conn.execute(sql, params).fetchall():
        e = evaluate_snap(dict(r))
        if e is not None:
            # Drive/W-L notes are added later. Keep the live-score note from the snap.
            e.notes = [n for n in e.notes if n.startswith("live score")]
            evals.append(e)
    return evals


def compute_all(conn: sqlite3.Connection) -> dict[str, Any]:
    """Pure compute (no writes): evals, drives, per-bucket weights, macros."""
    results = load_game_results(conn)
    evals = load_evals(conn)
    drives = assign_drives(evals)
    apply_drive_and_result_adjustments(evals, results)
    macro_rows = {}
    try:
        for r in conn.execute("SELECT id, macro FROM snaps").fetchall():
            macro_rows[int(r[0])] = r[1]
    except sqlite3.Error:
        pass

    buckets: dict[str, dict[tuple[str, str], dict[str, float]]] = {}
    macros: dict[str, dict[str, float]] = defaultdict(dict)
    by_opp: dict[str, list[SnapEval]] = defaultdict(list)
    for e in evals:
        by_opp[e.opponent_id].append(e)

    def _macro_weights(es: list[SnapEval], scale: float) -> dict[str, float]:
        acc: dict[str, _Acc] = defaultdict(_Acc)
        for e in es:
            # D snaps: the armed macro. O snaps (v1.15): the Active-8 custom adjustment the
            # live caller suggested with the play — same capped squash as every macro weight.
            if e.side not in ("defense", "offense"):
                continue
            m = (macro_rows.get(e.snap_id) or "").split(" [", 1)[0].strip().upper()
            if not m or m in ("NONE", "NULL"):
                continue
            acc[m].add(e.leverage, e.score)
        return {m: squash(a.mean, CAP_FAMILY) * scale for m, a in acc.items()}

    for opp, es in by_opp.items():
        w = weights_from_acc(aggregate(es))
        ar = anti_repeat_raw(es)
        if ar:
            w[("offense", "anti_repeat_penalty")] = {
                "weight": CAP_ANTI_REPEAT * math.tanh(ar / CAP_ANTI_REPEAT), "mean": ar, "n_obs": 0.0, "n": 0,
            }
        buckets[opp] = w
        macros[opp] = _macro_weights(es, 1.0)
    if evals:
        gw = weights_from_acc(aggregate(evals), scale=GLOBAL_SCALE)
        ar = anti_repeat_raw(evals)
        if ar:
            gw[("offense", "anti_repeat_penalty")] = {
                "weight": GLOBAL_SCALE * CAP_ANTI_REPEAT * math.tanh(ar / CAP_ANTI_REPEAT), "mean": ar, "n_obs": 0.0, "n": 0,
            }
        buckets.setdefault("global", {})
        # If an opponent is literally named "global" (never), merge; normal case = assign
        buckets["global"] = gw
        macros["global"] = _macro_weights(evals, GLOBAL_SCALE)
    return {
        "evals": evals,
        "drives": drives,
        "results": results,
        "buckets": buckets,
        "macros": dict(macros),
    }


def snapshot_weights(conn: sqlite3.Connection) -> dict[tuple[str, str, str], float]:
    try:
        return {
            (r[0], r[1], r[2]): float(r[3])
            for r in conn.execute("SELECT opponent_id, side, key, weight FROM gameplan_weights").fetchall()
        }
    except sqlite3.Error:
        return {}


def write_weights(conn: sqlite3.Connection, computed: dict[str, Any]) -> int:
    _ensure_stats_table(conn)
    n = 0
    with conn:
        conn.execute("DELETE FROM gameplan_weights")
        conn.execute("DELETE FROM learn_stats")
        for opp, w in computed["buckets"].items():
            for (side, key), d in w.items():
                if abs(d["weight"]) < 0.005:
                    continue
                conn.execute(
                    "INSERT OR REPLACE INTO gameplan_weights (opponent_id, side, key, weight) VALUES (?, ?, ?, ?)",
                    (opp, side, key, round(d["weight"], 4)),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO learn_stats (opponent_id, side, key, n, n_obs, mean) VALUES (?, ?, ?, ?, ?, ?)",
                    (opp, side, key, int(d["n"]), float(d["n_obs"]), float(d["mean"])),
                )
                n += 1
        conn.execute("UPDATE macro_weights SET weight = 0.0")
        for opp, mw in computed["macros"].items():
            for m, wv in mw.items():
                conn.execute(
                    """
                    INSERT INTO macro_weights (opponent_id, macro, weight, armed) VALUES (?, ?, ?, 0)
                    ON CONFLICT(opponent_id, macro) DO UPDATE SET weight = excluded.weight
                    """,
                    (opp, m, round(wv, 4)),
                )
    return n


def rebuild_all(db: Any, *, mark_version: bool = True) -> dict[str, Any]:
    """Recompute ALL learned weights from every logged snap + game result.

    Never deletes snaps / game_sessions. Idempotent.
    """
    conn = db.conn
    before = snapshot_weights(conn)
    computed = compute_all(conn)
    n = write_weights(conn, computed)
    after = snapshot_weights(conn)
    if mark_version:
        db.set_meta(META_RULES_KEY, RULES_VERSION)
        db.set_meta(META_REBUILT_AT_KEY, datetime.now().astimezone().isoformat(timespec="seconds"))
    computed.update({"before": before, "after": after, "rows_written": n})
    return computed


def weight_diff(
    before: dict[tuple[str, str, str], float],
    after: dict[tuple[str, str, str], float],
    *,
    opponent_id: str | None = None,
    side: str | None = "offense",
) -> list[tuple[str, float, float, float]]:
    keys = set(before) | set(after)
    out = []
    for k in keys:
        if opponent_id and k[0] != opponent_id:
            continue
        if side and k[1] != side:
            continue
        b, a = before.get(k, 0.0), after.get(k, 0.0)
        if abs(a - b) >= 0.01:
            out.append((k[2], b, a, a - b))
    out.sort(key=lambda t: -abs(t[3]))
    return out


# ===========================================================================
# One-time automatic rebuild (with backup)
# ===========================================================================


def backup_db_file(db: Any, *, when: datetime | None = None) -> Path | None:
    """Timestamped copy next to the DB (sqlite backup API = consistent copy)."""
    path = Path(getattr(db, "path", "") or "")
    if not path or not path.exists():
        return None
    stamp = (when or datetime.now()).strftime("%Y%m%d-%H%M%S")
    dest = path.with_name(f"{path.name}.bak-{stamp}")
    i = 1
    while dest.exists():
        dest = path.with_name(f"{path.name}.bak-{stamp}-{i}")
        i += 1
    try:
        target = sqlite3.connect(str(dest))
        try:
            db.conn.backup(target)
        finally:
            target.close()
    except sqlite3.Error:
        shutil.copy2(path, dest)
    return dest


def needs_rebuild(db: Any) -> bool:
    return (db.get_meta(META_RULES_KEY) or "") != RULES_VERSION


def ensure_rules_current(db: Any, *, printer=print, backup: bool = True) -> dict[str, Any] | None:
    """Run once per rules version on the next prep/play. Returns rebuild result or None."""
    if not needs_rebuild(db):
        return None
    n_snaps = int(db.conn.execute("SELECT COUNT(*) FROM snaps").fetchone()[0])
    if n_snaps == 0:
        # Fresh DB: nothing to back up or re-score — just stamp the rules version
        res = rebuild_all(db)
        res["backup_path"] = None
        return res
    bpath = backup_db_file(db) if backup else None
    res = rebuild_all(db)
    if bpath:
        db.set_meta(META_BACKUP_KEY, str(bpath))
    n_games = len([r for r in res["results"].values() if r.get("result_wl")])
    if printer:
        printer(
            f"Retrain rules updated to {RULES_VERSION}: rebuilt weights from {n_snaps} logged snaps"
            f" / {n_games} game results"
            + (f" (backup: {bpath})" if bpath else "")
        )
    res["backup_path"] = str(bpath) if bpath else None
    return res


# ===========================================================================
# Reading weights for recommendations
# ===========================================================================


def merged_macro_weights(db: Any, opponent_id: str) -> dict[str, float]:
    """Macro weights for one opponent, with the global bucket filling the gaps.

    Same rule as ``LearnedWeights.load``: this opponent's own number wins, so a
    James weight is not added on top of the global copy of those same snaps.
    A weight of 0 is "no opinion" and does not block the fill. CPU, and any
    other opponent with no weight for that macro, inherit the global bucket
    (``GLOBAL_SCALE`` of every logged snap). A human with fewer than 3 snaps
    also borrows a 0.25-diluted CPU prior before that fill.
    """
    if db is None:
        return {}

    def _load(bucket: str) -> dict[str, float]:
        out: dict[str, float] = {}
        try:
            rows = db.get_macro_weights(bucket)
        except Exception:  # noqa: BLE001
            return out
        for r in rows:
            w = float(r["weight"] or 0.0)
            if w == 0.0:
                continue
            out[str(r["macro"]).upper()] = w
        return out

    layers = [_load(opponent_id)]
    if opponent_id != "cpu":
        try:
            n_opp = int(db.count_snaps(opponent_id))
        except Exception:  # noqa: BLE001
            n_opp = 0
        if n_opp < 3:
            layers.append({k: v * 0.25 for k, v in _load("cpu").items()})
    layers.append(_load("global"))
    merged: dict[str, float] = {}
    for layer in layers:
        for key, weight in layer.items():
            merged.setdefault(key, weight)
    return merged


class LearnedWeights:
    """Merged learned weights for one opponent (global + cpu/opponent), clamped."""

    def __init__(self, weights: dict[str, float], stats: dict[str, dict[str, float]] | None = None) -> None:
        self.w = weights
        self.stats = stats or {}

    @classmethod
    def empty(cls) -> "LearnedWeights":
        return cls({}, {})

    @classmethod
    def load(cls, db: Any, opponent_id: str, side: str = "offense") -> "LearnedWeights":
        """Opponent-specific weights first; the global bucket only fills gaps.

        (Adding global on top would double-count CPU games, since the global
        bucket is built from the same snaps.) Human opponents with < 3 snaps also
        borrow a diluted CPU prior, mirroring ``gameplan.get_opponent_overlay``.
        """
        if db is None:
            return cls.empty()
        conn = db.conn
        _ensure_stats_table(conn)

        def _load(bucket: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
            w: dict[str, float] = {}
            st: dict[str, dict[str, float]] = {}
            try:
                for r in conn.execute(
                    "SELECT key, weight FROM gameplan_weights WHERE opponent_id = ? AND side = ?", (bucket, side)
                ).fetchall():
                    cap = cap_for_key(r[0])
                    w[r[0]] = max(-cap, min(cap, float(r[1])))
                for r in conn.execute(
                    "SELECT key, n, n_obs, mean FROM learn_stats WHERE opponent_id = ? AND side = ?", (bucket, side)
                ).fetchall():
                    st[r[0]] = {"n": int(r[1]), "n_obs": float(r[2]), "mean": float(r[3])}
            except sqlite3.Error:
                pass
            return w, st

        layers = [_load(opponent_id)]
        if opponent_id != "cpu":
            try:
                n_opp = int(conn.execute("SELECT COUNT(*) FROM snaps WHERE opponent_id = ?", (opponent_id,)).fetchone()[0])
            except sqlite3.Error:
                n_opp = 0
            if n_opp < 3:
                cw, cst = _load("cpu")
                layers.append(({k: v * 0.25 for k, v in cw.items()}, {k: {**d, "n_obs": d["n_obs"] * 0.25} for k, d in cst.items()}))
        layers.append(_load("global"))
        w: dict[str, float] = {}
        stats: dict[str, dict[str, float]] = {}
        for lw_, lst in layers:
            for k, v in lw_.items():
                w.setdefault(k, v)
            for k, d in lst.items():
                stats.setdefault(k, d)
        return cls(w, stats)

    def get(self, key: str) -> float:
        return self.w.get(key, 0.0)

    def norm(self, key: str) -> float:
        return self.get(key) / cap_for_key(key)

    def n_obs(self, key: str) -> float:
        return float((self.stats.get(key) or {}).get("n_obs", 0.0))

    def n(self, key: str) -> int:
        return int((self.stats.get(key) or {}).get("n", 0))

    def _store_key(self, store: str, base: str) -> str:
        return base if store == "general" else zone_key(store, base)

    def learned_score(
        self,
        zone: str,
        formation: str,
        play: str,
        *,
        coverage: str | None = None,
        coverage_source: str = "none",
    ) -> dict[str, float]:
        blend = REC_BLEND.get(zone, REC_BLEND[OPEN])
        fam = play_family(formation, play)
        pk = _k_play(formation, play)
        total = 0.0
        parts: dict[str, float] = {}
        for store, share in blend.items():
            p = self.norm(self._store_key(store, pk))
            f = self.norm(self._store_key(store, fam)) if fam else 0.0
            v = share * (REC_PLAY_SHARE * p + REC_FAMILY_SHARE * f)
            parts[f"{store}"] = v
            total += v
        cov_term = 0.0
        if coverage and coverage_source in ("live", "last"):
            mult = REC_COVERAGE_LIVE if coverage_source == "live" else REC_COVERAGE_LAST
            vk = _k_vs(formation, play, coverage)
            cov_term = mult * sum(share * self.norm(self._store_key(store, vk)) for store, share in blend.items())
            total += cov_term
        parts["coverage"] = cov_term
        zkey = pk if zone == OPEN else zone_key(zone, pk)
        parts["n_zone"] = float(self.n(zkey))
        parts["n_obs_zone"] = self.n_obs(zkey)
        parts["learned"] = total
        return parts


# ===========================================================================
# Postgame report
# ===========================================================================


def _fmt_key(key: str) -> str:
    k = key
    zone = "general"
    if k.startswith("zone::"):
        _, zone, rest = k.split("::", 2)
        k = rest
    if k.startswith("fam::"):
        k = "family " + k[5:]
    elif k.startswith("play::"):
        k = k[6:].replace("::", " — ")
    elif k.startswith("vs_look::"):
        f, p, c = k[9:].split("::", 2)
        k = f"{f} — {p} vs {c}"
    elif "::" not in k:
        k = "family " + k
    return f"[{ZONE_LABELS.get(zone, zone) if zone != 'general' else 'general'}] {k}"


def game_findings(evals: list[SnapEval], drives: list[dict[str, Any]], session_id: str | None) -> dict[str, Any]:
    es = [e for e in evals if (session_id is None or e.session_id == session_id) and e.side == "offense"]
    ds = [d for d in drives if (session_id is None or d["session_id"] == session_id) and d["side"] == "offense"]
    rz_trips = [d for d in ds if d["reached_rz"]]
    rz_tds = [d for d in rz_trips if d["result"] == "td"]
    rz = [e for e in es if e.zone in (RED_ZONE, GOAL_LINE)]
    gl = [e for e in es if e.zone == GOAL_LINE]
    return {
        "snaps": len(es),
        "rz_trips": len(rz_trips),
        "rz_tds": len(rz_tds),
        "rz_snaps": len(rz),
        "rz_success": sum(1 for e in rz if e.success),
        "gl_snaps": len(gl),
        "gl_success": sum(1 for e in gl if e.success),
        "turnovers": [e for e in es if "turnover" in e.tags],
        "sacks": [e for e in es if "sack" in e.tags],
        "rz_failures": [e for e in rz if not e.success],
        "drive_end_rz_fail": [e for e in es if "drive_end_rz_fail" in e.tags],
        "late_down_fails": [e for e in es if (e.down or 0) >= 3 and not e.success],
    }


def format_postgame_v2(
    db: Any,
    opponent_id: str,
    computed: dict[str, Any],
    *,
    session_id: str | None = None,
    top_n: int = 6,
) -> list[str]:
    """Plain-language postgame block: RZ/GL findings, turnovers, risers/fallers."""
    evals: list[SnapEval] = computed["evals"]
    drives = computed["drives"]
    res = (computed.get("results") or {}).get(session_id or "", {}) if session_id else {}
    f = game_findings(evals, drives, session_id)
    lines: list[str] = []
    scope = "this game" if session_id else "all logged games"
    lines.append(f"## Red zone / goal line ({scope})")
    lines.append(
        f"  Red-zone trips: {f['rz_trips']} → {f['rz_tds']} TD"
        + ("" if f["rz_trips"] == 0 else f" ({f['rz_tds'] / max(1, f['rz_trips']):.0%})")
        + f" | RZ snaps {f['rz_snaps']} ({f['rz_success']} successful)"
        + f" | goal-line snaps {f['gl_snaps']} ({f['gl_success']} successful)"
    )
    if f["drive_end_rz_fail"]:
        lines.append("  Drives that died inside the 20:")
        for e in f["drive_end_rz_fail"][:8]:
            lines.append(f"    - {e.label}")
    other_rz = [e for e in f["rz_failures"] if "drive_end_rz_fail" not in e.tags]
    if other_rz:
        lines.append("  Other red-zone failures:")
        for e in other_rz[:8]:
            lines.append(f"    - {e.label}")
    lines.append("## Turnovers & sacks")
    if not f["turnovers"] and not f["sacks"]:
        lines.append("  none — clean game")
    for e in f["turnovers"]:
        lines.append(f"  TURNOVER {e.label}  (x{abs(SCORE_TURNOVER):.1f} weight{', red zone' if e.zone != OPEN else ''})")
    for e in f["sacks"]:
        lines.append(f"  SACK {e.label}{' (red zone)' if e.zone != OPEN else ''}")
    if res and res.get("result_wl"):
        p = res.get("parsed") or {}
        mf = margin_factor(p.get("margin"))
        if res["result_wl"] == "loss":
            n_adj = sum(1 for e in evals if e.session_id == session_id and "loss_adjusted" in e.tags)
            lines.append(
                f"## Loss adjustment: {n_adj} stalled-drive-ending play(s) penalized "
                f"(margin {p.get('margin', '?')}, factor {mf:.2f}; red zone x{LOSS_STALL_RZ_MULT})"
            )
        elif res["result_wl"] == "win":
            n_adj = sum(1 for e in evals if e.session_id == session_id and "win_adjusted" in e.tags)
            lines.append(
                f"## Win adjustment: {n_adj} scoring-drive play(s) boosted (margin {p.get('margin', '?')}, factor {mf:.2f})"
            )
    diff = weight_diff(computed.get("before") or {}, computed.get("after") or {}, opponent_id=opponent_id)
    play_like = [d for d in diff if "play::" in d[0] or "fam::" in d[0] or "::" not in d[0]]
    general_keys = {d[0]: d for d in play_like if not d[0].startswith("zone::")}

    def _dup_of_general(d: tuple[str, float, float, float]) -> bool:
        # open-field store mostly mirrors general; hide near-identical duplicates
        if not d[0].startswith(f"zone::{OPEN}::"):
            return False
        base = d[0][len(f"zone::{OPEN}::"):]
        base = base[5:] if base.startswith("fam::") else base
        g = general_keys.get(base)
        return g is not None and abs(g[3] - d[3]) < 0.25

    play_like = [d for d in play_like if not _dup_of_general(d)]
    risers = sorted([d for d in play_like if d[3] > 0], key=lambda d: -d[3])[:top_n]
    fallers = sorted([d for d in play_like if d[3] < 0], key=lambda d: d[3])[:top_n]
    lines.append("## Top risers")
    lines += [f"  ↑ {_fmt_key(k)}: {b:+.2f} → {a:+.2f}" for k, b, a, _ in risers] or ["  (none)"]
    lines.append("## Top fallers")
    lines += [f"  ↓ {_fmt_key(k)}: {b:+.2f} → {a:+.2f}" for k, b, a, _ in fallers] or ["  (none)"]
    return lines


def zone_leaderboard(lw: LearnedWeights, zone: str, *, top: int = 6) -> dict[str, list[tuple[str, float, int]]]:
    """Best / worst plays in one zone store (for prep + report)."""
    prefix = f"zone::{zone}::play::" if zone != "general" else "play::"
    rows = []
    for k, v in lw.w.items():
        if k.startswith(prefix):
            rows.append((k[len(prefix):].replace("::", " — "), v, lw.n(k)))
    rows.sort(key=lambda r: -r[1])
    return {"best": rows[:top], "worst": sorted(rows, key=lambda r: r[1])[:top]}


__all__ = [
    "RULES_VERSION",
    "LearnedWeights",
    "SnapEval",
    "assign_drives",
    "backup_db_file",
    "compute_all",
    "constants_table",
    "ensure_rules_current",
    "evaluate_snap",
    "format_postgame_v2",
    "leverage_for",
    "margin_factor",
    "needs_rebuild",
    "parse_score",
    "play_family",
    "rebuild_all",
    "weight_diff",
    "zone_key",
    "zone_leaderboard",
]
