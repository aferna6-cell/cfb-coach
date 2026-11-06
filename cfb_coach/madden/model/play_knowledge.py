"""Research-only Madden play knowledge: catalog identity, player jobs, and transformations.

The catalog contains *names*, not proof of assignments. Never infer an exact
route, zone, read, or user responsibility from a play name. This module is
stdlib-only; it does not change the live caller, game DB, or model artifacts.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "madden27.play_knowledge.v1"
SIDES = frozenset({"offense", "defense"})
EVIDENCE = frozenset({"annotated", "source_documented", "user_verified"})
OFFENSE_JOBS = frozenset({"route", "block", "run", "motion", "play_action", "option", "decoy", "alignment"})
DEFENSE_JOBS = frozenset({"man", "zone", "match", "rush", "blitz", "spy", "contain", "run_fit", "alignment"})
# Vocabulary, not a claim that each route is available as a Madden hot route.
ROUTES = frozenset({
    "slant", "drag", "in", "out", "hitch", "curl", "comeback", "streak", "fade",
    "post", "corner", "flat", "wheel", "angle", "texas", "pivot", "zig", "option",
    "seam", "cross", "shallow_cross", "deep_cross", "swing", "screen", "smoke",
    "whip", "stick", "spot", "choice", "sluggo", "other",
})


class KnowledgeError(ValueError):
    """Invalid annotation or transformation; never silently invent a play fact."""


def play_key(side: str, book: str, formation: str, play: str) -> str:
    if side not in SIDES or not all((book, formation, play)):
        raise KnowledgeError("side, source book, formation and play are required")
    payload = json.dumps([side, book, formation, play], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class PlayerJob:
    position: str
    kind: str
    task: str
    evidence: str
    source: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, side: str) -> None:
        if not self.position.strip() or not self.task.strip():
            raise KnowledgeError("player position and task are required")
        if self.kind not in (OFFENSE_JOBS if side == "offense" else DEFENSE_JOBS):
            raise KnowledgeError(f"unsupported {side} job: {self.kind}")
        if self.evidence not in EVIDENCE or not self.source.strip():
            raise KnowledgeError("jobs require explicit evidence tier and a traceable source")
        if self.kind == "route" and self.task not in ROUTES:
            raise KnowledgeError(f"unrecognized route taxonomy value: {self.task}")
        try:
            json.dumps(self.params)
        except (ValueError, TypeError) as exc:
            raise KnowledgeError("job parameters must be JSON-serializable") from exc


@dataclass(frozen=True)
class ConceptTag:
    name: str
    evidence: str
    source: str

    def validate(self) -> None:
        if not self.name.strip() or self.evidence not in EVIDENCE or not self.source.strip():
            raise KnowledgeError("concept needs a name, evidence and source")


@dataclass(frozen=True)
class PlayKnowledge:
    key: str
    side: str
    book: str
    formation: str
    play: str
    catalog_source: str
    jobs: tuple[PlayerJob, ...] = ()
    read_order: tuple[str, ...] = ()
    read_source: str = ""
    concepts: tuple["ConceptTag", ...] = ()

    @property
    def completeness(self) -> str:
        if not self.jobs:
            return "catalog_only"
        # 11 distinct *user-verified* players is required, not eleven guesses.
        verified = {x.position for x in self.jobs if x.evidence == "user_verified" and x.kind != "alignment"}
        return "fully_verified" if len(verified) == 11 else "partial"

    def jobs_for(self, position: str) -> tuple[PlayerJob, ...]:
        return tuple(x for x in self.jobs if x.position == position)


@dataclass(frozen=True)
class AssignmentChange:
    """Exact edit to one player's job; the action must be a real Madden action.

    'hot_route' changes only one offensive route; 'defender_assignment' changes
    one defensive assignment. A saved macro is represented as explicitly verified
    subchanges, not guessed global effects.
    """
    method: str
    target: str
    kind: str
    task: str
    action_id: str
    user_verified: bool
    source: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def validate(self, side: str) -> None:
        if self.method not in {"hot_route", "defender_assignment"}:
            raise KnowledgeError("unknown individual adjustment transformation")
        if self.method == "hot_route" and (side != "offense" or self.kind != "route"):
            raise KnowledgeError("hot routes must replace an offensive route")
        if self.method == "defender_assignment" and side != "defense":
            raise KnowledgeError("defender assignments must modify defense")
        if not self.user_verified or not self.action_id or not self.source:
            raise KnowledgeError("transformation requires verified Madden action and source")
        PlayerJob(self.target, self.kind, self.task, "user_verified", self.source, self.params).validate(side)


def transform(play: PlayKnowledge, changes: Sequence[AssignmentChange], *,
              eligible_targets: Sequence[str]) -> PlayKnowledge:
    """Apply individually verified, non-conflicting edits; unknown jobs stay unknown.

    A hot route can replace a documented original route; it cannot invent a
    receiver's original assignment or turn an unknown play into a verified one.
    """
    if not changes:
        return play
    jobs = list(play.jobs)
    allowed = set(eligible_targets)
    seen: set[tuple[str, str]] = set()
    for change in changes:
        change.validate(play.side)
        if change.target not in allowed:
            raise KnowledgeError(f"{change.target}: not on the confirmed eligible target list")
        ident = (change.target, change.kind)
        if ident in seen:
            raise KnowledgeError(f"conflicting changes to {change.target}/{change.kind}")
        seen.add(ident)
        originals = [j for j in jobs if j.position == change.target and j.kind == change.kind]
        if not originals:
            raise KnowledgeError(f"{change.target}/{change.kind}: original assignment unknown")
        jobs = [j for j in jobs if (j.position, j.kind) != ident]
        jobs.append(PlayerJob(change.target, change.kind, change.task, "user_verified", change.source, change.params))
    # A modified play has no independently verified original QB progression.
    return replace(play, jobs=tuple(jobs), read_order=(), read_source="", concepts=())


@dataclass(frozen=True)
class MacroDefinition:
    name: str
    side: str
    slot: str
    installed: bool
    active: bool
    user_verified: bool
    changes: tuple[AssignmentChange, ...]
    source: str


def apply_macro(play: PlayKnowledge, macro: MacroDefinition, *,
                eligible_targets: Sequence[str]) -> PlayKnowledge:
    if play.side != macro.side:
        raise KnowledgeError("macro is for the wrong side of the ball")
    if not all((macro.installed, macro.active, macro.user_verified, macro.slot, macro.source)):
        raise KnowledgeError("macro is not user-verified, installed, active and assigned a slot")
    if not macro.changes:
        raise KnowledgeError("macro assignments are unknown; cannot assert exact transformation")
    return transform(play, macro.changes, eligible_targets=eligible_targets)


def offense_read(play: PlayKnowledge) -> str:
    """Only emit a read progression when it was sourced for this actual play."""
    if play.side != "offense" or not play.read_order or not play.read_source:
        return "READ UNKNOWN"
    route_positions = {j.position for j in play.jobs if j.kind == "route"}
    if any(p not in route_positions for p in play.read_order):
        return "READ UNKNOWN"
    return "READ: " + " -> ".join(play.read_order)


def defensive_user(play: PlayKnowledge, position: str) -> str:
    if play.side != "defense":
        return "USER RESPONSIBILITY UNKNOWN"
    jobs = play.jobs_for(position)
    known = [j for j in jobs if j.evidence in {"user_verified", "source_documented"}]
    if not known:
        return "USER RESPONSIBILITY UNKNOWN"
    return f"USER {position}: " + "; ".join(f"{j.kind} {j.task}" for j in known)


@dataclass(frozen=True)
class AdjustmentCandidate:
    """A *pre-scored* legal plan; scores come from the research evaluation engine."""
    label: str
    score: float
    verified: bool
    legal: bool
    source: str


def choose_adjustment(stock_score: float, candidates: Sequence[AdjustmentCandidate], *,
                      min_gain: float = 0.01) -> str:
    """Return one definitive selection. No speculative or non-executable advice."""
    if min_gain < 0:
        raise KnowledgeError("min_gain must not be negative")
    valid = [x for x in candidates if x.verified and x.legal and x.label and x.source and
             x.label != "NO ADJUSTMENT" and x.score == x.score]
    valid.sort(key=lambda x: (-x.score, x.label))
    if not valid or valid[0].score < stock_score + min_gain:
        return "NO ADJUSTMENT"
    return valid[0].label


class PlayKnowledgeDB:
    """Research-only local SQLite DB. Caller chooses path; never writes game DB."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT OR IGNORE INTO schema_meta VALUES ('version', 'madden27.play_knowledge.v1');
            CREATE TABLE IF NOT EXISTS plays (
                key TEXT PRIMARY KEY, side TEXT NOT NULL, book TEXT NOT NULL,
                formation TEXT NOT NULL, play TEXT NOT NULL, catalog_source TEXT NOT NULL,
                UNIQUE(side,book,formation,play));
            CREATE TABLE IF NOT EXISTS player_jobs (
                play_key TEXT NOT NULL REFERENCES plays(key), position TEXT NOT NULL,
                kind TEXT NOT NULL, task TEXT NOT NULL, evidence TEXT NOT NULL,
                source TEXT NOT NULL, params_json TEXT NOT NULL,
                PRIMARY KEY(play_key,position,kind,task));
            CREATE TABLE IF NOT EXISTS concepts (
                play_key TEXT NOT NULL REFERENCES plays(key), name TEXT NOT NULL,
                evidence TEXT NOT NULL, source TEXT NOT NULL,
                PRIMARY KEY(play_key,name));
            CREATE TABLE IF NOT EXISTS read_progressions (
                play_key TEXT PRIMARY KEY REFERENCES plays(key),
                positions_json TEXT NOT NULL, source TEXT NOT NULL);
        """)
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "PlayKnowledgeDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def import_catalog(self, payload: Mapping[str, Any]) -> dict[str, int]:
        """Import source-book play *names* without fabricating player jobs."""
        counts = {"offense": 0, "defense": 0}
        if not isinstance(payload.get("books"), dict):
            raise KnowledgeError("catalog needs books by side")
        with self.conn:
            for side in ("offense", "defense"):
                for book, book_data in (payload["books"].get(side) or {}).items():
                    for formation, form in (book_data.get("formations") or {}).items():
                        for play in dict.fromkeys(form.get("plays") or []):
                            key = play_key(side, book, formation, play)
                            self.conn.execute(
                                "INSERT OR IGNORE INTO plays VALUES (?,?,?,?,?,?)",
                                (key, side, book, formation, play,
                                 str(form.get("url") or book_data.get("url") or payload.get("source_url") or "unknown")))
                            counts[side] += 1
        return counts

    def annotate(self, key: str, jobs: Sequence[PlayerJob], *,
                 read_order: Sequence[str] = (), read_source: str = "",
                 concepts: Sequence[ConceptTag] = ()) -> None:
        row = self.conn.execute("SELECT side FROM plays WHERE key=?", (key,)).fetchone()
        if not row:
            raise KnowledgeError("play must exist in catalog before annotation")
        side = row["side"]
        if not jobs:
            raise KnowledgeError("empty annotation would erase useful evidence")
        for job in jobs:
            job.validate(side)
        for concept in concepts:
            concept.validate()
        if read_order and (side != "offense" or not read_source or
                           not set(read_order).issubset({j.position for j in jobs if j.kind == "route"})):
            raise KnowledgeError("sourced read progression must reference known offensive routes")
        with self.conn:
            self.conn.execute("DELETE FROM player_jobs WHERE play_key=?", (key,))
            self.conn.executemany(
                "INSERT INTO player_jobs VALUES (?,?,?,?,?,?,?)",
                [(key, j.position, j.kind, j.task, j.evidence, j.source,
                  json.dumps(j.params, sort_keys=True)) for j in jobs])
            self.conn.execute("DELETE FROM concepts WHERE play_key=?", (key,))
            self.conn.executemany("INSERT INTO concepts VALUES (?,?,?,?)",
                                  [(key, c.name, c.evidence, c.source) for c in concepts])
            self.conn.execute("DELETE FROM read_progressions WHERE play_key=?", (key,))
            if read_order:
                self.conn.execute("INSERT INTO read_progressions VALUES (?,?,?)",
                                  (key, json.dumps(list(read_order)), read_source))

    def get(self, key: str) -> PlayKnowledge | None:
        row = self.conn.execute("SELECT * FROM plays WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        jobs = tuple(PlayerJob(j["position"], j["kind"], j["task"], j["evidence"],
                               j["source"], json.loads(j["params_json"]))
                     for j in self.conn.execute("SELECT * FROM player_jobs WHERE play_key=? ORDER BY position,kind,task", (key,)))
        rr = self.conn.execute("SELECT * FROM read_progressions WHERE play_key=?", (key,)).fetchone()
        concepts = tuple(ConceptTag(r["name"], r["evidence"], r["source"])
                         for r in self.conn.execute("SELECT * FROM concepts WHERE play_key=? ORDER BY name", (key,)))
        return PlayKnowledge(key, row["side"], row["book"], row["formation"], row["play"],
                             row["catalog_source"], jobs,
                             tuple(json.loads(rr["positions_json"])) if rr else (), rr["source"] if rr else "",
                             concepts)

    def coverage(self) -> dict[str, Any]:
        rows = self.conn.execute("""SELECT p.side, COUNT(*) n,
            SUM(CASE WHEN EXISTS (SELECT 1 FROM player_jobs a WHERE a.play_key=p.key) THEN 1 ELSE 0 END) annotated,
            SUM(CASE WHEN (SELECT COUNT(DISTINCT j.position) FROM player_jobs j
                WHERE j.play_key=p.key AND j.evidence='user_verified')=11 THEN 1 ELSE 0 END) verified
            FROM plays p GROUP BY p.side""").fetchall()
        return {r["side"]: {"catalogued": r["n"], "partially_annotated": r["annotated"],
                             "all_11_verified": r["verified"]} for r in rows}