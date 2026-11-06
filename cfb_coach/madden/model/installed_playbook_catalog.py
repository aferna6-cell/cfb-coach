"""Faithful, provenance-aware installed Madden 27 playbook snapshot import.

Imports *recorded* installed names and source books into the research-only
PlayKnowledgeDB. Never infers play art or changes the active game database.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.play_knowledge import KnowledgeError, PlayKnowledgeDB, play_key

SCHEMA = "madden27.installed_playbooks.snapshot.v1"
SIDES = ("offense", "defense")


def _signature(side: str, formations: Mapping[str, Sequence[str]], sources: Mapping[str, str]) -> tuple[str, str]:
    body = [[name, sources[name], list(formations[name])] for name in sorted(formations)]
    payload = json.dumps([side, body], sort_keys=True, default=str).encode()
    digest = hashlib.sha256(payload).hexdigest().upper()
    fingerprint = ("OFF" if side == "offense" else "DEF") + f"-{digest[:4]}-{digest[4:8]}"
    return fingerprint, digest


def validate_snapshot(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Reject wrong inventories rather than silently fixing them."""
    if payload.get("schema") != SCHEMA:
        raise KnowledgeError("unknown installed-playbook snapshot schema")
    sides = payload.get("sides")
    if not isinstance(sides, dict) or set(sides) != set(SIDES):
        raise KnowledgeError("installed snapshot must explicitly contain offense and defense")
    out: dict[str, dict[str, Any]] = {}
    for side in SIDES:
        record = sides[side]
        if not isinstance(record, dict) or record.get("recorded_as_applied") is not True:
            raise KnowledgeError(f"{side}: backup does not attest recorded application")
        forms, sources = record.get("formations"), record.get("formation_sources")
        if not isinstance(forms, dict) or not forms or not isinstance(sources, dict) or set(forms) != set(sources):
            raise KnowledgeError(f"{side}: formations and source-book mapping must agree")
        for formation, plays in forms.items():
            if not isinstance(formation, str) or not formation.strip():
                raise KnowledgeError(f"{side}: invalid formation name")
            if not isinstance(sources[formation], str) or not sources[formation].strip():
                raise KnowledgeError(f"{side} {formation}: source book missing")
            if not isinstance(plays, list) or not plays or any(not isinstance(p, str) or not p.strip() for p in plays):
                raise KnowledgeError(f"{side} {formation}: invalid play list")
            if len(set(plays)) != len(plays):
                raise KnowledgeError(f"{side} {formation}: duplicate play names")
        fingerprint, digest = _signature(side, forms, sources)
        if record.get("inventory_fingerprint") != fingerprint:
            raise KnowledgeError(f"{side}: fingerprint mismatch: {record.get('inventory_fingerprint')!r} != {fingerprint}")
        out[side] = {"record": record, "fingerprint": fingerprint, "digest": digest}
    return out


def _find_source(catalog: Mapping[str, Any] | None, side: str, book: str, formation: str, play: str) -> tuple[str, str]:
    """Source catalogue is independent; discrepancies never alter user's inventory."""
    if catalog is None:
        return "source_catalog_unavailable", ""
    b = ((catalog.get("books") or {}).get(side) or {}).get(book)
    if b is None:
        return "source_book_not_found", ""
    f = (b.get("formations") or {}).get(formation)
    if f is None:
        return "source_formation_not_found", ""
    url = str(f.get("url") or b.get("url") or "")
    if play not in (f.get("plays") or []):
        return "source_play_not_found", url
    return "source_match", url


def import_installed(db: PlayKnowledgeDB, payload: Mapping[str, Any], *,
                     full_catalog: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Import 2 installed books as distinct research records and provenance links.

    Idempotent for an exact snapshot; never removes other research plays or
    existing sourced annotations. Both side inventories are transactionally
    validated before writes. The game DB remains read-only, external to this DB.
    """
    checked = validate_snapshot(payload)
    # Rows use the *recorded* source book. A missing/mismatched source does
    # not make the recorded installation disappear.
    rows = []
    catalog: dict[str, Any] = {"books": {"offense": {}, "defense": {}}}
    counts: dict[str, dict[str, int]] = {}
    for side in SIDES:
        rec = checked[side]["record"]
        for formation, plays in rec["formations"].items():
            book = rec["formation_sources"][formation]
            dest = catalog["books"][side].setdefault(book, {"formations": {}})["formations"]
            if formation in dest:
                raise KnowledgeError(f"{side}: duplicate source formation {book}/{formation}")
            # The source's URL is a reference, never a claim all plays agree.
            _, formation_url = _find_source(full_catalog, side, book, formation, plays[0])
            dest[formation] = {"plays": plays, "url": formation_url or f"backup:active_playbook_json/{side}"}
            for position, play in enumerate(plays):
                status, url = _find_source(full_catalog, side, book, formation, play)
                rows.append((checked[side]["digest"], side, book, formation, play,
                             play_key(side, book, formation, play), position,
                             status, url))
        counts[side] = {"formations": len(rec["formations"]),
                        "plays": sum(len(p) for p in rec["formations"].values())}
    # No destructive writes. The v1 base importer is idempotent.
    db.import_catalog(catalog)
    db.conn.executescript("""
        CREATE TABLE IF NOT EXISTS recorded_inventories (
            inventory_digest TEXT PRIMARY KEY, side TEXT NOT NULL,
            playbook_name TEXT NOT NULL, recorded_at TEXT,
            fingerprint TEXT NOT NULL, provenance TEXT NOT NULL,
            console_verified INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS recorded_installed_plays (
            inventory_digest TEXT NOT NULL REFERENCES recorded_inventories(inventory_digest),
            side TEXT NOT NULL, source_book TEXT NOT NULL,
            formation TEXT NOT NULL, play TEXT NOT NULL,
            play_key TEXT NOT NULL REFERENCES plays(key), slot_order INTEGER NOT NULL,
            source_check TEXT NOT NULL, source_formation_url TEXT NOT NULL,
            PRIMARY KEY(inventory_digest,source_book,formation,play));
        CREATE INDEX IF NOT EXISTS idx_installed_play_key ON recorded_installed_plays(play_key);
    """)
    with db.conn:
        for side in SIDES:
            rec = checked[side]["record"]
            db.conn.execute("""INSERT OR IGNORE INTO recorded_inventories
                (inventory_digest,side,playbook_name,recorded_at,fingerprint,provenance,console_verified)
                VALUES (?,?,?,?,?,?,0)""",
                (checked[side]["digest"], side, str(rec.get("recorded_playbook_name") or ""),
                 rec.get("recorded_at"), checked[side]["fingerprint"],
                 str(payload.get("truth_scope") or "recorded backup")))
        db.conn.executemany("""INSERT OR IGNORE INTO recorded_installed_plays
            (inventory_digest,side,source_book,formation,play,play_key,slot_order,source_check,source_formation_url)
            VALUES (?,?,?,?,?,?,?,?,?)""", rows)
    source_statuses: dict[str, dict[str, int]] = {side: {} for side in SIDES}
    for _, side, _, _, _, _, _, status, _ in rows:
        source_statuses[side][status] = source_statuses[side].get(status, 0) + 1
    return {"counts": counts,
            "fingerprints": {s: checked[s]["fingerprint"] for s in SIDES},
            "source_validation": source_statuses,
            "console_reverified": False,
            "scope": "backup-recorded installed inventory, not independently console-verified"}


def installed_coverage(db: PlayKnowledgeDB) -> dict[str, Any]:
    """Count ONLY recorded installed plays, not the full universe."""
    sql = """SELECT i.side AS side, COUNT(*) AS n,
        SUM(CASE WHEN EXISTS (SELECT 1 FROM player_jobs a WHERE a.play_key=i.play_key) THEN 1 ELSE 0 END) AS annotated,
        SUM(CASE WHEN (SELECT COUNT(DISTINCT j.position) FROM player_jobs j
           WHERE j.play_key=i.play_key AND j.evidence='user_verified')=11 THEN 1 ELSE 0 END) AS verified,
        SUM(CASE WHEN i.source_check='source_match' THEN 1 ELSE 0 END) AS source_match
        FROM recorded_installed_plays i GROUP BY i.side"""
    out = {}
    for r in db.conn.execute(sql):
        out[r["side"]] = {"catalogued": r["n"], "partially_annotated": r["annotated"],
                          "all_11_verified": r["verified"], "source_matches": r["source_match"]}
    return out


def annotation_queue(db: PlayKnowledgeDB) -> list[dict[str, Any]]:
    """Auditable to-do list for gameplay diagram annotation; no guessed jobs."""
    sql = """SELECT i.*, COUNT(j.kind) AS job_rows,
        COUNT(DISTINCT j.position) AS known_positions
        FROM recorded_installed_plays i
        LEFT JOIN player_jobs j ON j.play_key=i.play_key
        GROUP BY i.inventory_digest,i.source_book,i.formation,i.play
        ORDER BY i.side,i.formation,i.slot_order"""
    result = []
    for r in db.conn.execute(sql):
        result.append({"side": r["side"], "source_book": r["source_book"],
                       "formation": r["formation"], "play": r["play"],
                       "play_key": r["play_key"], "source_formation_url": r["source_formation_url"],
                       "source_catalog_check": r["source_check"],
                       "known_player_assignments": r["known_positions"],
                       "unrecorded_player_assignments": max(0, 11-r["known_positions"]),
                       "status": "partial" if r["job_rows"] else "catalog_only",
                       "console_verified": False})
    return result


def write_queue_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    if not rows:
        raise KnowledgeError("no installed play rows available to export")
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)