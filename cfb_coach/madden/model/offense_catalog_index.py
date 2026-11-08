"""Full offense universe with metadata-only entries, independent of verified assignment exports.

Membership uses only the in-repo stock-book (book -> formation -> plays) map.
No aliases, images, or metadata-declared book memberships are trusted.

``catalog_hash`` is SHA256 of UTF-8 JSON (sort_keys=True, separators=(',', ':'),
ensure_ascii=True) of a sorted list of permitted metadata dictionaries. Each
contains exactly: listing_id, formation, play, formation_slug, play_slug,
personnel_group, art_listed. Dictionaries are sorted by their canonical JSON,
preserving duplicates. Neither art locations/digests nor derived structure,
source-book joins, or provisional feed contents enter this listing identity.

``geometry_hash`` uses the same UTF-8 canonical JSON style, with payload
[{"listing_id": lid, "geometry": geom}, ...] sorted by listing_id and only
retained sanitized geometry objects. Without geometry its payload is [].
Geometry content never enters ``catalog_hash``.

Provisional v1 integration accepts a JSONL header followed by records, or a
single JSON object with ``entries`` as a list of records or a mapping
keyed by listing id. Records use listing_id (or id), and nonempty structure,
geometry, or assignments. References are JSON pointers into that separately
loaded feed, not paths supplied by a record. All such features remain partial
unless explicitly labeled high_confidence_extracted; even a claimed verified
label never becomes verified_assignments. Use only confidence-weighted priors,
never exact player identities, reads, or pixel training. No first-party structure
is claimed by this metadata loader. Missing optional feeds are harmless; invalid
present feeds fail clearly rather than silently dropping their records.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

from cfb_coach.madden import catalog
from cfb_coach.madden.model import ace_catalog

SCHEMA = "madden27.offense_catalog_index.v1"
PROVISIONAL_SCHEMA = "madden27.catalog_semantics.provisional.v1"


@dataclass(frozen=True)
class CatalogEntry:
    listing_id: str
    formation: str
    play: str
    formation_slug: str
    play_slug: str
    personnel_group: str | None
    feature_tier: str
    source_books: tuple[str, ...]
    source_book_status: str
    installable: bool
    art_listed: bool
    ace_image_derived: bool
    first_party_derived: bool
    model_use_permitted: bool
    blocked_reason: str | None
    structure_ref: str | None


@dataclass(frozen=True)
class CatalogIndex:
    schema: str
    n_indexed: int
    catalog_hash: str
    geometry_hash: str = field(default=hashlib.sha256(b"[]").hexdigest(), kw_only=True)
    breakdown: dict[str, int]
    entries: tuple[CatalogEntry, ...]
    blocked: tuple[dict, ...]
    _geometries: dict[str, dict] = field(default_factory=dict, repr=False, compare=False, kw_only=True)

    def geometry_for(self, listing_id: str) -> Mapping | None:
        """Return a detached sanitized geometry object, never its export wrapper.

        Retains markers, unowned_elements, play_flags and coordinate_frame;
        excludes art_url, art_sha256, source_detail_url and diagram_ref.
        """
        return deepcopy(self._geometries.get(listing_id))


def load_provisional_export(path: str | Path) -> dict:
    """Read the separate provisional feed; never invoke the verified exporter."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Producer envelope: a header object, then one entry per nonblank line.
        objects = []
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                objects.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid provisional export JSON at line {line_number}") from exc
        if not objects:
            raise ValueError("Invalid provisional export JSON: empty feed")
        data = objects[0]
        if isinstance(data, dict):
            data = {**data, "entries": objects[1:]}
    if not isinstance(data, dict) or data.get("schema") != PROVISIONAL_SCHEMA:
        raise ValueError(f"Expected provisional schema {PROVISIONAL_SCHEMA}")
    entries = data.get("entries", [])
    if not isinstance(entries, (list, dict)):
        raise ValueError("Provisional entries must be a list or listing-id mapping")
    if any(not isinstance(row, dict) for row in (entries.values() if isinstance(entries, dict) else entries)):
        raise ValueError("Each provisional entry must be an object")
    return _sanitized(data)


def _sanitized(value):
    """Remove private art locators/digests even in nested producer metadata."""
    if isinstance(value, dict):
        return {key: _sanitized(item) for key, item in value.items()
                if key not in {"art_sha256", "source_detail_url", "art_url", "diagram_ref"}}
    if isinstance(value, list):
        return [_sanitized(item) for item in value]
    return value


def _load_alias_audits(path: str | Path) -> dict:
    """Read advisory near-misses, never translate them into matching aliases.

    Literal listing ids (including double hyphens) remain untouched. The audit
    does not override the existing catalog.norm / ace_catalog.slug join policy.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid alias audit JSON") from exc
    if not isinstance(data, dict) or data.get("schema") != "madden27.catalog_alias_audit.v1":
        raise ValueError("Expected alias audit schema madden27.catalog_alias_audit.v1")
    return data


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _keys(*values: str) -> set[str]:
    return {key for value in values for key in
            (catalog.norm(value), catalog.norm(ace_catalog.slug(value))) if key}


def _provisional_rows(data: dict) -> dict[str, tuple[str | None, str, bool, dict | None]]:
    entries = data.get("entries", [])
    pairs = entries.items() if isinstance(entries, dict) else enumerate(entries)
    result = {}
    for key, row in pairs:
        listing_id = _text(row.get("listing_id") or row.get("id"))
        if not listing_id and isinstance(entries, dict):
            listing_id = _text(key)
        if not listing_id:
            raise ValueError("Provisional entry missing listing_id")
        if listing_id in result:
            raise ValueError("Duplicate provisional listing_id")
        permitted = row.get("model_use_permitted") is not False
        has_structure = permitted and any(row.get(field) for field in ("structure", "geometry", "assignments"))
        tier = row.get("feature_tier")
        if tier not in {"high_confidence_extracted", "partial", "name_only"}:
            tier = "partial" if has_structure else "name_only"
        if not has_structure:
            tier = "name_only"
        token = str(key).replace("~", "~0").replace("/", "~1")
        reference = f"provisional:#/entries/{token}" if has_structure else None
        geometry = row.get("geometry")
        result[listing_id] = (reference, tier, permitted,
                              geometry if isinstance(geometry, dict) else None)
    return result


def load_offense_universe(
    *,
    ace_store: str | Path = "/home/box/cfb_acemadden",
    playbooks: Mapping | None = None,
    provisional_export: str | Path | None = None,
    alias_audits: str | Path | None = None,
) -> CatalogIndex:
    """Index every offense row, including malformed-name/id and duplicate rows.

    Explicit playbooks mappings use the same shape as book_formations output:
    {book_name: {formation_name: [play_name, ...]}}. An empty mapping is honored.
    score_available reports eligibility for later scoring, not a computed score.
    """
    if alias_audits is not None:
        _load_alias_audits(alias_audits)  # Advisory only: do not apply aliases.
    if playbooks is None:
        playbooks = {book: catalog.book_formations("offense", book)
                     for book in catalog.book_names("offense")}
    membership: dict[tuple[str, str], set[str]] = {}
    for book, formations in playbooks.items():
        for formation, plays in formations.items():
            for play in plays:
                for f in _keys(formation):
                    for p in _keys(play):
                        membership.setdefault((f, p), set()).add(book)
    structures = {}
    if provisional_export is not None and Path(provisional_export).is_file():
        structures = _provisional_rows(load_provisional_export(provisional_export))

    geometries = {}
    entries = []
    metadata = []
    blocked = []
    breakdown = dict.fromkeys(("evaluated", "score_available", "structure_supported",
                               "verified_assignments", "high_confidence_extracted", "partial",
                               "name_only", "unknown",
                               "source_book_mapped", "installable"), 0)
    with (Path(ace_store) / "catalog.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid catalog JSON at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Catalog row must be an object at line {line_number}")
            if row.get("side") != "offense":
                continue
            listing_id = _text(row.get("id"))
            formation = _text(row.get("formation_label")) or _text(row.get("formation_slug"))
            play = _text(row.get("play_name")) or _text(row.get("play_slug"))
            formation_slug = _text(row.get("formation_slug")) or ace_catalog.slug(formation)
            play_slug = _text(row.get("play_slug")) or ace_catalog.slug(play)
            personnel = _text(row.get("personnel_group")) or None
            art_listed = bool(row.get("art_url") or row.get("art_sha256"))
            readable = bool(listing_id and formation and play)
            source_books = set()
            if readable:
                for f in _keys(formation, formation_slug):
                    for p in _keys(play, play_slug):
                        source_books.update(membership.get((f, p), ()))
            installable = bool(source_books)
            overlay = structures.get(listing_id, (None, "name_only", True, None))
            if overlay[3] is not None:
                geometries[listing_id] = overlay[3]
            structure_ref, tier, permitted = (
                overlay[:3]
                if readable else (None, "unknown", True)
            )
            reasons = []
            if not readable:
                reasons.append("structurally unreadable: missing id or formation/play name")
            if not permitted:
                reasons.append("provisional feed prohibits model use")
            if not installable:
                reasons.append("unresolved source-book membership: no verified stock-book triple")
            reason = "; ".join(reasons) or None
            entry = CatalogEntry(
                listing_id, formation, play, formation_slug, play_slug, personnel,
                tier, tuple(sorted(source_books)), "mapped" if installable else "unresolved",
                installable, art_listed, bool(structure_ref), False, permitted, reason, structure_ref,
            )
            entries.append(entry)
            metadata.append(dict(listing_id=listing_id, formation=formation, play=play,
                                 formation_slug=formation_slug, play_slug=play_slug,
                                 personnel_group=personnel, art_listed=art_listed))
            if reason:
                blocked.append(dict(listing_id=listing_id, formation=formation, play=play, reason=reason))
            for key, present in (("evaluated", readable), ("score_available", readable),
                                 ("structure_supported", bool(structure_ref)),
                                 ("high_confidence_extracted", tier == "high_confidence_extracted"),
                                 ("partial", tier == "partial"),
                                 ("name_only", tier == "name_only"), ("unknown", tier == "unknown"),
                                 ("source_book_mapped", installable), ("installable", installable)):
                breakdown[key] += int(present)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(canonical(sorted(metadata, key=canonical)).encode("utf-8")).hexdigest()
    geometry_payload = [{"listing_id": lid, "geometry": geom}
                        for lid, geom in sorted(geometries.items())]
    geometry_digest = hashlib.sha256(canonical(geometry_payload).encode("utf-8")).hexdigest()
    return CatalogIndex(SCHEMA, len(entries), digest, breakdown, tuple(entries), tuple(blocked),
                        geometry_hash=geometry_digest, _geometries=geometries)
