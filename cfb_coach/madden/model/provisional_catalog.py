"""Issue #132 PROVISIONAL / LOW-CONFIDENCE catalog feed.

Separate from catalog_semantics.export(), which stays verified-only. Nothing here is
ever tier ``verified``; geometry is unreviewed extractor output, a confidence-weighted
prior only. Slots, yards, reads, blocking and man/zone beaters are always UNKNOWN.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA = "madden27.catalog_semantics.provisional.v1"
TIERS = ("name_only", "high_confidence_automated", "partial_ambiguous",
         "geometry_extracted_unreviewed_withheld", "verified")
UNKNOWN = "UNKNOWN"
GEOMETRY_FIELDS = ("geometry",)
AMBIGUOUS_FLAGS = {"route_junction_ambiguous", "unknown_complex", "incomplete_trace",
                   "multiple_zones_share_marker_review"}
RIGHTS_GEOMETRY = {
    "basis": "Owner Aidan Fernandes written confirmation 2026-11-08 (relayed by dr eggbot): "
             "provisional/low-confidence export of Ace-derived route/coverage geometry and local VLM interpretation",
    "prior_scopes": ["metadata_bulk", "images_download", "images_local_storage"],
    "added_scopes": ["provisional_geometry_export_model_prior", "local_vlm_interpretation"],
    "limits": ["never verified; agreement is not verification", "no raw art/overlays/crops redistributed or sent to third parties",
               "no claimed player identity or QB reads", "no invented yards/blocking/slots/beaters"],
}


def play_key(side: str, formation_slug: str, play_slug: str, game: str = "madden27") -> str:
    return f"{game}|{side}|{formation_slug}|{play_slug}"


def _r(v: Any, n: int = 1) -> Any:
    if isinstance(v, float):
        return round(v, n)
    if isinstance(v, (list, tuple)):
        return [_r(x, n) for x in v]
    return v


def _element_conf(flags: Iterable[str], trace_complete: bool | None = True) -> tuple[str, bool]:
    amb = bool(set(flags) & AMBIGUOUS_FLAGS) or trace_complete is False
    return ("low" if amb else "medium"), amb


def compact_geometry(resp: Mapping[str, Any]) -> dict[str, Any]:
    """Compact, unreviewed geometry from a downloaded_art_responsibilities.v2 summary."""
    markers, any_amb = [], False
    for m in resp.get("markers", []):
        elems = []
        matched = m.get("matched", [])
        for e in (matched if isinstance(matched, list) else []):
            if not isinstance(e, dict):
                continue
            flags = list(e.get("association_flags", [])) + list(m.get("flags", []))
            conf, amb = _element_conf(flags, e.get("trace_complete"))
            any_amb |= amb
            el = {"kind": e.get("kind"), "colour_measured": e.get("colour"), "colour_meaning": UNKNOWN,
                  "confidence": conf, "ambiguous": amb, "flags": flags,
                  "owner_distance_px": _r(e.get("owner_distance_px"))}
            if e.get("kind") == "route_proposal":
                pl = _r(e.get("polyline") or [])
                el.update(polyline=pl, start=pl[0] if pl else None, end=pl[-1] if pl else None,
                          breaks=pl[1:-1], directed=True, trace_complete=e.get("trace_complete"),
                          route_family_candidate=e.get("suggested_shape") or UNKNOWN,
                          alternatives=[UNKNOWN] if amb else [])
            else:
                el.update(centroid=_r(e.get("centroid")), bbox=e.get("bbox"),
                          responsibility_hypothesis=("rush" if e.get("kind", "").startswith("rush") else
                                                     "zone" if "zone" in (e.get("kind") or "") else UNKNOWN),
                          man_match_hypothesis=UNKNOWN)
            elems.append(el)
        any_amb |= bool(set(m.get("flags", [])) & AMBIGUOUS_FLAGS)
        markers.append({"marker_index": m.get("marker_index_left_to_right"), "shape": m.get("marker_shape"),
                        "start_px": _r(m.get("position_px")), "identity_hypothesis": UNKNOWN,
                        "slot": UNKNOWN, "elements": elems})
    any_amb |= bool(set(resp.get("flags", [])) & AMBIGUOUS_FLAGS)
    unowned = [{"kind": k, "colour_measured": e.get("colour"), "polyline": _r(e.get("polyline")),
                "centroid": _r(e.get("centroid")), "flags": e.get("association_flags", []),
                "route_family_candidate": e.get("suggested_shape") or UNKNOWN, "owner": UNKNOWN, "confidence": "low"}
               for k, src in (("route", "unmatched_routes"), ("zone", "zones_unassociated"), ("rush", "rush_arrows_unassociated"))
               for e in resp.get(src, []) if isinstance(e, dict)]
    return {"coordinate_frame": "diagram_px_unreviewed", "markers": markers,
            "concept_interactions": {"crossing": UNKNOWN, "flood": UNKNOWN, "levels": UNKNOWN, "concept": UNKNOWN},
            "yards": UNKNOWN, "qb_read": UNKNOWN, "blocking": UNKNOWN,
            "any_ambiguous": any_amb, "play_flags": list(resp.get("flags", [])), "unowned_elements": unowned}


def _tier(has_geom: bool, include_geom: bool, ambiguous: bool) -> str:
    if not has_geom:
        return "name_only"
    if not include_geom:
        return "geometry_extracted_unreviewed_withheld"
    return "partial_ambiguous" if ambiguous else "high_confidence_automated"


def build_rows(catalog: Iterable[Mapping[str, Any]], *, raw_dir: Path | None, extraction_dir: Path | None,
               installed: set[tuple[str, str, str]] = frozenset(), alias_audits: Mapping[str, str] | None = None,
               formation_books: Mapping[str, list[str]] | None = None, side: str = "offense",
               include_ace_derived_geometry: bool = False, vlm: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """One row per indexed listing of ``side``; never drops a listing."""
    alias_audits, formation_books, vlm = alias_audits or {}, formation_books or {}, vlm or {}
    rows = []
    for r in catalog:
        if r.get("side") != side:
            continue
        fs, ps = r["formation_slug"], r["play_slug"]
        key = play_key(side, fs, ps, r.get("game", "madden27"))
        fname = f"{side}__{fs}__{ps}.png"
        png = raw_dir / fname if raw_dir else None
        downloaded = bool(png and png.exists())
        sha_ok = None
        if downloaded and r.get("art_sha256"):
            sha_ok = hashlib.sha256(png.read_bytes()).hexdigest() == r["art_sha256"]
        resp = {}
        if extraction_dir:
            p = extraction_dir / "responsibilities" / f"madden27__{side}__{fs}__{ps}.json"
            if p.exists():
                resp = json.loads(p.read_text())
        extracted = bool(resp) and resp.get("art_sha256") == r.get("art_sha256")
        geom = compact_geometry(resp) if extracted else None
        amb = bool(geom and geom["any_ambiguous"])
        ident = (side, fs, ps)
        books = formation_books.get(f"{side}|{fs}", [])
        if ident in installed:
            match = "matched"
        elif key in alias_audits:
            match = "unmatched_alias"
        else:
            match = "not_in_installed_set"
        blocked = None if r.get("art_url") else "no_art_url"
        if r.get("art_url") and not downloaded:
            blocked = "art_not_downloaded"
        elif downloaded and not extracted:
            blocked = "not_extracted"
        row = {
            "play_key": key, "listing_id": r["id"], "game": r.get("game"), "side": side,
            "formation_label": r.get("formation_label"), "formation_slug": fs, "personnel_group": r.get("personnel_group"),
            "play_name": r.get("play_name"), "play_slug": ps, "source": "acemadden",
            "source_detail_url": (r.get("source") or {}).get("detail_url"), "source_version": r.get("version") or "",
            "content_hash": r.get("content_hash"), "art_url_present": bool(r.get("art_url")),
            "art_sha256": r.get("art_sha256"), "downloaded": downloaded, "art_hash_ok": sha_ok,
            "extracted": extracted, "extractor_version": resp.get("extractor_version") if extracted else None,
            "diagram_ref": f"private:raw/{fname}" if downloaded else None,
            "evidence_tier": _tier(extracted, include_ace_derived_geometry, amb),
            "source_book_memberships": ([{"book": b, "level": "formation", "confidence": "low",
                                          "evidence": "public AceMadden team-book page links this formation; play-level membership unverified"}
                                         for b in books] or UNKNOWN),
            "source_book_mapped": bool(books), "installable": ident in installed,
            "installable_basis": "first-party installed_playbook_snapshot exact (formation, play, installed book) triple" if ident in installed else None,
            "installed_book_match": match, "unmatched_alias_notes": alias_audits.get(key),
            "rights_status": "provisional_geometry_permitted_2026-11-08" if include_ace_derived_geometry else "metadata_only",
            "model_use_permitted": bool(include_ace_derived_geometry and extracted),
            "model_use_scope": "confidence_weighted_prior_only" if include_ace_derived_geometry and extracted else "metadata_only",
            "verified": False, "vlm_pass": key in vlm, "blocked_reason": blocked,
        }
        # CatalogEntry-aligned fields for the Madden ML loader (#132 architecture map).
        tier = row["evidence_tier"]
        row.update({
            "formation": r.get("formation_label") or r.get("personnel_group"), "play": r.get("play_name"),
            "feature_tier": {"high_confidence_automated": "high_confidence_extracted", "partial_ambiguous": "partial",
                             "name_only": "name_only"}.get(tier, "unknown"),
            "source_books": [],  # verified play-level membership only; none verified yet
            "source_book_status": "unresolved" if books else "unknown",
            "art_listed": bool(r.get("art_url")), "ace_image_derived": bool(include_ace_derived_geometry and extracted),
            "first_party_derived": False,
            "structure_ref": row["diagram_ref"] and {"store": "private", "play_key": key,
                                                     "extractor_version": row["extractor_version"]},
        })
        if include_ace_derived_geometry:
            row["geometry"] = geom
            if key in vlm:
                row["vlm_hypotheses"] = vlm[key]
        rows.append(row)
    return rows


def export_provisional(catalog: list[Mapping[str, Any]], **kw: Any) -> dict[str, Any]:
    rows = build_rows(catalog, **kw)
    n = sum(1 for r in catalog if r.get("side") == kw.get("side", "offense"))
    if len(rows) != n:
        raise AssertionError(f"silent omission: {len(rows)} rows for {n} listings")
    if any(r["evidence_tier"] == "verified" or r["verified"] for r in rows):
        raise AssertionError("provisional feed may not carry verified rows")
    return {"schema": SCHEMA, "provisional": True, "verified_channel": "catalog_semantics.export (separate)",
            "include_ace_derived_geometry": bool(kw.get("include_ace_derived_geometry")),
            "rights": RIGHTS_GEOMETRY if kw.get("include_ace_derived_geometry") else {"scopes": RIGHTS_GEOMETRY["prior_scopes"]},
            "n_indexed": n, "rows": rows}
