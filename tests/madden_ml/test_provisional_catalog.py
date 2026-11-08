"""Issue #132 provisional feed: separate from the verified channel, every listing accounted for, never verified."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cfb_coach.madden.model import catalog_semantics as cs
from cfb_coach.madden.model import provisional_catalog as pc
from scripts.madden_ml.extract_downloaded_art import identity

ART = b"synthetic-not-real-art"
SHA = hashlib.sha256(ART).hexdigest()


def _cat():
    def rec(side, f, p, art=True):
        return {"id": f"madden27|{side}|{f}|{p}", "game": "madden27", "side": side, "formation_slug": f, "play_slug": p,
                "play_name": p.title(), "personnel_group": "G", "art_url": "u" if art else None,
                "art_sha256": SHA if art else None, "playbooks": []}
    return [rec("offense", "fa", "p-one"), rec("offense", "fa", "curl--outs"), rec("offense", "fb", "no-art", art=False),
            rec("offense", "fb", "amb"), rec("defense", "da", "cov")]


def _store(tmp_path):
    raw, ex = tmp_path / "raw", tmp_path / "ex"
    (ex / "responsibilities").mkdir(parents=True)
    raw.mkdir()
    route = {"kind": "route_proposal", "colour": "yellow", "polyline": [[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]],
             "suggested_shape": "vertical", "trace_complete": True, "association_flags": []}
    for f, p, flags in (("fa", "p-one", []), ("fa", "curl--outs", []), ("fb", "amb", ["route_junction_ambiguous"])):
        (raw / f"offense__{f}__{p}.png").write_bytes(ART)
        (ex / "responsibilities" / f"madden27__offense__{f}__{p}.json").write_text(json.dumps({
            "art_sha256": SHA, "extractor_version": "v2", "flags": flags, "unmatched_routes": [],
            "markers": [{"marker_index_left_to_right": 0, "marker_shape": "x", "position_px": [1.0, 2.0],
                         "matched": [route], "flags": []}, {"marker_index_left_to_right": 1, "matched": "unknown"}]}))
    return dict(raw_dir=raw, extraction_dir=ex)


def test_every_listing_accounted_and_never_verified(tmp_path):
    body = pc.export_provisional(_cat(), **_store(tmp_path), include_ace_derived_geometry=True)
    rows = {r["play_key"]: r for r in body["rows"]}
    assert body["schema"] == pc.SCHEMA and body["n_indexed"] == len(rows) == 4
    assert rows["madden27|offense|fb|no-art"]["feature_tier"] == "name_only"
    assert rows["madden27|offense|fb|no-art"]["blocked_reason"] == "no_art_url"
    assert rows["madden27|offense|fb|amb"]["feature_tier"] == "partial"
    assert rows["madden27|offense|fa|p-one"]["feature_tier"] == "high_confidence_extracted"
    for r in rows.values():
        assert not r["verified"] and r["feature_tier"] != "verified_assignments" and r["source_books"] == []
        assert not r["installable"]
        g = r.get("geometry")
        if g:
            assert g["yards"] == g["qb_read"] == g["blocking"] == pc.UNKNOWN
            assert all(m["identity_hypothesis"] == pc.UNKNOWN and m["slot"] == pc.UNKNOWN for m in g["markers"])
    assert "ART" not in json.dumps(body) and "synthetic-not-real-art" not in json.dumps(body)


def test_default_feed_excludes_ace_geometry(tmp_path):
    body = pc.export_provisional(_cat(), **_store(tmp_path))
    assert body["include_ace_derived_geometry"] is False
    for r in body["rows"]:
        assert not set(pc.GEOMETRY_FIELDS) & set(r) and "vlm_hypotheses" not in r
        assert r["model_use_permitted"] is False and not r["ace_image_derived"]
        assert r["evidence_tier"] in ("name_only", "geometry_extracted_unreviewed_withheld")


def test_verified_channel_still_empty():
    assert not any(row["eligible"] or row["features"] for row in cs.export(reviews=[])["feature_rows"])
    assert cs.SCHEMA != pc.SCHEMA


def test_double_hyphen_filename_kept_literally():
    assert identity(Path("offense__gun-spread-y-slot-wk__curl--outs.png")) == ("offense", "gun-spread-y-slot-wk", "curl--outs")
    assert identity(Path("offense__bad--form__x.png")) is None
    assert identity(Path("offense__f__-lead.png")) is None


def test_installable_only_for_installed_identity(tmp_path):
    rows = pc.build_rows(_cat(), **_store(tmp_path), installed={("offense", "fa", "p-one")},
                         alias_audits={"madden27|offense|fb|amb": "near miss, not aliased"})
    by = {r["play_key"]: r for r in rows}
    assert by["madden27|offense|fa|p-one"]["installable"] and by["madden27|offense|fa|p-one"]["installed_book_match"] == "matched"
    assert by["madden27|offense|fb|amb"]["installed_book_match"] == "unmatched_alias" and not by["madden27|offense|fb|amb"]["installable"]
