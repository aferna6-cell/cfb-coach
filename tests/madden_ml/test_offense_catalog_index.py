"""Synthetic metadata only; these tests never access the private store."""
import json
import sys
from types import ModuleType
from dataclasses import asdict

import pytest

from cfb_coach.madden import catalog
from cfb_coach.madden.model import ace_catalog, offense_catalog_index as index


@pytest.fixture
def rows():
    return [
        dict(id="one", side="offense", formation_label="Test Bunch", formation_slug="test-bunch",
             play_name="Test Cross!", play_slug="test-cross", personnel_group="11", playbooks=[],
             art_url="synthetic-art-marker", art_sha256=None),
        dict(id="two", side="offense", formation_slug="other-test", play_name="Test Dive",
             play_slug="test-dive", personnel_group="12", playbooks=[]),
        dict(id="three", side="offense", formation_slug="test-bunch", play_slug="test-cross",
             playbooks=["Untrusted Book"], art_sha256="synthetic-digest-marker"),
        dict(side="offense", formation_slug="test-bunch", play_name="Test Cross"),
        dict(id="no-name", side="offense", formation_slug="test-bunch"),
        dict(id="defense", side="defense", play_name="Not offense"),
    ]


@pytest.fixture
def books():
    return {"Synthetic Book": {"TEST BUNCH": ["Test Cross"]},
            "Second Synthetic": {"Test Bunch": ["Test Cross"]}}


def store(tmp_path, rows):
    (tmp_path / "catalog.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return tmp_path


def feed(tmp_path, entries):
    path = tmp_path / "synthetic-provisional.json"
    path.write_text(json.dumps(dict(schema=index.PROVISIONAL_SCHEMA, entries=entries)))
    return path


def test_zero_omissions_tiers_and_blocked(tmp_path, rows, books):
    result = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books)
    assert result.schema == "madden27.offense_catalog_index.v1"
    assert result.n_indexed == 5
    assert [r.listing_id for r in result.entries] == ["one", "two", "three", "", "no-name"]
    assert result.breakdown == dict(evaluated=3, score_available=3, structure_supported=0,
                                    verified_assignments=0, high_confidence_extracted=0, partial=0, name_only=3, unknown=2,
                                    source_book_mapped=2, installable=2)
    assert [r.feature_tier for r in result.entries] == ["name_only"] * 3 + ["unknown"] * 2
    assert all(r.model_use_permitted and not r.ace_image_derived and not r.first_party_derived
               for r in result.entries)
    assert {r["listing_id"] for r in result.blocked} == {"two", "", "no-name"}
    assert all(r["reason"] for r in result.blocked)
    assert all(r.blocked_reason for r in result.entries if not r.installable)


def test_join_uses_existing_helpers_and_empty_ace_books(tmp_path, rows, books, monkeypatch):
    calls = dict(norm=0, slug=0)
    for module, name in ((catalog, "norm"), (ace_catalog, "slug")):
        original = getattr(module, name)
        def wrapped(value, original=original, name=name):
            calls[name] += 1
            return original(value)
        monkeypatch.setattr(module, name, wrapped)
    result = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books)
    assert all(calls.values())
    matched, unmatched, slug_only = result.entries[:3]
    assert matched.installable and matched.source_book_status == "mapped"
    assert matched.source_books == ("Second Synthetic", "Synthetic Book")
    assert unmatched.source_book_status == "unresolved" and not unmatched.installable
    assert unmatched.source_books == ()
    assert slug_only.source_books == matched.source_books
    assert "Untrusted Book" not in slug_only.source_books


def test_art_boolean_only_and_hash_ignores_art_bytes(tmp_path, rows, books):
    first = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books)
    assert [r.art_listed for r in first.entries] == [True, False, True, False, False]
    serialized = json.dumps(asdict(first))
    assert "synthetic-art-marker" not in serialized
    assert "synthetic-digest-marker" not in serialized
    assert "art_url" not in serialized and "art_sha256" not in serialized
    rows[0]["art_url"] = "different-synthetic-marker"
    rows[2]["art_sha256"] = "different-digest-marker"
    rows.reverse()
    second = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books)
    assert first.catalog_hash == second.catalog_hash
    rows[-1]["personnel_group"] = "21"
    assert first.catalog_hash != index.load_offense_universe(
        ace_store=store(tmp_path, rows), playbooks=books).catalog_hash


def test_default_books_use_repo_helpers_only(tmp_path, rows, books, monkeypatch):
    monkeypatch.setattr(catalog, "book_names", lambda side: list(books) if side == "offense" else [])
    monkeypatch.setattr(catalog, "book_formations", lambda side, book: books[book])
    (tmp_path / "playbooks.json").write_text("not a source of membership")
    result = index.load_offense_universe(ace_store=store(tmp_path, rows))
    assert result.breakdown["installable"] == 2
    empty = index.load_offense_universe(ace_store=tmp_path, playbooks={})
    assert empty.breakdown["installable"] == 0


def test_missing_optional_provisional_does_not_block(tmp_path, rows, books):
    result = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books,
                                         provisional_export=tmp_path / "absent.json")
    assert result.n_indexed == 5 and result.breakdown["structure_supported"] == 0


@pytest.mark.parametrize("claimed,expected", [("verified_assignments", "partial"),
                                               ("high_confidence_extracted", "high_confidence_extracted"),
                                               ("partial", "partial")])
def test_provisional_never_verified(tmp_path, rows, books, monkeypatch, claimed, expected):
    catalog_semantics = ModuleType("catalog_semantics")
    def forbidden(*args, **kwargs):
        pytest.fail("verified-only exporter must never be called")
    catalog_semantics.export = forbidden
    monkeypatch.setitem(sys.modules, "catalog_semantics", catalog_semantics)
    monkeypatch.setitem(sys.modules, "cfb_coach.madden.model.catalog_semantics", catalog_semantics)
    path = feed(tmp_path, [dict(listing_id="one", structure={"synthetic": True}, feature_tier=claimed),
                           dict(listing_id="two", geometry={"synthetic": True}),
                           dict(listing_id="three", feature_tier="verified_assignments")])
    assert index.load_provisional_export(path)["schema"] == index.PROVISIONAL_SCHEMA
    result = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books,
                                         provisional_export=path)
    assert result.entries[0].feature_tier == expected
    assert result.entries[0].structure_ref == "provisional:#/entries/0"
    assert result.entries[0].ace_image_derived and not result.entries[0].first_party_derived
    assert result.entries[1].feature_tier == "partial"
    assert not result.entries[1].installable  # Structure cannot invent membership.
    assert result.entries[2].feature_tier == "name_only"  # A tier label is not structure.
    assert result.breakdown["structure_supported"] == 2
    assert result.breakdown["verified_assignments"] == 0
    assert result.catalog_hash == index.load_offense_universe(ace_store=tmp_path, playbooks=books).catalog_hash


def test_mapping_feed_and_explicit_structure_denial(tmp_path, rows, books):
    path = feed(tmp_path, {"one": dict(assignments={"synthetic": True}),
                           "two": dict(geometry={"synthetic": True}, model_use_permitted=False)})
    result = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books,
                                         provisional_export=path)
    assert result.entries[0].structure_ref == "provisional:#/entries/one"
    assert result.entries[1].structure_ref is None


@pytest.mark.parametrize("value", [{"schema": "wrong"}, [],
                                    {"schema": index.PROVISIONAL_SCHEMA, "entries": 1},
                                    {"schema": index.PROVISIONAL_SCHEMA, "entries": [1]}])
def test_invalid_feed(tmp_path, value):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="schema|entries|entry"):
        index.load_provisional_export(path)


def test_duplicate_rows_are_not_silently_deduplicated(tmp_path, rows):
    result = index.load_offense_universe(ace_store=store(tmp_path, rows[:1] * 2), playbooks={})
    assert result.n_indexed == 2 and len(result.blocked) == 2


def test_invalid_catalog_fails_with_line_number(tmp_path):
    (tmp_path / "catalog.jsonl").write_text("not json\n")
    with pytest.raises(ValueError, match="line 1"):
        index.load_offense_universe(ace_store=tmp_path, playbooks={})


def jsonl_feed(tmp_path, entries, schema=index.PROVISIONAL_SCHEMA):
    path = tmp_path / "synthetic-provisional.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in
                              [dict(schema=schema, n_indexed=len(entries), provisional=True), *entries]))
    return path


def test_jsonl_overlay_and_sanitization(tmp_path, rows, books):
    path = jsonl_feed(tmp_path, [
        dict(listing_id="one", geometry={"synthetic": True, "art_url": "private-marker"},
             feature_tier="high_confidence_extracted", verified=True, model_use_permitted=True,
             art_sha256="private-marker", source_detail_url="private-marker"),
        dict(listing_id="two", geometry={"synthetic": True}, feature_tier="partial",
             source_books=["Invented Book"], source_book_status="mapped", source_book_mapped=True,
             source_book_memberships=["Invented Book"], installable=True, installed_book_match=True),
        dict(listing_id="three", feature_tier="name_only", blocked_reason="no_art_url",
             model_use_permitted=False, vlm_pass=False),
    ])
    loaded = index.load_provisional_export(path)
    assert loaded["n_indexed"] == 3 and loaded["provisional"] is True
    assert len(loaded["entries"]) == 3
    assert "private-marker" not in json.dumps(loaded)
    result = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books,
                                         provisional_export=path)
    baseline = index.load_offense_universe(ace_store=tmp_path, playbooks=books)
    assert [e.feature_tier for e in result.entries[:3]] == ["high_confidence_extracted", "partial", "name_only"]
    assert result.breakdown["structure_supported"] == 2
    assert result.breakdown["verified_assignments"] == 0
    assert not result.entries[2].model_use_permitted
    assert result.entries[2].structure_ref is None
    assert result.entries[2].blocked_reason
    assert result.entries[1].source_books == () and not result.entries[1].installable
    assert [(e.source_books, e.installable) for e in result.entries] == [
        (e.source_books, e.installable) for e in baseline.entries]
    assert result.catalog_hash == baseline.catalog_hash


@pytest.mark.parametrize("schema", ["wrong", None])
def test_jsonl_schema_mismatch(tmp_path, schema):
    with pytest.raises(ValueError, match="schema"):
        index.load_provisional_export(jsonl_feed(tmp_path, [dict(listing_id="one")], schema))


def test_invalid_jsonl_reports_line(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text(json.dumps(dict(schema=index.PROVISIONAL_SCHEMA)) + "\nnot-json\n")
    with pytest.raises(ValueError, match="line 2"):
        index.load_provisional_export(path)


@pytest.mark.parametrize("book_play,listing_name,listing_slug", [
    ("L Spot", "Z Spot", "z-spot"),
    ("L Spot", "z-spot", "z-spot"),
    ("Zl Spot Shake", "Z Spot Shake", "z-spot-shake"),
    ("Zl Spot Shake", "z-spot-shake", "z-spot-shake"),
])
def test_l_z_near_misses_never_join(tmp_path, book_play, listing_name, listing_slug):
    rows = [dict(id="synthetic", side="offense", formation_slug="test-formation",
                 play_name=listing_name, play_slug=listing_slug)]
    path = jsonl_feed(tmp_path, [dict(listing_id="synthetic", installable=True,
                                    source_books=["Synthetic Book"], source_book_mapped=True)])
    result = index.load_offense_universe(ace_store=store(tmp_path, rows),
        playbooks={"Synthetic Book": {"Test Formation": [book_play]}}, provisional_export=path)
    assert not result.entries[0].installable and result.entries[0].source_books == ()
    # A distinct in-repo Z string is a legitimate join, not an L/Z alias.
    exact = index.load_offense_universe(ace_store=tmp_path,
        playbooks={"Distinct Book": {"Test Formation": [listing_name]}})
    assert exact.entries[0].source_books == ("Distinct Book",)


def test_advisory_audit_does_not_override_helpers_or_literal_slugs(tmp_path):
    audit = tmp_path / "synthetic-alias-audit.json"
    audit.write_text(json.dumps(dict(schema="madden27.catalog_alias_audit.v1",
        policy="no aliases invented; near-misses are NOT the same play",
        unresolved_installed_name_mismatches={"synthetic|portland-ywheel": "unresolved"},
        literal_double_hyphen_slugs=["literal"], notes={})))
    rows = [dict(id="wheel", side="offense", formation_slug="test-formation",
                 play_name="Portland Y Wheel", play_slug="portland-y-wheel"),
            dict(id="literal", side="offense", formation_slug="test-formation",
                 play_name="Curl & Outs", play_slug="curl--outs")]
    books = {"Synthetic Book": {"Test Formation": ["Portland Ywheel", "Curl & Outs"]}}
    result = index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks=books,
                                         alias_audits=audit)
    # Existing helper collision is reported, not replaced with a hand-built exclusion.
    assert catalog.norm("Portland Ywheel") == catalog.norm("Portland Y Wheel")
    assert ace_catalog.slug("Portland Ywheel") != ace_catalog.slug("Portland Y Wheel")
    assert result.entries[0].installable
    assert result.entries[1].play_slug == "curl--outs"
    assert result.entries[1].listing_id == "literal"


def test_invalid_alias_audit_is_not_silently_ignored(tmp_path, rows):
    audit = tmp_path / "audit.json"
    audit.write_text('{"schema": "wrong"}')
    with pytest.raises(ValueError, match="alias audit schema"):
        index.load_offense_universe(ace_store=store(tmp_path, rows), alias_audits=audit)


def test_jsonl_duplicate_denied_rows_still_fail(tmp_path, rows):
    path = jsonl_feed(tmp_path, [dict(listing_id="one", model_use_permitted=False)] * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        index.load_offense_universe(ace_store=store(tmp_path, rows), playbooks={}, provisional_export=path)
