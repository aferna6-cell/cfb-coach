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
                                    verified_assignments=0, name_only=3, unknown=2,
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
