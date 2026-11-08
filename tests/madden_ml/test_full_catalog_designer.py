"""Tiny synthetic universes only: never rank or publish a real Game 1 preview."""
from dataclasses import replace
from copy import deepcopy
from pathlib import Path
import argparse
import json
from unittest.mock import Mock

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.madden import catalog
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model import offense_catalog_index as index
from cfb_coach.madden.model import offense_portfolio as portfolio
from cfb_coach.madden.model import offense_score_cache as cache
from cfb_coach.madden.model.experimental_model import ExperimentalArtifact

BOOKS = {"Synthetic Stock": {"Gun Test Bunch": ["Mesh", "Inside Zone"]}}


@pytest.fixture
def tiny_index():
    row = index.CatalogEntry(
        listing_id="fixture-mesh", formation="Gun Test Bunch", play="Mesh",
        formation_slug="gun-test-bunch", play_slug="mesh", personnel_group="11",
        feature_tier="name_only", source_books=("Synthetic Stock",),
        source_book_status="mapped", installable=True, art_listed=False,
        ace_image_derived=False, first_party_derived=False, model_use_permitted=False,
        blocked_reason="provisional feed prohibits model use", structure_ref=None,
    )
    return index.CatalogIndex(index.SCHEMA, 2, "synthetic-catalog-v1",
        {"installable": 1}, (row, replace(row, listing_id="unmapped", play="Flood",
            play_slug="flood", source_books=(), installable=False)), ())


@pytest.fixture
def setup_design(tmp_path, monkeypatch):
    db = CoachDB(tmp_path / "fixture.db")
    monkeypatch.setattr(cache, "DEFAULT_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(catalog, "book_names", lambda side: list(BOOKS))
    monkeypatch.setattr(catalog, "book_formations", lambda side, book: BOOKS[book])
    yield db
    db.close()


def test_default_loads_full_index_and_passes_unfiltered(tiny_index, setup_design, monkeypatch):
    loader = Mock(return_value=tiny_index)
    monkeypatch.setattr(index, "load_offense_universe", loader)
    original = portfolio.select_formation_portfolio
    seen = []
    def select(*, catalog_index=None, priors_on=False, coverage_mix=None, **kwargs):
        seen.append(catalog_index)
        assert kwargs["all_books"] == BOOKS
        return original(catalog_index=catalog_index, priors_on=priors_on, coverage_mix=coverage_mix, **kwargs)
    monkeypatch.setattr(portfolio, "select_formation_portfolio", select)
    proposal = designer.design_offense(setup_design, artifact=ExperimentalArtifact(), skip_adjustment_design=True)
    loader.assert_called_once()
    assert seen == [tiny_index] and seen[0] is tiny_index
    assert seen[0].entries[0].blocked_reason and seen[0].entries[0].installable
    assert proposal["book"]["formations"] == BOOKS["Synthetic Stock"]
    assert proposal["n_indexed"] == 2 and proposal["n_installable"] == 1
    stored = list(cache.DEFAULT_CACHE_DIR.glob("*.json"))
    assert len(stored) == 1
    scores = cache.ScoreCache(stored[0].stem)
    scores.load()
    # Both legal and unresolved listings were scored, retained, and cached.
    for row in tiny_index.entries:
        assert scores.lookup(["catalog_listing_normal_probe.v1", row.listing_id, row.formation, row.play, "cpu"]) is not None
    for field in cache.PROVENANCE_FIELDS:
        assert proposal[field] == proposal["provenance"][field]
        assert proposal[field] == proposal["preparation_guide"]["provenance"][field]


def test_explicit_catalogue_and_injected_index_never_load_store(tiny_index, setup_design, monkeypatch):
    monkeypatch.setattr(index, "load_offense_universe", Mock(side_effect=AssertionError("Private store accessed")))
    hook = designer.design_offense(setup_design, catalogue=BOOKS, artifact=ExperimentalArtifact(), skip_adjustment_design=True)
    assert hook["catalog_hash"] == cache.digest(BOOKS)
    assert hook["n_indexed"] == hook["n_installable"] == 2
    injected = designer.design_offense(setup_design, catalog_index=tiny_index,
        artifact=ExperimentalArtifact(), skip_adjustment_design=True)
    assert injected["catalog_hash"] == tiny_index.catalog_hash
    assert injected["n_installable"] == 1


@pytest.mark.parametrize("field", cache.PROVENANCE_FIELDS + ("football_intelligence",))
def test_proposal_identity_signs_provenance(field, setup_design):
    proposal = designer.design_offense(setup_design, catalogue=BOOKS, artifact=ExperimentalArtifact(), skip_adjustment_design=True)
    changed = deepcopy(proposal)
    changed[field] = "different"
    assert designer._identity(changed) != proposal["proposal_id"]


def test_installable_count_does_not_use_block_reason_or_breakdown(tiny_index):
    dishonest_count = replace(tiny_index, breakdown={"installable": 90})
    metadata = cache.catalog_provenance(BOOKS, dishonest_count)
    assert metadata["n_installable"] == 1
    assert metadata["rights_feed_digest"] != cache.catalog_provenance(BOOKS,
        replace(tiny_index, entries=(replace(tiny_index.entries[0], model_use_permitted=True), tiny_index.entries[1])))["rights_feed_digest"]


@pytest.mark.parametrize("field", ["catalog_hash", "artifact_hash", "opponent_history_hash", "priors_on", "rights_feed_digest", "geometry_hash"])
def test_each_cache_key_component_invalidates(field):
    values = dict(catalog_hash="catalog", artifact_hash="artifact", opponent_history_hash="history",
                  priors_on=False, rights_feed_digest="rights")
    before = cache.cache_key(**values)
    values[field] = True if field == "priors_on" else "changed"
    assert before != cache.cache_key(**values)


def test_cache_load_store_and_repeated_adjustment_ranking(setup_design, monkeypatch, tmp_path):
    key = cache.cache_key(catalog_hash="c", artifact_hash="a", opponent_history_hash="o", priors_on=False, rights_feed_digest="r")
    scores = cache.ScoreCache(key, tmp_path / "scores")
    rate = Mock(return_value=.63)
    monkeypatch.setattr(designer, "_rate", rate)
    def rank(scores):
        return designer._auto_adjustment_pairs(None, BOOKS["Synthetic Stock"],
            {"Gun Test Bunch": "Synthetic Stock"}, "cpu", priors_on=False, coverage_mix={}, score_cache=scores)
    assert rank(scores) == [("Gun Test Bunch", "Mesh")]
    scores.store()
    reloaded = cache.ScoreCache(key, tmp_path / "scores")
    reloaded.load()
    assert rank(reloaded) == rank(scores)
    rate.assert_called_once()
    assert "Mesh" not in reloaded.path.read_text()
    assert not cache.DEFAULT_CACHE_DIR.is_relative_to(Path.cwd())


def test_artifact_and_history_digests_track_identity():
    art = ExperimentalArtifact()
    before = cache.artifact_digest(art)
    art._pregame_artifact_loader = "structural_nested_base"
    assert before != cache.artifact_digest(art)
    original = cache.opponent_history_digest({"cover_1": 1.0}, "opponent_history", .2)
    assert original != cache.opponent_history_digest({"cover_2": 1.0}, "opponent_history", .2)
    assert original != cache.opponent_history_digest({"cover_1": 1.0}, "explicit", .2)
    assert original != cache.opponent_history_digest({"cover_1": 1.0}, "opponent_history", .3)


@pytest.mark.parametrize("command", ["offense-design", "offense-coordinator"])
def test_default_command_uses_universe_without_catalogue(command, tiny_index, setup_design, monkeypatch, tmp_path, capsys):
    from cfb_coach.madden.model import cli

    loader = Mock(return_value=tiny_index)
    monkeypatch.setattr(index, "load_offense_universe", loader)
    parser = argparse.ArgumentParser()
    cli.build_ml_subparser(parser.add_subparsers())
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps(ExperimentalArtifact().to_dict()))
    extra = ["--text"] if command == "offense-design" else ["--full"]
    args = parser.parse_args(["ml", command, *extra, "--db", str(setup_design.path),
        "--artifact", str(artifact), "--skip-adjustment-design", "--max-formations", "1"])
    assert args.func(args) == 0
    loader.assert_called_once()
    payload, _ = json.JSONDecoder().raw_decode(capsys.readouterr().out)
    plan = payload["plan"] if command == "offense-coordinator" else payload
    guide = plan["preparation_guide"]
    assert guide["provenance"]["catalog_hash"] == tiny_index.catalog_hash
    assert {r["formation"]: r["stock_plays"] for r in guide["formations"]} == BOOKS["Synthetic Stock"]


@pytest.mark.parametrize("feed_exists", [True, False])
def test_optional_local_feed_and_aliases_are_forwarded_only_when_present(feed_exists, tiny_index, setup_design, monkeypatch):
    loader = Mock(return_value=tiny_index)
    monkeypatch.setattr(index, "load_offense_universe", loader)
    original = Path.is_file
    def is_file(path):
        if path.parent == Path("/workspace/madden_ml/issue_132_catalog_track"):
            return feed_exists
        return original(path)
    monkeypatch.setattr(Path, "is_file", is_file)
    designer.design_offense(setup_design, artifact=ExperimentalArtifact(), skip_adjustment_design=True)
    options = loader.call_args.kwargs
    assert set(options) == ({"provisional_export", "alias_audits"} if feed_exists else set())
