"""Issue #121 foundation: the football-universe repository keeps metadata, name structure, image geometry, assignments and
verification apart, never fabricates roles, and answers lookups fast enough for live code."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time

import pytest

from cfb_coach.madden import catalog
from cfb_coach.madden.model import football_universe as fu


@pytest.fixture(scope="module")
def u():
    return fu.load()


def test_committed_universe_equals_a_fresh_build_and_validates(u):
    fresh = fu.build()
    assert fresh["digest"] == u["digest"] and fu.validate(u) == []


def test_counts_are_derived_from_the_data_not_hardcoded(u):
    r = fu.coverage_report(u)
    for side in ("offense", "defense"):
        raw = {(f, p) for b in catalog.book_names(side) for f, ps in catalog.book_formations(side, b).items() for p in ps}
        installed_extra = {e["key"] for e in u["plays"] if e["side"] == side and "repo_playbook_catalog" not in e["sources"]}
        assert r["sides"][side]["unique_formation_play_pairs"] == len(raw) + len(installed_extra)
        assert r["sides"][side]["formation_play_listings_across_books"] == sum(
            len(catalog.book_formations(side, b)[f]) for b in catalog.book_names(side) for f in catalog.book_formations(side, b))


def test_the_units_are_reported_separately_and_honestly(u):
    o, d = (fu.coverage_report(u)["sides"][s] for s in ("offense", "defense"))
    assert o["metadata_indexed"]["pct"] == 100.0 and o["image_geometry"]["plays"] == 93 and o["roles_verified"]["plays"] == 0
    assert o["structure_known"]["plays"] == 8 and o["slot_level"]["by_tier"].get("source_documented", 0) == 40
    assert d["image_geometry"]["plays"] == 0 and d["roles_verified"]["plays"] == 0 and d["defender_slots_verified"] == 0
    assert o["image_geometry"]["plays"] < o["name_decomposed"]["plays"] < o["metadata_indexed"]["plays"]


def test_a_listed_but_undiagrammed_play_is_metadata_only_for_assignments(u):
    e = fu.search("Gun Tight Y Off", side="offense", installed=False, limit=1, u=u)[0]
    assert e["tiers"]["metadata_indexed"] and not e["tiers"]["image_geometry"] and not e["tiers"]["roles_verified"] and not e["installed"]
    assert "ev_image_geometry" not in e["features"]                     # the model is told how little is known


def test_spelling_variants_are_kept_apart_not_merged(u):
    groups = {}
    for e in u["plays"]:
        if "alias_group" in e:
            groups.setdefault(e["alias_group"], []).append(e)
    assert groups and all(len(v) >= 2 and len({x["formation"] + x["play"] for x in v}) == len(v) for v in groups.values())


def test_same_play_name_on_two_sides_never_collides(u):
    names_off = {e["play"] for e in u["plays"] if e["side"] == "offense"}
    names_def = {e["play"] for e in u["plays"] if e["side"] == "defense"}
    keys = [e["key"] for e in u["plays"]]
    assert len(keys) == len(set(keys))
    assert fu.play_key("offense", "X", "Same") != fu.play_key("defense", "X", "Same")


def test_validator_rejects_fabricated_roles(u):
    bad = copy.deepcopy(u)
    e = next(x for x in bad["plays"] if x["side"] == "defense" and not x["tiers"]["roles_verified"])
    e["tiers"]["roles_verified"] = True
    assert any("roles_verified" in m for m in fu.validate(bad))
    bad = copy.deepcopy(u)
    next(x for x in bad["plays"] if x["side"] == "defense")["tiers"]["image_geometry"] = True
    assert any("no defensive play art" in m for m in fu.validate(bad))


def test_installed_flags_match_the_owner_book_and_unseen_plays_are_queryable(u):
    assert sum(e["installed"] for e in u["plays"] if e["side"] == "offense") == 93
    unseen = fu.search("cloud", side="defense", installed=False, limit=5, u=u)
    assert unseen and all(not e["installed"] for e in unseen)
    row = fu.feature_row("defense", unseen[0]["formation"], unseen[0]["play"], u)
    assert row.get("ev_metadata_indexed") == 1.0 and fu.feature_row("offense", "Nope", "Nope", u) == {}


def test_sources_record_rights_and_acemadden_is_not_assumed_licensed(u):
    assert u["sources"]["acemadden_site"]["train_rights"] == "NOT_ESTABLISHED"
    assert "acemadden_site" not in {s for e in u["plays"] for s in e["sources"]}      # nothing was ingested from it


def test_cached_lookup_is_fast_enough_for_live_calls(u):
    e = u["plays"][100]
    t = time.perf_counter()
    for _ in range(2000):
        fu.feature_row(e["side"], e["formation"], e["play"], u)
    assert (time.perf_counter() - t) / 2000 < 0.001            # well under 1 ms each; the 150 ms live budget is untouched


def test_cli_search_and_report_run_read_only():
    out = subprocess.run([sys.executable, "-m", "cfb_coach", "ml", "universe", "search", "mesh", "--side", "offense", "--limit", "3"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and "legend" in out.stdout
    rep = subprocess.run([sys.executable, "-m", "cfb_coach", "ml", "universe", "report"], capture_output=True, text=True)
    assert json.loads(rep.stdout)["sides"]["offense"]["metadata_indexed"]["pct"] == 100.0


def test_pregame_proposal_carries_universe_evidence_without_changing_the_plan(tmp_path):
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import playbook
    from cfb_coach.madden.model import offense_designer as d
    from cfb_coach.madden.model.experimental_model import ExperimentalArtifact
    cat = {"Buccaneers": {"Gun Bunch": ["Mesh", "Inside Zone", "Flood"], "Gun Doubles": ["Slants", "Bench"]},
           "Shotgun Classic": {"Gun Tight Y Off": [f"P{i}" for i in range(24)]}}
    db = CoachDB(tmp_path / "c.db", seed={"opponents": {"cpu": {"display_name": "CPU", "team_now": "DET", "skill": "cpu", "confidence": "low", "profile_json": "{}"}}})
    playbook._save_state(db, {"applied": {"offense": {"side": "offense", "mode": "custom", "name": "x", "rev": 1, "source_book": "Buccaneers", "trimmed": True,
                                                        "formations": {"Gun Bunch": ["Mesh", "Inside Zone", "Flood"], "Gun Doubles": ["Slants", "Bench"]},
                                                        "audibles": {}, "core": ["Gun Bunch"], "locked_ts": "t"}}, "pending": {}})
    art = ExperimentalArtifact(model_version="t", n_supervised=10, evidence_quality="empirical_light", global_rate=0.5)
    plan = d.design_offense(db, opponent_id="cpu", max_formations=15, artifact=art, catalogue=cat, must_add=("Gun Tight Y Off",))
    assert {"Gun Bunch", "Gun Doubles", "Gun Tight Y Off"} <= set(plan["book"]["formations"]) and not [c for c in plan["changes"] if c["action"] == "REMOVE_FORMATION"]
    assert plan["universe_digest"] == fu.load()["digest"] and set(plan["universe_evidence"]) == set(plan["book"]["formations"])
    db.close()
