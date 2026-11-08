"""#132 QA F1 (slug stock lookup) and F2 (digest collisions). Fixture DBs only."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cfb_coach.db import CoachDB
from cfb_coach.madden import catalog, playbook
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model import offense_score_cache as cache
from cfb_coach.madden.model.offense_inventory import inventory_fingerprint
from cfb_coach.madden.model.experimental_model import ExperimentalArtifact

SLUG, DISPLAY, SRC = "gun-tight-doubles", "Gun Tight Doubles", "Buccaneers"


def _stock():
    return catalog.book_formations("offense", SRC)[DISPLAY]


def test_slug_lookup_matches_display_name():
    stock = _stock()
    assert len(stock) == 15 and "L Spot" in stock
    assert designer._stock_plays_for_source(SRC, SLUG) == stock
    assert designer._stock_plays_for_source(SRC, DISPLAY) == stock
    assert designer._stock_plays_for_source(SRC, "no-such-formation") == []


def test_auto_adjustment_pairs_accepts_slug():
    stock = _stock()
    eligible = {p for p in stock if not catalog.is_run(p) and "rpo" not in p.lower()}
    assert len(eligible) == 12 and "Cross Wheels" in eligible
    seen = []
    def rate(art, form, play, opp, **kw):
        seen.append(play)
        return 1.0 if play == "Cross Wheels" else 0.1
    with patch.object(designer, "_rate", side_effect=rate):
        pairs = designer._auto_adjustment_pairs(None, {SLUG: stock}, {SLUG: SRC}, "cpu",
                                                priors_on=False, coverage_mix={})
    assert pairs == [(SLUG, "Cross Wheels")]
    assert set(seen) == eligible


def test_confirm_installed_accepts_slug_key_on_fixture_db(tmp_path):
    db = CoachDB(tmp_path / "fixture.db", seed={"opponents": {}})
    try:
        current = {"side": "offense", "mode": "custom", "name": SRC, "source_book": SRC, "rev": 1,
                   "formations": {}, "audibles": {}, "core": [], "locked_ts": "t"}
        playbook._save_state(db, {"applied": {"offense": current}, "pending": {}})
        forms, sources = {SLUG: _stock()}, {SLUG: SRC}
        book = {**current, "rev": 2, "formations": forms, "formation_sources": sources,
                "inventory_id": inventory_fingerprint(forms, sources)}
        proposal = {"expected_applied_hash": designer._digest(current), "book": book,
                    "macro_blueprints": [], "opponent_id": "cpu"}
        proposal["proposal_id"] = designer._identity(proposal)
        db.set_meta(designer.META_PENDING, designer._canonical(proposal))
        out = designer.confirm_installed(db, proposal_id=proposal["proposal_id"],
                                         attestation="fixture attestation for slug lookup")
        assert out["confirmed"] and out["plays"] == 15
        # Completeness rule still holds: a partial slug formation is refused.
        forms = {SLUG: _stock()[:-1]}
        book = {**book, "formations": forms, "inventory_id": inventory_fingerprint(forms, sources)}
        applied = playbook._load_state(db)["applied"]["offense"]
        proposal = {"expected_applied_hash": designer._digest(applied), "book": book,
                    "macro_blueprints": [], "opponent_id": "cpu"}
        proposal["proposal_id"] = designer._identity(proposal)
        db.set_meta(designer.META_PENDING, designer._canonical(proposal))
        try:
            designer.confirm_installed(db, proposal_id=proposal["proposal_id"],
                                       attestation="fixture attestation for slug lookup")
        except ValueError as exc:
            assert "incomplete or diverged" in str(exc)
        else:
            raise AssertionError("partial formation accepted")
    finally:
        db.close()


def _art(**kw):
    return ExperimentalArtifact(model_version="m", n_supervised=1, evidence_quality="mixed", **kw)


def test_artifact_digest_global_rate_and_tables():
    lo, hi = _art(global_rate=0.2), _art(global_rate=0.8)
    assert cache.artifact_digest(lo) != cache.artifact_digest(hi)
    a, b = _art(global_rate=0.5), _art(global_rate=0.5)
    a.base, b.base = SimpleNamespace(global_rate=0.3), {"global_rate": 0.7}
    assert cache.artifact_digest(a) != cache.artifact_digest(b)
    c = _art(global_rate=0.5)
    c.play_family = {"pass": [3, 5]}
    assert cache.artifact_digest(c) != cache.artifact_digest(_art(global_rate=0.5))


def test_score_cache_isolated_by_artifact_digest(tmp_path):
    def key(art):
        return cache.cache_key(catalog_hash="c", artifact_hash=cache.artifact_digest(art),
                               opponent_history_hash="o", priors_on=False, rights_feed_digest="r")
    lo, hi = _art(global_rate=0.2), _art(global_rate=0.8)
    first = cache.ScoreCache(key(lo), tmp_path)
    assert first.get_or_compute(["probe"], lambda: lo.global_rate) == 0.2
    first.store()
    second = cache.ScoreCache(key(hi), tmp_path)
    second.load()
    assert second.path != first.path
    assert second.get_or_compute(["probe"], lambda: hi.global_rate) == 0.8


def _entry(structure_ref):
    return SimpleNamespace(listing_id="x", installable=True, model_use_permitted=False,
                           feature_tier="name_only", structure_ref=structure_ref)


def _idx(structure_ref="s1", **kw):
    return SimpleNamespace(entries=(_entry(structure_ref),), catalog_hash="h", n_indexed=1, **kw)


def test_rights_feed_digest_structure_ref_and_geometry_hash():
    assert (cache.catalog_provenance({}, _idx("s1"))["rights_feed_digest"]
            != cache.catalog_provenance({}, _idx("s2"))["rights_feed_digest"])
    absent = cache.catalog_provenance({}, _idx())
    assert "geometry_hash" not in absent
    g1 = cache.catalog_provenance({}, _idx(geometry_hash="g1"))
    g2 = cache.catalog_provenance({}, _idx(geometry_hash="g2"))
    assert g1["geometry_hash"] == "g1"
    assert len({absent["rights_feed_digest"], g1["rights_feed_digest"], g2["rights_feed_digest"]}) == 3
    keys = {cache.cache_key(catalog_hash="h", artifact_hash="a", opponent_history_hash="o",
                            priors_on=False, rights_feed_digest=p["rights_feed_digest"],
                            geometry_hash=p.get("geometry_hash") or "") for p in (g1, g2)}
    assert len(keys) == 2
