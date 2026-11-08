"""#132: the full-catalog portfolio can select a legal installable pair outside the installed 93; a 93-only system cannot.

Index entries are local SimpleNamespace rows shaped like CatalogEntry. Names come from the real installed snapshot and
real source-book stock lists; scores are mocked, not observed results."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cfb_coach.madden import catalog
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import offense_portfolio as portfolio

SNAPSHOT = Path(__file__).resolve().parents[2] / "cfb_coach/data/madden27/installed_playbook_snapshot.json"
OUT_SOURCE, OUT_FORMATION = "Cardinals", "Gun Wing Slot Offset"
BLOCKED_SOURCE, BLOCKED_FORMATION = "Cardinals", "Pistol Wing Slot"


def installed_books():
    off = json.loads(SNAPSHOT.read_text())["sides"]["offense"]
    books: dict = {}
    for form, plays in off["formations"].items():
        books.setdefault(off["formation_sources"][form], {})[form] = list(plays)
    return books


def entry(form, play, sources, *, installable=True, tier="name_only", blocked=None):
    return SimpleNamespace(
        listing_id=f"{form}|{play}", formation=form, play=play, formation_slug=form.lower(), play_slug=play.lower(),
        personnel_group=None, feature_tier=tier, source_books=tuple(sources),
        source_book_status="resolved" if sources else "unresolved", installable=installable, art_listed=True,
        ace_image_derived=False, first_party_derived=True, model_use_permitted=True, blocked_reason=blocked,
        structure_ref=None)


def fake_predict(art, *, formation, play, **kw):
    assert kw["coverage_source"] == "none" and kw["coverage_hint"] is None
    return {"probability": 0.9 if formation in (OUT_FORMATION, BLOCKED_FORMATION) else 0.3,
            "evidence_quality": "empirical"}


def run(books, index=None):
    with patch.object(em, "predict_success", side_effect=fake_predict):
        return portfolio.select_formation_portfolio(
            art=em.ExperimentalArtifact(), all_books=books, active={}, opponent_id="unknown", max_formations=3,
            coverage_mix={"cover_1_unspecified": 1}, priors_on=True, catalog_index=index)


def test_outside_triple_is_real_legal_and_not_installed():
    books = installed_books()
    assert sum(len(p) for f in books.values() for p in f.values()) == 93
    installed = {(f, p) for fs in books.values() for f, ps in fs.items() for p in ps}
    stock = catalog.book_formations("offense", OUT_SOURCE)[OUT_FORMATION]
    assert stock and not any(f == OUT_FORMATION for f, _ in installed)
    assert all((OUT_FORMATION, p) not in installed for p in stock)
    assert catalog.book_formations("offense", BLOCKED_SOURCE)[BLOCKED_FORMATION]


def test_full_catalog_selects_outside_pair_93_only_cannot():
    books = installed_books()
    only93 = run(books)
    assert OUT_FORMATION not in [r["formation"] for r in only93["chosen"]]

    stock = catalog.book_formations("offense", OUT_SOURCE)[OUT_FORMATION]
    blocked_stock = catalog.book_formations("offense", BLOCKED_SOURCE)[BLOCKED_FORMATION]
    index = SimpleNamespace(entries=tuple(
        [entry(f, p, [sb]) for sb, fs in books.items() for f, ps in fs.items() for p in ps]
        + [entry(OUT_FORMATION, p, [OUT_SOURCE]) for p in stock]
        + [entry(BLOCKED_FORMATION, p, [], installable=False, blocked="source book unresolved") for p in blocked_stock]
        + [entry("", "", [])]))
    full = run({**books, OUT_SOURCE: {OUT_FORMATION: list(stock)}}, index)
    chosen = {r["formation"]: r for r in full["chosen"]}
    assert OUT_FORMATION in chosen
    assert chosen[OUT_FORMATION]["source_book"] == OUT_SOURCE
    assert chosen[OUT_FORMATION]["plays"] == list(dict.fromkeys(stock))  # every stock play committed
    # high-scoring non-installable listing is evaluated and audited, never chosen
    assert BLOCKED_FORMATION not in chosen
    audited = {r["formation"] for r in full["evaluated_listings"]}
    assert {BLOCKED_FORMATION, OUT_FORMATION} <= audited
    blocked = [r for r in full["blocked_listings"] if r["formation"] == BLOCKED_FORMATION]
    assert len(blocked) == len(set(blocked_stock)) and all(r["evaluated"] for r in blocked)
    assert any(r["reason"].startswith("structurally unreadable") for r in full["blocked_listings"])
    assert len(full["chosen"]) <= 3


def test_installable_entry_without_full_book_commits_index_plays():
    stock = catalog.book_formations("offense", OUT_SOURCE)[OUT_FORMATION]
    index = [entry(OUT_FORMATION, p, [OUT_SOURCE]) for p in stock]
    out = run({}, index)
    assert [r["formation"] for r in out["chosen"]] == [OUT_FORMATION]
    assert out["chosen"][0]["plays"] == list(dict.fromkeys(stock))
