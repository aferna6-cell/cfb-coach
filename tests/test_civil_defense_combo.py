"""Guards for Civil match / Nickel 2-4 Wide stock-first source recipes.

These tests verify no fabricated user and no silently dropped manual adjustments.
They are not a performance validation or Xbox play-art verification.
"""
from cfb_coach.madden import civil_defense as d


def test_guide_counts_and_no_silent_nickel_transform():
    c = d.load_catalog()
    assert len(c["entries"]) == 20
    assert sum(e["kind"] == "nickel_recipe" for e in c["entries"]) == 10
    assert sum(e["kind"] == "match_guide" for e in c["entries"]) == 10
    assert not any(e["stock_first_eligible"] for e in c["entries"] if e["kind"] == "nickel_recipe")


def test_stock_match_user_exact():
    c = d.load_catalog()
    book = {"Dime 2-3": ["Cover 3 Match", "Cover 2 Match", "Cover 4 Quarters"]}
    r = d.resolve_user("Dime 2-3", "Cover 3 Match", "trips", book,
                       owner_match_checks_configured=True, catalog=c)
    assert r["ok"] and r["user"] == "Hook-Curl Defender"
    assert d.compact_card(r) == "PLAY: Cover 3 Match (Dime 2-3)\nUSER: Hook-Curl Defender"
    r = d.resolve_user("Dime 2-3", "Cover 2 Match", "5wr_tight", book,
                       owner_match_checks_configured=True, catalog=c)
    assert r["ok"] and r["user"] == "Short-side Deep Half Safety"


def test_unconfirmed_check_and_unknown_roles_block():
    c = d.load_catalog()
    book = {"Dime 2-3": ["Cover 3 Match", "Cover 2 Match", "Cover 4 Quarters"]}
    assert not d.resolve_user("Dime 2-3", "Cover 3 Match", "trips", book, catalog=c)["ok"]
    assert not d.resolve_user("Dime 2-3", "Cover 4 Quarters", "bunch", book,
                              owner_match_checks_configured=True, catalog=c)["ok"]
    assert not d.resolve_user("Dime 2-3", "Cover 2 Match", "trips", book,
                              owner_match_checks_configured=True, catalog=c)["ok"]
    assert d.resolve_user("Dime 2-3", "Cover 6", "trips", book,
                          owner_match_checks_configured=True, catalog=c)["reason"] == "NOT_INSTALLED"


def test_nickel_packages_blocked_even_if_macros_confirmed():
    c = d.load_catalog()
    book = {"Nickel 2-4 Wide": ["Cover 2 Match", "Cover 3 Cloud", "Tampa 2 Contain", "Cover 9 Show 2"],
            "Dime 2-3 Odd": ["DB Fire 2"]}
    for e in (e for e in c["entries"] if e["kind"] == "nickel_recipe"):
        r = d.resolve_user(e["formation"], e["play"], e["offense_shape"], book,
                           owner_match_checks_configured=True,
                           confirmed_macros=("24 Base", "DB Fire"), catalog=c)
        assert not r["ok"], (e["id"], r)


def test_stock_zone_fallback_needs_user_evidence():
    c = d.load_catalog()
    assert c["coverage_policy"]["stock_zone_fallback"]["callable"] is False
    try:
        d.compact_card({"ok": False, "reason": "UNKNOWN_USER"})
        assert False
    except ValueError:
        pass
