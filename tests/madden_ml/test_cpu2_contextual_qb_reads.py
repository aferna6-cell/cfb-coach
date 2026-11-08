"""Same play, different PRE-snap coverage/situation -> evidence-conditioned QB reads.

Provenance is explicit: these are football priors applied to known route
shapes, NOT measured success rates for individual receiver progressions.
"""
from __future__ import annotations

from cfb_coach.madden.model import offense_presnap_advice as advice
from cfb_coach.madden.model.offense_play_features import PlayStructure
from cfb_coach.madden.model.offense_taxonomy import RouteSpec
from cfb_coach.madden.model.offense_decision import Decision, NO_ADJUSTMENT


def _card(
    routes, documented, mix, *,
    down=1, distance=10, tier="source_documented",
):
    route_rows = tuple((slot, RouteSpec(family, side, band), tier)
                       for slot, family, side, band in routes)
    formation, play = "Gun Test", "Coverage Stress"
    structure = PlayStructure(
        formation=formation, play=play, routes=route_rows,
        skill_slots=tuple(slot for slot, *_ in routes),
        personnel_tier="unknown",
        hint={"play_type": "pass"},
        read_order=tuple(documented),
    )
    d = Decision(
        formation=formation, play=play,
        adjustment=NO_ADJUSTMENT, probability=0.52,
        stock_probability=0.52, interval=(0.40, 0.64),
        final_structure=structure,
    )
    return advice.build_card(d, mix, situation={"down": down, "distance": distance})


def test_same_documented_play_changes_primary_read_with_coverage_belief():
    routes = [
        ("SLWR", "drag", "left", "short"),
        ("TE1", "corner", "right", "deep"),
    ]
    published = ["SLWR", "TE1"]
    vs_man = _card(routes, published, {
        "cover_1_unspecified": 0.85, "cover_2_unspecified": 0.15,
    })
    vs_two_high_zone = _card(routes, published, {
        "cover_2_unspecified": 0.85, "cover_1_unspecified": 0.15,
    })
    assert vs_man["PRIMARY READ"].startswith("SLWR drag")
    assert vs_two_high_zone["PRIMARY READ"].startswith("TE1 corner")
    assert "SLWR drag" in vs_two_high_zone["PUBLISHED BASELINE"][0]
    assert "INFERRED contextual" in vs_two_high_zone["READ BASIS"]
    assert vs_two_high_zone["READ INPUTS"]["coverage_mix"].startswith("historical pre-snap")


def test_same_play_changes_with_down_distance_not_just_last_coverage():
    routes = [
        ("HB1", "flat", "right", "shallow"),
        ("WR1", "seam", "middle", "deep"),
    ]
    mix = {"cover_3_spot": 0.8, "cover_4_quarters": 0.2}
    first_down = _card(routes, ["HB1", "WR1"], mix, down=1, distance=10)
    third_long = _card(routes, ["HB1", "WR1"], mix, down=3, distance=12)
    assert first_down["PRIMARY READ"].startswith("HB1 flat")
    assert third_long["PRIMARY READ"].startswith("WR1 seam")
    assert "3rd/4th-and-long" in third_long["WHY"]


def test_ambiguous_coverage_preserves_documented_read_instead_of_overreacting():
    routes = [
        ("SLWR", "drag", "left", "short"),
        ("TE1", "corner", "right", "deep"),
    ]
    card = _card(routes, ["SLWR", "TE1"], {
        "unknown": 0.9, "cover_2_unspecified": 0.1,
    })
    assert card["PRIMARY READ"].startswith("SLWR drag")
    assert "defensive coverage uncertain" in card["WHY"]


def test_anonymous_geometry_remains_tentative_under_contextual_read_ranking():
    routes = [
        ("geom_0", "drag", "left", "short"),
        ("geom_1", "corner", "right", "deep"),
    ]
    card = _card(routes, [], {
        "cover_2_unspecified": 1.0,
    }, tier="partial")
    assert card["PRIMARY READ"].startswith("right corner")
    assert "receiver unverified" in card["PRIMARY READ"]
    assert "unreviewed play-diagram geometry" in card["READ BASIS"]
    assert card["PUBLISHED BASELINE"] is None


def test_unknown_routes_do_not_become_fabricated_receiver_reads():
    card = _card([], [], {"cover_2_unspecified": 1.0})
    assert card["PRIMARY READ"].startswith("UNKNOWN")
    assert card["READ BASIS"] == "unknown"
