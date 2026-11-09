"""The defensive objective learns from recorded events, never invents missing targets."""
from __future__ import annotations

from cfb_coach.madden.model import defense_v2_pilot as v2


def snap(i: int, *, yards: int | None = None, td: bool | None = None,
         expl: bool | None = None, turnover: bool | None = None,
         convert: bool | None = None):
    return v2.Snap(
        "g1", "cpu-a", "2026-11-01", i, "Nickel Over", "Cover 3 Sky",
        "single_high", "long", "cross", bool(td), yards=yards,
        touchdown=td, explosive=expl, turnover=turnover, converted=convert,
    )


def test_missing_targets_do_not_become_zero_events():
    art = v2.fit([snap(i) for i in range(1, 22)])
    assert v2.auxiliary_risk(art, "long", "single_high") == {}


def test_verified_multi_target_statistics_are_smoothed_and_bounded():
    rows = [snap(i, yards=22 if i % 2 == 0 else 0, td=(i % 10 == 0),
                 expl=(i % 2 == 0), turnover=(i % 7 == 0),
                 convert=(i % 3 == 0)) for i in range(1, 31)]
    art = v2.fit(rows)
    aux = v2.auxiliary_risk(art, "long", "single_high")
    assert aux["yards_n"] == 30
    assert 0.0 < aux["td"] < 1.0
    assert 0.0 < aux["explosive"] < 1.0
    assert aux["expected_yards"] > 5.0
    assert aux["turnover_n"] == 30
    assert "conversion" in aux
    assert v2.defensive_utility_risk(0.4, aux) != 0.4


def test_rank_contains_independent_success_and_multi_target_risk():
    rows = [snap(i, yards=20, td=False, expl=True, turnover=False,
                 convert=True) for i in range(1, 22)]
    art = v2.fit(rows)
    belief = {c: 1.0 / len(v2.CONCEPTS) for c in v2.CONCEPTS}
    top = v2.rank(art, context="long", belief=belief,
                  calls=[{"formation": "Nickel Over", "play": "Cover 3 Sky",
                          "installed": True, "legal": True}])["ranked"][0]
    assert "predicted_offense_success" in top
    assert "defensive_utility_risk" in top
    assert "secondary_outcomes" in top
    assert top["secondary_outcomes"]["explosive_n"] == 21
