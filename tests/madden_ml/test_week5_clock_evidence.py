"""Week 5: time-dependent strategy requires actual game-clock evidence.

Synthetic replays of the Week 4 3rd-and-5; never rewrite real logs or artifacts.
"""
from types import SimpleNamespace

from cfb_coach.madden.model.football_situation import (
    coordinator_components, evaluate_situation, selection_adjustment,
)
from cfb_coach.madden.model.offense_strategy import _objective


def situation(
    *, down=3, distance=5, yardline=70, quarter=4,
    score_us=21, score_them=7, clock_seconds=None, two_minute=False,
):
    return SimpleNamespace(
        side="offense", down=down, distance=distance,
        yardline=yardline, quarter=quarter, clock_seconds=clock_seconds,
        score_us=score_us, score_them=score_them, two_minute=two_minute,
        red_zone=yardline >= 80, goal_line=yardline >= 97,
        coverage_hint=None, coverage_source="none", extras={},
    )


def test_week4_third_and_five_without_clock_does_not_claim_clock_management():
    sit = situation()
    picture = evaluate_situation(sit)
    assert picture["clock_seconds"] is None
    assert "clock_management" not in picture["situations"]
    assert "late_lead_run" not in {
        p["id"] for p in coordinator_components("HB Stretch", picture)
    }
    assert _objective(sit, {"state": "unknown"}, None) == "convert_critical_first_down"
    # Week 4's inside-the-twenty penalty must not leak to the open field.
    assert selection_adjustment("HB Stretch", sit)[0] == 0.0


def test_explicit_fourth_quarter_clock_retains_lead_protection_on_normal_down():
    sit = situation(down=1, distance=10, clock_seconds=420)
    picture = evaluate_situation(sit)
    assert "clock_management" in picture["situations"]
    assert "late_lead_run" in {
        p["id"] for p in coordinator_components("HB Stretch", picture)
    }
    assert _objective(sit, {"state": "unknown"}, None) == "protect_late_lead"


def test_trailing_without_clock_does_not_claim_time_pressure():
    sit = situation(down=2, distance=8, score_us=7, score_them=21)
    picture = evaluate_situation(sit)
    assert "clock_management" not in picture["situations"]
    assert "late_trail_pass" not in {
        p["id"] for p in coordinator_components("Inside Cross", picture)
    }
    assert _objective(sit, {"state": "unknown"}, None) != "conserve_clock_while_trailing"


def test_trailing_with_clock_retains_pass_preference():
    sit = situation(down=2, distance=8, score_us=7, score_them=21,
                    clock_seconds=315)
    picture = evaluate_situation(sit)
    assert "clock_management" in picture["situations"]
    assert "late_trail_pass" in {
        p["id"] for p in coordinator_components("Inside Cross", picture)
    }
    assert _objective(sit, {"state": "unknown"}, None) == "conserve_clock_while_trailing"


def test_explicit_two_minute_flag_is_sufficient_even_without_clock_reading():
    sit = situation(down=1, distance=10, two_minute=True)
    picture = evaluate_situation(sit)
    assert picture["two_minute"] is True
    assert "clock_management" in picture["situations"]
    assert _objective(sit, {"state": "unknown"}, None) == "protect_late_lead"


def test_clock_from_extras_is_not_dropped():
    sit = situation(down=1, distance=10)
    sit.extras["clock_seconds"] = 500
    picture = evaluate_situation(sit)
    assert picture["clock_seconds"] == 500
    assert "clock_management" in picture["situations"]
    assert _objective(sit, {"state": "unknown"}, None) == "protect_late_lead"


def test_medium_critical_down_priority_remains_even_with_known_clock():
    sit = situation(down=4, distance=6, clock_seconds=420)
    assert _objective(sit, {"state": "unknown"}, None) == "convert_critical_first_down"
