"""Week 4: goal-to-go medium yardage and legal adjustment diagnostics.

Synthetic situations only. Never mutate a real game, artifact or database.
"""
from types import SimpleNamespace

from cfb_coach.madden.model.football_situation import (
    coordinator_components, evaluate_situation, selection_adjustment,
)
from cfb_coach.madden.model.offense_joint_decision import choose_joint_action


def situation(*, down=3, distance=6, yardline=94, goal_line=True, red_zone=True):
    return SimpleNamespace(
        side="offense", down=down, distance=distance, yardline=yardline,
        red_zone=red_zone, goal_line=goal_line, two_minute=False, quarter=2,
        score_us=7, score_them=3, coverage_hint=None,
        coverage_source="none", extras={},
    )


def test_goal_to_go_third_and_six_run_is_penalized_not_rewarded():
    sit = situation()
    run, reason = selection_adjustment("HB Mid Zone", sit)
    quick, _ = selection_adjustment("Inside Cross", sit)
    assert run == -0.065
    assert "third/fourth-and-4-to-6" in reason
    assert quick == 0.0
    ev = evaluate_situation(sit)
    run_parts = coordinator_components("HB Mid Zone", ev)
    assert "goal_line_run" not in {part["id"] for part in run_parts}


def test_third_and_short_goal_line_run_remains_competitive():
    sit = situation(distance=2, yardline=98)
    run, _ = selection_adjustment("HB Dive", sit)
    assert run > 0
    ev = evaluate_situation(sit)
    assert "goal_line_run" in {
        part["id"] for part in coordinator_components("HB Dive", ev)
    }


def test_first_down_goal_line_run_still_fits():
    sit = situation(down=1, distance=9, yardline=91)
    run, _ = selection_adjustment("HB Mid Zone", sit)
    assert run == 0.0
    ev = evaluate_situation(sit)
    assert "goal_line_run" in {
        part["id"] for part in coordinator_components("HB Mid Zone", ev)
    }


def test_medium_yardage_penalty_only_in_scoring_territory():
    normal = situation(down=3, distance=6, yardline=50, goal_line=False, red_zone=False)
    run, _ = selection_adjustment("HB Mid Zone", normal)
    assert run == 0.0
    red = situation(down=4, distance=5, yardline=81, goal_line=False, red_zone=True)
    value, _ = selection_adjustment("HB Mid Zone", red)
    assert value == -0.065


def test_fourth_and_long_existing_guard_unchanged():
    sit = situation(down=4, distance=9, yardline=94)
    run, _ = selection_adjustment("HB Mid Zone", sit)
    assert run <= -0.45
    ev = evaluate_situation(sit)
    assert "goal_line_run" not in {
        part["id"] for part in coordinator_components("HB Mid Zone", ev)
    }


def test_joint_action_reports_why_no_actions_can_fire():
    sit = situation()
    book = {"Pistol Trips": ["HB Dive", "Inside Cross"]}
    ranked = [
        {"formation": "Pistol Trips", "play": "HB Dive",
         "selection_score": 0.50, "probability": 0.50, "uncertainty": 1.0},
        {"formation": "Pistol Trips", "play": "Inside Cross",
         "selection_score": 0.52, "probability": 0.52, "uncertainty": 1.0},
    ]
    result = choose_joint_action(
        ranked=ranked, anchor=ranked[0], sit=sit, book=book,
        active=[], db=None, allow_macros=False,
        use_knowledge=False, use_strategy=False,
        use_opponent_learning=False, use_expert_learning=False,
        use_diversity=False,
    )
    audit = result["action_availability"]
    assert audit["sourced_action_candidates"] == 0
    assert audit["observed_current_coverage"] is False
    assert audit["armed_macro_count"] == 0
    assert audit["reason"] == "no_current_coverage_and_no_armed_macro"
    assert audit["no_unsupported_action_invented"] is True
    assert result["kind"] == "none"
    assert result["plans_compared"] == 2
