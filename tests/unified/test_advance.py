"""Situation projection should not guess possession changes or double-advance yardlines."""
from __future__ import annotations

from cfb_coach.madden.unified.advance import next_situation, next_situation_line
from cfb_coach.madden.unified.inputs import parse_snap_line
from cfb_coach.madden.unified.types import GameState, OFF, DEF


def state(side=OFF, down=1, distance=10, yardline=41):
    return GameState(game_id="advance-test", seq=1, our_side=side, opponent_id="cpu",
                     down=down, distance=distance, yardline=yardline)


def expect(st, result, line):
    projected, reason = next_situation_line(st, result)
    assert reason == ""
    assert projected == line
    parsed = parse_snap_line(projected, None)
    assert parsed.side == st.our_side
    assert parsed.down == next_situation(st, result)["down"]
    assert parsed.distance == next_situation(st, result)["distance"]
    assert parsed.yardline == next_situation(st, result)["offense_yardline"]


def test_exact_user_example():
    expect(state(OFF, 1, 10, 41), "gain 2", "o 2&8 my 43")


def test_cpu_offense_result_moves_toward_our_goal_and_uses_opp_side():
    expect(state(DEF, 1, 10, 41), "gain 2", "d 2&8 opp 43")


def test_first_down_and_midfield_crossing():
    expect(state(OFF, 3, 7, 48), "gain 12", "o 1&10 opp 40")


def test_incomplete_and_negative_yardage():
    expect(state(OFF, 1, 10, 41), "incomplete", "o 2&10 my 41")
    expect(state(OFF, 2, 8, 43), "loss 3", "o 3&11 my 40")
    expect(state(OFF, 2, 8, 43), "sack -6", "o 3&14 my 37")


def test_goal_to_go_uses_remaining_distance():
    expect(state(OFF, 1, 4, 96), "gain 2", "o 2&2 opp 2")
    expect(state(OFF, 3, 2, 94), "gain 3", "o 1&3 opp 3")


def test_fourth_down_conversion_but_not_failure():
    expect(state(DEF, 4, 2, 43), "gain 3", "d 1&10 opp 46")
    projection = next_situation(state(DEF, 4, 4, 43), "incomplete")
    assert not projection["automatic"] and "Fourth-down" in projection["reason"]


def test_turnovers_scores_unknown_and_unclear_sacks_require_manual():
    s = state(OFF, 1, 10, 41)
    for outcome in ("td", "int", "fumble lost", "stop", "convert", "sack", "gain 2 td"):
        assert next_situation_line(s, outcome)[0] is None
    assert next_situation_line(state(OFF, 2, 3, 99), "gain 2")[0] is None
    assert next_situation_line(state(OFF, 2, 3, 2), "loss 3")[0] is None


def test_same_base_twice_does_not_accumulate_yards():
    s = state(OFF, 1, 10, 41)
    assert next_situation(s, "gain 2")["offense_yardline"] == 43
    assert next_situation(s, "gain 5")["offense_yardline"] == 46
