"""Synthetic scenarios: behaviour and consistency only, not evidence of better play."""

from cfb_coach.madden.model.defense_strategy_eval import (
    EVIDENCE_WEIGHT,
    SITUATION_THREAT_PRIOR,
    assess_candidates,
    threat_distribution,
)
from cfb_coach.madden.model.strategic_context import GameHistory, build_pre_snap_context
from cfb_coach.situation import Situation

BOOK = [
    {"formation": "Goal Line 5-3-3", "play": "Cover 1 Press"},
    {"formation": "4-3 Over", "play": "Cover 3 Sky"},
    {"formation": "Nickel 2-4-5", "play": "Cover 4 Quarters"},
    {"formation": "Nickel 2-4-5", "play": "Tampa 2"},
    {"formation": "Dime 1-3-7", "play": "Cover 6"},
    {"formation": "Nickel 3-3-5", "play": "Mid Blitz 0"},
    {"formation": "Nickel 2-4-5", "play": "Cover 1 Robber"},
]
RUN, VERT, CROSS, FLOOD = "HB Inside Zone", "Four Verticals", "Mesh", "Flood"


def _run(down, distance, yardline, concepts=(), **kw):
    sit = Situation(raw="t", side="defense", down=down, distance=distance, yardline=yardline, **kw)
    ctx = build_pre_snap_context(sit, history=GameHistory(opponent_observed_concepts=tuple(concepts)),
                                 legal_candidates=BOOK)
    return assess_candidates(ctx, BOOK, sit=sit)


def _by_play(a):
    return {c.play: c for c in a.candidates}


def test_first_and_ten_against_a_run_heavy_opponent():
    a = _run(1, 10, 60, [RUN] * 8)
    assert a.threat["run"] > 0.5
    c = _by_play(a)
    assert not c["Cover 6"].legal
    assert c["Cover 3 Sky"].rank < c["Cover 4 Quarters"].rank
    assert a.candidates[0].front != "light"


def test_third_and_long_against_a_pass_heavy_opponent():
    a = _run(3, 12, 65, [VERT, VERT, FLOOD, VERT, VERT, FLOOD, VERT, VERT])
    c = _by_play(a)
    assert a.threat["vert"] > a.threat["run"]
    assert c["Cover 4 Quarters"].rank < c["Mid Blitz 0"].rank
    assert c["Mid Blitz 0"].explosive_exposure > c["Cover 4 Quarters"].explosive_exposure


def test_goal_line_against_a_frequent_runner():
    a = _run(2, 2, 2, [RUN] * 6)
    c = _by_play(a)
    assert a.situation_key == "goal_line"
    assert not c["Cover 6"].legal
    assert a.candidates[0].front == "heavy"
    assert all(x.legal for x in a.candidates if x.rank < c["Cover 6"].rank)


def test_red_zone_against_quick_passing():
    a = _run(2, 8, 15, [CROSS, "Slants", CROSS, "Drive", CROSS, "Slants"])
    assert a.situation_key == "red_zone"
    assert a.candidates[0].coverage_family in ("man", "cover2")


def test_two_minute_defense_while_leading_guards_against_explosives():
    lead = _run(1, 10, 50, score_us=24, score_them=17, extras={"quarter": 4, "clock_seconds": 90})
    trail = _run(1, 10, 50, score_us=17, score_them=24, extras={"quarter": 4, "clock_seconds": 90})
    assert lead.situation_key == "two_minute"
    gap = lambda a: _by_play(a)["Cover 4 Quarters"].score - _by_play(a)["Mid Blitz 0"].score  # noqa: E731
    assert gap(lead) > gap(trail) > 0
    assert any("leading late" in x.note for x in _by_play(lead)["Cover 4 Quarters"].components)


def test_tendency_change_within_a_game_follows_recent_plays():
    early_runs = [RUN] * 8 + [VERT, FLOOD, VERT, CROSS, VERT, FLOOD]
    early_pass = [VERT, FLOOD, VERT, CROSS, VERT, FLOOD] + [RUN] * 8
    a = _run(1, 10, 60, early_runs)
    b = _run(1, 10, 60, early_pass)
    assert a.threat["run"] < b.threat["run"]


def test_no_history_uses_the_situation_prior_and_says_so():
    a = _run(1, 10, 60)
    assert a.threat_sources["game_plays"] == 0
    for f, v in SITUATION_THREAT_PRIOR["early"].items():
        assert abs(a.threat[f] - v / sum(SITUATION_THREAT_PRIOR["early"].values())) < 1e-9
    assert any("no opponent history" in u for u in a.uncertainty)


def test_no_call_is_called_measured_without_confirmed_executions():
    a = _run(1, 10, 60, [RUN] * 4)
    assert any("no confirmed defensive executions" in u for u in a.uncertainty)
    assert all(c.evidence != "measured" for x in a.candidates for c in x.components)
    assert all(c.evidence in EVIDENCE_WEIGHT for x in a.candidates for c in x.components)


def test_measured_outcomes_enter_only_with_enough_confirmed_executions():
    sit = Situation(raw="t", side="defense", down=1, distance=10, yardline=60)
    ctx = build_pre_snap_context(sit)
    few = assess_candidates(ctx, BOOK, sit=sit, measured={"man": {"n": 5, "offense_success": 0.2}})
    many = assess_candidates(ctx, BOOK, sit=sit, measured={"man": {"n": 40, "offense_success": 0.2}})
    man_few = _by_play(few)["Cover 1 Robber"]
    man_many = _by_play(many)["Cover 1 Robber"]
    assert not any(c.evidence == "measured" for c in man_few.components)
    assert any(c.evidence == "measured" for c in man_many.components)
    assert man_many.score > man_few.score


def test_illegal_calls_never_outrank_legal_ones_and_explanations_exist():
    for a in (_run(1, 10, 60, [RUN] * 8), _run(2, 2, 2, [RUN] * 6)):
        ranks = [(c.legal, c.rank) for c in a.candidates]
        worst_legal = max(r for ok, r in ranks if ok)
        assert all(r > worst_legal for ok, r in ranks if not ok)
        assert all(c.versus_next for c in a.candidates[:-1])


def test_unrecognized_coverage_never_outranks_assessed_calls():
    book = BOOK + [{"formation": "46 Normal", "play": "Engage Eight"}]
    sit = Situation(raw="t", side="defense", down=2, distance=4, yardline=56)
    a = assess_candidates(build_pre_snap_context(sit), book, sit=sit)
    unknown = _by_play(a)["Engage Eight"]
    assert not unknown.assessed
    assert all(c.rank < unknown.rank for c in a.candidates if c.legal and c.assessed)


def test_opponent_profile_adjustment_is_labeled_observed():
    sit = Situation(raw="t", side="defense", down=1, distance=10, yardline=60)
    profile = {"n": 30, "run_share": 0.8, "shrink": 0.5, "scope": "all downs", "concepts": {"run": 0.8, "cross": 0.2}}
    a = assess_candidates(build_pre_snap_context(sit), BOOK, sit=sit, opponent_profile=profile)
    comps = [c for x in a.candidates for c in x.components if c.name == "opponent_profile"]
    assert comps and all(c.evidence == "observed_opponent" for c in comps)


def test_latency_for_a_large_installed_book():
    sit = Situation(raw="t", side="defense", down=3, distance=7, yardline=40)
    ctx = build_pre_snap_context(sit, history=GameHistory(opponent_observed_concepts=(RUN, VERT) * 10))
    big = BOOK * 12
    assess_candidates(ctx, big, sit=sit)
    a = assess_candidates(ctx, big, sit=sit)
    assert len(a.candidates) == len(big)
    assert a.latency_ms < 50.0


def test_threat_distribution_sums_to_one():
    sit = Situation(raw="t", side="defense")
    t, _ = threat_distribution(build_pre_snap_context(sit))
    assert abs(sum(t.values()) - 1.0) < 1e-9
