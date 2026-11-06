import pytest

from cfb_coach.madden.model.strategic_context import (
    POST_SNAP_FIELDS,
    PostSnapLeak,
    assert_pre_snap,
    build_pre_snap_context,
    history_from_snaps,
    opponent_context,
)
from cfb_coach.situation import Situation


def _sit(**kw):
    return Situation(raw="t", **kw)


def test_defense_field_position_is_distance_to_the_goal_we_defend():
    ctx = build_pre_snap_context(_sit(side="defense", down=1, distance=10, yardline=8))
    assert ctx.yards_to_goal.value == 8
    assert ctx.red_zone.value is True
    assert ctx.backed_up.value is False


def test_offense_field_position_counts_from_our_goal():
    ctx = build_pre_snap_context(_sit(side="offense", down=1, distance=10, yardline=8))
    assert ctx.yards_to_goal.value == 92
    assert ctx.red_zone.value is False
    assert ctx.backed_up.value is True


def test_missing_values_stay_missing_and_are_listed():
    ctx = build_pre_snap_context(_sit(side="defense"))
    for name in ("down", "distance", "yards_to_goal", "quarter", "clock_seconds", "score_diff",
                 "red_zone", "two_minute", "short_yardage", "legal_candidates"):
        cv = getattr(ctx, name)
        assert cv.status == "missing" and cv.value is None, name
        assert name in ctx.missing
    assert "opponent_run_share" in ctx.missing
    f = ctx.as_features()
    assert f["down_missing"] == 1.0 and f["distance_missing"] == 1.0
    assert f["opp_run_share_missing"] == 1.0
    assert ctx.inconsistencies == ()


def test_partial_context_derives_only_what_it_can():
    ctx = build_pre_snap_context(_sit(side="defense", down=3, distance=8, extras={"quarter": 4}))
    assert ctx.long_yardage.value is True
    assert ctx.two_minute.status == "missing"
    assert ctx.red_zone.status == "missing"


def test_explicit_flags_used_when_field_position_missing():
    ctx = build_pre_snap_context(_sit(side="defense", goal_line=True, two_minute=True))
    assert ctx.goal_line.value is True and ctx.goal_line.source == "explicit_flag"
    assert ctx.red_zone.value is True
    assert ctx.two_minute.value is True


def test_two_minute_and_score_from_clock_and_scores():
    ctx = build_pre_snap_context(_sit(side="defense", down=1, distance=10, yardline=60, score_us=21,
                                      score_them=17, extras={"quarter": 4, "clock_seconds": 95}))
    assert ctx.two_minute.value is True
    assert ctx.score_diff.value == 4


def test_one_score_is_not_a_score_differential():
    ctx = build_pre_snap_context(_sit(side="defense", score_us=7))
    assert ctx.score_diff.status == "missing"


@pytest.mark.parametrize("kw, problem", [
    (dict(down=5, distance=10, yardline=50), "down_out_of_range"),
    (dict(down=1, distance=0, yardline=50), "distance_out_of_range"),
    (dict(down=1, distance=10, yardline=0), "yards_to_goal_out_of_range"),
    (dict(down=1, distance=10, yardline=50, extras={"quarter": 9}), "quarter_out_of_range"),
    (dict(down=1, distance=10, yardline=50, extras={"clock_seconds": 2000}), "clock_out_of_range"),
    (dict(down=1, distance=10, yardline=50, score_us=-3, score_them=0), "negative_score"),
    (dict(down=1, distance=10, yardline=6), "exceeds_yards_to_goal"),
    (dict(down=1, distance=10, yardline=40, red_zone=True), "red_zone_flag_contradicts"),
])
def test_inconsistent_inputs_are_reported_not_trusted(kw, problem):
    ctx = build_pre_snap_context(_sit(side="defense", **kw))
    assert any(problem in p for p in ctx.inconsistencies), ctx.inconsistencies


def test_out_of_range_values_are_dropped_not_clamped():
    ctx = build_pre_snap_context(_sit(side="defense", down=7, distance=10, yardline=50))
    assert ctx.down.value is None and ctx.down.status == "missing"


def test_live_look_versus_previous_snap():
    live = build_pre_snap_context(_sit(side="defense", concept_hint="Mesh", concept_source="live"))
    last = build_pre_snap_context(_sit(side="defense", concept_hint="Mesh", concept_source="last"))
    assert live.live_look.status == "known"
    assert last.live_look.status == "inferred"


def test_history_keeps_our_calls_apart_from_opponent_observations():
    earlier = [
        {"side": "defense", "formation": "Nickel", "play": "Cover 3", "concept_seen": "HB Inside Zone"},
        {"side": "defense", "formation": "Dime", "play": "Cover 2", "concept_seen": "Mesh"},
        {"side": "defense", "formation": "Nickel", "play": "Cover 1", "concept_seen": None},
    ]
    h = history_from_snaps(earlier, side="defense")
    assert len(h.our_previous_calls) == 3
    assert h.opponent_observed_concepts == ("HB Inside Zone", "Mesh")
    assert h.opponent_run_pass == (0, 1)
    f = build_pre_snap_context(_sit(side="defense"), history=h).as_features()
    assert f["game_opp_pass_share"] == 0.5 and f["last_opp_pass"] == 1.0


def test_our_call_is_never_opponent_evidence():
    h = history_from_snaps([{"side": "defense", "formation": "Goal Line", "play": "HB Dive"}], side="defense")
    assert h.opponent_observed_concepts == ()
    assert build_pre_snap_context(_sit(side="defense"), history=h).as_features()["last_opp_missing"] == 1.0


def test_opponent_profile_sources():
    assert opponent_context("someone", None).run_share.status == "missing"
    pooled = opponent_context("someone", {"n": 120, "run_share": 0.4, "scope": "pooled human opponents"})
    assert pooled.run_share.status == "inferred" and pooled.own_plays == 0 and pooled.pooled_plays == 120
    blended = opponent_context("someone", {"n": 6, "run_share": 0.5, "scope": "his 6 plays + league prior"})
    assert blended.run_share.status == "inferred" and blended.own_plays == 6
    own = opponent_context("someone", {"n": 60, "run_share": 0.7, "scope": "all downs"})
    assert own.run_share.status == "derived"
    assert opponent_context(None, None).kind == "unknown"


def test_post_snap_fields_rejected_as_inputs():
    assert_pre_snap({"down": 1, "distance": 10})
    for name in ("concept_seen", "result", "executed_play"):
        assert name in POST_SNAP_FIELDS
        with pytest.raises(PostSnapLeak):
            assert_pre_snap({"down": 1, name: "x"})


def test_context_from_snap_reads_only_pre_snap_columns():
    from cfb_coach.madden.model.strategic_context import context_from_snap

    row = {"side": "defense", "down": 2, "distance": 7, "yardline": 81, "quarter": 3,
           "situation_raw": "2&7 opp 19 mesh", "score_for": 14, "score_against": 10,
           "concept_seen": "HB Inside Zone", "result": "td", "yards": 81}
    earlier = [{"side": "defense", "formation": "Nickel", "play": "Cover 3", "concept_seen": "Mesh"}]
    ctx = context_from_snap(row, earlier=earlier)
    assert ctx.yards_to_goal.value == 81 and ctx.backed_up.value is False
    assert ctx.score_diff.value == 4 and ctx.quarter.value == 3
    assert ctx.live_look.status == "inferred"
    assert ctx.history.opponent_observed_concepts == ("Mesh",)
    assert ctx.as_features()["last_opp_pass"] == 1.0


def test_features_contain_no_post_snap_names():
    ctx = build_pre_snap_context(_sit(side="defense", down=2, distance=4, yardline=30))
    assert not set(ctx.as_features()) & POST_SNAP_FIELDS
    assert ctx.legal_candidates.status == "missing"
    with_cands = build_pre_snap_context(_sit(side="defense"), legal_candidates=["a", "b"])
    assert with_cands.legal_candidates.value == 2
