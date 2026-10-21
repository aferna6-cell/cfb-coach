"""Week 3 five-priority regression gates. Synthetic; no user game DB is modified."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

from cfb_coach import live_server
from cfb_coach.madden.model import football_situation, offense_joint_decision
from cfb_coach.madden.model.joint_override_guard import review_override


def test_short_yardage_competing_options_are_not_all_runs() -> None:
    sit = SimpleNamespace(down=3, distance=2, yardline=98, extras={},
                          red_zone=True, goal_line=True, two_minute=False,
                          score_us=7, score_them=7)
    options = ["HB Dive", "Mesh", "PA Boot Y Leak", "RPO Alert Bubble"]
    details = {p: football_situation.selection_adjustment(p, sit) for p in options}
    assert all(delta > 0 for delta, _ in details.values())
    assert "run" in details["HB Dive"][1]
    assert "quick pass" in details["Mesh"][1]
    assert "play action" in details["PA Boot Y Leak"][1]
    assert "RPO" in details["RPO Alert Bubble"][1]


def test_joint_no_action_override_requires_margin() -> None:
    anchor = {"formation": "Pistol Trips", "play": "X Spot"}
    reps = [
        {"formation": "Pistol Trips", "play": "X Spot", "joint_score": 0.70,
         "plan": {"plan_id": "NO_ADJUSTMENT"}},
        {"formation": "Pistol Trips", "play": "HB Draw", "joint_score": 0.71,
         "plan": {"plan_id": "NO_ADJUSTMENT"}},
    ]
    result, audit = review_override(chosen=reps[1], anchor=anchor, representatives=reps)
    assert result is reps[0]
    assert audit["guard_kept_anchor"] is True
    assert audit["joint_gain"] == 0.01
    assert audit["score_claim"] == "model_proxy_not_counterfactual_outcome"

    reps[1]["joint_score"] = 0.78
    result, audit = review_override(chosen=reps[1], anchor=anchor, representatives=reps)
    assert result is reps[1]
    assert audit["guard_kept_anchor"] is False
    reps[1]["joint_score"] = 0.71
    reps[1]["plan"] = {"plan_id": "VERIFIED_ARMED_MACRO"}
    result, audit = review_override(chosen=reps[1], anchor=anchor, representatives=reps)
    assert result is reps[1]


def test_game_clock_is_stamped_before_model_and_not_a_play_clock() -> None:
    source = inspect.getsource(live_server.LivePlayController.call_only)
    assert source.index("self._stamp_live_context(sit, game_clock") < source.index("call = self._make(sit)")
    source2 = inspect.getsource(live_server.LivePlayController.result_and_call)
    assert source2.index("self._stamp_live_context(sit, game_clock") < source2.index("call = self._make(sit)")
    assert 'extras["clock_seconds"] = seconds' in inspect.getsource(
        live_server.LivePlayController._stamp_live_context
    )
    block = inspect.getsource(offense_joint_decision._clock_blocks)
    assert 'evaluation.get("play_clock_seconds")' in block
    assert 'evaluation.get("clock_seconds")' not in block


def test_compact_pad_confirmation_is_explicit_and_resets() -> None:
    html = inspect.getsource(live_server.render_live_html)
    assert 'id="btn-ran-recommended"' in html
    assert 'id="btn-ran-different"' in html
    assert 'execVerifyTouched = true;' in html
    assert 'setExecSelection("used_recommended")' in html
    assert 'setExecSelection("used_different")' in html
    assert 'execVerifyTouched = false;' in html
    assert 'verify_touched: execVerifyTouched' in html


def test_joint_audit_is_attached_to_model_record() -> None:
    from cfb_coach.madden.model import experimental_live
    joint_source = inspect.getsource(offense_joint_decision.choose_joint_action)
    assert 'review_override(' in joint_source
    assert 'decision["override_audit"] = override_audit' in joint_source
    applied = inspect.getsource(experimental_live.apply_experimental_offense)
    assert '"joint_override_audit": joint.get("override_audit")' in applied


def test_historical_joint_report_keeps_unseen_scores_unknown() -> None:
    import json
    import sqlite3

    from scripts.madden_ml.review_joint_history import summarize

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE ml_decisions (session_id TEXT, model_version TEXT, "
        "decision_json TEXT, policy_source TEXT, id INTEGER PRIMARY KEY)"
    )
    def item(candidates):
        return json.dumps({
            "experimental_offense": {
                "selection_audit": {"top_selected": ["Gun", "Mesh"]},
                "offense_action": {
                    "kind": "none",
                    "joint": {"formation": "Gun", "play": "Draw"},
                    "candidates": candidates,
                },
            }
        })
    conn.execute(
        "INSERT INTO ml_decisions (session_id,model_version,policy_source,decision_json) "
        "VALUES (?,?,?,?)",
        ("game-one", "offense.52.empirical", "model", item([
            {"formation": "Gun", "play": "Mesh", "id": "NO_ADJUSTMENT", "score": 0.55},
            {"formation": "Gun", "play": "Draw", "id": "NO_ADJUSTMENT", "score": 0.56},
        ])),
    )
    conn.execute(
        "INSERT INTO ml_decisions (session_id,model_version,policy_source,decision_json) "
        "VALUES (?,?,?,?)",
        ("game-two", "offense.52.empirical", "model", item([])),
    )
    report = summarize(conn)
    counts = report["counts"]
    assert counts["inspectable"] == 2
    assert counts["no_adjustment_changed_play"] == 2
    assert counts["score_pair_retained"] == 1
    assert counts["score_pair_not_retained"] == 1
    assert counts["below_new_margin_with_retained_scores"] == 1
    assert len(report["examples"]) == 1
