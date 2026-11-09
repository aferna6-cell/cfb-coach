"""The no-macro ML path may recommend only verified, bounded Xbox hot routes."""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_taxonomy as tx
from cfb_coach.madden.model.offense_live_single_route import maybe_add_single_route


def _structure(tier="diagram_verified"):
    return pf.PlayStructure(
        formation="Gun Test", play="Mesh", routes=(
            ("WR1", tx.RouteSpec("corner", "left", "deep"), tier),
            ("HB1", tx.RouteSpec("flat", "right", "shallow"), tier),
        ),
        skill_slots=("WR1", "HB1"), personnel_tier="verified",
        hint={"play_type": "pass"}, read_order=("WR1", "HB1"),
    )


def _decided(actions, executable=True):
    return od.Decision(
        formation="Gun Test", play="Mesh",
        adjustment="HOT ROUTE", probability=0.61,
        stock_probability=0.52, interval=(0.48, 0.72),
        actions=tuple(actions), executable_now=executable, latency_ms=7.0,
        final_structure=_structure(),
    )


def _inputs():
    joint = {"kind": "none", "joint": {"formation": "Gun Test", "play": "Mesh"}}
    ctx = SimpleNamespace(
        applied=True, art=SimpleNamespace(residual_active=False),
        priors=True, mix={"cover_2_unspecified": 0.8, "unknown": 0.2},
        learned={},
    )
    sit = SimpleNamespace(
        down=3, distance=10, yardline=48,
        extras={"play_clock_seconds": 25},
    )
    return joint, ctx, sit


def test_known_receiver_single_route_can_be_proposed_if_evidence_and_controls_pass():
    joint, ctx, sit = _inputs()
    action = ex.PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"})
    with (
        mock.patch.object(sm, "structure_any", return_value=_structure()),
        mock.patch.object(od, "decide", return_value=_decided([action])) as decide,
        mock.patch.object(ex, "validate_plan", return_value=[]),
    ):
        actual = maybe_add_single_route(
            joint, route_ctx=ctx, sit=sit, opponent_id="cpu",
            remaining_budget_ms=40, confirmations={"verified": True},
        )
    assert actual["kind"] == "adjustment"
    assert actual["adjustment"]["target"] == "WR1"
    assert actual["adjustment"]["route"] == "Slant"
    assert "Press Y" in actual["adjustment"]["buttons"]
    assert actual["adjustment"]["not_a_macro"] is True
    assert actual["scores_are"] == "model_structural_extrapolation_not_learned_action_effect"
    assert decide.call_args.kwargs["budget_ms"] <= 12.0


def test_partial_diagram_has_no_verified_xbox_receiver_target():
    joint, ctx, sit = _inputs()
    with (
        mock.patch.object(sm, "structure_any", return_value=_structure("partial")),
        mock.patch.object(od, "decide", side_effect=AssertionError("must not score")),
    ):
        result = maybe_add_single_route(
            joint, route_ctx=ctx, sit=sit, opponent_id="cpu",
            remaining_budget_ms=40,
        )
    assert result == joint


def test_no_time_budget_preserves_unchanged_play():
    joint, ctx, sit = _inputs()
    result = maybe_add_single_route(
        joint, route_ctx=ctx, sit=sit, opponent_id="cpu",
        remaining_budget_ms=6,
    )
    assert result == joint


def test_no_route_model_or_priors_means_no_hot_route_prediction():
    joint, ctx, sit = _inputs()
    ctx.priors = False
    result = maybe_add_single_route(
        joint, route_ctx=ctx, sit=sit, opponent_id="cpu",
        remaining_budget_ms=40,
    )
    assert result == joint


def test_unverified_controls_refuse_adjustment_even_when_model_favors_it():
    joint, ctx, sit = _inputs()
    action = ex.PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"})
    with (
        mock.patch.object(sm, "structure_any", return_value=_structure()),
        mock.patch.object(od, "decide", return_value=_decided([action])),
        mock.patch.object(ex, "validate_plan", return_value=["route has not been editor verified"]),
    ):
        result = maybe_add_single_route(
            joint, route_ctx=ctx, sit=sit, opponent_id="cpu",
            remaining_budget_ms=40,
        )
    assert result == joint


def test_conflicting_multi_receiver_payload_is_not_silently_executed():
    joint, ctx, sit = _inputs()
    actions = [
        ex.PlannedAction("off.hot_route", {"target": "WR1", "route": "Slant"}),
        ex.PlannedAction("off.hot_route", {"target": "HB1", "route": "Flat"}),
    ]
    with (
        mock.patch.object(sm, "structure_any", return_value=_structure()),
        mock.patch.object(od, "decide", return_value=_decided(actions)),
    ):
        result = maybe_add_single_route(
            joint, route_ctx=ctx, sit=sit, opponent_id="cpu",
            remaining_budget_ms=40,
        )
    assert result == joint
