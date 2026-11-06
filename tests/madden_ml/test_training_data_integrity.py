"""Regression checks: outcome labels and executed-vs-recommended separation.

Pure synthetic examples: no access to the user's private Madden database.
"""

from __future__ import annotations

import pytest

from cfb_coach.madden.model import dataset, experimental_model


def _row(*, side: str = "offense", result: str = "td", yards=None,
         executed: bool = True, play: str = "Mesh") -> dict:
    return {
        "session_id": "synthetic-game",
        "snap_id": f"synthetic-{side}-{result}",
        "side": side,
        "down": 1,
        "distance": 10,
        "formation": "Gun Bunch",
        "play": play,
        "result": result,
        "yards": yards,
        "executed_status": "identified" if executed else "unknown",
        "executed_verification": "verified" if executed else "unknown",
        "executed_formation": "Gun Bunch" if executed else None,
        "executed_play": play if executed else None,
    }


@pytest.mark.parametrize("side,result,yards,expected", [
    ("offense", "td", None, "true"),
    ("offense", "td", 0, "true"),
    ("defense", "td", None, "false"),
    ("defense", "td", 0, "false"),
    ("offense", "convert", None, "true"),
    ("defense", "convert", None, "false"),
    ("offense", "stop", None, "false"),
    ("defense", "stop", None, "true"),
    ("offense", "incomplete", None, "false"),
    ("defense", "incomplete", None, "true"),
    ("offense", "int", 8, "false"),  # Never mistake return yards for a gain.
    ("defense", "int", 8, "true"),
    ("offense", "sack", None, "false"),
    ("defense", "sack", None, "true"),
    ("offense", "-2", None, "false"),
    ("defense", "-2", None, "true"),
    ("offense", "+4", None, "true"),   # 1st-and-10: 40% success threshold.
    ("offense", "+3", None, "false"),
])
def test_outcome_label_is_situation_aware_and_terminal_first(
    side: str, result: str, yards: int | None, expected: str
) -> None:
    row = dataset._normalize_row(_row(side=side, result=result, yards=yards),
                                 provenance="synthetic.test")
    assert row["success"] == expected
    assert row["label_available"] is True


def test_verified_touchdown_is_eligible_supervised_training() -> None:
    row = dataset._normalize_row(_row(result="td"), provenance="synthetic.test")
    assert row["eligibility"] == dataset.ELIGIBILITY_VERIFIED_EXECUTION
    assert row["supervised_eligible"] is True
    assert row["action_play"] == "Mesh"
    assert row["success"] == "true"


def test_unverified_touchdown_never_credits_recommended_play() -> None:
    row = dataset._normalize_row(_row(result="td", executed=False),
                                 provenance="synthetic.test")
    assert row["success"] == "true"  # We do know the offense scored.
    assert row["supervised_eligible"] is False
    assert row["eligibility"] == dataset.ELIGIBILITY_OUTCOME_UNCERTAIN_EXEC
    assert row["action_play"] is None


def test_unknown_outcome_is_not_invented_as_success() -> None:
    row = dataset._normalize_row(_row(result="unrecognized event"),
                                 provenance="synthetic.test")
    assert row["success"] == "unknown"
    assert row["supervised_eligible"] is False


def test_uncertain_result_affects_context_not_specific_play_learning() -> None:
    verified = dataset._normalize_row(_row(result="incomplete", play="Mesh"),
                                      provenance="synthetic.test")
    uncertain = dataset._normalize_row(
        _row(result="td", play="Y Cross", executed=False),
        provenance="synthetic.test",
    )
    model = experimental_model.train_experimental([verified, uncertain])
    assert model.n_supervised == 1
    assert model.n_discounted_priors == 1
    # Only the verified Mesh execution can teach which concept was used.
    assert "mesh" in model.play_concept
    assert "cross" not in model.play_concept
    assert "cross|1_long" not in model.concept_down_distance
    # The uncertain TD still contributes reduced-weight situation evidence.
    assert "1_long" in model.down_distance
