"""Defensive V2 model-built pregame book tests: no fixed mandatory rotation."""
from __future__ import annotations

import pytest
from cfb_coach.madden.model import defense_v2_pilot as v2
from cfb_coach.madden.model import defense_v2_playbook as designer


def model():
    rows = [
        v2.Snap("g1", "cpu", "2026-11-01", 1, "Nickel Over",
                "Cover 3 Sky", "single_high", "early", "cross", True),
        v2.Snap("g2", "cpu", "2026-11-02", 1, "Dime",
                "Cover 2 Man", "man", "long", "vert", False),
    ]
    return v2.fit(rows)


def test_stock_formations_can_be_selected_without_installing():
    catalogue = {
        "A": {"Nickel Over": ["Cover 3 Sky"], "Dime": ["Cover 2 Man"]},
        "B": {"Dollar": ["Cover 4 Quarters"], "4-3": ["Cover 2 Invert"]},
    }
    proposal = designer.propose(catalogue=catalogue, model=model(), max_formations=3)
    assert 1 <= proposal["n_formations"] <= 3
    assert proposal["n_plays"] >= proposal["n_formations"]
    assert proposal["installed"] is False
    assert proposal["requires_physical_editor_installation"] is True
    for formation, plays in proposal["formations"].items():
        assert set(plays) == set(catalogue[proposal["formation_sources"][formation]][formation])
    assert len(proposal["proposal_fingerprint"]) == 64


def test_excessive_formation_limit_and_invalid_model_rejected():
    with pytest.raises(ValueError):
        designer.propose(catalogue={"A": {"Dime": ["Cover 2 Man"]}},
                         model=model(), max_formations=16)
    with pytest.raises(ValueError):
        designer.propose(catalogue={"A": {"Dime": ["Cover 2 Man"]}},
                         model={}, max_formations=1)


def test_duplicate_formation_from_multiple_books_only_used_once():
    catalog = {"A": {"Nickel Over": ["Cover 3 Sky"]},
               "B": {"Nickel Over": ["Cover 2 Man"]}}
    proposal = designer.propose(catalogue=catalog, model=model(), max_formations=15)
    assert proposal["n_formations"] == 1


def test_unknown_calls_not_silently_treated_as_proven():
    with pytest.raises(ValueError):
        designer.propose(catalogue={"A": {"Mystery": ["Obscure New Call"]}},
                         model=model(), max_formations=15)



def test_model_chooses_multiple_formations_when_distinct_threats_require_them(monkeypatch):
    from cfb_coach.madden.model import defense_v2_pilot as v2

    def outcome(model, context, formation, play, family, concept):
        run_front = formation.startswith("Goal Line")
        p = 0.15 if (run_front and concept == "run") or (not run_front and concept == "vert") else 0.8
        return p, {}
    monkeypatch.setattr(v2, "offense_risk", outcome)
    cat = {
        "Heavy": {"Goal Line 5-3": ["Cover 0"]},
        "Pass": {"Dime Normal": ["Cover 2 Man"]},
    }
    book = designer.propose(catalogue=cat, model=model(), max_formations=15)
    assert set(book["formations"]) == {"Goal Line 5-3", "Dime Normal"}
    assert book["n_formations"] == 2
    assert book["installed"] is False
