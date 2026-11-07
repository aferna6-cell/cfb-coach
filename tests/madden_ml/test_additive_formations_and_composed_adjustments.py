"""Issue #120 (owner corrections): additive formations until the 15 cap; composed, unbounded-count adjustment search.

Search tests stub the LEARNED gain so they exercise search, legality, conflicts and timing, not model quality.
"""
from __future__ import annotations

import pytest

from cfb_coach.madden import catalog
from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_presnap_advice as adv
from tests.madden_ml.test_offense_intel_model import ANNOTATED, SIT, art, rows  # noqa: F401


# ------------------------------------------------------------------------------------------------------------ formations
def _cat(n_installed):
    book = {f"F{i}": [f"P{i}a", f"P{i}b", f"P{i}c"] for i in range(30)}
    book["Gun Tight Y Off"] = [f"GTYO{i}" for i in range(24)]
    return {"Shotgun Classic": {k: v for k, v in book.items()}}, [f"F{i}" for i in range(n_installed)]


def _active(cat, installed):
    return {"formations": {f: list(cat["Shotgun Classic"][f]) for f in installed}, "formation_sources": {f: "Shotgun Classic" for f in installed}}


def _ranked(cat, scores):
    return [{"formation": f, "source_book": "Shotgun Classic", "score": scores.get(f, 0.0), "plays": cat["Shotgun Classic"][f]} for f in cat["Shotgun Classic"]]


@pytest.mark.parametrize("n", [1, 2, 5, 9, 14])
def test_never_removes_before_the_cap_even_when_a_formation_scores_terribly(n):
    cat, inst = _cat(n)
    scores = {f: (-5.0 if i == 0 else 1.0) for i, f in enumerate(cat["Shotgun Classic"])}      # F0 is awful
    chosen = {r["formation"]: r["plays"] for r in _ranked(cat, scores)[n:n + 2]}
    f, s, swaps, notes = designer.additive_book(active=_active(cat, inst), chosen_formations=chosen,
                                                chosen_sources={k: "Shotgun Classic" for k in chosen}, ranked=_ranked(cat, scores),
                                                catalogue=cat, max_formations=15)
    assert set(inst) <= set(f) and swaps == [] and len(f) <= 15
    assert f["F0"] == cat["Shotgun Classic"]["F0"]                       # retained, unused is allowed


def test_gun_tight_y_off_is_added_as_the_sixth_formation_and_all_five_stay():
    cat, inst = _cat(5)
    f, s, swaps, notes = designer.additive_book(active=_active(cat, inst), chosen_formations={}, chosen_sources={}, ranked=_ranked(cat, {}),
                                                catalogue=cat, max_formations=15, must_add=("Gun Tight Y Off",))
    assert list(f)[:5] == inst and len(f) == 6 and f["Gun Tight Y Off"] == cat["Shotgun Classic"]["Gun Tight Y Off"]
    assert len(f["Gun Tight Y Off"]) == 24 and s["Gun Tight Y Off"] == "Shotgun Classic"
    assert sum(len(v) for v in f.values()) == 5 * 3 + 24 and swaps == []


def test_real_catalogue_has_gun_tight_y_off_with_24_plays_in_shotgun_classic():
    assert len(catalog.book_formations("offense", "Shotgun Classic")["Gun Tight Y Off"]) == 24


def test_at_the_cap_additions_stop_and_swaps_are_optional_evidence_backed_proposals_only():
    cat, inst = _cat(15)
    scores = {f: 0.5 for f in cat["Shotgun Classic"]}
    scores["F3"], scores["F20"] = 0.10, 0.60                            # clearly weak installed, clearly better candidate
    f, s, swaps, notes = designer.additive_book(active=_active(cat, inst), chosen_formations={"F20": cat["Shotgun Classic"]["F20"]},
                                                chosen_sources={"F20": "Shotgun Classic"}, ranked=_ranked(cat, scores), catalogue=cat, max_formations=15)
    assert set(f) == set(inst) and len(f) == 15                        # nothing added, nothing removed automatically
    assert swaps and swaps[0]["action"] == "OPTIONAL_SWAP" and swaps[0]["remove"] == "F3" and swaps[0]["add"] == "F20"
    assert swaps[0]["requires_owner_approval"] and swaps[0]["evidence"]["gap"] >= designer.SWAP_MIN_SCORE_GAP


def test_unknown_must_add_is_reported_not_invented_and_diverged_installs_are_completed_not_dropped():
    cat, inst = _cat(2)
    active = _active(cat, inst)
    active["formations"]["F0"] = ["P0a"]                                # partial install
    f, s, _, notes = designer.additive_book(active=active, chosen_formations={}, chosen_sources={}, ranked=_ranked(cat, {}), catalogue=cat,
                                            max_formations=15, must_add=("Nonexistent",))
    assert f["F0"] == cat["Shotgun Classic"]["F0"] and "Nonexistent" not in f and any("Nonexistent" in n for n in notes)


# -------------------------------------------------------------------------------------------------- composed adjustments
@pytest.fixture()
def play():
    s = pf.structure_lookup(*ANNOTATED[0])
    return s, pf.eligible_hot_route_targets(s)


class _Stub:
    """Learned gain stub: +G logit per changed receiver (route structure only). Search/legality/timing are the real code."""
    G = 0.12

    def __init__(self, *a, **k):
        self.base = a[1] if len(a) > 1 else None

    def score(self, mod):
        return self.G * len(mod.adjustments), 0.01, []


def _decide(art, s, monkeypatch, **kw):
    monkeypatch.setattr(od.sm, "DeltaScorer", lambda artifact, st, **k: _Stub(artifact, st))
    return od.decide(art, SIT, [(s.formation, s.play)], mode="research", margin=0.0, z=0.0, budget_ms=5000, **kw)


def test_three_or_more_independent_hot_routes_are_chosen_with_no_numeric_cap(art, play, monkeypatch):
    s, targets = play
    if len(targets) < 3:
        pytest.skip("play has fewer than 3 eligible receivers")
    d = _decide(art, s, monkeypatch)
    hot = [a for a in d.actions if a.action_id == "off.hot_route"]
    assert len(hot) >= 3 and len({a.params["target"] for a in hot}) == len(hot)        # distinct receivers only
    assert d.adjustment.count("HOT ROUTE") == len(hot)


def test_final_structure_and_reads_are_recomputed_from_the_composed_play(art, play, monkeypatch):
    s, targets = play
    d = _decide(art, s, monkeypatch)
    fs = d.final_structure
    assert fs.adjustments and fs.read_order == ()                       # stock read order is NOT reused
    card = adv.build_card(d, {})
    assert "INFERRED" in card["READ BASIS"] or card["PRIMARY READ"].startswith("UNKNOWN")
    for a in d.actions:
        assert fs.route_for(a.params["target"]).family != s.route_for(a.params["target"]).family


def test_no_adjustment_competitor_wins_when_nothing_helps(art, play, monkeypatch):
    s, _ = play
    monkeypatch.setattr(_Stub, "G", 0.0)
    d = _decide(art, s, monkeypatch)
    assert d.adjustment == od.NO_ADJUSTMENT and d.actions == ()


def test_conflicting_assignments_are_rejected_not_merged(play):
    s, targets = play
    t = targets[0]
    macro = od.MacroEffect("M", {t: "Slant"}, True, "user_ingame:test")
    acts = (ex.PlannedAction("off.saved.M"), ex.PlannedAction("off.hot_route", {"target": t, "route": "Post"}))
    with pytest.raises(pf.StructureError, match="conflicting assignments"):
        od._apply_actions(s, acts, {"M": macro})
    two = (ex.PlannedAction("off.hot_route", {"target": t, "route": "Post"}), ex.PlannedAction("off.hot_route", {"target": t, "route": "Corner"}))
    with pytest.raises(pf.StructureError, match="conflicting assignments"):
        od._apply_actions(s, two, {})


def test_play_clock_cutoff_limits_the_plan_by_time_not_by_a_constant(art, play, monkeypatch):
    s, targets = play
    if len(targets) < 3:
        pytest.skip("play has fewer than 3 eligible receivers")
    free = _decide(art, s, monkeypatch)
    tight = _decide(art, s, monkeypatch, situation_clock=None) if False else None
    monkeypatch.setattr(od.sm, "DeltaScorer", lambda artifact, st, **k: _Stub(artifact, st))
    sit = od.Situation(down=3, distance=7, yardline=45, opponent_id="cpu", opponent_type="cpu", play_clock_remaining=9.0)
    d = od.decide(art, sit, [(s.formation, s.play)], mode="research", margin=0.0, z=0.0, budget_ms=5000)
    n_free = len([a for a in free.actions if a.action_id == "off.hot_route"])
    n_tight = len([a for a in d.actions if a.action_id == "off.hot_route"])
    assert n_tight < n_free
    need = ex.plan_seconds([ex.registry()["off.hot_route"]] * n_tight)
    assert need <= 9.0


def test_protection_and_hot_routes_compose_and_a_macro_conflicts_with_route_changes(art, play, monkeypatch):
    s, targets = play
    d = _decide(art, s, monkeypatch, extra_actions=(ex.PlannedAction("off.pass_protection", {"protection": "Half Slide"}),),
                non_route_gain={"off.pass_protection": (0.2, 0.01)})
    ids = [a.action_id for a in d.actions]
    assert "off.pass_protection" in ids and "off.hot_route" in ids          # mixed protection + several routes in one sequence
    assert od._apply_actions(s, d.actions, {}).adjustments                    # structure still transformed from the verified original
    t = targets[0]
    lo = ex.Loadout.from_dict({"offense": [{"name": "SLANTS", "slot": "A", "installed": True, "active": True, "confirmed_by_user": True}]})
    ctx = ex.PlayContext("offense", s.formation, s.play, "pass", eligible_receivers=s.skill_slots)
    both = (ex.PlannedAction("off.saved.SLANTS"), ex.PlannedAction("off.hot_route", {"target": t, "route": "Post"}))
    assert any("cannot both be applied" in e for e in ex.validate_plan(both, ctx, loadout=lo))   # Madden's own rule, from the registry
    mixed = (ex.PlannedAction("off.saved.SLANTS"), ex.PlannedAction("off.pass_protection", {"protection": "Half Slide"}))
    assert ex.validate_plan(mixed, ctx, loadout=lo) == []


def test_search_stays_inside_the_latency_budget_and_never_enumerates_blindly(art, play, monkeypatch):
    s, targets = play
    d = _decide(art, s, monkeypatch)
    assert d.latency_ms < 5000
    assert len(d.plans) < 2 ** max(1, len(targets) * 3)                  # far fewer plans than the power set of (target x option)
