"""Owned #130 slice: actual transforms, conservative proposals and live locks."""
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import catalog, playbook
from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import experimental_model as em
from cfb_coach.madden.model import offense_adjustment_designer as ad
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_designer as designer
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.playcaller import MaddenCall

PAIR = ("Gun Trips X Nasty", "Hi Lo Cross")
PROBE = SimpleNamespace(down=1, distance=10, yardline=35)


@dataclass(frozen=True)
class Score:
    # Exact architecture contract; this is an injected deterministic test double,
    # NOT another real matcher/model.
    formation: str
    play: str
    base_probability: float
    matchup_edge: float
    total_probability: float
    mix: dict
    mix_source: str
    assignment_status: str
    assignment_confidence: float
    cells: list = field(default_factory=list)
    components: dict = field(default_factory=dict)


class Scorer:
    def __init__(self, gain=.08, uncertainty=0.0):
        self.calls = []
        self.gain = gain
        self.uncertainty = uncertainty

    def __call__(self, art, *, formation, play, probe, mix, opponent_id,
                 priors_on, learned=None, structure=None):
        self.calls.append((structure, priors_on, mix))
        count = sum(":drag->slant" in a or ":in->slant" in a for a in structure.adjustments)
        return Score(formation, play, .5, 0., .5 + min(3, count) * self.gain,
                     dict(mix or {}), "counterfactual" if mix else "unknown",
                     "verified" if structure.n_known else "unknown", 1. if structure.n_known else 0.,
                     components={"edge_se": 0., "model_uncertainty": self.uncertainty})


def design(**kw):
    scorer = kw.pop("score_pair_fn", Scorer())
    return ad.design_adjustments(None, formation=PAIR[0], play=PAIR[1], source_book="Buccaneers",
                                 probe=PROBE, opponent_id="cpu", score_pair_fn=scorer, **kw)


def confirms():
    return {"off.hot_route": True, **{"off.hot_route|" + r: True for r in ("Slant", "Flat", "Out")}}


def test_scores_actual_adjusted_structures_and_compatible_multi_routes():
    scorer = Scorer()
    result = design(score_pair_fn=scorer, priors_on=True, coverage_mix={"cover_1_unspecified": 1.})
    selected = result["selected"]
    assert selected["kind"] == "multi_route"
    assert selected["gain"] == pytest.approx(.24)
    assert len(selected["changed_slots"]) == 3
    assert all(s.formation == PAIR[0] and s.play == PAIR[1] for s, _, _ in scorer.calls)
    assert all(on and mix == {"cover_1_unspecified": 1.} for _, on, mix in scorer.calls)
    assert all(not s.read_order for s, _, _ in scorer.calls if s.adjustments)
    assert len({a["params"]["target"] for a in selected["actions"]}) == len(selected["actions"])
    assert selected["net_gain"] < selected["gain"]


def test_unconfirmed_actions_stay_draft_and_cannot_be_live_selection():
    result = design(installed_pairs=[PAIR])
    assert result["selected"]["status"] == ad.DRAFT
    assert result["selected"]["activation"] == "NOT_ARMED"
    assert not result["selected"]["executable_now"]
    assert result["selected"]["controls"].startswith("UNKNOWN")
    assert result["live_selection"]["label"] == od.NO_ADJUSTMENT


def test_confirmation_is_independent_of_stock_install():
    result = design(confirmations=confirms())
    assert not result["selected"]["executable_now"]
    result = design(confirmations=confirms(), installed_pairs=[PAIR])
    assert result["live_selection"]["kind"] == "multi_route"
    assert result["live_selection"]["executable_now"]


@pytest.mark.parametrize("kw", [
    {"score_pair_fn": Scorer(gain=0.)},
    {"score_pair_fn": Scorer(gain=.009)},
    {"cost_per_second": .2},
    {"score_pair_fn": Scorer(gain=.018, uncertainty=1.)},
])
def test_no_adjustment_wins_when_benefit_does_not_clear_cost_uncertainty(kw):
    assert design(**kw)["selected"]["label"] == od.NO_ADJUSTMENT


def test_protection_is_compared_but_never_given_fabricated_route_gain():
    result = design()
    protections = [r for r in result["candidates"] if r["kind"] == "protection"]
    assert len(protections) == 2
    assert {r["actions"][0]["params"]["protection"] for r in protections} == {"Half Slide", "Full Slide"}
    assert all(r["probability"] is None and r["reads"] == "UNKNOWN" for r in protections)
    assert all(not r["executable_now"] for r in protections)


def test_verified_macro_uses_exact_transform_and_unverified_macro_abstains():
    loadout = ex.Loadout((ex.LoadoutEntry("offense", "known", "slot-1", True, True, True),
                          ex.LoadoutEntry("offense", "opaque", "slot-2", True, True, True)))
    effects = {"known": od.MacroEffect("known", {"WR1": "Slant"}, True, "owner attested route effect")}
    result = design(loadout=loadout, macro_effects=effects, installed_pairs=[PAIR])
    macros = {r["label"]: r for r in result["candidates"] if r["kind"] == "existing_macro"}
    assert macros["MACRO known"]["probability"] == pytest.approx(.58)
    assert macros["MACRO known"]["changed_slots"] == ["WR1:drag->slant"]
    assert macros["MACRO known"]["executable_now"]
    assert macros["MACRO opaque"]["probability"] is None
    assert macros["MACRO opaque"]["reads"] == "UNKNOWN"


def test_new_macros_are_always_draft_even_if_constituents_are_confirmed():
    result = design(confirmations=confirms(), installed_pairs=[PAIR], proposed_macros=[{"name": "unverified"}])
    proposals = [r for r in result["candidates"] if r["kind"] == "proposed_macro"]
    assert len(proposals) == 2
    assert all(r["status"] == ad.DRAFT and not r["executable_now"] for r in proposals)
    assert proposals[0]["reads"] == "UNKNOWN"
    assert proposals[1]["source_ids"]


def test_unknown_assignments_do_not_invent_hot_routes(monkeypatch):
    stock = sm.structure_any(*PAIR)
    unknown = replace(stock, routes=(), read_order=(), sourced_concepts=(), skill_slots=())
    monkeypatch.setattr(sm, "structure_any", lambda *a: unknown)
    scorer = Scorer()
    result = design(score_pair_fn=scorer, priors_on=True)
    assert result["selected"]["label"] == od.NO_ADJUSTMENT
    assert len(scorer.calls) == 1
    assert not any(r["kind"] in ("hot_route", "multi_route") for r in result["candidates"])


def test_non_stock_pairs_rejected_before_scoring():
    with pytest.raises(ValueError, match="stock source-book"):
        ad.design_adjustments(None, formation=PAIR[0], play="Invented Play", source_book="Buccaneers",
                              probe=PROBE, opponent_id="cpu", score_pair_fn=Scorer())


def test_search_budget_is_honest():
    result = design(max_combinations=0)
    assert result["search_truncated"]
    assert not any(r["kind"] == "multi_route" for r in result["candidates"])


def test_explicit_priors_default_does_not_read_live_environment(monkeypatch):
    monkeypatch.setenv("CFB_COACH_FOOTBALL_PRIORS", "1")
    scorer = Scorer()
    result = design(score_pair_fn=scorer)
    assert not result["priors_on"] and all(not on for _, on, _ in scorer.calls)
    assert all(mix is None for _, _, mix in scorer.calls)


def test_install_guide_contains_exact_stock_source_plays_and_no_fabricated_slots():
    book = {"formations": {PAIR[0]: [PAIR[1]]}, "formation_sources": {PAIR[0]: "Buccaneers"}}
    proposal = {"book": book, "suggested_audibles": [{"formation": PAIR[0], "play": PAIR[1]}],
                "macro_blueprints": [{"name": "draft", "source_ids": ["ea-controls-xb-off"]}]}
    guide = designer.preparation_guide(proposal)
    assert guide["formations"][0]["stock_plays"] == [PAIR[1]]
    assert guide["formations"][0]["source_book"] == "Buccaneers"
    assert guide["formations"][0]["status"] == "MANUAL_INSTALL_AND_CONFIRM_REQUIRED"
    assert designer.preparation_guide(proposal, active=book)["formations"][0]["status"] == "CONFIRMED_INSTALLED"
    assert all(r["arm_slot"] is None for r in guide["audibles"] + guide["macros"])
    assert guide["macros"][0]["activation"] == "NOT_ARMED"


@pytest.fixture
def db(tmp_path):
    db = CoachDB(tmp_path / "inventory.db")
    yield db
    db.close()


def install(db):
    # Use the existing explicit stage/confirm transaction, not direct meta writes.
    p = designer.design_offense(db, artifact=em.ExperimentalArtifact(),
        catalogue={"Buccaneers": {PAIR[0]: catalog.book_formations("offense", "Buccaneers")[PAIR[0]]}},
        max_formations=1)
    designer.stage_design(db, p)
    if not playbook.load_books(db).get("offense"):
        with pytest.raises(ValueError, match="confirmed installed"):
            ad.validate_live_call(db, MaddenCall("offense", *PAIR, "No adj", ""), opponent_id="cpu")
    designer.confirm_installed(db, proposal_id=p["proposal_id"],
                              attestation="I installed every listed formation and all plays in Madden")
    return p


def test_live_gate_requires_confirmed_inventory_not_staged_proposal(db):
    call = MaddenCall("offense", *PAIR, "No adj", "")
    with pytest.raises(ValueError, match="confirmed installed"):
        ad.validate_live_call(db, call, opponent_id="cpu")
    install(db)
    ad.validate_live_call(db, call, opponent_id="cpu")
    call.play = "Uninstalled Play"
    with pytest.raises(ValueError, match="confirmed installed"):
        ad.validate_live_call(db, call, opponent_id="cpu")


def test_live_hot_route_and_macro_require_separate_user_confirmation(db):
    install(db)
    call = MaddenCall("offense", *PAIR, "No adj", "", adjustment={
        "id": "hot_slant_vs_man", "target": "WR1", "route": "Slant", "kind": "hot_route"})
    with pytest.raises(ValueError, match="not confirmed"):
        ad.validate_live_call(db, call, opponent_id="cpu")
    ad.validate_live_call(db, call, opponent_id="cpu", confirmations=confirms())
    call.adjustment = None
    call.macro = "unarmed"
    with pytest.raises(ValueError, match="installed and armed"):
        ad.validate_live_call(db, call, opponent_id="cpu")


def test_controller_rejects_uninstalled_call_before_exposing_it(db):
    call = MaddenCall("offense", *PAIR, "No adj", "")
    controller = LivePlayController(db, "cpu", lambda *a, **kw: call, lambda *a: None, lambda *a: "")
    with pytest.raises(ValueError, match="confirmed installed"):
        controller._make(SimpleNamespace(side="offense"))
    assert controller.last_call is None
    install(db)
    assert controller._make(SimpleNamespace(side="offense")) is call


def test_designer_only_passes_supported_shared_portfolio_keywords(db, monkeypatch):
    from cfb_coach.madden.model import offense_portfolio as portfolio
    original = portfolio.select_formation_portfolio
    books = {"Buccaneers": {PAIR[0]: catalog.book_formations("offense", "Buccaneers")[PAIR[0]]}}
    common = dict(artifact=em.ExperimentalArtifact(), catalogue=books, max_formations=1)
    observed = []

    def old(*, art, all_books, active, opponent_id, max_formations, db=None):
        return original(art=art, all_books=all_books, active=active, opponent_id=opponent_id,
                        max_formations=max_formations, db=db)

    monkeypatch.setattr(portfolio, "select_formation_portfolio", old)
    assert designer.design_offense(db, **common)["preparation_guide"]["formations"]
    with pytest.raises(ValueError, match="not integrated yet"):
        designer.design_offense(db, priors_on=True, coverage_mix={"cover_3_match": 1.}, **common)

    def new(*, art, all_books, active, opponent_id, max_formations, db=None,
            priors_on=False, coverage_mix=None):
        observed.append((priors_on, coverage_mix))
        return old(art=art, all_books=all_books, active=active, opponent_id=opponent_id,
                   max_formations=max_formations, db=db)

    monkeypatch.setattr(portfolio, "select_formation_portfolio", new)
    designer.design_offense(db, priors_on=True, coverage_mix={"cover_3_match": 1.}, **common)
    assert observed == [(True, {"cover_3_match": 1.})]
