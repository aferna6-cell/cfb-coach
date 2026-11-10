"""Offline, source-book constrained adjustment proposals; never installs or arms.

Value comes only from the shared pregame scorer. Execution cost and a conservative
uncertainty hurdle decide whether its predicted lift is worth proposing. Protection
and opaque macros have UNKNOWN modeled effects, not an invented route delta.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import asdict
from typing import Any, Callable, Mapping, Sequence

from cfb_coach.madden import catalog, research_db
from cfb_coach.madden.adjustments import executable as ex
from cfb_coach.madden.model import offense_decision as od
from cfb_coach.madden.model import offense_play_features as pf
from cfb_coach.madden.model import offense_structural_model as sm
from cfb_coach.madden.model import offense_taxonomy as tx

DRAFT = "DRAFT_NEEDS_IN_GAME_VERIFICATION"


def _shared_score(*args: Any, **kwargs: Any) -> Any:
    # Lazy seam: Strategist owns the implementation. No local fallback scorer.
    from cfb_coach.madden.model.offense_pregame_scoring import score_pair
    return score_pair(*args, **kwargs)


def _uncertainty(stock: Any, adjusted: Any) -> float:
    """Conservative probability-scale screening hurdle, NOT a confidence interval.

    Worst-case logit->probability slope is 1/4. Assignment extrapolation and
    model uncertainty add a penalty; a missing uncertainty is never zero.
    """
    components = adjusted.components or {}
    se = math.hypot(float((stock.components or {}).get("edge_se", 0.0)),
                    float(components.get("edge_se", 0.0))) / 4
    uncertainty = components.get("model_uncertainty")
    uncertainty = 1.0 if uncertainty is None else max(0.0, min(1.0, float(uncertainty)))
    confidence = max(0.0, min(1.0, float(adjusted.assignment_confidence)))
    return se + .02 * (1 - confidence) + .02 * uncertainty


def design_adjustments(
    art: Any, *, formation: str, play: str, source_book: str, probe: Any,
    opponent_id: str, priors_on: bool = False,
    coverage_mix: Mapping[str, float] | None = None, learned: Mapping | None = None,
    loadout: ex.Loadout | None = None, confirmations: Mapping[str, Any] | None = None,
    macro_effects: Mapping[str, od.MacroEffect] | None = None,
    proposed_macros: Sequence[Mapping[str, Any]] = (),
    installed_pairs: Sequence[tuple[str, str]] = (),
    score_pair_fn: Callable[..., Any] | None = None,
    margin: float = .02, cost_per_second: float = .002,
    max_combinations: int = 256,
) -> dict[str, Any]:
    """Compare stock, exact hot routes/multi-routes, protection and macro plans.

    The selected proposal can be a draft; ``live_selection`` can only be confirmed.
    New macros are ALWAYS drafts even when their constituent routes are confirmed.
    Multi-route search is bounded and reports truncation (never a latency claim).
    """
    if play not in catalog.book_formations("offense", source_book).get(formation, []):
        raise ValueError("Adjustment design requires a known stock source-book pair")
    if margin < 0 or cost_per_second < 0 or max_combinations < 0:
        raise ValueError("Adjustment margins, costs and search budget must be nonnegative")
    score = score_pair_fn or _shared_score
    stock = sm.structure_any(formation, play)
    effects = {k: v for k, v in (macro_effects or {}).items() if v.verified and v.source}
    installed = (formation, play) in installed_pairs
    kw = dict(formation=formation, play=play, probe=probe, opponent_id=opponent_id,
              priors_on=priors_on, mix=coverage_mix, learned=learned)
    base = score(art, structure=stock, **kw)
    ctx = ex.PlayContext("offense", formation, play, od._play_type(stock),
                         eligible_receivers=stock.skill_slots,
                         play_clock_remaining=getattr(probe, "play_clock_remaining", None))
    rows = [{"kind": "none", "label": od.NO_ADJUSTMENT, "actions": [],
             "probability": base.total_probability, "gain": 0.0, "uncertainty_hurdle": 0.0,
             "execution_cost": 0.0, "net_gain": 0.0, "beneficial": True,
             "executable_now": installed, "status": "CONFIRMED_INSTALLED" if installed else DRAFT,
             "activation": "STOCK_PLAY", "reads": "stock", "score": asdict(base)}]

    def consider(actions: tuple[ex.PlannedAction, ...], kind: str, *, draft: bool = False,
                 label: str | None = None) -> dict[str, Any]:
        row = {"kind": kind, "label": label or od.label_for(actions),
               "actions": [{"id": a.action_id, "params": dict(a.params)} for a in actions],
               "status": DRAFT, "activation": "NOT_ARMED", "executable_now": False,
               "probability": None, "gain": None, "net_gain": None, "beneficial": False,
               "reads": "UNKNOWN", "controls": "UNKNOWN: verify in Madden"}
        rows.append(row)
        errors = ex.validate_plan(actions, ctx, loadout=loadout, confirmations=confirmations)
        specs = [ex._spec_for(a, "offense", loadout) for a in actions]
        if any(a.action_id in od.NON_ROUTE_ACTIONS for a in actions):
            row["reason"] = "UNKNOWN: protection/other non-route effects not represented by route transforms"
            return row
        try:
            adjusted = od._apply_actions(stock, actions, effects)
        except pf.StructureError as exc:
            row["reason"] = "UNKNOWN: " + str(exc)
            return row
        if errors or any(s is None for s in specs):
            row["reason"] = "; ".join(errors) or "Unknown registry action"
            return row
        sc = score(art, structure=adjusted, **kw)
        seconds = ex.plan_seconds(specs)
        uncertainty = _uncertainty(base, sc)
        cost = seconds * cost_per_second
        gain = sc.total_probability - base.total_probability
        net = gain - uncertainty - cost
        live_errors = ex.validate_plan(actions, ctx, loadout=loadout,
                                       confirmations=confirmations, require_live=True)
        executable = installed and not live_errors and not draft
        row.update(probability=sc.total_probability, gain=gain, uncertainty_hurdle=uncertainty,
                   execution_seconds=seconds, execution_cost=cost, net_gain=net,
                   beneficial=net > margin, executable_now=executable,
                   status="CONFIRMED_INSTALLED" if executable else DRAFT,
                   activation="CONFIRMED_ACTIVE" if executable else "NOT_ARMED",
                   reads="INFERRED: recompute from adjusted routes", score=asdict(sc),
                   changed_slots=list(adjusted.adjustments),
                   reason="benefit clears uncertainty and input cost" if net > margin else
                          "benefit does not justify uncertainty and input cost")
        if executable:
            row["controls"] = [ex.render_inputs(s, a.params, loadout) for s, a in zip(specs, actions)]
        return row

    singles = []
    if od._play_type(stock) == "pass":
        for target, route in itertools.product(pf.eligible_hot_route_targets(stock), tx.HOT_ROUTE_OPTIONS):
            a = ex.PlannedAction("off.hot_route", {"target": target, "route": route})
            r = consider((a,), "hot_route")
            if r["probability"] is not None:
                singles.append((r, a))
    # Keep two best alternatives per known target; combinations are checked for
    # exact route conflicts AND registry exclusivity before any numeric scoring.
    pool = []
    for target in pf.eligible_hot_route_targets(stock):
        pool.extend(sorted((x for x in singles if x[1].params["target"] == target),
                           key=lambda x: -x[0]["probability"])[:2])
    tried = 0
    truncated = False
    for size in range(2, len(pf.eligible_hot_route_targets(stock)) + 1):
        for combo in itertools.combinations(pool, size):
            actions = tuple(x[1] for x in combo)
            if len({a.params["target"] for a in actions}) != size:
                continue
            if tried >= max_combinations:
                truncated = True
                break
            consider(actions, "multi_route")
            tried += 1
        if truncated:
            break
    protection = ex.registry().get("off.pass_protection")
    if protection and od._play_type(stock) == "pass":
        for option in protection.params.get("protection", {}).get("options", []):
            consider((ex.PlannedAction(protection.id, {"protection": option}),), "protection")
    if loadout:
        for entry in loadout.entries:
            if entry.side == "offense":
                consider((ex.PlannedAction("off.saved." + entry.name),), "existing_macro")
    for proposal in proposed_macros:
        # A caller-provided blueprint is never promoted to a verified MacroEffect.
        row = {"kind": "proposed_macro", "label": str(proposal.get("name") or "unnamed draft"),
               "status": DRAFT, "activation": "NOT_ARMED", "executable_now": False,
               "probability": None, "beneficial": False, "net_gain": None,
               "reads": "UNKNOWN", "controls": "UNKNOWN: verify in Madden",
               "reason": "Blueprint effects/slots are not user-verified"}
        rows.append(row)
    # A new exact macro blueprint can reuse a scored compatible multi-route plan,
    # never an invented assignment or Xbox slot. Its score remains route-effect
    # extrapolation; macro execution cost is UNKNOWN until physically verified.
    multis = [r for r in rows if r["kind"] == "multi_route" and r.get("beneficial")]
    if multis:
        best = max(multis, key=lambda r: r["net_gain"])
        rows.append({**best, "kind": "proposed_macro", "label": "DRAFT: " + best["label"],
                     "status": DRAFT, "activation": "NOT_ARMED", "executable_now": False,
                     "controls": "UNKNOWN: verify macro settings and slot in Madden",
                     "source_ids": list(ex.registry()["off.hot_route"].sources),
                     "macro_execution_cost": "UNKNOWN; score uses constituent hot-route input cost"})
    eligible = [r for r in rows[1:] if r.get("beneficial") and r["kind"] != "proposed_macro"]
    selected = max(eligible, key=lambda r: r["net_gain"], default=rows[0])
    executable = [r for r in eligible if r["executable_now"]]
    live = max(executable, key=lambda r: r["net_gain"], default=rows[0])
    return {"formation": formation, "play": play, "source_book": source_book,
            "priors_on": priors_on, "coverage_mix": dict(coverage_mix or {}),
            "candidates": rows, "selected": selected, "live_selection": live,
            "search_truncated": truncated, "basis": od.BASIS}


def _named_audible_play(row: Mapping[str, Any]) -> str | None:
    """Play named by an audible suggestion, or None. Never invents a destination."""
    for key in ("audible_to", "play", "to"):
        value = row.get(key)
        if isinstance(value, str) and value.strip() and value.strip() != "run audible":
            return value.strip()
    label = str(row.get("label") or "")
    if "→" in label and label.lower().startswith("audible"):
        dest = label.split("→", 1)[1].strip()
        return dest or None
    target = row.get("target")
    if isinstance(target, str) and target.strip() and target.strip() != "run audible":
        return target.strip()
    return None


def validate_live_call(db: Any, call: Any, *, opponent_id: str,
                       confirmations: Mapping[str, Any] | None = None,
                       loadout: ex.Loadout | None = None, yardline: Any = None) -> None:
    """Fail closed before displaying a Madden offense call outside confirmed inventory.

    The play-installed lock always applies. Heuristic display suggestions are not
    rejected when confirmations and loadout are unset and the call is not a
    selected joint action (no ml_offense_action). When inventory is set or the
    call carries ml_offense_action, unconfirmed adjustments and macros fail closed.
    """
    from cfb_coach.madden import playbook
    from cfb_coach.madden.model.offense_designer import verified_created_macros

    if isinstance(getattr(call, "scheme_variant", None), Mapping):
        # civil_scheme_only: re-validate the exact variant through the one legal filter.
        from cfb_coach.madden.scheme import catalog as scheme_catalog
        from cfb_coach.madden.scheme import legality as scheme_legality

        scheme_legality.revalidate_call(db, call, scheme_catalog.load_cached(), yardline=yardline)
        return
    book = playbook.load_books(db).get("offense") or {}
    installed = (book.get("formations") or {}).get(call.formation, [])
    if call.play not in installed:
        raise ValueError("Live offense requires a confirmed installed formation/play")
    inventory_set = confirmations is not None or loadout is not None
    joint = isinstance(getattr(call, "ml_offense_action", None), dict)
    if not inventory_set and not joint:
        return
    if getattr(call, "macro", None):
        macro = str(call.macro)
        approved = any(m["name"] == macro and any(
            p.get("formation") == call.formation and p.get("play") == call.play
            for p in m.get("base_pairs") or []) for m in verified_created_macros(db, opponent_id))
        if not approved and not (loadout and loadout.usable("offense", macro)):
            raise ValueError("Live macro is not confirmed installed and armed in a slot")
    adjustment = getattr(call, "adjustment", None) or {}
    if not adjustment:
        return
    researched = {a["id"]: a for a in research_db.offense_adjustments()}
    actions = []
    for aid in adjustment.get("parts") or [adjustment.get("id")]:
        row = researched.get(aid) or {}
        if not adjustment.get("parts"):
            row = {**row, **adjustment}
        kind = row.get("kind") or row.get("type")
        if kind == "hot_route" and row.get("target") and row.get("route"):
            actions.append(ex.PlannedAction("off.hot_route", {"target": row["target"], "route": row["route"]}))
        elif kind == "pass_protection" and row.get("route"):
            actions.append(ex.PlannedAction("off.pass_protection", {"protection": row["route"]}))
        elif kind == "audible":
            dest = _named_audible_play(row)
            if dest is None or dest == call.play or dest not in installed:
                raise ValueError("Live audible is not a confirmed installed play")
        else:
            raise ValueError("Live adjustment has UNKNOWN executable effects")
    if not actions:
        return
    stock = sm.structure_any(call.formation, call.play)
    ctx = ex.PlayContext("offense", call.formation, call.play, od._play_type(stock),
                         eligible_receivers=stock.skill_slots)
    errors = ex.validate_plan(actions, ctx, loadout=loadout, confirmations=confirmations, require_live=True)
    if errors:
        raise ValueError("Live adjustment is not confirmed: " + "; ".join(errors))
