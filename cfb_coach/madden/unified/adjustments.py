"""Joint play + individual-adjustment candidates, built on the verified Madden 27 action registry.

* NO ADJUSTMENT is always present and always competes.
* An adjustment is ``executable`` only when the owner has confirmed it in-game
  (``adjustments.executable.readiness == live_ready``) and the registry says it is legal for
  the exact play context. Unconfirmed ones can be generated for analysis (``include_unconfirmed``)
  but are marked ``executable=False`` and are never selectable live.
* Effect sizes are weak priors from the registry's ``addresses`` / ``exposes`` tags (labelled
  ``weak_prior``), or learned residuals once enough executed-adjustment data exists.
* Offense hot routes are only generated for slots whose route is documented/verified;
  the changed route is carried in ``modified_features["routes"]`` so the card shows the real change.
"""
from __future__ import annotations

import itertools
from typing import Any, Iterable, Mapping, Sequence

from cfb_coach.madden.adjustments import executable as X
from cfb_coach.madden.unified.types import DEF, OFF, Adjustment, FootballAction, FootballPlay, GameState

PAIR_LIMIT = 2
LEARN_MIN_N = 15
CONCEPT_TAGS = {"four_verts": "vert", "wheel": "vert", "dagger": "vert", "mesh": "cross", "drive": "cross",
                "deep_cross": "cross", "levels": "cross", "flood_sail": "flood", "screen": "screen", "rpo": "rpo",
                "smash": "flood", "curl_flat": "flood"}


def opponent_tags(side_of_opponent: str, signature: Sequence[str]) -> set[str]:
    sig = tuple(signature) + ("unknown",) * 3
    tags: set[str] = set()
    if side_of_opponent == DEF:
        ctype, fam = sig[0], sig[1]
        if ctype == "man":
            tags.add("man")
        if ctype == "pressure" or "blitz" in sig[2]:
            tags.add("pressure")
        if fam in ("cover1", "cover3", "cover0"):
            tags.add("single_high")
        if fam in ("cover2", "cover4", "cover6", "cover9"):
            tags.add("two_high")
        tags.add(fam)
    else:
        ptype, concept = sig[0], sig[1]
        if ptype in ("run", "rpo", "screen"):
            tags.add(ptype)
        if ptype == "play_action":
            tags.add("vert")
        if concept in CONCEPT_TAGS:
            tags.add(CONCEPT_TAGS[concept])
    return tags


def _effects(spec: X.ActionSpec, params: Mapping[str, str]) -> Mapping[str, Any]:
    for v in params.values():
        if v in spec.effects_by_param:
            return spec.effects_by_param[v]
    return spec.effects


class AdjustmentPlanner:
    def __init__(self, confirmations: Mapping[str, Any] | None = None, *, include_unconfirmed: bool = False) -> None:
        self.confirmations = dict(confirmations or {})
        self.include_unconfirmed = include_unconfirmed
        self.learned: dict[tuple[str, str], list[float]] = {}  # (action_id, opp tag) -> [n, sum_dy]

    # ---- candidate generation --------------------------------------------------------
    def _context(self, side: str, play: FootballPlay) -> X.PlayContext:
        if side == OFF:
            ptype = "run" if play.play_type == "run" else "pass"
            recv = tuple(a.slot for a in play.assignments if a.role not in ("UNKNOWN", "block"))
            return X.PlayContext(side=OFF, formation=play.formation, play=play.play, play_type=ptype, eligible_receivers=recv)
        return X.PlayContext(side=DEF, formation=play.formation, play=play.play, play_type="coverage")

    def _singles(self, side: str, play: FootballPlay) -> list[tuple[Adjustment, dict[str, Any]]]:
        ctx = self._context(side, play)
        out: list[tuple[Adjustment, dict[str, Any]]] = []
        for spec in X.registry().values():
            if spec.side != side or spec.kind in ("stock", "preset", "saved_custom") or spec.inputs_grade in ("undocumented", "conflict"):
                continue
            grids = [{}]
            for pname, meta in spec.params.items():
                opts = meta.get("options")
                if isinstance(opts, list):
                    grids = [dict(g, **{pname: o}) for g in grids for o in opts]
                else:
                    grids = []  # dynamic option lists (e.g. formation audibles) are not enumerated here
            if spec.requires.get("target") == "eligible_receiver":
                grids = [dict(g, target=r) for g in (grids or [{}]) for r in ctx.eligible_receivers]
            if spec.requires.get("target") == "defender":
                grids = []  # defender identities are not known without verified rosters/assignments
            for g in grids:
                plan = [X.PlannedAction(spec.id, g)]
                errs = X.validate_plan(plan, ctx, confirmations=self.confirmations, require_live=False)
                if errs:
                    continue
                option = next((g[k] for k in spec.params if k in g), None)
                status, _ = X.readiness(spec, option, loadout=None, confirmations=self.confirmations)
                live = status == "live_ready"
                if not live and not self.include_unconfirmed:
                    continue
                adj = Adjustment(spec.id, spec.name + (f": {option}" if option else ""), g, "weak_prior", spec.est_seconds or 2.0)
                out.append((adj, {"live": live, "spec": spec}))
        return out

    def candidates(self, side: str, play: FootballPlay, st: GameState | None = None) -> list[FootballAction]:
        acts = [FootballAction(play)]  # NO ADJUSTMENT, always
        singles = self._singles(side, play)
        ctx = self._context(side, play)
        for adj, meta in singles:
            acts.append(self._apply(play, (adj,), side, executable=meta["live"]))
        live = [(a, m) for a, m in singles if m["live"] or self.include_unconfirmed]
        for combo in itertools.combinations(live, 2) if PAIR_LIMIT >= 2 else []:
            plan = [X.PlannedAction(a.action_id, a.params) for a, _ in combo]
            if X.validate_plan(plan, ctx, confirmations=self.confirmations):
                continue
            acts.append(self._apply(play, tuple(a for a, _ in combo), side, executable=all(m["live"] for _, m in combo)))
        return acts

    @staticmethod
    def _apply(play: FootballPlay, adjs: tuple[Adjustment, ...], side: str, *, executable: bool) -> FootballAction:
        mod: dict[str, Any] = {}
        routes = {a.slot: a.role for a in play.assignments}
        for a in adjs:
            if a.action_id == "off.hot_route" and "target" in a.params:
                routes[a.params["target"]] = a.params.get("route", "UNKNOWN").lower()
                mod["routes"] = dict(routes)
                mod["changed_routes"] = {a.params["target"]: a.params.get("route")}
        return FootballAction(play, adjs, mod, True, None, executable)

    # ---- effects -------------------------------------------------------------------
    def effect(self, adj: Adjustment, side: str, opp_signature: Sequence[str]) -> tuple[float, float, str]:
        """(delta yards in OUR favour, sd, support). Weak prior unless learned data exists."""
        spec = X.registry().get(adj.action_id)
        tags = opponent_tags(DEF if side == OFF else OFF, opp_signature)
        dy = 0.0
        if spec is not None:
            eff = _effects(spec, adj.params)
            add, exp = eff.get("addresses", {}), eff.get("exposes", {})
            dy = 4.0 * (sum(add.get(t, 0.0) for t in tags) - sum(exp.get(t, 0.0) for t in tags))
        n_tot, s_tot = 0.0, 0.0
        for t in tags:
            rec = self.learned.get((adj.action_id, t))
            if rec:
                n_tot += rec[0]
                s_tot += rec[1]
        if n_tot >= LEARN_MIN_N:
            w = n_tot / (n_tot + 20.0)
            return w * (s_tot / n_tot) + (1 - w) * dy, 0.9 / (1 + n_tot / 10) ** 0.5, "learned"
        return dy, 0.9, "weak_prior"

    def learn(self, adj: Adjustment, opp_signature: Sequence[str], side: str, dy_vs_stock: float) -> None:
        for t in opponent_tags(DEF if side == OFF else OFF, opp_signature):
            rec = self.learned.setdefault((adj.action_id, t), [0.0, 0.0])
            rec[0] += 1
            rec[1] += dy_vs_stock


# ------------------------------------------------------------------------------------------------ status table
_EXISTS = {"ea_documented": "documented by EA", "ea_entry_documented": "entry documented by EA (menu item unconfirmed)",
           "ea_named_path_untested": "named in EA material, path untested"}


def status_table(confirmations: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """For each registry action: does it exist in Madden 27, has the OWNER verified it, may live advice recommend it."""
    conf = dict(confirmations or {})
    from cfb_coach.madden.unified import knowledge as K

    off_with_routes = sum(1 for (f, p) in K.offense_assignment_index() if len(K.offense_play(f, p).assignments) >= 2)
    out = []
    for spec in X.registry().values():
        if spec.kind == "stock":
            continue
        options = []
        for meta in spec.params.values():
            if isinstance(meta.get("options"), list):
                options = list(meta["options"])
        entry_ok = X.confirmed(conf, spec.id)
        opt_ok = [o for o in options if X.confirmed(conf, spec.id, o)]
        exists = spec.inputs_grade in _EXISTS
        if not exists:
            status, why = "UNAVAILABLE", spec.blocked_reason or f"no verified Madden 27 input ({spec.inputs_grade})"
        elif spec.kind in ("preset", "saved_custom"):
            status, why = "UNAVAILABLE", "saved Custom Adjustment slots are not recommended by this coordinator (no verified slot contents)"
        elif spec.requires.get("target") == "defender":
            status, why = "UNAVAILABLE", "needs verified defender identities for this play (none in the repo)"
        elif spec.requires.get("target") == "eligible_receiver" and not off_with_routes and spec.side == OFF:
            status, why = "UNAVAILABLE", "needs documented receiver slots; none available"
        elif not entry_ok:
            status, why = "NOT_OWNER_VERIFIED", "exists, but not confirmed by the owner in-game: never recommended live"
        elif options and not opt_ok:
            status, why = "NOT_OWNER_VERIFIED", "entry confirmed but no menu option confirmed"
        else:
            status, why = "LIVE_RECOMMENDABLE", "owner-verified; chosen only if its benefit survives uncertainty vs NO ADJUSTMENT"
        needs = ("documented receiver slots: " + str(off_with_routes) + " catalog plays") if spec.requires.get("target") == "eligible_receiver" and spec.side == OFF else \
                "an eligible receiver target (offense receivers are unknown on defense)" if spec.requires.get("target") == "eligible_receiver" else None
        out.append({"id": spec.id, "name": spec.name, "side": spec.side, "category": spec.category, "exists_in_madden": exists,
                    "existence_evidence": _EXISTS.get(spec.inputs_grade, spec.inputs_grade), "owner_verified_entry": entry_ok,
                    "owner_verified_options": opt_ok, "options": options, "status": status, "reason": why, "needs": needs,
                    "optimization_basis": "weak registry prior unless executed-adjustment data exists (then learned)"})
    return out


def confirmations_template() -> dict[str, Any]:
    t: dict[str, Any] = {"_README": "Set a key to true ONLY after you have performed that exact adjustment on your Xbox in Madden 27 and it worked. "
                                    "Anything left false is never recommended. Option keys look like 'action_id|Option'."}
    for r in status_table():
        if r["status"] == "UNAVAILABLE":
            continue
        t[r["id"]] = False
        for o in r["options"]:
            t[f"{r['id']}|{o}"] = False
    return t
