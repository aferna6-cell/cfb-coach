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
