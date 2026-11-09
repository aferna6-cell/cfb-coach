"""UnifiedCoordinator: ONE implementation serving the offensive and defensive coordinator roles.

recommend() -> record_execution() -> record_opponent() -> close_snap() is the whole live cycle.
The card is built from the solver's chosen action only; there is no later substitution.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

from cfb_coach.madden.model.defense_v2_live import installed_fingerprint
from cfb_coach.madden.unified import knowledge as K
from cfb_coach.madden.unified.adjustments import AdjustmentPlanner
from cfb_coach.madden.unified.opponent import Belief, OpponentModel
from cfb_coach.madden.unified.personal import PersonalExecution
from cfb_coach.madden.unified.reads import ReadPlan, UserAssignment, defense_user, offense_reads
from cfb_coach.madden.unified.solver import Decision, Solver, SolverConfig
from cfb_coach.madden.unified.store import DuplicateSnapError, UnifiedStore
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, EXEC_MODIFIED, EXEC_UNKNOWN, OFF, Adjustment, FootballAction,
                                            GameState, OpponentAction, Outcome, SnapObservation, other_side, stable_hash)
from cfb_coach.madden.unified.world_model import WorldModel, records_from_observation

COORDINATOR_NAME = "unified_v1"
LATENCY_BUDGET_MS = 150.0
MAX_FORMATIONS = 15


class InstalledBookError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def inventory_fp(side: str, inventory: Mapping[str, Sequence[str]]) -> str:
    return installed_fingerprint(inventory).replace("DEF-INST-", f"{side[:3].upper()}-INST-")


@dataclass
class RecommendationCard:
    side: str
    snap_id: str
    decision_model: str
    model_version: str
    artifact: str
    football_evidence_confidence: float
    evidence_note: str
    opponent_prediction_source: str
    fallback: dict[str, Any]
    formation: str
    play: str
    adjustments: list[str]
    changed_routes: dict[str, str]
    reads: ReadPlan | None
    user: UserAssignment | None
    explanation: str
    expected: dict[str, float]
    alternatives: list[dict[str, Any]]
    policy_prob: float
    candidate_set_id: str
    latency_ms: float
    action_key: str
    decision_id: str

    def render(self) -> str:
        L = [f"[{self.decision_model} | {self.model_version} | {self.artifact}] {self.side.upper()}",
             f"CALL: {self.formation} -> {self.play}",
             "ADJUST: " + ("; ".join(self.adjustments) if self.adjustments else "NO ADJUSTMENT")]
        if self.changed_routes:
            L.append("ROUTE CHANGES: " + ", ".join(f"{k}->{v}" for k, v in self.changed_routes.items()))
        if self.reads:
            L += [f"READ 1: {self.reads.primary}", f"READ 2: {self.reads.secondary}", f"(reads: {self.reads.basis})"]
        if self.user:
            L += [f"USER: {self.user.position}", f"PRIMARY: {self.user.primary}", f"SECONDARY: {self.user.secondary}",
                  f"({'VERIFIED' if self.user.verified else 'UNVERIFIED'}: {self.user.basis})"]
        L += [f"WHY: {self.explanation}", f"OPPONENT MODEL: {self.opponent_prediction_source}",
              f"EVIDENCE CONFIDENCE: {self.football_evidence_confidence:.2f} ({self.evidence_note})",
              f"FALLBACK: {self.fallback['reason'] if self.fallback['occurred'] else 'none'}"]
        return "\n".join(L)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedCoordinator:
    def __init__(self, our_side: str, inventory: Mapping[str, Sequence[str]], *, confirmed_fingerprint: str,
                 sources: Mapping[str, str] | None = None, model: WorldModel | None = None,
                 opponent_model: OpponentModel | None = None, planner: AdjustmentPlanner | None = None,
                 personal: PersonalExecution | None = None, config: SolverConfig | None = None,
                 store: UnifiedStore | None = None, roles_file: str | None = None, artifact: str = "cold_start_prior_v1",
                 fallback_fn: Callable[[GameState], Any] | None = None, latency_budget_ms: float = LATENCY_BUDGET_MS,
                 snaps: dict[str, SnapObservation] | None = None, history: dict[str, list[str]] | None = None) -> None:
        if our_side not in (OFF, DEF):
            raise ValueError("our_side must be offense or defense")
        if len(inventory) > MAX_FORMATIONS:
            raise InstalledBookError(f"{len(inventory)} formations installed; the limit is {MAX_FORMATIONS} per side")
        fp = inventory_fp(our_side, inventory)
        if fp != confirmed_fingerprint:
            raise InstalledBookError(f"installed-book fingerprint mismatch: inventory={fp} confirmed={confirmed_fingerprint}")
        self.side, self.inventory, self.fingerprint = our_side, {f: list(dict.fromkeys(p)) for f, p in inventory.items()}, fp
        self.plays = K.plays_from_inventory(our_side, self.inventory, sources)
        self.model = model or WorldModel()
        self.opp = opponent_model or OpponentModel(other_side(our_side))
        self.planner = planner or AdjustmentPlanner()
        self.personal = personal or PersonalExecution()
        self.solver = Solver(self.model, self.planner, self.personal, config)
        self.store, self.roles_file, self.artifact = store, roles_file, artifact
        self.fallback_fn, self.budget = fallback_fn, latency_budget_ms
        # In a two-sided session both coordinators share these two dicts: one monotone snap history per game.
        self.snaps: dict[str, SnapObservation] = snaps if snaps is not None else {}
        self.history: dict[str, list[str]] = history if history is not None else {}  # game_id -> ordered snap ids
        self._cands: list[FootballAction] | None = None
        self.warmup()

    # ---- candidates ---------------------------------------------------------------
    def legal_candidates(self) -> list[FootballAction]:
        if self._cands is None:
            acts: list[FootballAction] = []
            for p in self.plays:
                acts += self.planner.candidates(self.side, p)
            self._cands = [a for a in acts if a.legal]
        return self._cands

    def warmup(self) -> None:
        """Precompute candidates and touch every lazy cache so the first live call is as fast as the rest."""
        cands = self.legal_candidates()
        st = GameState("__warmup__", 1, self.side, "__warmup__", 2, 7, 50)
        self.solver.decide(st, self.side, cands, self.opp.beliefs(st))

    def stock_action(self, formation: str, play: str) -> FootballAction:
        """NO-ADJUSTMENT action for an installed (formation, play); anything else is an error, never a guess."""
        for a in self.legal_candidates():
            if a.no_adjustment and a.play.formation == formation and a.play.play == play:
                return a
        raise KeyError(f"{formation} | {play} is not in the installed {self.side} book")

    def last_same_side(self, game_id: str, before_seq: int | None = None) -> SnapObservation | None:
        """Most recent earlier snap in this game where WE were on the same side of the ball (sequence/reaction context)."""
        for sid in reversed(self.history.get(game_id, [])):
            o = self.snaps[sid]
            if o.state.our_side == self.side and (before_seq is None or o.state.seq < before_seq):
                return o
        return None

    # ---- decision -----------------------------------------------------------------
    def recommend(self, st: GameState) -> RecommendationCard:
        if st.our_side != self.side:
            raise ValueError(f"coordinator plays {self.side}; state says {st.our_side}")
        if st.playbook_fingerprint is not None and st.playbook_fingerprint != self.fingerprint:
            raise InstalledBookError("state playbook fingerprint does not match the confirmed installed book")
        if st.snap_id in self.snaps:
            raise DuplicateSnapError(f"snap {st.snap_id} was already decided")
        ids = self.history.get(st.game_id, [])
        if ids and self.snaps[ids[-1]].state.seq >= st.seq:
            raise DuplicateSnapError(f"snap sequence must increase: last {self.snaps[ids[-1]].state.seq}, got {st.seq}")
        t0 = time.perf_counter()
        prev = self.last_same_side(st.game_id)
        prev_ids = [i for i in ids if self.snaps[i].state.our_side == self.side]
        our_prev = None
        if prev is not None and prev.recommended is not None and prev.executed_status != "not_executed":
            ex = prev.executed or prev.recommended
            our_prev = ex.play.signature[1] if ex.play.signature[1:] else ex.play.signature[0]
        recent = [self.snaps[i].recommended.play.concept_path[-2 if len(self.snaps[i].recommended.play.concept_path) > 1 else 0]
                  for i in prev_ids[-3:] if self.snaps[i].recommended is not None]
        belief = self.opp.beliefs(st, our_prev_family=our_prev)
        fb = {"occurred": False, "reason": None}
        try:
            dec = self.solver.decide(st, self.side, self.legal_candidates(), belief, recent_own_families=recent,
                                     deadline_ms=self.budget)
            if dec.fallback:
                fb = {"occurred": True, "reason": dec.fallback}
        except Exception as e:  # noqa: BLE001 - technical safety fallback, always surfaced
            if self.fallback_fn is None:
                raise
            self.fallback_fn(st)
            raise RuntimeError(f"unified decision failed ({e!r}); caller must use legacy path") from e
        latency = (time.perf_counter() - t0) * 1e3
        card = self._card(st, dec, belief, fb, latency)
        obs = SnapObservation(st, recommended=dec.chosen.action, decision_id=card.decision_id)
        self.snaps[st.snap_id] = obs
        self.history.setdefault(st.game_id, []).append(st.snap_id)
        if self.store:
            self.store.log_decision(card.decision_id, st, dec.chosen.action, card.to_dict(), policy_prob=dec.policy_prob,
                                    candidate_set_id=dec.candidate_set_id, model_version=card.model_version, seed=dec.seed,
                                    latency_ms=latency, fell_back=fb["occurred"], fallback_reason=fb["reason"], ts=now())
        return card

    def _card(self, st: GameState, dec: Decision, belief: Belief, fb: dict[str, Any], latency: float) -> RecommendationCard:
        a = dec.chosen.action
        reads = offense_reads(a, belief) if self.side == OFF else None
        user = defense_user(a, belief, self.roles_file) if self.side == DEF else None
        conf = a.play.confidence if dec.chosen.source != "learned" else max(a.play.confidence, 0.7)
        note = f"play knowledge: {a.play.provenance}; unknown: {list(a.play.unknown_fields)}; value source {dec.chosen.source}"
        why = (reads.explanation if reads else user.explanation) + \
              f"; robust value {dec.chosen.robust:+.3f}, uncertainty {dec.chosen.uncertainty:.3f}, exec cost {dec.chosen.exec_cost:.3f}"
        alts = [{"action": s.action.key, "score": round(s.score, 4), "ev": round(s.ev, 4), "unc": round(s.uncertainty, 4)}
                for s in dec.ranked[1:4]]
        did = stable_hash([st.snap_id, a.key, dec.candidate_set_id, dec.seed], 14)
        return RecommendationCard(self.side, st.snap_id, COORDINATOR_NAME, dec.model_version, self.artifact, conf, note,
                                  dec.belief_sources, fb, a.play.formation, a.play.play, [x.label for x in a.adjustments],
                                  dict(a.modified_features.get("changed_routes") or {}), reads, user, why,
                                  {k: round(v, 4) for k, v in dec.chosen.expected.items()}, alts, dec.policy_prob,
                                  dec.candidate_set_id, round(latency, 3), a.key, did)

    # ---- lifecycle -----------------------------------------------------------------
    def record_execution(self, st: GameState, executed: FootballAction | None, status: str,
                         actual_adjustments: tuple[Adjustment, ...] | None = None, *, amend: bool = False) -> SnapObservation:
        """What WE actually ran. A recommendation is never auto-promoted to an execution."""
        o = self.snaps[st.snap_id].with_execution(executed, status, actual_adjustments)
        if self.store:  # persist first: an AlreadyRecorded refusal must leave memory untouched
            self.store.log_execution(st.snap_id, executed, status, actual_adjustments, now(), amend=amend)
        self.snaps[st.snap_id] = o
        return o

    def record_opponent(self, st: GameState, oa: OpponentAction, *, amend: bool = False) -> SnapObservation:
        """What the OTHER side was revealed to run. Never touches our executed action."""
        o = self.snaps[st.snap_id].with_opponent(oa)
        if self.store:
            self.store.log_opponent(st.snap_id, oa, now(), amend=amend)
        self.snaps[st.snap_id] = o
        return o

    def close_snap(self, st: GameState, outcome: Outcome) -> SnapObservation:
        """Close the snap, update beliefs immediately (no retraining) and the shared world model."""
        if self.snaps[st.snap_id].closed:
            raise DuplicateSnapError(f"snap {st.snap_id} is already closed")
        o = replace(self.snaps[st.snap_id], outcome=outcome, closed=True, closed_ts=now(), cutoff_seq=st.seq)
        if self.store:
            self.store.log_outcome(st.snap_id, outcome, now())
        self.snaps[st.snap_id] = o
        self.learn_from(o, self.last_same_side(st.game_id, st.seq))
        return o

    def learn_from(self, o: SnapObservation, prev_same_side: SnapObservation | None) -> None:
        """Opponent beliefs for OUR side's opponent + the shared world model (tagged by provenance)."""
        if o.state.our_side == self.side:
            self.opp.observe(o, prev_obs=prev_same_side)
        for state, prof, d, oc, w, side in records_from_observation(o):
            self.model.update(state, prof, d, oc, weight=w, our_side=side, opponent_class=state.opponent_class)
            self.model.provenance[f"our_{side}_executed"] = self.model.provenance.get(f"our_{side}_executed", 0) + 1

    def ingest_history(self, games: Sequence[Sequence[SnapObservation]]) -> dict[str, int]:
        """Replay finished games chronologically.

        Opponent beliefs: only observations from THIS coordinator's side (the other side's opponent model
        is fed by the other coordinator). Shared world model: verified rows from BOTH sides, once each —
        call this on exactly one coordinator per shared model (``UnifiedSession`` does the routing).
        """
        used = 0
        for game in games:
            last_by_side: dict[str, SnapObservation] = {}
            for o in game:
                if o.closed:
                    if o.state.our_side == self.side:
                        self.opp.observe(o, prev_obs=last_by_side.get(self.side))
                    for state, prof, d, oc, w, side in records_from_observation(o):
                        self.model.update(state, prof, d, oc, weight=w, our_side=side, opponent_class=state.opponent_class)
                        self.model.provenance[f"our_{side}_executed"] = self.model.provenance.get(f"our_{side}_executed", 0) + 1
                        used += 1
                last_by_side[o.state.our_side] = o
            if game:
                self.opp.close_game(game[0].state.game_id)
        return {"games": len(games), "world_model_rows": used, "opponent_records": len(self.opp.records)}

    def end_game(self, game_id: str) -> None:
        self.opp.close_game(game_id)
