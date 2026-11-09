"""One continuous two-sided Madden game: offense and defense coordinators on SHARED state.

Shared across both coordinators: one game id, one monotone snap sequence, one snap history, one world model,
one personal-execution model, one adjustment planner. Separate by design: the two opponent models (what
they run on defense vs on offense) and the provenance of every learned row.

Loop:  start_snap(situation text)  ->  [owner runs the play]  ->  finish_snap(executed, result, opponent call)
The side is taken from the possession the owner types; nothing is relaunched on a possession change.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from cfb_coach.madden.unified.adjustments import AdjustmentPlanner
from cfb_coach.madden.unified.coordinator import InstalledBookError, RecommendationCard, UnifiedCoordinator
from cfb_coach.madden.unified.inputs import InputError, parse_opponent_call, parse_result, parse_snap_line
from cfb_coach.madden.unified.opponent import OpponentModel
from cfb_coach.madden.unified.personal import PersonalExecution
from cfb_coach.madden.unified.solver import SolverConfig
from cfb_coach.madden.unified.store import DuplicateSnapError, UnifiedStore
from cfb_coach.madden.unified.types import (DEF, EXEC_CONFIRMED, EXEC_MODIFIED, EXEC_NOT_RUN, EXEC_UNKNOWN, OFF, FootballAction, GameState,
                                            SnapObservation, other_side)
from cfb_coach.madden.unified.world_model import WorldModel


class SessionError(RuntimeError):
    pass


class UnifiedSession:
    def __init__(self, *, books: Mapping[str, Mapping[str, Any]], opponent_id: str, store: UnifiedStore, model: WorldModel | None = None,
                 artifact: str = "cold_start_prior_v1", game_id: str | None = None, resume: bool = False, opponent_class: str | None = None,
                 roles_file: str | None = None, adjustment_confirmations: Mapping[str, Any] | None = None,
                 config: SolverConfig | None = None, latency_budget_ms: float = 150.0, ingest: bool = True) -> None:
        self._kw = dict(books=books, opponent_id=opponent_id, store=store, model=model, artifact=artifact, opponent_class=opponent_class,
                        roles_file=roles_file, adjustment_confirmations=adjustment_confirmations, config=config, latency_budget_ms=latency_budget_ms, ingest=ingest)
        for side in (OFF, DEF):
            if side not in books:
                raise InstalledBookError(f"no owner-confirmed installed {side} book; both books are required for a two-sided game")
        self.opponent_id = opponent_id
        self.opponent_class = opponent_class or ("cpu" if opponent_id.lower().startswith("cpu") else "human")
        self.store, self.artifact = store, artifact
        self.game_id = game_id or f"{opponent_id}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        known = store.known_games()
        if self.game_id in known and not resume:
            raise DuplicateSnapError(f"game id {self.game_id!r} already exists in this store; choose a new one or pass resume=True")
        self.model = model or WorldModel()
        self.personal = PersonalExecution()
        self.planner = AdjustmentPlanner(adjustment_confirmations)
        self.snaps: dict[str, SnapObservation] = {}
        self.history: dict[str, list[str]] = {}
        self.coord: dict[str, UnifiedCoordinator] = {}
        for side in (OFF, DEF):
            b = books[side]
            self.coord[side] = UnifiedCoordinator(
                side, b["inventory"], confirmed_fingerprint=b["fingerprint"], model=self.model,
                opponent_model=OpponentModel(other_side(side)), planner=self.planner, personal=self.personal, config=config,
                store=store, roles_file=roles_file, artifact=artifact, latency_budget_ms=latency_budget_ms,
                snaps=self.snaps, history=self.history)
        self.history_report: dict[str, Any] = {}
        if ingest:
            closed = store.closed_games()
            self.history_report = self.ingest_history([g for g in store.load_games() if g and g[0].state.game_id in closed])
        self.seq = store.max_seq(self.game_id) + 1
        if resume and self.seq > 1:  # re-load this game's own earlier snaps so previous-snap context is exact
            for g in store.load_games():
                if g and g[0].state.game_id == self.game_id:
                    for o in g:
                        self.snaps[o.snap_id] = o
                        self.history.setdefault(self.game_id, []).append(o.snap_id)
                        if o.closed:
                            self.coord[o.state.our_side].learn_from(o, self.coord[o.state.our_side].last_same_side(self.game_id, o.state.seq))
        store.mark_game(self.game_id, opponent_id, _now())
        self.pending: tuple[GameState, Any] | None = None  # (state, RecommendationCard | its stored dict after a restart)
        self.last_side: str | None = None
        self.model_rows_at_start = self.model.n_records
        ids = self.history.get(self.game_id, [])
        if ids:
            last = self.snaps[ids[-1]]
            self.last_side = last.state.our_side
            if not store.has_row("executions", last.snap_id) and not store.has_row("outcomes", last.snap_id) and last.snap_id not in store.voided():
                self.pending = (last.state, store.card_json(last.snap_id))  # an interrupted snap: still waiting for its result

    # ---- shared-state history replay: routes each observation to the right opponent model, world model once ----------
    def ingest_history(self, games: Sequence[Sequence[SnapObservation]]) -> dict[str, Any]:
        rows = 0
        for game in games:
            last: dict[str, SnapObservation] = {}
            for o in game:
                if o.closed:
                    c = self.coord[o.state.our_side]
                    c.opp.observe(o, prev_obs=last.get(o.state.our_side))
                    from cfb_coach.madden.unified.world_model import records_from_observation
                    for state, prof, d, oc, w, side in records_from_observation(o):
                        self.model.update(state, prof, d, oc, weight=w, our_side=side, opponent_class=state.opponent_class)
                        self.model.provenance[f"our_{side}_executed"] = self.model.provenance.get(f"our_{side}_executed", 0) + 1
                        rows += 1
                last[o.state.our_side] = o
            if game:
                for c in self.coord.values():
                    c.opp.close_game(game[0].state.game_id)
        return {"games": len(games), "world_model_rows": rows, "provenance": dict(self.model.provenance),
                "opponent_records": {s: len(c.opp.records) for s, c in self.coord.items()}}

    # ---- live loop -------------------------------------------------------------------------------------------------------
    def start_snap(self, text: str, *, quarter: int | None = None, score_us: int | None = None, score_opp: int | None = None) -> RecommendationCard:
        if self.pending is not None:
            self.abandon_pending("new situation entered before a result")
        sit = parse_snap_line(text, self.last_side)
        st = GameState(self.game_id, self.seq, sit.side, self.opponent_id, sit.down, sit.distance, sit.yardline,
                       opponent_class=self.opponent_class, quarter=quarter,
                       score_us=score_us if score_us is not None else sit.score_us, score_opp=score_opp if score_opp is not None else sit.score_opp,
                       playbook_fingerprint=self.coord[sit.side].fingerprint)
        card = self.coord[sit.side].recommend(st)
        self.pending = (st, card)
        self.last_side = sit.side
        self.seq += 1  # allocated only after a successful decision; never reused
        return card

    def abandon_pending(self, reason: str = "abandoned") -> None:
        """Leave the snap decided but unlearned: no execution, no outcome, never trains."""
        self.pending = None

    def finish_snap(self, *, executed: str | None = None, result: str | None = None, opponent_call: str | None = None) -> dict[str, Any]:
        if self.pending is None:
            raise SessionError("no snap is waiting for a result; enter a situation first")
        st, card = self.pending
        co = self.coord[st.our_side]
        rec = co.snaps[st.snap_id].recommended
        # validate EVERYTHING before writing anything, so a typo leaves the pending snap untouched
        ex_action, ex_status, actual = self._parse_executed(co, rec, card, executed)
        opp_action = parse_opponent_call(opponent_call, other_side(st.our_side)) if opponent_call and opponent_call.strip() else None
        outcome = parse_result(result, st) if result and result.strip() else None
        co.record_execution(st, ex_action, ex_status, actual)
        if opp_action is not None:
            co.record_opponent(st, opp_action)
        learned = False
        if outcome is not None:
            co.close_snap(st, outcome)
            learned = co.snaps[st.snap_id].labelled_for_own_learning and opp_action is not None
        self.pending = None
        return {"snap_id": st.snap_id, "side": st.our_side, "executed_status": ex_status, "opponent_recorded": opp_action is not None,
                "outcome_recorded": outcome is not None, "world_model_row": learned,
                "belief_evidence_next": {s: round(c.opp.beliefs(GameState(self.game_id, self.seq, s, self.opponent_id, 1, 10, 50, opponent_class=self.opponent_class)).n_effective, 2)
                                         for s, c in self.coord.items()}}

    @staticmethod
    def _parse_executed(co: UnifiedCoordinator, rec: FootballAction, card: RecommendationCard, text: str | None):
        t = (text or "u").strip()
        low = t.lower()
        if low in ("y", "yes", "as called"):
            return rec, EXEC_CONFIRMED, tuple(rec.adjustments)
        if low in ("y noadj", "y no-adj", "y noadjustment"):
            stock = co.stock_action(rec.play.formation, rec.play.play)
            return (stock, EXEC_MODIFIED, ()) if rec.adjustments else (stock, EXEC_CONFIRMED, ())
        if low in ("n", "no", "not run", "didn't"):
            return None, EXEC_NOT_RUN, None
        if low in ("u", "unknown", ""):
            return None, EXEC_UNKNOWN, None
        if "|" in t:
            f, p = (x.strip() for x in t.split("|", 1))
            try:
                act = co.stock_action(f, p)
            except KeyError as e:
                raise InputError(str(e)) from e
            return act, (EXEC_CONFIRMED if (f, p) == (rec.play.formation, rec.play.play) and not rec.adjustments else EXEC_MODIFIED), ()
        raise InputError("executed? y (ran it as called) / y noadj / n / u (unknown) / 'formation | play' (what you really ran)")

    def void_snap(self, snap_id: str, reason: str = "owner undo") -> None:
        """Undo = append corrected revisions that make the snap carry NO verified facts, then rebuild from disk.

        Nothing is deleted and the snap id stays consumed (no id is ever reused). The voided snap no longer trains
        anything, because its execution is unknown, its outcome has no yardage and its opponent call is unverified.
        """
        from cfb_coach.madden.unified.types import OpponentAction, Outcome
        o = self.snaps[snap_id]
        st = self.store
        if st.has_row("executions", snap_id):
            st.log_execution(snap_id, None, EXEC_UNKNOWN, None, _now(), amend=True)
        if st.has_row("opponent_obs", snap_id):
            st.log_opponent(snap_id, OpponentAction(other_side(o.state.our_side), None, None, (), "unverified", 0.0), _now(), amend=True)
        if st.has_row("outcomes", snap_id):
            st.log_outcome(snap_id, Outcome(), _now(), amend=True)
        st.void(snap_id, _now(), reason)

    def reload(self) -> "UnifiedSession":
        """A fresh session over the same store and game: every earlier observation is replayed from disk exactly once."""
        return UnifiedSession(**self._kw, game_id=self.game_id, resume=True)

    def end_game(self) -> None:
        if self.pending is not None:
            self.abandon_pending("game ended")
        for c in self.coord.values():
            c.opp.close_game(self.game_id)
        self.store.mark_game(self.game_id, self.opponent_id, _now(), closed=True)

    def status(self) -> dict[str, Any]:
        return {"game_id": self.game_id, "next_seq": self.seq, "last_side": self.last_side, "artifact": self.artifact,
                "world_model_rows": self.model.n_records, "provenance": dict(self.model.provenance),
                "store": self.store.counts()}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
