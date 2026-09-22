"""Orchestrate PRE_SNAP → snap → ACTIVE → end → POST_PLAY; emit PlayRecord."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cfb_coach.vision.play_end import PlayEndDetector
from cfb_coach.vision.play_family import classify_family, family_to_concept_hint
from cfb_coach.vision.play_record import PlayRecord
from cfb_coach.vision.play_state import PlayStateTracker, canonicalize
from cfb_coach.vision.result_detect import detect_result
from cfb_coach.vision.snap_detect import SnapDetector


OnPlayEnd = Callable[[PlayRecord], None]


@dataclass
class PlayTracker:
    """High-level play lifecycle over VisionPipeline observations."""

    state: PlayStateTracker = field(default_factory=PlayStateTracker)
    snapper: SnapDetector = field(default_factory=SnapDetector)
    ender: PlayEndDetector = field(default_factory=PlayEndDetector)
    session_id: str = ""
    game_id: str = ""
    opponent_id: str = ""
    on_play_end: OnPlayEnd | None = None

    _open: PlayRecord | None = None
    _plays: list[PlayRecord] = field(default_factory=list)
    _last_hud: dict[str, Any] = field(default_factory=dict)
    _play_index: int = 0

    def configure(
        self,
        *,
        session_id: str = "",
        game_id: str = "",
        opponent_id: str = "",
        on_play_end: OnPlayEnd | None = None,
    ) -> None:
        self.session_id = session_id or self.session_id
        self.game_id = game_id or self.game_id or self.session_id
        self.opponent_id = opponent_id or self.opponent_id
        if on_play_end is not None:
            self.on_play_end = on_play_end

    @property
    def plays(self) -> list[PlayRecord]:
        return list(self._plays)

    @property
    def last_play(self) -> PlayRecord | None:
        return self._plays[-1] if self._plays else None

    def feed(
        self,
        *,
        now: float | None = None,
        motion: float | None = None,
        centroids: list[tuple[float, float]] | None = None,
        play_clock: str | None = None,
        hud: dict[str, Any] | None = None,
        formation: str = "unknown",
        formation_side: str = "",
        shell: str = "unknown",
        pressure: str = "unknown",
        family_hint: str | None = None,
        coach_rec: str = "",
        our_call: str = "",
        macro: str = "",
        side: str = "offense",
        motion_present: bool | None = None,
        motion_direction: str = "UNKNOWN",
        menu_like: bool = False,
    ) -> tuple[str, PlayRecord | None]:
        """Ingest one analysis tick. Returns (state, completed_play|None)."""
        now = time.time() if now is None else now
        hud = hud or {}
        hud_changed = bool(hud) and hud != self._last_hud and self._last_hud
        completed: PlayRecord | None = None

        # Snap detect when pre-snap
        snap = self.snapper.update(
            now=now,
            motion=motion,
            centroids=centroids,
            play_clock=play_clock,
            play_state=self.state.state,
        )
        end_sig = self.ender.update(
            now=now,
            motion=motion,
            hud_changed=hud_changed,
            play_state=self.state.state,
        )

        st = self.state.update(
            now=now,
            motion=motion,
            hud_visible=True,
            menu_like=menu_like,
            snap_signal=snap.snapped,
            end_signal=end_sig.ended,
        )

        # Open play on snap transition
        if snap.snapped and self._open is None:
            self.ender.on_snap()
            self._play_index += 1
            zone = _field_zone(hud.get("yardline"))
            self._open = PlayRecord(
                game_id=self.game_id,
                session_id=self.session_id,
                play_id=f"{self.session_id or 'g'}-{self._play_index:04d}",
                opponent_id=self.opponent_id,
                side=side,
                down=hud.get("down"),
                distance=hud.get("distance"),
                yardline=hud.get("yardline"),
                quarter=hud.get("quarter"),
                field_zone=zone,
                score_us=hud.get("score_us"),
                score_them=hud.get("score_them"),
                formation=formation or "unknown",
                formation_side=formation_side or "",
                shell=shell or "unknown",
                pressure=pressure or "unknown",
                motion_present=motion_present,
                motion_direction=motion_direction or "UNKNOWN",
                coach_rec=coach_rec,
                our_call=our_call,
                macro=macro,
                snap_confidence=snap.confidence,
                confidence={
                    "snap": snap.confidence,
                    "formation": 0.5,
                    "shell": 0.5,
                },
                ts_snap=now,
                notes=",".join(snap.reasons),
            )
            if canonicalize(st) != "PLAY_ACTIVE":
                self.state.force("PLAY_ACTIVE", now)

        # Update open play pre-snap look if still early
        if self._open is not None and canonicalize(st) == "PLAY_ACTIVE":
            if formation and formation != "unknown":
                self._open.formation = formation
            if shell and shell != "unknown":
                self._open.shell = shell
            if pressure and pressure != "unknown":
                self._open.pressure = pressure

        # Close play on end
        if end_sig.ended and self._open is not None:
            completed = self._finalize(
                self._open,
                now=now,
                end_conf=end_sig.confidence,
                hud=hud,
                family_hint=family_hint,
                end_reasons=end_sig.reasons,
            )
            self._open = None
            if canonicalize(st) == "PLAY_ACTIVE":
                self.state.force("PLAY_ENDING", now)

        if hud:
            self._last_hud = dict(hud)
        return self.state.state, completed

    def _finalize(
        self,
        rec: PlayRecord,
        *,
        now: float,
        end_conf: float,
        hud: dict[str, Any],
        family_hint: str | None,
        end_reasons: list[str],
    ) -> PlayRecord:
        yards_delta = None
        yards_conf = 0.0
        # Prefer explicit hint yards; HUD delta only when both yardlines present
        if rec.yardline is not None and hud.get("yardline") is not None:
            try:
                yards_delta = int(hud["yardline"]) - int(rec.yardline)
                # Heuristic: offense gaining → depends on side of field; keep None if absurd
                if abs(yards_delta) > 80:
                    yards_delta = None
                else:
                    yards_conf = 0.55
            except (TypeError, ValueError):
                yards_delta = None

        res = detect_result(
            hint=family_hint,
            yards_delta=yards_delta,
            yards_conf=yards_conf,
            score_us_delta=_delta(hud.get("score_us"), rec.score_us),
            score_them_delta=_delta(hud.get("score_them"), rec.score_them),
            family=None,
        )
        fam = classify_family(
            hint=family_hint,
            formation=rec.formation,
            result_type=res.result_type,
            yards=res.yards,
            confidence=0.55 if family_hint else 0.3,
        )
        # Re-run result with family
        res = detect_result(
            hint=family_hint,
            yards_delta=yards_delta,
            yards_conf=yards_conf,
            score_us_delta=_delta(hud.get("score_us"), rec.score_us),
            score_them_delta=_delta(hud.get("score_them"), rec.score_them),
            family=fam.family,
            motion_escape=fam.family == "SCRAMBLE",
        )
        rec.play_family = fam.family
        rec.concept_tags = list(fam.concept_tags)
        rec.result_type = res.result_type
        rec.yards = res.yards
        rec.scramble_dir = res.scramble_dir
        rec.end_confidence = end_conf
        rec.ts_end = now
        rec.confidence["end"] = end_conf
        rec.confidence["family"] = fam.confidence
        rec.confidence["result"] = res.confidence
        rec.explosive = _is_explosive(rec.play_family, rec.yards)
        if end_reasons:
            rec.notes = (rec.notes + ";" if rec.notes else "") + ",".join(end_reasons)

        # Prevent double-count: only append if new play_id
        if not self._plays or self._plays[-1].play_id != rec.play_id:
            self._plays.append(rec)
            if self.on_play_end:
                try:
                    self.on_play_end(rec)
                except Exception:
                    pass
        return rec

    def force_end(
        self,
        *,
        now: float | None = None,
        family_hint: str | None = None,
        result_hint: str | None = None,
    ) -> PlayRecord | None:
        """Manual / test close of open play."""
        if self._open is None:
            return None
        now = time.time() if now is None else now
        hint = family_hint or result_hint
        completed = self._finalize(
            self._open,
            now=now,
            end_conf=0.8,
            hud=self._last_hud,
            family_hint=hint,
            end_reasons=["force_end"],
        )
        self._open = None
        self.state.force("POST_PLAY", now)
        return completed

    def correct_last(self, **fields: Any) -> PlayRecord | None:
        """Manual correction of last PlayRecord (R/P/S/I/X etc.)."""
        if not self._plays:
            return None
        rec = self._plays[-1]
        for k, v in fields.items():
            if hasattr(rec, k) and v is not None:
                setattr(rec, k, v)
        rec.corrected = True
        # Recompute family if hint-like fields changed
        if "play_family" in fields or "result_type" in fields or "concept_tags" in fields:
            pass
        elif fields.get("family_hint"):
            fam = classify_family(
                hint=str(fields["family_hint"]),
                formation=rec.formation,
                result_type=rec.result_type,
                yards=rec.yards,
                confidence=0.85,
            )
            rec.play_family = fam.family
            rec.concept_tags = fam.concept_tags
        rec.explosive = _is_explosive(rec.play_family, rec.yards)
        return rec

    def concept_hint_for(self, rec: PlayRecord) -> str:
        return family_to_concept_hint(rec.play_family, rec.concept_tags)


def _delta(a: Any, b: Any) -> int | None:
    if a is None or b is None:
        return None
    try:
        return int(a) - int(b)
    except (TypeError, ValueError):
        return None


def _field_zone(yardline: Any) -> str:
    if yardline is None:
        return "open"
    try:
        y = int(yardline)
    except (TypeError, ValueError):
        return "open"
    if y <= 5:
        return "gl"
    if y <= 20:
        return "rz"
    if y >= 80:
        return "backed"
    return "open"


def _is_explosive(family: str, yards: int | None, *, run_n: int = 10, pass_n: int = 15) -> bool:
    if yards is None:
        return False
    if family in ("RUN_INSIDE", "RUN_OUTSIDE", "QB_RUN", "SCRAMBLE", "RPO"):
        return yards >= run_n
    return yards >= pass_n
