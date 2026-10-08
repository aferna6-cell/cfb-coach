"""Future opt-in defensive ML control pilot — architecture only.

Sprint 5.2 implements the control path shape (authorize → select → rollback)
but **refuses activation**. Live defensive calls stay heuristic. Instant
rollback always returns the heuristic call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from cfb_coach.madden.model import defense_ca, defense_shadow

META_PILOT = "ml_defense_control_pilot"  # must stay off
META_PILOT_AUTH = "ml_defense_control_pilot_auth"  # separate authorization token slot
META_ROLLBACK = "ml_defense_control_rollback"  # "heuristic" instant path


@dataclass
class ControlPilotDecision:
    """Result of a (disabled) control attempt."""

    controlled: bool
    formation: str | None
    play: str | None
    adjustment: str | None
    source: str  # "heuristic_rollback" | "refused" | "pilot" (never pilot this sprint)
    reason: str
    rolled_back: bool = True
    refused: bool = True
    shadow: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "controlled": False,  # hard: never True this sprint
            "formation": self.formation,
            "play": self.play,
            "adjustment": self.adjustment,
            "source": self.source,
            "reason": self.reason,
            "rolled_back": True,
            "refused": True,
            "shadow": dict(self.shadow),
        }


def pilot_enabled(db: Any | None) -> bool:
    """True only when meta explicitly requests the pilot — still refused below."""
    if db is None:
        return False
    try:
        return str(db.get_meta(META_PILOT) or "off").lower() in ("on", "1", "true", "pilot")
    except Exception:  # noqa: BLE001
        return False


def authorization_present(db: Any | None) -> bool:
    if db is None:
        return False
    try:
        raw = db.get_meta(META_PILOT_AUTH)
        return bool(raw) and str(raw).strip().lower() not in ("", "off", "none", "0")
    except Exception:  # noqa: BLE001
        return False


def refuse_pilot_activation(db: Any | None = None) -> dict[str, Any]:
    """Always refuse in this sprint — even if meta bits are flipped."""
    shadow_refuse = defense_shadow.refuse_activation(db)
    return {
        "ok": False,
        "activated": False,
        "pilot": "off",
        "control": "off",
        "reason": (
            "Sprint 5.2: defensive ML control pilot is implemented but disabled. "
            "Activation requires a future authorized sprint after Stage 3 readiness "
            "criteria and separate user confirmation. Instant heuristic rollback remains "
            "the only live path."
        ),
        "shadow_refuse": shadow_refuse,
        "current_meta": {
            META_PILOT: (db.get_meta(META_PILOT) if db is not None else None),
            META_PILOT_AUTH: (db.get_meta(META_PILOT_AUTH) if db is not None else None),
            META_ROLLBACK: (db.get_meta(META_ROLLBACK) if db is not None else None),
            defense_shadow.META_CONTROL: (
                db.get_meta(defense_shadow.META_CONTROL) if db is not None else None
            ),
        },
        "rollback": "heuristic",
    }


def try_activate_pilot(
    db: Any,
    *,
    authorization: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Opt-in activation entrypoint — always refuses this sprint.

    Never writes control=on. ``force`` and ``authorization`` are accepted only
    so callers can be tested; they cannot bypass the refuse gate here.
    """
    del authorization, force
    result = refuse_pilot_activation(db)
    # Ensure metas stay off even if a caller expected a write.
    try:
        if db is not None:
            if str(db.get_meta(META_PILOT) or "").lower() in ("on", "1", "true", "pilot"):
                db.set_meta(META_PILOT, "off")
            if str(db.get_meta(defense_shadow.META_CONTROL) or "").lower() in (
                "on",
                "1",
                "true",
                "control",
            ):
                db.set_meta(defense_shadow.META_CONTROL, "off")
            db.set_meta(META_ROLLBACK, "heuristic")
    except Exception:  # noqa: BLE001
        pass
    return result


def instant_rollback_to_heuristic(
    *,
    heuristic_formation: str | None,
    heuristic_play: str | None,
    heuristic_adjustment: str | None = None,
    reason: str = "instant heuristic rollback",
    shadow: Mapping[str, Any] | None = None,
) -> ControlPilotDecision:
    """Always-available rollback: display the heuristic call unchanged."""
    return ControlPilotDecision(
        controlled=False,
        formation=heuristic_formation,
        play=heuristic_play,
        adjustment=heuristic_adjustment,
        source="heuristic_rollback",
        reason=reason,
        rolled_back=True,
        refused=True,
        shadow=dict(shadow or {}),
    )


def select_controlled_defense(
    *,
    sit: Any,
    heuristic_call: Any,
    book: Mapping[str, Any],
    armed_macros: list[str],
    opponent_id: str,
    db: Any = None,
) -> ControlPilotDecision:
    """Control selection path — builds a shadow rec then **refuses** to apply it.

    Architecture for a future authorized pilot:
    1. Build shadow recommendation under the 150 ms budget.
    2. Require adjustment ``control_eligible`` (provenance + compat) when present.
    3. Apply only when pilot auth + readiness pass (not this sprint).
    4. On any failure / timeout / refusal → instant heuristic rollback.
    """
    heur_form = getattr(heuristic_call, "formation", None)
    heur_play = getattr(heuristic_call, "play", None)
    heur_adj = getattr(heuristic_call, "macro", None) or getattr(
        heuristic_call, "adj_or_macro", None
    )
    if heur_adj in (None, "", "No adj", "—"):
        heur_adj = None

    # Hard refuse before any control application.
    if pilot_enabled(db) or defense_shadow.control_enabled(db) or authorization_present(db):
        refuse = refuse_pilot_activation(db)
        return instant_rollback_to_heuristic(
            heuristic_formation=heur_form,
            heuristic_play=heur_play,
            heuristic_adjustment=heur_adj,
            reason=refuse["reason"],
        )

    rec = defense_shadow.build_shadow_recommendation(
        sit=sit,
        heuristic_call=heuristic_call,
        book=book,
        armed_macros=armed_macros,
        opponent_id=opponent_id,
        db=db,
    )
    shadow = rec.to_dict()

    # Even in a future pilot, never apply an adjustment that fails provenance.
    if rec.adjustment:
        view = defense_ca.inspect_armed_macro(rec.adjustment)
        ok, why = defense_ca.play_adjustment_compatible(
            view,
            formation=rec.formation,
            play=rec.play,
            book=book,
        )
        if not (view.provenance_ok_for_control and ok):
            shadow["control_adj_blocked"] = why or "provenance/compat insufficient"
            # Future pilot would drop adj or full-rollback; this sprint always rollbacks.

    # Sprint gate: never take authority.
    return instant_rollback_to_heuristic(
        heuristic_formation=heur_form,
        heuristic_play=heur_play,
        heuristic_adjustment=heur_adj,
        reason=(
            "control path exercised then refused — live call remains heuristic "
            f"(shadow would have been {rec.formation}/{rec.play}"
            + (f"|{rec.adjustment}" if rec.adjustment else "")
            + ")"
        ),
        shadow=shadow,
    )
