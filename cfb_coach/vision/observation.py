"""GameObservation — structured live-vision state (Milestone 1).

Maps into DefenseLook for the coach tip contract. Classical CV only;
no LLM / cloud vision per frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from cfb_coach.vision.look import DefenseLook, PRESSURES, SHELLS

# Formation families (offense alignment we see)
FORMATIONS = ("bunch", "trips", "empty", "2x2", "other", "unknown")

# Shell from high safeties (coarse)
OBS_SHELLS = ("one_high", "two_high", "unknown")

# Coarse pressure vocab (maps to DefenseLook PRESSURES)
OBS_PRESSURES = ("none", "show", "left", "right", "middle", "all_out", "unknown")

PLAY_STATES = (
    "UNKNOWN",
    "MENU",
    "BETWEEN_PLAYS",
    "PRE_SNAP",
    "PLAY_ACTIVE",
    "PLAY_ENDING",
    "POST_PLAY",
    # M1 compat aliases (accepted in normalize)
    "PLAY_ENDED",
    "OTHER",
)

# Confidence below this → force field to unknown
DEFAULT_CONF_THRESHOLD = 0.35


@dataclass
class SituationHUD:
    """Coarse HUD fields (OCR optional / stubbed)."""

    down: int | None = None
    distance: int | None = None
    quarter: int | None = None
    clock: str | None = None
    score_us: int | None = None
    score_them: int | None = None
    side: str | None = None  # "O" | "D" | None


@dataclass
class GameObservation:
    """Full frame observation fed into coach via DefenseLook bridge."""

    situation: SituationHUD = field(default_factory=SituationHUD)
    formation: str = "unknown"  # bunch|trips|empty|2x2|other|unknown
    formation_side: str = ""  # L|R| for bunch/trips bias
    shell: str = "unknown"  # one_high|two_high|unknown
    pressure: str = "unknown"  # none|show|left|right|middle|all_out|unknown
    play_state: str = "OTHER"
    confidence: dict[str, float] = field(default_factory=dict)
    source: str = "stub"
    notes: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    ts: float | None = None  # monotonic or wall time from pipeline

    def conf(self, key: str, default: float = 0.0) -> float:
        return float(self.confidence.get(key, default))

    def normalized(self, *, conf_threshold: float = DEFAULT_CONF_THRESHOLD) -> GameObservation:
        """Normalize vocab; low-confidence fields → unknown."""
        formation = (self.formation or "unknown").lower().strip()
        shell = (self.shell or "unknown").lower().strip()
        pressure = (self.pressure or "unknown").lower().strip()
        play_state = (self.play_state or "OTHER").upper().strip()

        formation_aliases = {
            "bunch_l": "bunch",
            "bunch_r": "bunch",
            "trips_l": "trips",
            "trips_r": "trips",
            "trips_left": "trips",
            "trips_right": "trips",
            "bunch_left": "bunch",
            "bunch_right": "bunch",
            "doubles": "2x2",
            "spread": "empty",
            "empty_backfield": "empty",
        }
        shell_aliases = {
            "1h": "one_high",
            "single_high": "one_high",
            "one": "one_high",
            "2h": "two_high",
            "two": "two_high",
        }
        pressure_aliases = {
            "off": "none",
            "clean": "none",
            "0": "none",
            "sim": "show",
            "show_blitz": "show",
            "l": "left",
            "blitz_left": "left",
            "r": "right",
            "blitz_right": "right",
            "mid": "middle",
            "blitz_middle": "middle",
            "a": "all_out",
            "zero_blitz": "all_out",
            "heat": "middle",
            "blitz": "middle",
        }

        formation = formation_aliases.get(formation, formation)
        shell = shell_aliases.get(shell, shell)
        pressure = pressure_aliases.get(pressure, pressure)

        if formation not in FORMATIONS:
            formation = "unknown"
        if shell not in OBS_SHELLS:
            shell = "unknown"
        if pressure not in OBS_PRESSURES:
            pressure = "unknown"
        # Canonicalize M1 aliases
        _alias = {"OTHER": "UNKNOWN", "PLAY_ENDED": "PLAY_ENDING"}
        play_state = _alias.get(play_state, play_state)
        _canonical = (
            "UNKNOWN", "MENU", "BETWEEN_PLAYS", "PRE_SNAP",
            "PLAY_ACTIVE", "PLAY_ENDING", "POST_PLAY",
        )
        if play_state not in _canonical:
            play_state = "UNKNOWN"

        conf = {k: max(0.0, min(1.0, float(v))) for k, v in (self.confidence or {}).items()}

        # Force unknown when confidence is low
        if formation != "unknown" and conf.get("formation", 1.0) < conf_threshold:
            formation = "unknown"
        if shell != "unknown" and conf.get("shell", 1.0) < conf_threshold:
            shell = "unknown"
        if pressure != "unknown" and conf.get("pressure", 1.0) < conf_threshold:
            pressure = "unknown"
        if play_state not in ("UNKNOWN", "MENU") and conf.get("play_state", 1.0) < conf_threshold:
            play_state = "UNKNOWN"

        side = (self.formation_side or "").upper().strip()
        if side not in ("L", "R", ""):
            side = ""

        sit = self.situation
        if not isinstance(sit, SituationHUD):
            sit = SituationHUD()

        return GameObservation(
            situation=SituationHUD(
                down=sit.down,
                distance=sit.distance,
                quarter=sit.quarter,
                clock=sit.clock,
                score_us=sit.score_us,
                score_them=sit.score_them,
                side=sit.side,
            ),
            formation=formation,
            formation_side=side,
            shell=shell,
            pressure=pressure,
            play_state=play_state,
            confidence=conf,
            source=self.source,
            notes=self.notes,
            extras=dict(self.extras),
            ts=self.ts,
        )

    def to_defense_look(self) -> DefenseLook:
        """Bridge to existing coach tip contract."""
        n = self.normalized()
        shell_map = {
            "one_high": "single_high",
            "two_high": "two_high",
            "unknown": "unknown",
        }
        pressure_map = {
            "none": "none",
            "show": "show",
            "left": "blitz_left",
            "right": "blitz_right",
            "middle": "blitz_middle",
            "all_out": "all_out",
            "unknown": "unknown",
        }
        # Overall confidence = min of available major fields, or mean
        keys = [k for k in ("shell", "pressure", "formation") if k in n.confidence]
        if keys:
            overall = sum(n.confidence[k] for k in keys) / len(keys)
        else:
            overall = 0.0

        extras = dict(n.extras)
        extras["formation"] = n.formation
        extras["formation_side"] = n.formation_side
        extras["play_state"] = n.play_state
        extras["obs_shell"] = n.shell
        extras["obs_pressure"] = n.pressure
        extras["situation"] = asdict(n.situation)
        extras["confidence"] = dict(n.confidence)

        look = DefenseLook(
            front="unknown",  # front ID deferred; keep coarse
            shell=shell_map.get(n.shell, "unknown"),
            pressure=pressure_map.get(n.pressure, "unknown"),
            confidence=overall,
            source=n.source if n.source != "stub" else "capture",
            notes=n.notes,
            extras=extras,
        )
        return look.normalized()

    def short_line(self) -> str:
        """Live HUD-style line: 3&7 | BUNCH R | 2-HIGH | PRESSURE L"""
        n = self.normalized()
        sit = n.situation
        if sit.down is not None and sit.distance is not None:
            sit_s = f"{sit.down}&{sit.distance}"
        elif sit.down is not None:
            sit_s = f"{sit.down}&?"
        else:
            sit_s = "?&?"

        form = (n.formation or "unknown").upper()
        if form == "2X2":
            form = "2x2"
        if n.formation_side:
            form = f"{form} {n.formation_side}"

        shell_disp = {
            "one_high": "1-HIGH",
            "two_high": "2-HIGH",
            "unknown": "?-HIGH",
        }.get(n.shell, "?-HIGH")

        press_disp = {
            "none": "PRESSURE NONE",
            "show": "PRESSURE SHOW",
            "left": "PRESSURE L",
            "right": "PRESSURE R",
            "middle": "PRESSURE MID",
            "all_out": "PRESSURE ALL",
            "unknown": "PRESSURE ?",
        }.get(n.pressure, "PRESSURE ?")

        return f"{sit_s} | {form} | {shell_disp} | {press_disp}"

    def to_dict(self) -> dict[str, Any]:
        n = self.normalized()
        d = asdict(n)
        return d


# Silence unused import lint for SHELLS/PRESSURES re-export consumers
_ = (SHELLS, PRESSURES)
