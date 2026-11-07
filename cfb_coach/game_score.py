"""Optional live score (us vs them) for situational play-calling.

Aidan can set the score once per session (CLI flag, `score 21-14` in the sit>
loop, or the HTML us/them boxes). It sticks until he updates it. Down, distance,
field zone, quarter/clock, and the opponent persona stay in charge; the score
only leans the mix when the game is actually score-sensitive (late, two-minute,
or a two-score margin). A close first half stays neutral.

This is not the end-of-game score. Game over still stores a winner-first final
(`24-17` on a win) for retrain. Live input is always us-them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_US_THEM = re.compile(r"\bus\s+(\d{1,3})\s+them\s+(\d{1,3})\b", re.I)
_PAIR = re.compile(r"(\d{1,3})\s*(?:-|–|—|\bto\b)\s*(\d{1,3})", re.I)
_PAIR_ONLY = re.compile(r"(\d{1,3})\s+(\d{1,3})")
_INLINE_SCORE = re.compile(
    r"\bscore\s+(\d{1,3})\s*(?:-|–|—|\bto\b)\s*(\d{1,3})\b",
    re.I,
)
_QTR = re.compile(
    r"\bq(?:tr|uarter)?\s*([1-5])\b|\b([1-5])q\b|\b([1-4])(?:st|nd|rd|th)\s+quarter\b|\b(?:ot|overtime)\b",
    re.I,
)
_SNAP_NOTE = re.compile(r"score_us=(\d+)\s+score_them=(\d+)")

# Margins where a 2-point try (not the extra point) is the football decision
# after a touchdown. Advisory only — the caller has no separate 2-pt play hook.
_TWO_POINT_MARGINS = frozenset({-2, -5, -8, -10, -16})

_HEAT_PLAY = re.compile(r"blitz|fire|sim\b|mug", re.I)


@dataclass(frozen=True)
class GameScore:
    """Current score from our perspective: `us` is Aidan, `them` is the opponent."""

    us: int
    them: int

    @property
    def margin(self) -> int:
        return self.us - self.them

    @property
    def label(self) -> str:
        return f"{self.us}-{self.them}"


@dataclass
class LiveContext:
    """Score and quarter remembered across snaps in one play session."""

    score: GameScore | None = None
    quarter: int | None = None

    def describe(self) -> str:
        bits: list[str] = []
        if self.score is not None:
            bits.append(f"score {self.score.label} (us-them)")
        if self.quarter is not None:
            bits.append(f"Q{self.quarter}" if self.quarter < 5 else "OT")
        return ", ".join(bits)


@dataclass(frozen=True)
class ScoreContext:
    us: int
    them: int
    margin: int
    quarter: int | None
    late: bool
    phase: str  # neutral | protect | prevent | trail
    strength: float  # 0..1, before down/field scaling


def parse_live_score(text: str | None) -> GameScore | None:
    """Parse an us-them score. Accepts `21-14`, `21 to 14`, `us 21 them 14`, `21 14`."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw or raw.lower() in {"clear", "off", "none", "reset", "?"}:
        return None
    m = _US_THEM.search(raw)
    if m:
        return GameScore(int(m.group(1)), int(m.group(2)))
    m = _PAIR.search(raw)
    if m:
        return GameScore(int(m.group(1)), int(m.group(2)))
    m = _PAIR_ONLY.fullmatch(raw)
    if m:
        return GameScore(int(m.group(1)), int(m.group(2)))
    return None


def context_from_args(args: object) -> LiveContext:
    """Build a session context from `--score` / `--quarter`. Bad `--score` exits."""
    raw = getattr(args, "score", None)
    score = None
    if raw not in (None, ""):
        score = parse_live_score(str(raw))
        if score is None:
            raise SystemExit(f"--score must be us-them like 21-14 (got {raw!r})")
    q = getattr(args, "quarter", None)
    quarter = None
    if q not in (None, ""):
        try:
            quarter = int(q)
        except (TypeError, ValueError):
            raise SystemExit(f"--quarter must be 1-5 (5 = OT), got {q!r}") from None
        if not 1 <= quarter <= 5:
            raise SystemExit("--quarter must be 1-5 (5 = OT)")
    return LiveContext(score=score, quarter=quarter)


def interpret_live_command(raw: str, ctx: LiveContext) -> str | None:
    """Handle `score …` / `quarter …` as a whole line. None means it is a situation."""
    text = (raw or "").strip()
    if not text:
        return None
    m = re.fullmatch(r"(?:score|pts)(?:\s+(.*))?$", text, re.I)
    if m:
        rest = (m.group(1) or "").strip()
        if not rest or rest.lower() in {"?", "show"}:
            shown = ctx.describe() or "not set"
            return f"  score: {shown}. Set with `score 21-14` (us-them). `score clear` to drop it."
        if rest.lower() in {"clear", "off", "none", "reset"}:
            ctx.score = None
            return "  score cleared — calls ignore score until you set it again"
        parsed = parse_live_score(rest)
        if parsed is None:
            return f"  score not understood: {rest!r} — use us-them, e.g. score 21-14 or score us 21 them 14"
        ctx.score = parsed
        return f"  score → {parsed.label} (us-them). Sticks for the rest of this session."
    m = re.fullmatch(r"(?:quarter|qtr|q)\s*([1-5]|ot)", text, re.I)
    if m:
        tok = m.group(1).lower()
        ctx.quarter = 5 if tok == "ot" else int(tok)
        label = "OT" if ctx.quarter == 5 else f"Q{ctx.quarter}"
        return f"  quarter → {label}. Sticks until you change it."
    if re.fullmatch(r"(?:quarter|qtr)\s+(?:clear|off|none)", text, re.I):
        ctx.quarter = None
        return "  quarter cleared"
    m = re.fullmatch(r"us\s+(\d{1,3})\s+them\s+(\d{1,3})", text, re.I)
    if m:
        ctx.score = GameScore(int(m.group(1)), int(m.group(2)))
        return f"  score → {ctx.score.label} (us-them). Sticks for the rest of this session."
    return None


def sit_prompt(side: str, ctx: LiveContext) -> str:
    """Terminal prompt. With no score/quarter this stays `[O] sit>`."""
    tag = (side or "offense")[:1].upper()
    extra: list[str] = []
    if ctx.score is not None:
        extra.append(ctx.score.label)
    if ctx.quarter is not None:
        extra.append("OT" if ctx.quarter >= 5 else f"Q{ctx.quarter}")
    head = tag if not extra else f"{tag} {' '.join(extra)}"
    return f"[{head}] sit> "


def parse_inline_score(text: str) -> GameScore | None:
    """Score written inside a situation line: `score 21-14` or `us 21 them 14`."""
    if not text:
        return None
    m = _US_THEM.search(text)
    if m:
        return GameScore(int(m.group(1)), int(m.group(2)))
    m = _INLINE_SCORE.search(text)
    if m:
        return GameScore(int(m.group(1)), int(m.group(2)))
    return None


def parse_inline_quarter(text: str) -> int | None:
    """`q4`, `4q`, `4th quarter`, `ot` — not `4th and 1`."""
    if not text:
        return None
    m = _QTR.search(text)
    if not m:
        return None
    if m.group(0).lower() in {"ot", "overtime"}:
        return 5
    for g in m.groups():
        if g:
            return int(g)
    return None


def quarter_of(sit: object) -> int | None:
    extras = getattr(sit, "extras", None) or {}
    if not isinstance(extras, dict):
        return None
    q = extras.get("quarter")
    try:
        return int(q) if q not in (None, "") else None
    except (TypeError, ValueError):
        return None


def absorb_and_stamp(sit: object, ctx: LiveContext) -> None:
    """Copy an inline score/quarter onto the session, and fill gaps from the session.

    A snap that names its own score wins for that snap and becomes the new session
    score, so Aidan can update mid-game without a separate command.
    """
    us = getattr(sit, "score_us", None)
    them = getattr(sit, "score_them", None)
    if us is not None and them is not None:
        ctx.score = GameScore(int(us), int(them))
    elif ctx.score is not None:
        sit.score_us = ctx.score.us  # type: ignore[attr-defined]
        sit.score_them = ctx.score.them  # type: ignore[attr-defined]
    extras = getattr(sit, "extras", None)
    if not isinstance(extras, dict):
        return
    q = extras.get("quarter")
    if q not in (None, ""):
        try:
            ctx.quarter = int(q)
        except (TypeError, ValueError):
            pass
    elif ctx.quarter is not None:
        extras["quarter"] = ctx.quarter


def classify(sit: object) -> ScoreContext | None:
    """How hard the score should lean this snap. None when no score is set."""
    us = getattr(sit, "score_us", None)
    them = getattr(sit, "score_them", None)
    if us is None or them is None:
        return None
    margin = int(us) - int(them)
    q = quarter_of(sit)
    late = bool(getattr(sit, "two_minute", False)) or (q is not None and q >= 4)
    blowout_up = margin >= 14
    blowout_down = margin <= -14
    if not late and not blowout_up and not blowout_down:
        phase, strength = "neutral", 0.0
    elif blowout_up:
        phase, strength = "prevent", (1.0 if late else 0.7)
    elif late and margin >= 8:
        phase, strength = "protect", 0.9
    elif late and margin >= 4:
        phase, strength = "protect", 0.55
    elif late and margin > 0:
        phase, strength = "protect", 0.4
    elif blowout_down or (late and margin <= -3):
        if late and margin <= -8:
            phase, strength = "trail", 1.0
        elif blowout_down and not late:
            phase, strength = "trail", 0.65
        else:
            phase, strength = "trail", 0.75
    elif late and margin < 0:
        phase, strength = "trail", 0.55
    else:
        phase, strength = "neutral", 0.0
    return ScoreContext(
        us=int(us), them=int(them), margin=margin, quarter=q, late=late, phase=phase, strength=strength,
    )


def situation_scale(sit: object) -> float:
    """Down/field still wins. Score is a lean, not a new menu, on GL, short, and 3rd-and-long."""
    if getattr(sit, "goal_line", False):
        return 0.2
    if getattr(sit, "red_zone", False) and getattr(sit, "short_yardage", False):
        return 0.3
    if getattr(sit, "short_yardage", False):
        return 0.4
    dist = getattr(sit, "distance", None)
    long = bool(getattr(sit, "long_yardage", False)) or (
        getattr(sit, "down", None) in (3, 4) and dist is not None and dist >= 7
    )
    if long:
        return 0.45
    return 1.0


def offense_score_bonus(sit: object, play: str, *, run: bool = False, deep: bool = False) -> float:
    """Additive sit-bonus. 0 when the score is unset or the game is early and close.

    Protect / up big: runs and quick game up, explosives down (clock).
    Trailing: explosives and tempo up, pure runs down. Short yardage and the goal
    line scale this down so the existing D&D menu stays in front.
    """
    ctx = classify(sit)
    if ctx is None or ctx.phase == "neutral":
        return 0.0
    eff = ctx.strength * situation_scale(sit)
    if eff < 0.15:
        return 0.0
    if ctx.phase in ("protect", "prevent"):
        if run:
            b = 0.16 * eff
        elif deep:
            b = -0.18 * eff
        else:
            b = 0.05 * eff
        if ctx.phase == "prevent":
            b += (0.04 if run else (-0.04 if deep else 0.0)) * eff
    else:
        if deep:
            b = 0.18 * eff
        elif run:
            b = -0.10 * eff
        else:
            b = 0.06 * eff
    if getattr(sit, "down", None) == 4 and not getattr(sit, "goal_line", False):
        if ctx.phase == "trail" and ctx.late and (deep or not run):
            b += 0.06 * eff
        elif ctx.phase in ("protect", "prevent") and run:
            b += 0.06 * eff
        elif ctx.phase in ("protect", "prevent") and deep:
            b -= 0.06 * eff
    return max(-0.22, min(0.22, round(b, 3)))


def shift_defense_mix(mix: dict[str, float], sit: object) -> dict[str, float]:
    """Lean a Madden coverage-family mix. Returns the same shares when score is neutral."""
    ctx = classify(sit)
    if ctx is None or ctx.phase == "neutral":
        return dict(mix)
    eff = ctx.strength * situation_scale(sit)
    if eff < 0.35:
        return dict(mix)
    out = {k: float(v) for k, v in mix.items()}
    if ctx.phase in ("prevent", "protect"):
        soft = 0.28 if ctx.phase == "prevent" else 0.14
        heat = 0.22 if ctx.phase == "prevent" else 0.12
        out["two_high"] = out.get("two_high", 0.0) + soft * eff
        out["cover2"] = out.get("cover2", 0.0) + 0.08 * eff
        out["pressure"] = max(0.02, out.get("pressure", 0.0) - heat * eff)
        out["man"] = max(0.02, out.get("man", 0.0) - 0.08 * eff)
    else:
        out["pressure"] = out.get("pressure", 0.0) + 0.24 * eff
        out["man"] = out.get("man", 0.0) + 0.12 * eff
        out["two_high"] = max(0.04, out.get("two_high", 0.0) - 0.14 * eff)
    total = sum(max(0.0, v) for v in out.values()) or 1.0
    return {k: max(0.0, v) / total for k, v in out.items()}


def adjust_cfb_defense(
    sit: object,
    form: str,
    play: str,
    macro: str,
    user: str,
    rationale: str,
) -> tuple[str, str, str, str, str]:
    """Shift the CFB shell when the score is loud. Repeated-tendency macros stay.

    Goal line and true short yardage keep the heavy package. Obvious passing downs
    keep the quarters shell when trailing (don't blitz into a bomb).
    """
    ctx = classify(sit)
    if ctx is None or ctx.phase == "neutral":
        return form, play, macro, user, rationale
    eff = ctx.strength * situation_scale(sit)
    if eff < 0.45:
        return form, play, macro, user, rationale
    if getattr(sit, "goal_line", False):
        return form, play, macro, user, rationale
    if getattr(sit, "short_yardage", False) and getattr(sit, "down", None) in (3, 4) and (getattr(sit, "distance", None) or 99) <= 2:
        return form, play, macro, user, rationale
    if "6-1" in (form or "") or form.startswith("4-3"):
        return form, play, macro, user, rationale
    armed = (macro or "none") not in ("none", "", "HEAT")
    if armed:
        return form, play, macro, user, rationale
    long = bool(getattr(sit, "long_yardage", False)) or (
        getattr(sit, "down", None) in (3, 4) and (getattr(sit, "distance", None) or 0) >= 8
    )
    soft_already = long or "quarter" in (play or "").lower() or "dime" in (form or "").lower()
    if ctx.phase in ("prevent", "protect") and soft_already:
        if macro == "HEAT" or _HEAT_PLAY.search(play or ""):
            return "Nickel Over", "Cover 4 Quarters", "none", "User seam", rationale
        return form, play, macro, user, rationale
    if ctx.phase == "prevent" or (ctx.phase == "protect" and eff >= 0.75):
        return "Nickel Over", "Cover 4 Quarters", "none", "User seam", rationale
    if ctx.phase == "protect":
        if macro == "HEAT" or _HEAT_PLAY.search(play or ""):
            return "Nickel Over", "Cover 3 Sky", "none", "User hook", rationale
        return form, play, macro, user, rationale
    if long:
        return form, play, macro, user, rationale
    return "Nickel Double Mug", "Mid Blitz 0", "HEAT", "User hot", rationale


def score_call_note(sit: object) -> str:
    """One rationale clause. Empty when no score is set."""
    ctx = classify(sit)
    if ctx is None:
        return ""
    head = f"score {ctx.us}-{ctx.them}"
    if ctx.quarter is not None:
        head += " OT" if ctx.quarter >= 5 else f" Q{ctx.quarter}"
    elif getattr(sit, "two_minute", False):
        head += " 2min"
    side = getattr(sit, "side", "offense") or "offense"
    if ctx.phase == "neutral":
        return f"{head} neutral — early/close, score not moving the call"
    if side == "defense":
        if ctx.phase == "prevent":
            body = "prevent — soft shell, keep everything in front"
        elif ctx.phase == "protect":
            body = "protect the lead — sound coverage, less heat"
        else:
            body = "trailing — aggression to get the ball back"
    else:
        if ctx.phase == "prevent":
            body = "up big — milk the clock, safer mix"
        elif ctx.phase == "protect":
            body = "protect the lead — clock and play-clock, fewer explosives"
        else:
            body = "trailing — tempo and explosives"
    bits = [f"{head} {body}"]
    bits.extend(_decision_bits(sit, ctx))
    return " | ".join(bits)


def _decision_bits(sit: object, ctx: ScoreContext) -> list[str]:
    """4th-down and 2-pt advisories. There is no separate 2-pt play to call."""
    if ctx.phase == "neutral" or ctx.strength < 0.35:
        return []
    out: list[str] = []
    side = getattr(sit, "side", "offense") or "offense"
    if side == "offense" and getattr(sit, "down", None) == 4 and not getattr(sit, "goal_line", False):
        if ctx.phase == "trail" and ctx.late:
            out.append("4th down: go for it — trailing late")
        elif ctx.phase in ("protect", "prevent"):
            out.append("4th down: don't force it — protect the lead")
    if (
        side == "offense"
        and ctx.late
        and (getattr(sit, "red_zone", False) or getattr(sit, "goal_line", False))
        and ctx.margin in _TWO_POINT_MARGINS
    ):
        out.append(f"2-pt live if this drive scores (down {abs(ctx.margin)})")
    return out


def snap_notes_for(sit: object) -> str | None:
    """Notes column text so ingest can see the live score without touching the final W/L score."""
    us = getattr(sit, "score_us", None)
    them = getattr(sit, "score_them", None)
    if us is None or them is None:
        return None
    return f"score_us={int(us)} score_them={int(them)}"


def parse_snap_score_note(notes: str | None) -> tuple[int, int] | None:
    if not notes:
        return None
    m = _SNAP_NOTE.search(str(notes))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


__all__ = [
    "GameScore",
    "LiveContext",
    "ScoreContext",
    "absorb_and_stamp",
    "adjust_cfb_defense",
    "classify",
    "context_from_args",
    "interpret_live_command",
    "offense_score_bonus",
    "parse_inline_quarter",
    "parse_inline_score",
    "parse_live_score",
    "parse_snap_score_note",
    "score_call_note",
    "shift_defense_mix",
    "sit_prompt",
    "situation_scale",
    "snap_notes_for",
]
