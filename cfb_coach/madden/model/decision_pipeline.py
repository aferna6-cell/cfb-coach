"""Single sealed decision assembly — disabled by default.

When ``CFB_COACH_SEALED_PIPELINE=1`` or meta ``ml_sealed_pipeline=1``, live
callers may use :func:`assemble_decision` instead of the post-hoc VOD swap
path. Heuristic compatibility mode keeps the existing displayed call and
only records a parity report.

Do not activate hybrid selection from this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from cfb_coach.madden.model.schema import (
    CandidatePlay,
    CoachingMode,
    legal_candidates,
)

FLAG_ENV = "CFB_COACH_SEALED_PIPELINE"
FLAG_META = "ml_sealed_pipeline"


def pipeline_enabled(db: Any = None) -> bool:
    env = (os.environ.get(FLAG_ENV) or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    if db is not None:
        try:
            raw = db.get_meta(FLAG_META)
            if str(raw or "").strip().lower() in ("1", "true", "yes", "on"):
                return True
        except Exception:  # noqa: BLE001
            return False
    return False


@dataclass
class SealedCall:
    formation: str | None
    play: str | None
    macro: str | None
    side: str
    reads: tuple[str, ...] = ()
    adjustments: tuple[str, ...] = ()
    rationale: str = ""
    source: str = "heuristic"
    book_legal: bool = True


@dataclass
class ParityReport:
    matched: bool
    differences: list[str] = field(default_factory=list)
    old: dict[str, Any] = field(default_factory=dict)
    new: dict[str, Any] = field(default_factory=dict)


def assemble_decision(
    *,
    situation: Any,
    opponent_id: str,
    db: Any,
    make_heuristic: Callable[..., Any],
    candidates: Sequence[CandidatePlay],
    mode: CoachingMode = CoachingMode.HEURISTIC,
    vod_prior: CandidatePlay | None = None,
) -> tuple[Any, SealedCall, ParityReport | None]:
    """Build one sealed call: candidates → heuristic → optional prior → reads.

    Returns ``(legacy_call, sealed, parity)``. When the flag is off, returns
    the legacy call unchanged and an empty parity report.
    """
    legacy = make_heuristic(situation)
    sealed = SealedCall(
        formation=getattr(legacy, "formation", None),
        play=getattr(legacy, "play", None),
        macro=getattr(legacy, "macro", None),
        side=getattr(legacy, "side", "offense") or "offense",
        rationale=str(getattr(legacy, "rationale", "") or ""),
        source="heuristic",
        book_legal=True,
    )
    if not pipeline_enabled(db):
        return legacy, sealed, None

    legal = legal_candidates(candidates)
    # VOD prior may only select an already-legal candidate — never invent names.
    if vod_prior is not None and vod_prior.in_applied_book.value == "true":
        for cand in legal:
            if cand.formation == vod_prior.formation and cand.play == vod_prior.play:
                sealed = SealedCall(
                    formation=cand.formation,
                    play=cand.play,
                    macro=getattr(legacy, "macro", None),
                    side=sealed.side,
                    rationale="vod_prior+" + sealed.rationale,
                    source="vod_prior",
                    book_legal=True,
                )
                break

    # Reads/macros stay attached to the sealed candidate (no post-seal swap).
    # Compatibility: compare against the legacy call that already applied VOD.
    parity = compare_calls(legacy, sealed)
    if mode is CoachingMode.HEURISTIC:
        # Keep displaying the legacy call until parity is proven.
        return legacy, sealed, parity
    return legacy, sealed, parity


def compare_calls(legacy: Any, sealed: SealedCall) -> ParityReport:
    old = {
        "formation": getattr(legacy, "formation", None),
        "play": getattr(legacy, "play", None),
        "macro": getattr(legacy, "macro", None),
        "side": getattr(legacy, "side", None),
    }
    new = {
        "formation": sealed.formation,
        "play": sealed.play,
        "macro": sealed.macro,
        "side": sealed.side,
    }
    diffs = [k for k in old if old[k] != new[k]]
    return ParityReport(matched=not diffs, differences=diffs, old=old, new=new)


def optional_vod_lookup(raw_name: str, models_dir: Any = None) -> CandidatePlay | None:
    """Optional PR #19 mapping boundary. Returns None when mappings are absent."""
    try:
        from cfb_coach.vod_model.mappings import load_mappings, lookup  # type: ignore
    except Exception:  # noqa: BLE001 — PR #19 not merged
        return None
    try:
        index = load_mappings(models_dir)
        mapped = lookup(index, "madden27", raw_name)
        if mapped is None:
            return None
        # Mapped play is a candidate identity, not a concept prediction.
        return None  # Caller must resolve via candidate_in_book with applied book.
    except Exception:  # noqa: BLE001
        return None
