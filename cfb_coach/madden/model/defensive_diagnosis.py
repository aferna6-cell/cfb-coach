"""Defensive picture for the offensive coordinator.

Observed, inferred, and unknown stay separate. A two-high shell is not
Cover 2 or Quarters. A previous snap is not this snap. A formation name
does not identify coverage.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

DIAGNOSIS_VERSION = "defensive_diagnosis.v1"
SHELLS: tuple[str, ...] = (
    "cover_0", "cover_1", "cover_2", "cover_3", "cover_4", "cover_6",
)
# One pseudo-count per shell. Sparse samples stay near the uniform prior.
PRIOR_COUNT = 1.0
# Do not publish an inferred shell until the sample actually moves the prior.
MIN_INFER_CONFIDENCE = 0.40
MIN_INFER_MARGIN = 0.15

_COVER_0 = re.compile(r"cover\s*0|\bzero\b", re.I)
_COVER_1 = re.compile(r"cover\s*1", re.I)
_COVER_2 = re.compile(r"cover\s*2|\btampa\b", re.I)
_COVER_3 = re.compile(r"cover\s*3", re.I)
_COVER_4 = re.compile(r"cover\s*4|\bquarters\b|\bpalms\b", re.I)
_COVER_6 = re.compile(r"cover\s*6", re.I)
_MAN = re.compile(r"\bman\b", re.I)
_ZONE = re.compile(r"\bzone\b", re.I)
_MATCH = re.compile(r"\bmatch\b|pattern[\s-]*match|zone[\s-]*match", re.I)
_TWO_HIGH = re.compile(r"two[\s-]*high|split[\s-]*field", re.I)
_SINGLE_HIGH = re.compile(r"single[\s-]*high|\bmofc\b", re.I)
_PRESS = re.compile(r"\bpress\b", re.I)
_OFF = re.compile(r"\boff\b|\bbail\b", re.I)
_LEVERAGE = re.compile(r"\binside\b|\boutside\b|leverage", re.I)
_PRESSURE = re.compile(r"pressure|blitz|\bfire\b|\bmug\b", re.I)
_BOX = re.compile(r"(\d)\s*man\s*box|\bbox\s*(\d)", re.I)


def _shell_from_text(text: str) -> str | None:
    """Explicit cover name only. Two-high words are not a shell."""
    if _COVER_0.search(text):
        return "cover_0"
    if _COVER_1.search(text):
        return "cover_1"
    if _COVER_6.search(text):
        return "cover_6"
    if _COVER_4.search(text):
        return "cover_4"
    if _COVER_2.search(text) and not _TWO_HIGH.search(text):
        return "cover_2"
    if _COVER_2.search(text) and _COVER_2.search(text):
        return "cover_2"
    if _COVER_3.search(text):
        return "cover_3"
    return None


def _structure(text: str, shell: str | None) -> str | None:
    if shell in ("cover_0", "cover_1", "cover_3"):
        return "single_high"
    if shell in ("cover_2", "cover_4", "cover_6"):
        return "two_high"
    if _TWO_HIGH.search(text):
        return "two_high"
    if _SINGLE_HIGH.search(text):
        return "single_high"
    return None


def parse_look(text: str | None) -> dict[str, Any]:
    """Parse words that were actually present. Do not complete the call."""
    raw = text or ""
    shell = _shell_from_text(raw) if raw else None
    # "two-high" alone must not fall through a Cover 2 regex.
    if raw and _TWO_HIGH.search(raw) and not re.search(r"cover\s*[0-9]", raw, re.I):
        shell = None
    principles: list[str] = []
    if shell == "cover_0" or _PRESSURE.search(raw):
        principles.append("pressure")
    if _MAN.search(raw) or shell == "cover_1":
        principles.append("man")
    if _ZONE.search(raw) or shell in ("cover_2", "cover_3", "cover_4", "cover_6"):
        principles.append("zone")
    if _MATCH.search(raw):
        principles.append("match")
    return {
        "shell": shell,
        "structure": _structure(raw, shell),
        "principles": principles,
        "press": True if _PRESS.search(raw) else (False if _OFF.search(raw) else None),
        "leverage": "mentioned" if _LEVERAGE.search(raw) else None,
        "box": _box(raw),
        "pressure": bool(_PRESSURE.search(raw) or shell == "cover_0"),
        "match_family": (
            "pattern_match" if re.search(r"pattern", raw, re.I)
            else "zone_match" if re.search(r"zone[\s-]*match", raw, re.I)
            else "match_unspecified" if _MATCH.search(raw)
            else None
        ),
    }


def _box(text: str) -> int | None:
    match = _BOX.search(text or "")
    if not match:
        return None
    raw = match.group(1) or match.group(2)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if 5 <= value <= 9 else None


def shrink_shells(counts: Mapping[str, int]) -> dict[str, Any]:
    """Posterior over explicit shells. Uniform prior, reported sample size."""
    observed = {shell: int(counts.get(shell) or 0) for shell in SHELLS}
    total = sum(observed.values())
    denominator = total + PRIOR_COUNT * len(SHELLS)
    posterior = {
        shell: round((observed[shell] + PRIOR_COUNT) / denominator, 4)
        for shell in SHELLS
    }
    ordered = sorted(SHELLS, key=lambda shell: (-posterior[shell], shell))
    mode = ordered[0]
    margin = posterior[mode] - posterior[ordered[1]]
    confidence = round(total / denominator, 4) if denominator else 0.0
    publish = confidence >= MIN_INFER_CONFIDENCE and margin >= MIN_INFER_MARGIN
    return {
        "sample_size": total,
        "prior_per_shell": PRIOR_COUNT,
        "posterior": posterior,
        "confidence": confidence,
        "mode": mode if publish else None,
        "margin": round(margin, 4),
        "state": "inferred" if publish else "unknown",
        "note": (
            "Shrinkage toward a uniform shell prior. A thin sample does not "
            "become a coverage call."
        ),
    }


def _historical_counts(memory: Mapping[str, Any] | None) -> dict[str, int]:
    counts = {shell: 0 for shell in SHELLS}
    if not memory:
        return counts
    labels = list(memory.get("observed_labels") or [])
    if not labels and memory.get("distribution"):
        # coverage_class buckets are not shells. Do not promote them.
        return counts
    for label in labels:
        parsed = parse_look(str(label))
        shell = parsed.get("shell")
        if shell in counts:
            counts[shell] += 1
    return counts


def diagnose_defense(
    sit: Any = None,
    memory: Mapping[str, Any] | None = None,
    *,
    formation_name: str | None = None,
) -> dict[str, Any]:
    """Current snap first. History can only be inferred. Formation is ignored."""
    del formation_name  # a formation name is not a coverage call
    hint = getattr(sit, "coverage_hint", None) if sit is not None else None
    source = getattr(sit, "coverage_source", None) if sit is not None else None
    source = source or "none"
    parsed = parse_look(hint if source == "live" else None)
    history = shrink_shells(_historical_counts(memory))
    observed = source == "live" and bool(hint) and (
        parsed["shell"] or parsed["structure"] or parsed["principles"] or parsed["pressure"]
    )
    inferred_shell = None if observed else history["mode"]
    if source == "last":
        current_state = "unknown"
        note = "previous snap only; not the current coverage"
    elif observed:
        current_state = "observed"
        note = "pre-snap words on this snap; unspecified shells stay unknown"
    elif history["state"] == "inferred":
        current_state = "inferred"
        note = "historical explicit shells after shrinkage; not this snap's coverage"
    else:
        current_state = "unknown"
        note = "current coverage was not established"
    return {
        "version": DIAGNOSIS_VERSION,
        "state": current_state,
        "observed": {
            "shell": parsed["shell"] if observed else None,
            "structure": parsed["structure"] if observed else None,
            "principles": list(parsed["principles"]) if observed else [],
            "pressure": bool(parsed["pressure"]) if observed else False,
            "press": parsed["press"] if observed else None,
            "leverage": parsed["leverage"] if observed else None,
            "box": parsed["box"] if observed else None,
            "match_family": parsed["match_family"] if observed else None,
            "label": hint if observed else None,
        },
        "inferred": {
            "shell": inferred_shell,
            "sample_size": history["sample_size"],
            "confidence": history["confidence"] if inferred_shell else 0.0,
            "posterior": history["posterior"],
            "margin": history["margin"],
        },
        "unknown": [
            name for name, present in (
                ("shell", not (observed and parsed["shell"]) and not inferred_shell),
                ("structure", not (observed and parsed["structure"])),
                ("front", True),
                ("run_fit", True),
                ("leverage", not (observed and parsed["leverage"])),
                ("box", not (observed and parsed["box"])),
            ) if present
        ],
        "two_high_is_not_a_confirmed_shell": True,
        "formation_name_ignored": True,
        "previous_snap_not_copied": source != "live",
        "note": note,
    }
