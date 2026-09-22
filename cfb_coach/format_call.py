"""Strict call formatting — UX lock for Aidan's live syntax."""

from __future__ import annotations

import re

_ARROW = " → "


def _two_reads(raw: str) -> str:
    """Normalize to exactly 'Read1 → Read2' (never a single read)."""
    text = (raw or "").strip()
    if not text:
        return f"Primary{_ARROW}Checkdown"
    # Already has arrow variants
    for sep in (" → ", "->", "→", " => ", "=>"):
        if sep in text:
            parts = [p.strip() for p in text.replace("=>", "→").replace("->", "→").split("→")]
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                return f"{parts[0]}{_ARROW}{parts[1]}"
            if len(parts) == 1:
                return f"{parts[0]}{_ARROW}Checkdown"
    # Slash or comma lists
    for sep in (" / ", "/", ", ", ","):
        if sep in text:
            parts = [p.strip() for p in text.split(sep[0] if len(sep) == 1 else sep)]
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                return f"{parts[0]}{_ARROW}{parts[1]}"
    # Single token — invent a sensible second read
    return f"{text}{_ARROW}Checkdown"


def format_offense(formation: str, play: str, adj: str, reads: str) -> str:
    adj_s = (adj or "No adj").strip() or "No adj"
    return f"{formation} — {play} | {adj_s} | {_two_reads(reads)}"


def format_defense(formation: str, play: str, macro: str, user_job: str) -> str:
    macro_s = (macro or "none").strip() or "none"
    # One user job — strip chained arrows / multi-jobs
    job = (user_job or "User hook").strip()
    job = re.split(r"\s*→\s*|\s*>\s*|\s*/\s*", job)[0].strip()
    if not job.lower().startswith("user"):
        job = f"User {job}"
    return f"{formation} — {play} | {macro_s} | {job}"
