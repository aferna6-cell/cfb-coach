"""Tendency confidence — one tell bumps; hard-counter only on live repeats."""

from __future__ import annotations

from cfb_coach.db import CoachDB
from cfb_coach.situation import Situation


def situation_bucket(sit: Situation) -> str:
    """Coarse field/D&D bucket for same-situation tendency matching."""
    if sit.goal_line:
        zone = "gl"
    elif sit.red_zone:
        zone = "rz"
    else:
        zone = "open"
    if sit.short_yardage:
        dd = "short"
    elif sit.long_yardage:
        dd = "long"
    else:
        dd = "normal"
    down = str(sit.down) if sit.down else "x"
    return f"{zone}|{dd}|{down}"


def _cov_match(stored: str, want: str) -> bool:
    a, b = stored.lower(), want.lower()
    return a == b or a in b or b in a


def _sum_live(db: CoachDB, opponent_id: str, buckets: list[str], key: str, match_cov: bool) -> int:
    total = 0
    for bucket in buckets:
        for row in db.get_tendencies(opponent_id, bucket):
            if bucket.endswith("_seed"):
                continue
            ok = _cov_match(row["key"], key) if match_cov else (
                key.lower() in row["key"].lower() or row["key"].lower() in key.lower()
            )
            if ok:
                total += int(row["count"])
    return total


def coverage_count(
    db: CoachDB | None,
    opponent_id: str,
    coverage: str,
    sit: Situation | None = None,
) -> int:
    """Live coverage weight only (seed priors excluded)."""
    if not db or not coverage:
        return 0
    buckets = ["their_coverage"]
    if sit:
        buckets.append(f"their_coverage|{situation_bucket(sit)}")
    return _sum_live(db, opponent_id, buckets, coverage, match_cov=True)


def concept_count(db: CoachDB | None, opponent_id: str, concept: str) -> int:
    """Live concept weight only (seed priors excluded)."""
    if not db or not concept:
        return 0
    return _sum_live(db, opponent_id, ["offense_concept"], concept, match_cov=False)



def is_user_opponent(opponent_id: str | None) -> bool:
    """True for human dynasty opponents (not CPU volume-lab)."""
    from cfb_coach.opponents import is_cpu_opponent

    return not is_cpu_opponent(opponent_id)


def in_game_coverage_count(
    db: CoachDB | None,
    opponent_id: str,
    coverage: str,
    *,
    window: int = 12,
) -> int:
    """Count coverage_seen in the most recent snaps (this-game proxy)."""
    if not db or not coverage:
        return 0
    snaps = db.get_recent_snaps(opponent_id, limit=window)
    n = 0
    want = coverage.lower()
    for s in snaps:
        seen = (s["coverage_seen"] or "").lower()
        if not seen:
            continue
        if want == seen or want in seen or seen in want:
            n += 1
    return n


def in_game_concept_count(
    db: CoachDB | None,
    opponent_id: str,
    concept: str,
    *,
    window: int = 12,
) -> int:
    """Count concept_seen in the most recent snaps (this-game proxy)."""
    if not db or not concept:
        return 0
    snaps = db.get_recent_snaps(opponent_id, limit=window)
    n = 0
    want = concept.lower()
    for s in snaps:
        seen = (s["concept_seen"] or "").lower()
        if not seen:
            continue
        if want == seen or want in seen or seen in want:
            n += 1
    return n


def is_repeated_coverage(
    db: CoachDB | None,
    opponent_id: str,
    coverage: str,
    sit: Situation | None = None,
    *,
    threshold: int = 2,
) -> bool:
    """True after the same shell recurred enough in LIVE logs.

    For user opponents (Alabama humans ~once/season): prefer THIS-game window
    (recent snaps) so we can pivot macros after 2+ tells mid-game without
    needing a thick career dataset. Still no single-snap whiplash.
    """
    if is_user_opponent(opponent_id):
        if in_game_coverage_count(db, opponent_id, coverage) >= threshold:
            return True
    return coverage_count(db, opponent_id, coverage, sit) >= threshold



def is_repeated_concept(
    db: CoachDB | None,
    opponent_id: str,
    concept: str,
    *,
    threshold: int = 2,
) -> bool:
    """True after the same concept recurred enough in LIVE logs.

    User opponents: 2+ tells in the current-game window unlock targeted macros.
    Still no single-snap whiplash. CPU/lab keeps career live counts.
    """
    if is_user_opponent(opponent_id):
        if in_game_concept_count(db, opponent_id, concept) >= threshold:
            return True
    return concept_count(db, opponent_id, concept) >= threshold



def mild_bump_coverage(
    db: CoachDB,
    opponent_id: str,
    coverage: str,
    sit: Situation | None = None,
) -> None:
    """One strong tell → +1 live global and +1 situational. Not a hard-counter."""
    if not coverage:
        return
    db.bump_tendency(opponent_id, "their_coverage", coverage, amount=1)
    if sit:
        db.bump_tendency(
            opponent_id,
            f"their_coverage|{situation_bucket(sit)}",
            coverage,
            amount=1,
        )


def mild_bump_concept(
    db: CoachDB,
    opponent_id: str,
    concept: str,
    sit: Situation | None = None,
    *,
    success: bool = False,
) -> None:
    if not concept:
        return
    db.bump_tendency(opponent_id, "offense_concept", concept, success=success, amount=1)
    if sit:
        db.bump_tendency(
            opponent_id,
            f"offense_concept|{situation_bucket(sit)}",
            concept,
            success=success,
            amount=1,
        )


def describe_coverage_policy(
    coverage: str | None,
    *,
    live: bool,
    last_snap: bool,
    repeated: bool,
) -> str:
    if not coverage:
        return "no coverage signal — base situational"
    if last_snap and not repeated:
        return (
            f"last snap showed {coverage} — logged/mild bump only; "
            "next call stays base situational (no hard-counter)"
        )
    if last_snap and repeated:
        # Still do not auto-apply last-snap as this snap's forced beater;
        # repeated unlocks optional hard-counter only when live THIS snap too.
        return (
            f"last snap {coverage} (family is repeated in log) — "
            "still base unless this snap shows it live again"
        )
    if repeated:
        return f"REPEATED live {coverage} in-situation — coverage-specific answer OK"
    if live:
        return (
            f"live look {coverage} (single/unconfirmed) — soft lean only; "
            "not a hard-counter beater"
        )
    return f"{coverage} noted — soft prior only"
