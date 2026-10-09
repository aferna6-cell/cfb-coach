"""Pregame formation portfolio. The model ranks the whole catalog.

Installing a formation keeps every verified source-book play. A formation's
score is its value across situations, not one highly rated play. The user
still installs and attests the book; this module only proposes.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from cfb_coach.madden import catalog
from cfb_coach.madden.model import experimental_model
from cfb_coach.madden.model.football_situation import (
    SITUATION_NAMES,
    evaluate_situation,
    explain_play,
    roster_evidence,
)

# Probes are generic pre-snap states. None of them invent a defensive shell.
PORTFOLIO_PROBES: dict[str, SimpleNamespace] = {
    "normal": SimpleNamespace(
        down=1, distance=10, yardline=35, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    ),
    "short_yardage": SimpleNamespace(
        down=3, distance=1, yardline=55, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    ),
    "third_and_long": SimpleNamespace(
        down=3, distance=10, yardline=45, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    ),
    "red_zone": SimpleNamespace(
        down=1, distance=10, yardline=85, red_zone=True, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    ),
    "goal_line": SimpleNamespace(
        down=1, distance=1, yardline=98, red_zone=True, goal_line=True,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    ),
    "backed_up": SimpleNamespace(
        down=1, distance=10, yardline=8, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    ),
    "two_minute": SimpleNamespace(
        down=2, distance=7, yardline=45, red_zone=False, goal_line=False,
        two_minute=True, score_us=17, score_them=24, coverage_hint=None,
        coverage_source="none", extras={"quarter": 4, "clock_seconds": 70, "timeouts_us": 1},
    ),
    "clock_management": SimpleNamespace(
        down=1, distance=10, yardline=40, red_zone=False, goal_line=False,
        two_minute=True, score_us=24, score_them=17, coverage_hint=None,
        coverage_source="none", extras={"quarter": 4, "clock_seconds": 90, "timeouts_us": 2},
    ),
}


def load_verified_roster(db: Any) -> dict[str, Any]:
    """Read an explicit roster snapshot. Absence is unknown, not a guess."""
    if db is None:
        return roster_evidence(None)
    raw = None
    try:
        raw = db.get_meta("ml_verified_roster.v1")
    except Exception:  # noqa: BLE001
        raw = None
    if not raw:
        return roster_evidence(None)
    import json
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return roster_evidence(None)
    return roster_evidence(payload if isinstance(payload, dict) else None)


def opponent_defense_profile(db: Any, opponent_id: str) -> dict[str, Any]:
    """Historical looks only. Sample size and confidence are reported."""
    empty = {
        "state": "unknown",
        "sample_size": 0,
        "confidence": 0.0,
        "distribution": {},
        "inferred_look": None,
        "note": "no verified defensive observations for this opponent",
    }
    if db is None or not opponent_id:
        return empty
    try:
        rows = db.conn.execute(
            "SELECT coverage_seen FROM snaps WHERE opponent_id=? AND side='offense' "
            "AND coverage_seen IS NOT NULL AND coverage_seen != ''",
            (opponent_id,),
        ).fetchall()
    except Exception:  # noqa: BLE001
        return empty
    from cfb_coach.madden.playcaller import coverage_class

    counts: dict[str, int] = {}
    for row in rows:
        kind = coverage_class(row["coverage_seen"])
        if not kind:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    sample = sum(counts.values())
    if sample <= 0:
        return empty
    inferred = max(counts, key=lambda k: (counts[k], k))
    # Below eight classified looks the distribution is too thin to move a formation.
    if sample < 8:
        return {
            "state": "inferred_low_sample",
            "sample_size": sample,
            "confidence": round(sample / 24.0, 3),
            "distribution": counts,
            "inferred_look": None,
            "note": "defensive sample is too small to change the formation portfolio",
        }
    return {
        "state": "inferred",
        "sample_size": sample,
        "confidence": round(min(1.0, sample / 24.0), 3),
        "distribution": counts,
        "inferred_look": inferred,
        "note": "historical tendency, not the next snap's coverage",
    }


def _zone_for(name: str) -> str:
    if name == "goal_line":
        return "gl"
    if name == "red_zone":
        return "rz"
    return "open"


def _situation_value(
    art: Any,
    formation: str,
    plays: list[str],
    opponent_id: str,
    situation: str,
    probe: Any,
) -> float:
    """Mean of the best few concepts that actually fit this situation."""
    zone = _zone_for(situation)
    best: dict[str, float] = {}
    for play in plays:
        if situation not in ("goal_line", "red_zone") and not catalog.zone_fit(play, zone):
            continue
        if situation == "goal_line" and catalog.is_deep(play) and "goal" not in play.lower():
            # Still scored, but the evaluator applies the compressed-space penalty.
            pass
        pred = experimental_model.predict_success(
            art, formation=formation, play=play,
            down=getattr(probe, "down", None),
            distance=getattr(probe, "distance", None),
            yardline=getattr(probe, "yardline", None),
            opponent_id=opponent_id,
            coverage_hint=None, coverage_source="none",
            heuristic_bonus=0.0,
        )
        fit = explain_play(play, probe)
        score = float(pred["probability"]) + float(fit["selection_delta"]) + float(fit["coordinator_delta"])
        concept = experimental_model._play_concept(play)
        if concept not in best or score > best[concept]:
            best[concept] = score
    if not best:
        return 0.0
    ordered = sorted(best.values(), reverse=True)
    take = ordered[: min(4, len(ordered))]
    return sum(take) / len(take)


def _tendency_bonus(plays: list[str], profile: Mapping[str, Any]) -> tuple[float, str]:
    look = profile.get("inferred_look")
    confidence = float(profile.get("confidence") or 0.0)
    if profile.get("state") != "inferred" or not look or confidence <= 0:
        return 0.0, "no confident defensive tendency"
    quick = 0
    power = 0
    for play in plays:
        if catalog.is_run(play):
            power += 1
        elif experimental_model._play_family(play) in ("screen", "pass"):
            quick += 1
    if look == "pressure" and quick:
        return round(0.02 * confidence, 5), f"portfolio notes historical pressure ({profile['sample_size']} looks)"
    if look in ("man", "cover2", "two_high", "single_high") and (quick or power):
        return round(0.01 * confidence, 5), f"portfolio notes historical {look} ({profile['sample_size']} looks)"
    return 0.0, "tendency does not match a supported concept family"


def _roster_bonus(plays: list[str], roster: Mapping[str, Any]) -> tuple[float, str]:
    if roster.get("state") != "verified":
        return 0.0, "roster unknown"
    runs = sum(catalog.is_run(p) for p in plays)
    passes = len(plays) - runs
    lean = float(roster["pass_strength"]) - float(roster["run_strength"])
    if abs(lean) < 0.15 or not plays:
        return 0.0, "roster has no strong pass/run lean"
    share = (passes if lean > 0 else runs) / len(plays)
    delta = round(max(-0.02, min(0.02, 0.03 * lean * share)), 5)
    return delta, "verified roster lean applied as a small portfolio weight"


def select_formation_portfolio(
    *,
    art: Any,
    all_books: Mapping[str, Mapping[str, list[str]]],
    active: Mapping[str, Any],
    opponent_id: str,
    max_formations: int,
    db: Any = None,
) -> dict[str, Any]:
    """Greedy coverage across every catalogued formation. No favorite list."""
    roster = load_verified_roster(db)
    profile = opponent_defense_profile(db, opponent_id)
    current = active.get("formations") or {}
    ranked: list[dict[str, Any]] = []
    for source, formations in all_books.items():
        for form, plays in formations.items():
            unique = list(dict.fromkeys(plays or []))
            if not unique:
                continue
            by_situation: dict[str, float] = {}
            for name in SITUATION_NAMES:
                by_situation[name] = round(_situation_value(
                    art, form, unique, opponent_id, name, PORTFOLIO_PROBES[name],
                ), 6)
            if sum(by_situation.values()) <= 0:
                continue
            tendency, tendency_why = _tendency_bonus(unique, profile)
            roster_delta, roster_why = _roster_bonus(unique, roster)
            continuity = 0.01 if form in current else 0.0
            spread = len({experimental_model._play_concept(p) for p in unique})
            ranked.append({
                "formation": form,
                "source_book": source,
                "plays": unique,
                "has_run": any(catalog.is_run(p) for p in unique),
                "has_pass": any(not catalog.is_run(p) for p in unique),
                "situation_scores": by_situation,
                "tendency_bonus": tendency,
                "tendency_reason": tendency_why,
                "roster_bonus": roster_delta,
                "roster_reason": roster_why,
                "continuity": continuity,
                "concept_count": spread,
                "top_play": max(unique, key=lambda p: by_situation["normal"]),
                "model_top_probability": round(max(by_situation.values()), 6),
            })
    # Same formation offered by two books: keep the higher multi-situation mean.
    best: dict[str, dict[str, Any]] = {}
    for row in ranked:
        row["score"] = round(
            sum(row["situation_scores"].values()) / len(SITUATION_NAMES)
            + row["tendency_bonus"] + row["roster_bonus"] + row["continuity"]
            + 0.002 * row["concept_count"],
            6,
        )
        prev = best.get(row["formation"])
        if prev is None or (row["score"], row["source_book"]) > (prev["score"], prev["source_book"]):
            best[row["formation"]] = row
    pool = sorted(best.values(), key=lambda r: (-r["score"], r["formation"]))
    coverage = {name: 0.0 for name in SITUATION_NAMES}
    chosen: list[dict[str, Any]] = []
    remaining = list(pool)
    while remaining and len(chosen) < max_formations:
        def marginal(row: dict[str, Any]) -> tuple[float, float, str]:
            gain = 0.0
            for name in SITUATION_NAMES:
                gain += max(0.0, float(row["situation_scores"][name]) - coverage[name])
            gain += row["tendency_bonus"] + row["roster_bonus"] + row["continuity"]
            return (gain, row["score"], row["formation"])

        pick = max(remaining, key=marginal)
        remaining.remove(pick)
        if marginal(pick)[0] <= 0 and chosen:
            break
        chosen.append(pick)
        for name in SITUATION_NAMES:
            coverage[name] = max(coverage[name], float(pick["situation_scores"][name]))
    if chosen and not any(r["has_run"] for r in chosen):
        runner = next((r for r in remaining if r["has_run"]), None)
        if runner:
            chosen[-1] = runner
    if chosen and not any(r["has_pass"] for r in chosen):
        passer = next((r for r in remaining if r["has_pass"]), None)
        if passer:
            chosen[-1] = passer
    rationales = []
    addressed = {name: [] for name in SITUATION_NAMES}
    for row in chosen:
        best_situations = sorted(SITUATION_NAMES, key=lambda n: -row["situation_scores"][n])[:3]
        for name in SITUATION_NAMES:
            if row["situation_scores"][name] > 0:
                addressed[name].append(row["formation"])
        runs = [p for p in row["plays"] if catalog.is_run(p)]
        rationales.append({
            "formation": row["formation"],
            "source_book": row["source_book"],
            "plays": list(row["plays"]),
            "why": (
                f"Covers {', '.join(best_situations)} better than the formations already chosen. "
                f"{row['tendency_reason']}. {row['roster_reason']}."
            ),
            "situations": best_situations,
            "situation_scores": row["situation_scores"],
            "defensive_tendency": profile.get("inferred_look"),
            "tendency_confidence": profile.get("confidence"),
            "suggested_audible": runs[0] if runs else None,
            "audible_status": "suggestion_only_not_installed",
        })
    return {
        "chosen": chosen,
        "ranked": pool,
        "rationales": rationales,
        "situation_coverage": {
            name: {
                "best_score": round(coverage[name], 6),
                "formations": addressed[name],
                "addressed": bool(addressed[name]),
            }
            for name in SITUATION_NAMES
        },
        "roster": roster,
        "opponent_defense": profile,
        "probes": list(SITUATION_NAMES),
        "uncertainty": (
            "Formation scores mix the experimental success model with labeled "
            "situation proxies. They are not verified win rates. A tendency is "
            "historical. Roster data counts only when a verified snapshot exists."
        ),
    }
