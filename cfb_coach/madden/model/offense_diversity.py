"""Evidence-aware variety for the offensive coordinator.

Every eligible installed play is already scored. This module measures how
concentrated the recommendations are, and it widens near-tie sampling only
when recent calls have piled into one formation, concept, or play.

It does not rotate the playbook, impose a minimum quota, or move a play
whose learned probability is clearly better.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.experimental_model import _play_concept, _play_family
from cfb_coach.madden.model.football_knowledge import CONCEPTS, profile_for_play

# A learned-probability gap this large is a football decision, not a tie.
CLEAR_SUPERIORITY = 0.10
# Added to the 0.012 indifference band. Still well under a clear gap.
MAX_EXTRA_BAND = 0.06
_MIN_CALLS = 4
_CONCENTRATION_FLOOR = 0.45

# User-reported recommendations from an earlier coordinator. Not a Sprint 12
# result, and not a target to chase by calling worse plays.
RECORDED_BASELINE = {
    "game_id": "9f2ebdeb9d8f4e2d",
    "status": "user_reported",
    "era": "earlier_ml_behavior",
    "not_evidence_for_sprint_12": True,
    "not_a_decision_quality_benchmark": True,
    "reported_result": "won 28-7",
    "recommendations": 63,
    "formations": 3,
    "labeled_concept_families": 3,
    "top_formation": "Gun Doubles Clamp Stack",
    "top_formation_recommendations": 46,
    "top_formation_share": round(46 / 63, 4),
    "note": (
        "Diversity compared with this game is not evidence that a more "
        "varied call sheet would have scored more than 28."
    ),
}


def package_family(play: str | None) -> str:
    """Run, pass, play-action, screen, or RPO. Unknown stays unknown."""
    concept = profile_for_play(play).get("concept_id")
    if concept == "play_action":
        return "play_action"
    if concept == "screen":
        return "screen"
    if concept == "rpo":
        return "rpo"
    family = _play_family(play)
    if family == "run":
        return "run"
    if family == "screen":
        return "screen"
    if family == "rpo":
        return "rpo"
    if family == "pass":
        return "pass"
    if concept:
        principle = CONCEPTS.get(str(concept)) or {}
        if principle.get("family") == "run":
            return "run"
        if principle.get("family") == "pass":
            return "pass"
    return "unknown"


def _as_calls(recent: Sequence[Any] | None) -> list[tuple[str, str, str]]:
    calls: list[tuple[str, str, str]] = []
    for item in recent or []:
        if isinstance(item, Mapping):
            form = str(item.get("formation") or "")
            play = str(item.get("play") or "")
            kind = str(item.get("adjustment_kind") or item.get("kind") or "none")
        else:
            seq = tuple(item)
            form = str(seq[0]) if seq else ""
            play = str(seq[1]) if len(seq) > 1 else ""
            kind = str(seq[2]) if len(seq) > 2 else "none"
        if form and play:
            calls.append((form, play, kind))
    return calls


def _share(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if not total:
        return 0.0
    return max(counts.values()) / total


def concentration_stats(recent: Sequence[Any] | None) -> dict[str, Any]:
    """How bunched the recent recommendations are. Empty history is not concentrated."""
    calls = _as_calls(recent)
    forms: Counter[str] = Counter(form for form, _play, _kind in calls)
    plays: Counter[str] = Counter(f"{form} — {play}" for form, play, _kind in calls)
    concepts: Counter[str] = Counter(
        str(profile_for_play(play).get("concept_id") or _play_concept(play))
        for _form, play, _kind in calls
    )
    packages: Counter[str] = Counter(package_family(play) for _form, play, _kind in calls)
    adjustments: Counter[str] = Counter(kind for _form, _play, kind in calls)
    last = calls[0] if calls else None
    last_concept = None
    if last is not None:
        last_concept = profile_for_play(last[1]).get("concept_id") or _play_concept(last[1])
    return {
        "calls": len(calls),
        "top_formation_share": round(_share(forms), 4),
        "top_play_share": round(_share(plays), 4),
        "top_concept_share": round(_share(concepts), 4),
        "top_package_share": round(_share(packages), 4),
        "formation_counts": dict(forms),
        "concept_counts": dict(concepts),
        "package_counts": dict(packages),
        "adjustment_counts": dict(adjustments),
        "last_concept": last_concept,
        "fixed_rotation": False,
        "minimum_quota": False,
    }


def extra_band_width(
    stats: Mapping[str, Any] | None,
    leader: Mapping[str, Any] | None,
) -> float:
    """Widen near-tie sampling only after real concentration, and never for a confident lock."""
    if not stats or int(stats.get("calls") or 0) < _MIN_CALLS:
        return 0.0
    row = leader or {}
    uncertainty = float(row.get("uncertainty", 1.0) if row.get("uncertainty") is not None else 1.0)
    quality = row.get("evidence_quality")
    if uncertainty < 0.35 and quality in ("empirical", "verified"):
        return 0.0
    peak = max(
        float(stats.get("top_formation_share") or 0.0),
        float(stats.get("top_concept_share") or 0.0),
        float(stats.get("top_play_share") or 0.0),
    )
    if peak < _CONCENTRATION_FLOOR:
        return 0.0
    span = (peak - _CONCENTRATION_FLOOR) / (1.0 - _CONCENTRATION_FLOOR)
    return round(min(MAX_EXTRA_BAND, 0.02 + 0.04 * span), 5)


def sampling_weight(
    *,
    base_weight: float,
    formation: str,
    concept_id: str | None,
    stats: Mapping[str, Any] | None,
    apply_concentration: bool,
) -> float:
    """Mild down-weight for a repeated look. Only used inside an already-close band."""
    if not apply_concentration or not stats:
        return base_weight
    forms = stats.get("formation_counts") or {}
    concepts = stats.get("concept_counts") or {}
    # Light enough that the better close play stays available. A hard
    # down-weight would be a quota against the repeated call.
    weight = base_weight / (1.0 + 0.12 * float(forms.get(formation) or 0.0)) ** 0.5
    if concept_id:
        weight /= (1.0 + 0.08 * float(concepts.get(concept_id) or 0.0)) ** 0.5
        principle = CONCEPTS.get(str(stats.get("last_concept") or "")) or {}
        if concept_id in set(principle.get("complements") or []) and concept_id != stats.get("last_concept"):
            weight *= 1.15
    return weight


def _top(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    if not total:
        return {"distinct": 0, "top_value": None, "top_count": 0, "top_share": None, "counts": {}}
    value, count = counts.most_common(1)[0]
    return {
        "distinct": len(counts),
        "top_value": value,
        "top_count": count,
        "top_share": round(count / total, 4),
        "counts": dict(counts),
    }


def diversity_report(
    calls: Sequence[Mapping[str, Any]],
    book: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Concentration of recommendations, plus how much of the installed book was in the decision."""
    eligible = [(form, play) for form, plays in book.items() for play in plays]
    eligible_plays = len(eligible)
    recommended_plays = [str(row.get("play") or "") for row in calls if row.get("play")]
    recommended_forms = [str(row.get("formation") or "") for row in calls if row.get("formation")]
    concepts = [
        str(profile_for_play(play).get("concept_id") or _play_concept(play))
        for play in recommended_plays
    ]
    labeled = [_play_concept(play) for play in recommended_plays]
    packages = [package_family(play) for play in recommended_plays]
    adjustments = [str(row.get("adjustment_kind") or "none") for row in calls]
    considered = [int(row.get("plays_considered") or 0) for row in calls]
    distinct_recommended = len({(form, play) for form, play in zip(recommended_forms, recommended_plays)})
    mean_considered = (sum(considered) / len(considered)) if considered else 0.0
    probabilities = [float(row["probability"]) for row in calls if row.get("probability") is not None]
    bests = [float(row["best_probability"]) for row in calls if row.get("best_probability") is not None]
    mean_p = sum(probabilities) / len(probabilities) if probabilities else None
    mean_best = sum(bests) / len(bests) if bests else None
    return {
        "recommendations": len(calls),
        "eligible_installed_plays": eligible_plays,
        "plays_considered_mean": round(mean_considered, 3),
        "consideration_rate": (
            round(mean_considered / eligible_plays, 4) if eligible_plays else None
        ),
        "recommended_play_coverage": (
            round(distinct_recommended / eligible_plays, 4) if eligible_plays else None
        ),
        "formations": _top(Counter(recommended_forms)),
        "plays": _top(Counter(
            f"{form} — {play}" for form, play in zip(recommended_forms, recommended_plays)
        )),
        "concept_families": _top(Counter(concepts)),
        "labeled_concept_families": _top(Counter(labeled)),
        "packages": _top(Counter(packages)),
        "adjustments": _top(Counter(adjustments)),
        "situations": _top(Counter(str(row.get("situation") or "unspecified") for row in calls)),
        "repeated_call_max_streak": _max_streak(list(zip(recommended_forms, recommended_plays))),
        "mean_chosen_probability": None if mean_p is None else round(mean_p, 4),
        "mean_best_probability": None if mean_best is None else round(mean_best, 4),
        "mean_probability_drop": (
            None if mean_p is None or mean_best is None else round(mean_best - mean_p, 4)
        ),
        "fixed_rotation": False,
        "minimum_quota": False,
    }


def _max_streak(keys: list[tuple[str, str]]) -> int:
    longest = 0
    streak = 0
    prev = None
    for key in keys:
        streak = streak + 1 if key == prev else 1
        longest = max(longest, streak)
        prev = key
    return longest


def compare_to_recorded_baseline(report: Mapping[str, Any]) -> dict[str, Any]:
    """Diversity against the 28–7 recommendation log. Not a quality or win-rate claim."""
    formations = int((report.get("formations") or {}).get("distinct") or 0)
    concepts = int((report.get("labeled_concept_families") or {}).get("distinct") or 0)
    share = (report.get("formations") or {}).get("top_share")
    recorded_share = RECORDED_BASELINE["top_formation_share"]
    return {
        "recorded_baseline": dict(RECORDED_BASELINE),
        "formations_vs_recorded_3": formations - 3,
        "labeled_concepts_vs_recorded_3": concepts - 3,
        "top_formation_share": share,
        "recorded_top_formation_share": recorded_share,
        "less_concentrated_than_recorded_top_formation": (
            share is not None and float(share) < float(recorded_share)
        ),
        "decision_quality_not_inferred": True,
        "not_a_win_rate_claim": True,
        "note": (
            "Beating 3 formations, 3 concept labels, or a 46/63 formation share "
            "only says the call sheet was less concentrated. It does not say "
            "the offense would have been more effective than that 28–7 game."
        ),
    }


def _sit(name: str) -> Any:
    from types import SimpleNamespace

    table = {
        "normal": dict(down=1, distance=10, yardline=35),
        "third_long": dict(down=3, distance=12, yardline=48),
        "short": dict(down=3, distance=1, yardline=55),
        "second_medium": dict(down=2, distance=6, yardline=42),
        "goal_line": dict(down=1, distance=1, yardline=99, goal_line=True, red_zone=True),
        "red_zone": dict(down=2, distance=7, yardline=85, red_zone=True),
        "two_minute_trail": dict(
            down=2, distance=8, yardline=45, two_minute=True,
            score_us=14, score_them=21, extras={"quarter": 4},
        ),
        "two_minute_lead": dict(
            down=1, distance=10, yardline=40, two_minute=True,
            score_us=21, score_them=14, extras={"quarter": 4},
        ),
    }
    payload = dict(
        down=1, distance=10, yardline=40, red_zone=False, goal_line=False,
        two_minute=False, score_us=None, score_them=None, coverage_hint=None,
        coverage_source="none", extras={},
    )
    payload.update(table[name])
    return SimpleNamespace(**payload)


def _demo_book() -> dict[str, list[str]]:
    return {
        "Gun Doubles Clamp Stack": ["Mesh", "Inside Zone", "Four Verticals", "PA Boot", "HB Slip Screen"],
        "Gun Trips": ["Flood", "Stick", "Outside Zone"],
        "Gun Bunch": ["Dagger", "Slants", "Counter"],
        "Singleback Wing": ["Smash", "Stretch", "Play Action"],
        "Gun Tight": ["Levels", "HB Dive", "Curl Flat"],
        "I Form": ["Power", "Texas", "Screen"],
        "Gun Wing": ["Drive", "RPO Glance", "Wheel"],
        "Pistol": ["Spacing", "Trap", "Sail"],
    }


def _demo_rows(book: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    rows = []
    for form, plays in book.items():
        for play in plays:
            probability = 0.60 if form == "Gun Doubles Clamp Stack" else 0.57
            rows.append({
                "formation": form,
                "play": play,
                "probability": probability,
                "uncertainty": 0.62,
                "evidence_quality": "prior_driven",
                "play_concept": _play_concept(play),
            })
    return rows


def _run_policy(
    *,
    use_diversity: bool,
    snaps: int = 32,
) -> dict[str, Any]:
    from cfb_coach.madden.catalog import is_run
    from cfb_coach.madden.model.offense_joint_decision import choose_joint_action
    from cfb_coach.madden.model.offense_selection_policy import choose_model_play

    book = _demo_book()
    base_rows = _demo_rows(book)
    names = (
        "normal", "third_long", "short", "second_medium",
        "goal_line", "red_zone", "two_minute_trail", "two_minute_lead",
    )
    recent: list[tuple[str, str]] = []
    calls = []
    third_long_runs = 0
    for index in range(1, snaps + 1):
        name = names[(index - 1) % len(names)]
        sit = _sit(name)
        ranked, _audit = choose_model_play(
            base_rows, recent_calls=list(recent), sit=sit,
            opponent_type="cpu", session_id=f"diversity-{int(use_diversity)}",
            snap_seq=index,
        )
        best_probability = max(float(row["probability"]) for row in ranked)
        decision = choose_joint_action(
            ranked=ranked, anchor=ranked[0], sit=sit, book=book,
            active=[], db=None, opponent_id="synthetic",
            session_id=f"diversity-{int(use_diversity)}", snap_seq=index,
            opponent_type="cpu", use_diversity=use_diversity,
            recent_calls=[{"formation": form, "play": play} for form, play in recent],
        )
        form = str(decision["joint"]["formation"])
        play = str(decision["joint"]["play"])
        chosen = next(row for row in ranked if row["formation"] == form and row["play"] == play)
        if name == "third_long" and is_run(play):
            third_long_runs += 1
        calls.append({
            "situation": name,
            "formation": form,
            "play": play,
            "adjustment_kind": decision.get("kind") or "none",
            "plays_considered": decision.get("plays_considered"),
            "probability": float(chosen["probability"]),
            "best_probability": best_probability,
            "extra_band": (decision.get("exploration") or {}).get("extra_band"),
            "why": (decision.get("football_intelligence") or {}).get("summary"),
        })
        recent.insert(0, (form, play))
        recent = recent[:20]
    report = diversity_report(calls, book)
    report["third_long_run_calls"] = third_long_runs
    report["policy"] = "sprint_12_diversity" if use_diversity else "sprint_12_no_diversity_band"
    return report


def clear_superiority_probe() -> dict[str, Any]:
    """A much stronger learned play still wins after a long run of that same call."""
    from cfb_coach.madden.model.offense_joint_decision import choose_joint_action

    book = {
        "Gun Doubles Clamp Stack": ["Mesh"],
        "Gun Trips": ["Flood", "Stick"],
        "Gun Bunch": ["Dagger"],
    }
    ranked = [
        {"formation": "Gun Doubles Clamp Stack", "play": "Mesh", "probability": 0.82,
         "selection_score": 0.82, "uncertainty": 0.62, "evidence_quality": "prior_driven"},
        {"formation": "Gun Trips", "play": "Flood", "probability": 0.55,
         "selection_score": 0.55, "uncertainty": 0.62, "evidence_quality": "prior_driven"},
        {"formation": "Gun Trips", "play": "Stick", "probability": 0.54,
         "selection_score": 0.54, "uncertainty": 0.62, "evidence_quality": "prior_driven"},
        {"formation": "Gun Bunch", "play": "Dagger", "probability": 0.53,
         "selection_score": 0.53, "uncertainty": 0.62, "evidence_quality": "prior_driven"},
    ]
    recent = [
        {"formation": "Gun Doubles Clamp Stack", "play": "Mesh"} for _ in range(12)
    ]
    decision = choose_joint_action(
        ranked=ranked, anchor=ranked[0], sit=_sit("normal"), book=book,
        active=[], db=None, session_id="clear", snap_seq=13, opponent_type="cpu",
        recent_calls=recent, use_diversity=True,
    )
    return {
        "chosen_play": decision["joint"]["play"],
        "held_clear_leader": decision["joint"]["play"] == "Mesh",
        "plays_considered": decision.get("plays_considered"),
        "eligible": sum(len(plays) for plays in book.values()),
        "extra_band": (decision.get("exploration") or {}).get("extra_band"),
        "note": "Concentration did not displace a play that is clearly ahead on learned probability.",
    }


def concentrated_history_probe(*, snaps: int = 24) -> dict[str, Any]:
    """Same close scores and the same recent pile-up. Only the diversity band changes."""
    from cfb_coach.madden.model.offense_joint_decision import choose_joint_action

    book = {
        "Gun Doubles Clamp Stack": ["Mesh", "Inside Zone"],
        "Gun Trips": ["Flood"],
        "Gun Bunch": ["Dagger"],
        "Singleback Wing": ["Smash"],
    }
    ranked = []
    for form, plays in book.items():
        for play in plays:
            probability = 0.60 if play == "Mesh" else 0.57
            ranked.append({
                "formation": form, "play": play, "probability": probability,
                "selection_score": probability, "uncertainty": 0.62,
                "evidence_quality": "prior_driven",
            })
    recent = [{"formation": "Gun Doubles Clamp Stack", "play": "Mesh"} for _ in range(8)]

    def _calls(use_diversity: bool) -> list[dict[str, Any]]:
        out = []
        for seq in range(1, snaps + 1):
            decision = choose_joint_action(
                ranked=ranked, anchor=ranked[0], sit=_sit("normal"), book=book,
                active=[], db=None, session_id="concentrated", snap_seq=seq,
                opponent_type="cpu", recent_calls=recent, use_diversity=use_diversity,
                use_knowledge=False, use_strategy=False,
            )
            out.append({
                "formation": decision["joint"]["formation"],
                "play": decision["joint"]["play"],
                "adjustment_kind": "none",
                "plays_considered": decision.get("plays_considered"),
                "probability": next(
                    row["probability"] for row in ranked
                    if row["play"] == decision["joint"]["play"]
                    and row["formation"] == decision["joint"]["formation"]
                ),
                "best_probability": 0.60,
                "situation": "normal",
                "extra_band": (decision.get("exploration") or {}).get("extra_band"),
            })
        return out

    locked = diversity_report(_calls(False), book)
    opened = diversity_report(_calls(True), book)
    return {
        "without_diversity_band": locked,
        "with_diversity": opened,
        "band_changes_the_call_sheet": (
            locked["formations"]["distinct"] == 1 and opened["formations"]["distinct"] > 1
        ),
        "best_play_still_used": (opened["plays"]["counts"].get("Gun Doubles Clamp Stack — Mesh") or 0) > 0,
        "note": (
            "The 0.03 probability gap is not a clear superiority. Sampling "
            "other formations does not ban the better play."
        ),
    }


def evaluate_strategic_diversity() -> dict[str, Any]:
    """Paired synthetic call sheets. Variety is reported apart from effectiveness."""
    without = _run_policy(use_diversity=False)
    with_band = _run_policy(use_diversity=True)
    history = concentrated_history_probe()
    return {
        "evidence": "synthetic_representative_situations",
        "not_a_win_rate_claim": True,
        "decision_quality_separate_from_diversity": True,
        "recorded_baseline": dict(RECORDED_BASELINE),
        "without_diversity_band": without,
        "with_diversity": with_band,
        "concentrated_history": history,
        "baseline_comparison": compare_to_recorded_baseline(with_band),
        "clear_superiority_probe": clear_superiority_probe(),
        "note": (
            "Both policies score every installed play. The diversity band only "
            "samples calls that are already close. A lower formation share than "
            "the recorded 28–7 game is not a claim of better offense."
        ),
    }
