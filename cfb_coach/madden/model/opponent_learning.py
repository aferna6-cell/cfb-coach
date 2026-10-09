"""Conservative opponent tendencies and concept evidence.

Snap N is fit only from records the caller already limited to earlier snaps.
Sparse samples stay uncertain. A recommendation is never a training row.
Video observations are not accepted here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.learning import SUCCESS_NEED
from cfb_coach.madden.model.defensive_observation import structure_observation
from cfb_coach.madden.model.football_knowledge import profile_for_play
from cfb_coach.outcome import parse_outcome

LEARNER_VERSION = "opponent_tendency.v1"
CONCEPT_EVIDENCE_VERSION = "concept_evidence.v1"
LEARNING_CAP = 0.02
PRIOR_STRENGTH = 2.0
PRIOR_MEAN = 0.5
MIN_CONTEXT_N = 8
MIN_CHANGE_HALF = 4
CHANGE_MARGIN = 0.25
MIN_CONCEPT_N = 6
POOR_CONVERSION = 0.35

_QUICK = frozenset({
    "mesh", "stick", "slant_flat", "curl_flat", "texas_angle", "spacing",
})
_PASSING_DOWNS = frozenset({3, 4})


def readonly_madden_db_path() -> Path:
    """Locate the Madden log without creating directories or the file."""
    raw = os.environ.get("CFB_COACH_MADDEN_DB")
    if raw:
        return Path(raw).expanduser()
    env = os.environ.get("CFB_COACH_DB")
    if env:
        return Path(env).expanduser().resolve().parent / "madden27.db"
    return Path.home() / ".cfb-coach" / "madden27.db"


def _shrink(successes: float, n: int) -> float:
    return (successes + PRIOR_STRENGTH * PRIOR_MEAN) / (n + PRIOR_STRENGTH)


def _effective_n(n: int) -> float:
    if n <= 0:
        return 0.0
    return (n * n) / (n + PRIOR_STRENGTH)


def _confidence(n: int, published: bool) -> float:
    if not published or n <= 0:
        return 0.0
    n_eff = _effective_n(n)
    return round(min(0.85, n_eff / (n_eff + 8.0)), 3)


def _labels(record: Mapping[str, Any]) -> dict[str, Any]:
    """Success uses the existing contract. Conversion requires the first down."""
    if "success" in record and "conversion" in record and record.get("prelabeled"):
        return {
            "success": record.get("success"),
            "conversion": record.get("conversion"),
            "yards": record.get("yards"),
            "turnover": bool(record.get("turnover")),
            "sack": bool(record.get("sack")),
            "kind": record.get("kind"),
        }
    text = record.get("outcome")
    if isinstance(text, Mapping):
        text = text.get("result") or text.get("outcome")
    parsed = parse_outcome(None if text is None else str(text))
    down = record.get("down")
    distance = record.get("distance")
    yards = record.get("yards") if record.get("yards") is not None else parsed.yards
    kind = parsed.kind
    turnover = parsed.is_turnover
    sack = kind == "sack"
    if kind in ("int", "fumble", "sack", "incomplete"):
        success: bool | None = False
        conversion: bool | None = False
    elif kind == "td":
        success = True
        conversion = True
    elif kind == "convert":
        success = True
        conversion = True
    else:
        success = None
        conversion = None
        if yards is not None and down is not None and distance is not None:
            try:
                need = SUCCESS_NEED.get(int(down))
                if need is not None:
                    success = float(yards) >= float(need) * float(distance)
            except (TypeError, ValueError):
                success = None
        if yards is not None and distance is not None:
            try:
                conversion = float(yards) >= float(distance)
            except (TypeError, ValueError):
                conversion = None
    return {
        "success": success,
        "conversion": conversion,
        "yards": yards,
        "turnover": turnover,
        "sack": sack,
        "kind": kind,
    }


def records_from_memory_events(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Verified executions only. Recommendations stay out of the fit."""
    records: list[dict[str, Any]] = []
    for row in events:
        if row.get("verification") != "verified_execution":
            continue
        presnap = row.get("presnap") or {}
        play = row.get("executed_play")
        concept = None
        if play:
            concept = profile_for_play(play).get("concept_id")
        labeled = _labels({
            "down": presnap.get("down"),
            "distance": presnap.get("distance"),
            "outcome": row.get("outcome"),
        })
        observation = structure_observation(
            row.get("observed_defense"),
            timing="historical",
            source="historical",
        )
        records.append({
            "snap_seq": row.get("snap_seq"),
            "snap_id": row.get("snap_id"),
            "down": presnap.get("down"),
            "distance": presnap.get("distance"),
            "yardline": presnap.get("yardline"),
            "clock_seconds": presnap.get("clock_seconds"),
            "score_us": presnap.get("score_us"),
            "score_them": presnap.get("score_them"),
            "formation": row.get("executed_formation"),
            "play": play,
            "concept_id": concept,
            "observed_defense": row.get("observed_defense"),
            "observation": observation,
            "outcome": row.get("outcome"),
            "model_probability": None,
            "verified_execution": True,
            "recommendation_only": False,
            **labeled,
        })
    return records


def _pressure_bit(record: Mapping[str, Any]) -> bool | None:
    observation = record.get("observation")
    if not isinstance(observation, Mapping):
        observation = structure_observation(
            record.get("observed_defense"), timing="historical", source="historical",
        )
    if observation.get("legacy_ambiguous") and observation.get("pressure") is None:
        return None
    if observation.get("pressure") is None and observation.get("blitz") is None:
        return None
    return bool(observation.get("pressure") or observation.get("blitz"))


def _zone_shell(record: Mapping[str, Any]) -> bool | None:
    observation = record.get("observation") or {}
    shell = observation.get("coverage_shell")
    if observation.get("legacy_ambiguous") or not shell:
        return None
    return shell in ("cover_2", "cover_3", "cover_4", "cover_6")


def _estimate(name: str, context: str, successes: int, n: int) -> dict[str, Any]:
    published = n >= MIN_CONTEXT_N
    posterior = _shrink(successes, n) if n else None
    return {
        "name": name,
        "context": context,
        "sample_size": n,
        "event_count": successes,
        "effective_sample_size": round(_effective_n(n), 3),
        "raw_rate": None if n == 0 else round(successes / n, 3),
        "shrunk_rate": None if posterior is None else round(posterior, 3),
        "prior_strength": PRIOR_STRENGTH,
        "prior_mean": PRIOR_MEAN,
        "published": published,
        "state": "estimated" if published else "insufficient",
        "confidence": _confidence(n, published),
    }


def scope_training_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    opponent_id: str,
    game_id: str | None = None,
) -> dict[str, Any]:
    """Keep one opponent, and one game when a game id was requested.

    A blank opponent id is not the requested opponent. CPU and human rows
    are never pooled.
    """
    kept: list[Mapping[str, Any]] = []
    excluded_missing_opponent = 0
    excluded_other_opponent = 0
    excluded_other_game = 0
    for row in rows:
        opp = row.get("opponent_id")
        if opp is None or str(opp).strip() == "":
            excluded_missing_opponent += 1
            continue
        if str(opp) != str(opponent_id):
            excluded_other_opponent += 1
            continue
        if game_id is not None:
            gid = row.get("game_id") or row.get("session_id")
            if str(gid or "") != str(game_id):
                excluded_other_game += 1
                continue
        kept.append(row)
    return {
        "rows": kept,
        "excluded_missing_opponent": excluded_missing_opponent,
        "excluded_other_opponent": excluded_other_opponent,
        "excluded_other_game": excluded_other_game,
    }


def _change(labeled: list[tuple[int, bool]], metric: str) -> dict[str, Any] | None:
    ordered = sorted(labeled, key=lambda item: item[0])
    if len(ordered) < MIN_CHANGE_HALF * 2:
        return None
    mid = len(ordered) // 2
    early, late = ordered[:mid], ordered[mid:]
    if len(early) < MIN_CHANGE_HALF or len(late) < MIN_CHANGE_HALF:
        return None
    early_k = sum(1 for _, value in early if value)
    late_k = sum(1 for _, value in late if value)
    early_s = _shrink(early_k, len(early))
    late_s = _shrink(late_k, len(late))
    early_raw = early_k / len(early)
    late_raw = late_k / len(late)
    if abs(late_s - early_s) < CHANGE_MARGIN:
        return None
    if (late_raw - early_raw) * (late_s - early_s) <= 0:
        return None
    direction = "increased" if late_s > early_s else "decreased"
    return {
        "metric": metric,
        "direction": direction,
        "early_n": len(early),
        "late_n": len(late),
        "early_rate": round(early_raw, 3),
        "late_rate": round(late_raw, 3),
        "early_shrunk": round(early_s, 3),
        "late_shrunk": round(late_s, 3),
        "shrunk_delta": round(late_s - early_s, 3),
        "hypothesis": True,
        "automatic_conclusion": False,
        "within_one_game": True,
        "crossed_game_boundary": False,
        "confidence": round(min(0.8, (len(early) + len(late)) / 40.0), 3),
    }


def _changes_respecting_games(
    triples: Sequence[tuple[int, bool, str | None]],
    metric: str,
) -> list[dict[str, Any]]:
    """Split each game on its own timeline. Never compare across games."""
    named = {game for _, _, game in triples if game}
    groups: dict[str | None, list[tuple[int, bool]]] = {}
    if not named:
        groups[None] = [(order, value) for order, value, _game in triples]
    else:
        for order, value, game in triples:
            if not game:
                continue
            groups.setdefault(str(game), []).append((order, value))
    found: list[dict[str, Any]] = []
    for game, pairs in groups.items():
        change = _change(pairs, metric)
        if not change:
            continue
        tagged = dict(change)
        tagged["game_id"] = game
        found.append(tagged)
    return found


def _is_verified_execution(row: Mapping[str, Any]) -> bool:
    """Missing verification is not verification."""
    if row.get("recommendation_only"):
        return False
    if row.get("source") in ("video", "human_confirmed_film"):
        return False
    return row.get("verified_execution") is True


def _is_approved_film_observation(row: Mapping[str, Any]) -> bool:
    return (
        row.get("defense_observation_approved") is True
        and row.get("human_verification") == "verified_human"
        and row.get("source") == "human_confirmed_film"
        and row.get("verified_execution") is not True
    )


def _concept_row(concept_id: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    successes = [row.get("success") for row in rows if row.get("success") is not None]
    conversions = [row.get("conversion") for row in rows if row.get("conversion") is not None]
    yards = [float(row["yards"]) for row in rows if row.get("yards") is not None]
    success_k = sum(1 for value in successes if value)
    conversion_k = sum(1 for value in conversions if value)
    shells: dict[str, int] = {}
    formations: dict[str, int] = {}
    probs: list[float] = []
    hits: list[float] = []
    for row in rows:
        shell = (row.get("observation") or {}).get("coverage_shell")
        if shell:
            shells[str(shell)] = shells.get(str(shell), 0) + 1
        form = row.get("formation")
        if form:
            formations[str(form)] = formations.get(str(form), 0) + 1
        if row.get("model_probability") is not None and row.get("success") is not None:
            probs.append(float(row["model_probability"]))
            hits.append(1.0 if row.get("success") else 0.0)
    published = n >= MIN_CONCEPT_N and len(conversions) >= MIN_CONCEPT_N
    shrunk_conversion = _shrink(conversion_k, len(conversions)) if conversions else None
    calibration = None
    if len(probs) >= MIN_CONTEXT_N:
        mean_p = sum(probs) / len(probs)
        mean_y = sum(hits) / len(hits)
        brier = sum((p - y) ** 2 for p, y in zip(probs, hits)) / len(probs)
        calibration = {
            "n": len(probs),
            "mean_predicted": round(mean_p, 3),
            "empirical_success": round(mean_y, 3),
            "brier": round(brier, 4),
            "state": "estimated",
        }
    elif probs:
        calibration = {"n": len(probs), "state": "insufficient"}
    return {
        "concept_id": concept_id,
        "sample_size": n,
        "effective_sample_size": round(_effective_n(n), 3),
        "success_n": len(successes),
        "success_rate": None if not successes else round(success_k / len(successes), 3),
        "shrunk_success": None if not successes else round(_shrink(success_k, len(successes)), 3),
        "conversion_n": len(conversions),
        "conversion_rate": None if not conversions else round(conversion_k / len(conversions), 3),
        "shrunk_conversion": None if shrunk_conversion is None else round(shrunk_conversion, 3),
        "yards_known": len(yards),
        "yards_mean": None if not yards else round(sum(yards) / len(yards), 3),
        "turnover_rate": round(sum(1 for row in rows if row.get("turnover")) / n, 3) if n else None,
        "sack_rate": round(sum(1 for row in rows if row.get("sack")) / n, 3) if n else None,
        "by_shell": shells,
        "by_formation": formations,
        "personnel": "unknown",
        "calibration": calibration,
        "published": published,
        "state": "estimated" if published else "insufficient",
        "confidence": _confidence(len(conversions), published),
        "poor_conversion": bool(
            published and shrunk_conversion is not None and shrunk_conversion < POOR_CONVERSION
        ),
    }


def fit_opponent_model(memory: Mapping[str, Any] | None = None,
                       records: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Partial-pool tendencies and concept results. Empty evidence stays uncertain."""
    if records is None:
        records = list((memory or {}).get("verified_snap_records") or [])
    verified_rows = [row for row in records if _is_verified_execution(row)]
    film_rows = [row for row in records if _is_approved_film_observation(row)]
    usable = verified_rows
    tendency_rows = verified_rows + film_rows
    pressure_all: list[tuple[int, bool, str | None]] = []
    pressure_passing: list[tuple[int, bool, str | None]] = []
    zone_shells: list[tuple[int, bool, str | None]] = []
    by_formation: dict[str, list[tuple[int, bool]]] = {}
    concepts: dict[str, list[Mapping[str, Any]]] = {}
    seq_fallback = 0
    unscoped_beside_named_games = 0
    named_games = {
        str(row.get("game_id") or row.get("session_id"))
        for row in tendency_rows
        if row.get("game_id") or row.get("session_id")
    }
    for row in tendency_rows:
        seq_fallback += 1
        seq = row.get("snap_seq")
        order = seq_fallback if seq is None else int(seq)
        game_key = row.get("game_id") or row.get("session_id")
        game_key = str(game_key) if game_key else None
        if game_key is None and named_games:
            unscoped_beside_named_games += 1
        bit = _pressure_bit(row)
        if bit is not None:
            pressure_all.append((order, bit, game_key))
            try:
                passing = int(row.get("down")) in _PASSING_DOWNS
            except (TypeError, ValueError):
                passing = False
            if passing:
                pressure_passing.append((order, bit, game_key))
        zone = _zone_shell(row)
        if zone is not None:
            zone_shells.append((order, zone, game_key))
        if not _is_verified_execution(row):
            continue
        form = row.get("formation")
        if form and bit is not None:
            by_formation.setdefault(str(form), []).append((order, bit))
        concept = row.get("concept_id")
        if concept and row.get("play"):
            concepts.setdefault(str(concept), []).append(row)

    estimates = [
        _estimate(
            "pressure", "pressure_observations",
            sum(1 for _, value, _game in pressure_all if value), len(pressure_all),
        ),
        _estimate(
            "pressure", "passing_downs",
            sum(1 for _, value, _game in pressure_passing if value), len(pressure_passing),
        ),
        _estimate(
            "zone_shell", "explicit_shells",
            sum(1 for _, value, _game in zone_shells if value), len(zone_shells),
        ),
    ]
    for form, pairs in sorted(by_formation.items()):
        estimates.append(_estimate(
            "pressure", f"offensive_formation:{form}",
            sum(1 for _, value in pairs if value), len(pairs),
        ))
    changes = []
    for metric, triples in (
        ("pressure_on_passing_downs", pressure_passing),
        ("pressure_overall", pressure_all),
        ("zone_shell", zone_shells),
    ):
        changes.extend(_changes_respecting_games(triples, metric))
    concept_rows = {
        concept: _concept_row(concept, rows) for concept, rows in sorted(concepts.items())
    }
    quick_poor = any(
        concept_rows.get(concept, {}).get("poor_conversion") for concept in _QUICK
    )
    pressure_up = next(
        (
            row for row in changes
            if row["metric"] == "pressure_on_passing_downs" and row["direction"] == "increased"
        ),
        None,
    )
    hypothesis = None
    if pressure_up and quick_poor:
        hypothesis = {
            "id": "revised_away_from_quick_pressure",
            "text": (
                "Passing-down pressure increased earlier, but verified quick-concept "
                "results no longer support that response."
            ),
            "confidence": pressure_up["confidence"],
            "revised": True,
        }
    elif pressure_up:
        hypothesis = {
            "id": "learned_passing_down_pressure",
            "text": (
                "Historical evidence suggests the opponent is increasing pressure on "
                "passing downs. Quick concepts may offer a favorable response, but "
                "the current defensive look remains uncertain."
            ),
            "confidence": pressure_up["confidence"],
            "revised": False,
        }
    uncertain = [
        f"{row['name']} ({row['context']}): n={row['sample_size']}"
        for row in estimates if not row["published"]
    ]
    enough = [
        f"{row['name']} ({row['context']}): n={row['sample_size']}, n_eff={row['effective_sample_size']}"
        for row in estimates if row["published"]
    ]
    missing = []
    if not usable:
        missing.append("verified_executions_with_outcomes")
    if unscoped_beside_named_games:
        missing.append("game_id_missing_on_some_snaps_excluded_from_change_detection")
    if not pressure_all:
        missing.append("defensive_labels_that_state_pressure_or_no_pressure")
    if not any(row.get("down") is not None for row in usable):
        missing.append("down_and_distance_on_verified_snaps")
    if not any(row.get("yards") is not None for row in usable):
        missing.append("verified_yardage")
    if not concept_rows:
        missing.append("executed_play_names_that_map_to_concepts")
    missing.append("personnel_and_player_attributes")
    missing.append("route_diagrams")
    return {
        "version": LEARNER_VERSION,
        "concept_version": CONCEPT_EVIDENCE_VERSION,
        "usable_verified_snaps": len(usable),
        "approved_film_observations": len(film_rows),
        "trained_on_recommendations": False,
        "video_observations_included": False,
        "change_detection": "within_each_game",
        "estimates": estimates,
        "changes": changes,
        "concepts": concept_rows,
        "situations_with_enough_evidence": enough,
        "uncertain_situations": uncertain,
        "strategy_hypothesis": hypothesis,
        "missing_data": missing,
        "adjustment_causality": "not_estimated_without_explicit_execution_verification",
        "note": (
            "Rates are shrunk toward 0.5 with a prior strength of 2. "
            "A context is published only at n>=8. A change needs at least "
            "4 labeled snaps in each half of the same game and a shrunk gap of 0.25. "
            "Snaps from different games are not compared. Two or three snaps stay uncertain. "
            "A missing verification flag is not treated as verified."
        ),
    }


def learning_adjustment(
    play: str,
    sit: Any,
    model: Mapping[str, Any] | None,
    *,
    knowledge_delta: float = 0.0,
    strategy_delta: float = 0.0,
    current_pressure_observed: bool = False,
) -> dict[str, Any]:
    """Bounded nudge. Zero when the sample is thin or another prior already scored it."""
    profile = profile_for_play(play)
    concept = profile.get("concept_id")
    reasons = ["no verified tendency moves this play"]
    withheld: list[str] = []
    delta = 0.0
    if not model or not concept:
        return _adjustment(play, concept, 0.0, reasons, withheld)
    evidence = (model.get("concepts") or {}).get(concept) or {}
    if evidence.get("poor_conversion"):
        return _adjustment(
            play, concept, 0.0,
            [f"verified {concept} conversion is too poor to prefer; sample {evidence.get('sample_size')}"],
            ["poor_verified_conversion"],
        )
    hypothesis = model.get("strategy_hypothesis") or {}
    if hypothesis.get("id") == "revised_away_from_quick_pressure":
        return _adjustment(
            play, concept, 0.0,
            ["learned pressure response was revised after contradictory results"],
            ["strategy_revised"],
        )
    change = next(
        (
            row for row in (model.get("changes") or [])
            if row.get("metric") == "pressure_on_passing_downs" and row.get("direction") == "increased"
        ),
        None,
    )
    try:
        down = int(getattr(sit, "down", None))
    except (TypeError, ValueError):
        down = None
    if change and down in _PASSING_DOWNS and concept in _QUICK:
        if current_pressure_observed:
            withheld.append("current pressure is already handled by the coordinator")
        elif float(knowledge_delta) > 0 or float(strategy_delta) > 0:
            withheld.append("knowledge or strategy already scored this concept")
        else:
            span = abs(float(change.get("shrunk_delta") or 0.0))
            scale = min(1.0, span / 0.50)
            sample = int(change.get("early_n") or 0) + int(change.get("late_n") or 0)
            if sample < 16:
                scale *= 0.5
            delta = LEARNING_CAP * scale
            reasons = [
                "historical passing-down pressure increased; quick concept is a bounded response; "
                "the current defensive look remains uncertain"
            ]
    return _adjustment(play, concept, delta, reasons, withheld)


def _adjustment(
    play: str,
    concept: str | None,
    delta: float,
    reasons: list[str],
    withheld: list[str],
) -> dict[str, Any]:
    return {
        "play": play,
        "concept_id": concept,
        "delta": round(min(LEARNING_CAP, max(0.0, delta)), 5),
        "reasons": reasons,
        "withheld": withheld,
        "cap": LEARNING_CAP,
        "not_a_script": True,
        "causal_adjustment_claim": False,
    }


def _row_to_record(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if row.get("eligibility") != "verified_execution":
        return None
    if str(row.get("executed_verification") or "").lower() != "verified":
        return None
    if str(row.get("executed_status") or "").lower() != "identified":
        return None
    if row.get("trusted_vod"):
        return None
    play = row.get("executed_play")
    concept = profile_for_play(play).get("concept_id") if play else None
    labeled = _labels({
        "down": row.get("down"),
        "distance": row.get("distance"),
        "yards": row.get("yards"),
        "outcome": row.get("result") or row.get("outcome"),
    })
    return {
        "snap_seq": row.get("snap_seq") or row.get("id"),
        "snap_id": row.get("snap_id") or row.get("ml_snap_id"),
        "game_id": row.get("game_id") or row.get("session_id"),
        "opponent_id": row.get("opponent_id"),
        "down": row.get("down"),
        "distance": row.get("distance"),
        "formation": row.get("executed_formation"),
        "play": play,
        "concept_id": concept,
        "observed_defense": row.get("coverage_seen") or row.get("coverage_hint"),
        "observation": structure_observation(
            row.get("coverage_seen") or row.get("coverage_hint"),
            timing="post_snap" if row.get("coverage_seen") else "historical",
            source="historical",
        ),
        "model_probability": row.get("probability"),
        "verified_execution": True,
        "recommendation_only": False,
        "source": "verified_execution",
        **labeled,
    }


def opponent_learning_report(
    db: Any = None,
    *,
    opponent_id: str = "cpu",
    game_id: str | None = None,
    film_store: str | None = None,
    include_admitted_film: bool = False,
) -> dict[str, Any]:
    """Read-only tactical report. Does not write gameplay history."""
    from cfb_coach.madden.model.offense_strategy import current_strategy

    rows: list[dict[str, Any]] = []
    read_error = None
    if db is not None:
        try:
            from cfb_coach.madden.model.dataset import build_rows

            rows = build_rows(db=db)
        except Exception as exc:  # noqa: BLE001 — report the gap, do not invent rows
            read_error = str(exc)
            rows = []
    scoped = scope_training_rows(rows, opponent_id=opponent_id, game_id=game_id)
    rows = list(scoped["rows"])
    eligibility: dict[str, int] = {}
    for row in rows:
        key = str(row.get("eligibility") or "unknown")
        eligibility[key] = eligibility.get(key, 0) + 1
    records = [record for record in (_row_to_record(row) for row in rows) if record]
    admitted_ids: list[str] = []
    if include_admitted_film and film_store:
        from cfb_coach.madden.model.film_evidence import load_admitted_records

        admitted = load_admitted_records(
            film_store, game_id=game_id, opponent_id=opponent_id,
        )
        records.extend(admitted)
        admitted_ids = [str(row.get("evidence_id")) for row in admitted if row.get("evidence_id")]
    model = fit_opponent_model(records=records)
    plan = current_strategy(None, None, learned=model)
    influence = (
        "A published passing-down pressure increase can add at most "
        f"{LEARNING_CAP:.3f} to a quick concept on 3rd or 4th down when the "
        "current look is still unknown and knowledge or strategy did not already "
        "score that concept. Fewer than 16 labeled snaps use half of that. "
        "A clearly stronger learned play is unchanged. Recommendations and "
        "video rows are not training evidence."
    )
    return {
        "version": LEARNER_VERSION,
        "opponent_id": opponent_id,
        "game_id": game_id,
        "opponent_scope": "exact_opponent_id",
        "game_scope": "requested_game_only" if game_id else "all_games_for_this_opponent",
        "excluded_missing_opponent": scoped["excluded_missing_opponent"],
        "excluded_other_opponent": scoped["excluded_other_opponent"],
        "excluded_other_game": scoped["excluded_other_game"],
        "read_only": True,
        "history_modified": False,
        "learned_from_video": False,
        "usable_verified_snaps": model["usable_verified_snaps"],
        "approved_film_observations": model.get("approved_film_observations", 0),
        "admitted_film_evidence_ids": admitted_ids,
        "eligibility_counts": eligibility,
        "recommendation_only": eligibility.get("recommendation_only", 0),
        "trusted_vod_rows_excluded": eligibility.get("trusted_vod", 0),
        "tendencies": model["estimates"],
        "changes": model["changes"],
        "concepts": model["concepts"],
        "situations_with_enough_evidence": model["situations_with_enough_evidence"],
        "uncertain_situations": model["uncertain_situations"],
        "current_strategy_hypothesis": plan.get("hypothesis"),
        "hypothesis_id": plan.get("hypothesis_id"),
        "how_this_could_influence_decisions": influence,
        "missing_data": model["missing_data"] + (
            [f"database_read_failed: {read_error}"] if read_error else []
        ),
        "reusable_film_components": {
            "vod_success": "trains a separate success file and does not enter the live coordinator",
            "vod_model": "sample-gated call prior; silent below n=8; not called on this path",
            "activated_for_sprint_13": False,
        },
        "held_out_evaluation": "insufficient",
        "win_rate_claim": False,
        "note": model["note"],
    }
