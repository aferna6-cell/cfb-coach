"""Verified defensive snaps for ``ml train-defense`` and ``ml eval-defense``.

Verified means one of two things, and nothing else:

* an executed/accepted live defensive row, or
* a ``concept_seen`` label on a VOD row whose ``review_status`` is in the
  reviewed set or whose ``trainable`` flag is true, that is not
  ``excluded_from_training``, and whose concept_seen confidence is at least
  ``MIN_CONCEPT_SEEN_CONFIDENCE``.

``permitted_use`` is a rights field. It does not make a VOD row verified.
Live defense rows and VOD observed-defense rows are separate sources. A
recommendation, an auto-labelled non-trainable row, and the blocked VOD
source are not verified. Zero verified snaps is a refusal. Training never
runs on that empty set, and nothing here promotes the established defense
caller.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

MIN_CONCEPT_SEEN_CONFIDENCE = 0.5
VERIFIED_DEFINITION = (
    "A defensive snap is verified when it is an executed/accepted live row, "
    "or it has a concept_seen label, review_status is in the reviewed set or "
    "trainable is true, excluded_from_training is not true, and concept_seen "
    f"confidence is at least {MIN_CONCEPT_SEEN_CONFIDENCE}."
)
P95_LIMIT_MS = 150.0
SMALL_N = 8
_ACCEPTED = frozenset({
    "accepted", "executed", "used_recommended", "yes", "true",
})
_NOT_APPLIED = frozenset({
    "rejected", "no", "false", "declined", "different", "used_different",
})
_LIVE_STATUS = frozenset({"identified", "accepted", "executed"})


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def _side(row: Mapping[str, Any]) -> str:
    return str(row.get("side") or "").strip().lower()


def _is_defense(row: Mapping[str, Any]) -> bool:
    side = _side(row)
    return side in ("", "defense", "d") or side.startswith("defense")


def _source_text(row: Mapping[str, Any]) -> str:
    parts = [
        row.get("source"), row.get("vod_id"), row.get("video_id"),
        row.get("youtube_id"), row.get("source_id"), row.get("url"),
        row.get("provenance"), row.get("data_source"),
    ]
    return " ".join(str(part) for part in parts if part not in (None, ""))


def _excluded_vod(row: Mapping[str, Any]) -> bool:
    from cfb_coach.madden.model.dataset import vod_source_excluded

    return vod_source_excluded(row)


def _concept_seen_confidence(row: Mapping[str, Any]) -> float | None:
    """Numeric concept_seen confidence, or None when the row has none."""
    raw = row.get("concept_seen_confidence")
    if raw is None:
        fields = row.get("field_confidence")
        if isinstance(fields, Mapping):
            raw = fields.get("concept_seen")
    if raw is None or isinstance(raw, bool) or _blank(raw):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _excluded_from_training(row: Mapping[str, Any]) -> bool:
    flag = row.get("excluded_from_training")
    if flag is True or flag == 1:
        return True
    if isinstance(flag, str) and flag.strip().lower() in {"true", "1", "yes"}:
        return True
    return False


def _vod_quality(row: Mapping[str, Any]) -> bool:
    """Quality gate for a VOD row. ``permitted_use`` is not consulted."""
    from cfb_coach.madden.model.dataset import REVIEWED_STATUSES

    reviewed = str(row.get("review_status") or "") in REVIEWED_STATUSES
    if not (reviewed or row.get("trainable") is True):
        return False
    if _excluded_from_training(row):
        return False
    confidence = _concept_seen_confidence(row)
    return confidence is not None and confidence >= MIN_CONCEPT_SEEN_CONFIDENCE


def _looks_like_vod(row: Mapping[str, Any]) -> bool:
    if row.get("trusted_vod") or row.get("vod_id") or row.get("recording_id"):
        return True
    if row.get("observed_defense") not in (None, ""):
        return True
    if row.get("review_status") or row.get("permitted_use"):
        return True
    text = _source_text(row).lower()
    return "vod" in text


def _executed_or_accepted(row: Mapping[str, Any]) -> bool:
    if row.get("recommendation_only") is True:
        return False
    status = str(row.get("executed_status") or "").strip().lower()
    verification = str(row.get("executed_verification") or "").strip().lower()
    accepted = str(row.get("accepted") or "").strip().lower()
    if verification == "verified" and status in _LIVE_STATUS:
        return True
    return accepted in _ACCEPTED


def verification_source(row: Mapping[str, Any]) -> str | None:
    """``live`` or ``vod`` when the row is a verified defensive snap, else None."""
    if not isinstance(row, Mapping) or row.get("exclude") or row.get("invalid"):
        return None
    if not _is_defense(row) or _side(row) in ("offense", "o"):
        return None
    if _excluded_vod(row):
        return None
    live = _executed_or_accepted(row) and not _looks_like_vod(row)
    if live:
        return "live"
    concept = str(row.get("concept_seen") or "").strip()
    if concept and _looks_like_vod(row) and _vod_quality(row):
        return "vod"
    return None


def partition_verified(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Split verified snaps by source. Unverified rows are omitted."""
    buckets: dict[str, list[dict[str, Any]]] = {"live": [], "vod": []}
    for row in rows:
        source = verification_source(row)
        if source:
            buckets[source].append(dict(row))
    return buckets


def verified_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    parts = partition_verified(rows)
    live, vod = len(parts["live"]), len(parts["vod"])
    return {"live": live, "vod": vod, "total": live + vod}


def format_verified_counts(counts: Mapping[str, int]) -> str:
    return "\n".join((
        f"verified_definition: {VERIFIED_DEFINITION}",
        f"verified_live: {int(counts.get('live') or 0)}",
        f"verified_vod: {int(counts.get('vod') or 0)}",
        f"verified_total: {int(counts.get('total') or 0)}",
    ))


def _success_text(row: Mapping[str, Any]) -> str | None:
    raw = row.get("success")
    if raw is None:
        raw = row.get("stop")
    if isinstance(raw, bool):
        return "true" if raw else "false"
    if _blank(raw):
        return None
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "y", "success", "stop"):
        return "true"
    if text in ("0", "false", "no", "n", "fail", "failure"):
        return "false"
    return None


def training_view(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one verified snap into the stop-model row shape."""
    out = dict(row)
    out["side"] = "defense"
    play = (
        out.get("executed_play") or out.get("action_play") or out.get("play")
        or out.get("observed_defense") or ""
    )
    out["executed_play"] = str(play) if play else ""
    success = _success_text(row)
    if success is not None:
        out["success"] = success
    return out


def _match_id(row: Mapping[str, Any]) -> str:
    for key in ("match_id", "video_id", "game_id", "vod_id", "session_id"):
        value = row.get(key)
        if not _blank(value):
            return str(value)
    return "unknown"


def record_defensive_execution(
    row: Mapping[str, Any] | None = None,
    *,
    executed_status: str | None = None,
    executed_play: str | None = None,
    executed_formation: str | None = None,
    recommended_play: str | None = None,
    accepted: str | None = None,
    executed_adjustment_id: str | None = None,
    touched: bool = False,
    adjustment_confirmed: bool = False,
) -> dict[str, Any]:
    """Optional accept/execute stamp for a defensive call.

    Reuses the offense helpers. Omitting every option leaves ``accepted`` and
    ``executed_status`` at ``unknown`` and does not store an executed
    adjustment. Passing the same flags the offense path already accepts is
    what records a confirmation.
    """
    from cfb_coach.madden.model.call_outcome_log import (
        acceptance_for_log, executed_adjustment_for_log,
    )

    out = dict(row or {})
    out.setdefault("side", "defense")
    opted = (
        touched or adjustment_confirmed
        or executed_status is not None or accepted is not None
    )
    if not opted:
        out.setdefault("accepted", "unknown")
        out.setdefault("executed_status", "unknown")
        out.setdefault("executed_verification", "unknown")
        return out
    status = executed_status if executed_status is not None else out.get("executed_status")
    status = str(status or "unknown").strip().lower() or "unknown"
    played = executed_play if executed_play is not None else out.get("executed_play")
    shown = recommended_play
    if shown is None:
        shown = out.get("recommended_play") or out.get("play")
    verification = str(out.get("executed_verification") or "unknown")
    if status == "used_recommended":
        status = "identified"
        played = played or shown
        verification = "verified"
    elif status == "identified" and touched:
        verification = "verified"
    out["executed_status"] = status
    out["executed_verification"] = verification
    if played is not None:
        out["executed_play"] = played
    if executed_formation is not None:
        out["executed_formation"] = executed_formation
    out["accepted"] = acceptance_for_log(
        executed_status=status,
        executed_play=None if played is None else str(played),
        recommended_play=None if shown is None else str(shown),
        explicit=accepted,
        touched=touched,
    )
    adjustment = executed_adjustment_id
    if adjustment is None:
        adjustment = out.get("recommended_adjustment_id") or out.get("macro")
    if adjustment_confirmed:
        out["executed_adjustment_id"] = executed_adjustment_for_log(
            None if adjustment is None else str(adjustment),
            confirmed=True,
        )
    return out


def _situation(row: Mapping[str, Any]) -> str:
    from cfb_coach.madden.model.defense_coordinator import _bucket

    return _bucket(
        row.get("down"), row.get("distance"),
        red_zone=bool(row.get("red_zone")), goal_line=bool(row.get("goal_line")),
        two_minute=bool(row.get("two_minute")),
    )


def _family(play: str) -> str | None:
    from cfb_coach.madden.defense_select import FAMILIES, call_family

    fam = call_family(play)
    return fam if fam in FAMILIES else None


def _rate(won: int, total: int) -> float:
    return (won + 4.0) / (total + 8.0)


def _fit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = [0, 0]
    by_sit: dict[str, list[int]] = {}
    by_fam: dict[str, list[int]] = {}
    by_sit_fam: dict[str, list[int]] = {}
    for row in rows:
        success = _success_text(row)
        if success is None:
            continue
        ok = int(success == "true")
        sit = _situation(row)
        fam = _family(str(row.get("executed_play") or ""))
        overall[0] += ok
        overall[1] += 1
        slot = by_sit.setdefault(sit, [0, 0])
        slot[0] += ok
        slot[1] += 1
        if fam:
            fam_slot = by_fam.setdefault(fam, [0, 0])
            fam_slot[0] += ok
            fam_slot[1] += 1
            both = by_sit_fam.setdefault(sit + "|" + fam, [0, 0])
            both[0] += ok
            both[1] += 1
    return {
        "overall": overall, "by_sit": by_sit, "by_fam": by_fam,
        "by_sit_fam": by_sit_fam,
    }


def _predict(fit: Mapping[str, Any], row: Mapping[str, Any]) -> tuple[float, float, bool]:
    """Return ``(model_p, situation_baseline_p, used_model)`` for the executed play.

    The baseline is the situation stop rate, which is what the established
    caller uses. The model uses that situation's rate for the executed
    play's coverage family. ``used_model`` is false when that family slot is
    missing and the score falls back to the situation baseline.
    """
    sit = _situation(row)
    base = fit["by_sit"].get(sit) or fit["overall"]
    baseline = _rate(int(base[0]), int(base[1])) if base else 0.5
    fam = _family(str(row.get("executed_play") or ""))
    slot = None
    if fam:
        slot = fit["by_sit_fam"].get(sit + "|" + fam) or fit["by_fam"].get(fam)
    if not slot:
        return baseline, baseline, False
    return _rate(int(slot[0]), int(slot[1])), baseline, True


def _unlabelled_reason(row: Mapping[str, Any]) -> str:
    raw = row.get("success")
    if raw is None:
        raw = row.get("stop")
    if raw is None or _blank(raw):
        return "missing_success"
    return "unparseable_success"


def _brier(y: float, p: float) -> float:
    return (y - p) ** 2


def _ci(samples: Sequence[float]) -> list[float] | None:
    ordered = sorted(float(item) for item in samples)
    if not ordered:
        return None
    if len(ordered) == 1:
        return [round(ordered[0], 6), round(ordered[0], 6)]
    lo = ordered[int(0.025 * (len(ordered) - 1))]
    hi = ordered[int(0.975 * (len(ordered) - 1))]
    return [round(lo, 6), round(hi, 6)]


def _percentile(values: Sequence[float], fraction: float) -> float:
    from cfb_coach.madden.model.dual_control import percentile

    return percentile([float(item) for item in values], fraction)


def evaluate_defense(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 7, n_boot: int = 400,
) -> dict[str, Any]:
    """Hold out by match and compare with the situation baseline.

    The paired difference is baseline Brier minus model Brier. Positive means
    the model is closer to the outcome than the established situation rate.
    The gate passes only when that 95% CI lies entirely above 0 and scoring
    p95 is under 150 ms. Passing does not enable the caller.
    """
    labeled = []
    drop_reasons: dict[str, int] = {}
    for row in rows:
        if _success_text(row) is None:
            reason = _unlabelled_reason(row)
            drop_reasons[reason] = drop_reasons.get(reason, 0) + 1
            continue
        labeled.append(training_view(row))
    matches = sorted({_match_id(row) for row in labeled})
    report: dict[str, Any] = {
        "definition": VERIFIED_DEFINITION,
        "holdout": "match",
        "n_verified_labeled": len(labeled),
        "n_dropped_unlabelled": sum(drop_reasons.values()),
        "unlabelled_drop_reasons": drop_reasons,
        "n_matches": len(matches),
        "beats_baseline": False,
        "ci_excludes_0": False,
        "diff": None,
        "ci95": None,
        "p95_ms": None,
        "p95_under_150": False,
        "gate": "fail",
        "situations": {},
        "auto_promote": False,
    }
    if len(matches) < 2 or not labeled:
        report["reason"] = "need at least 2 matches with outcomes to hold out by match"
        return report
    by_match: dict[str, list[dict[str, Any]]] = {match: [] for match in matches}
    for row in labeled:
        by_match[_match_id(row)].append(row)
    _family(str(labeled[0].get("executed_play") or ""))
    _situation(labeled[0])
    scored: list[dict[str, Any]] = []
    latencies: list[float] = []
    for held in matches:
        train = [row for match, group in by_match.items() if match != held for row in group]
        fit = _fit(train)
        for row in by_match[held]:
            started = time.perf_counter()
            model_p, base_p, used_model = _predict(fit, row)
            latencies.append((time.perf_counter() - started) * 1000.0)
            y = 1.0 if _success_text(row) == "true" else 0.0
            diff = _brier(y, base_p) - _brier(y, model_p)
            scored.append({
                "match": held, "situation": _situation(row), "diff": diff, "y": y,
                "model_used": used_model,
            })
    point = sum(item["diff"] for item in scored) / len(scored)
    rng = random.Random(seed)
    grouped = {
        match: [item["diff"] for item in scored if item["match"] == match]
        for match in matches
    }
    samples = []
    for _ in range(n_boot):
        drawn = [rng.choice(matches) for _ in matches]
        bag = [value for match in drawn for value in grouped[match]]
        if bag:
            samples.append(sum(bag) / len(bag))
    ci = _ci(samples)
    lo = None if not ci else ci[0]
    hi = None if not ci else ci[1]
    excludes = bool(ci) and lo is not None and hi is not None and (hi < 0 or lo > 0)
    beats = bool(lo is not None and lo > 0)
    p95 = round(_percentile(latencies, 0.95), 3) if latencies else None
    under = bool(p95 is not None and p95 < P95_LIMIT_MS)
    situations: dict[str, Any] = {}
    for sit in sorted({item["situation"] for item in scored}):
        sit_rows = [item for item in scored if item["situation"] == sit]
        sit_point = sum(item["diff"] for item in sit_rows) / len(sit_rows)
        sit_samples = []
        sit_matches = sorted({item["match"] for item in sit_rows})
        grouped = {
            match: [item["diff"] for item in sit_rows if item["match"] == match]
            for match in sit_matches
        }
        for _ in range(n_boot):
            drawn = [rng.choice(sit_matches) for _ in sit_matches]
            bag = [value for match in drawn for value in grouped[match]]
            if bag:
                sit_samples.append(sum(bag) / len(bag))
        sit_ci = _ci(sit_samples)
        n_model = sum(1 for item in sit_rows if item["model_used"])
        situations[sit] = {
            "n": len(sit_rows),
            "n_model_used": n_model,
            "n_baseline_fallback": len(sit_rows) - n_model,
            "diff": round(sit_point, 6),
            "ci95": sit_ci,
        }
    report.update({
        "diff": round(point, 6),
        "ci95": ci,
        "ci_excludes_0": excludes,
        "beats_baseline": beats,
        "p95_ms": p95,
        "p95_under_150": under,
        "gate": "pass" if beats and excludes and under else "fail",
        "situations": situations,
        "reason": (
            "model beats the situation baseline; CI excludes 0; p95 under 150 ms"
            if beats and excludes and under else
            "gate refused: need a CI entirely above 0 and p95 under 150 ms"
        ),
    })
    return report


def report_allows_enable(model: Mapping[str, Any] | None) -> tuple[bool, str]:
    """True only when the installed artifact's eval report clears the gate."""
    report = (model or {}).get("eval_report") if isinstance(model, Mapping) else None
    if not isinstance(report, Mapping) or not report:
        return False, "installed artifact has no eval report"
    ci = report.get("ci95")
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return False, "eval report does not beat the baseline with a CI excluding 0"
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError):
        return False, "eval report does not beat the baseline with a CI excluding 0"
    excludes = (
        report.get("ci_excludes_0") is True
        and report.get("beats_baseline") is True
        and lo > 0
        and hi > 0
    )
    if not excludes:
        return False, "eval report does not beat the baseline with a CI excluding 0"
    try:
        p95 = float(report.get("p95_ms"))
    except (TypeError, ValueError):
        return False, "eval report p95 is not under 150 ms"
    if p95 >= P95_LIMIT_MS or report.get("p95_under_150") is not True:
        return False, "eval report p95 is not under 150 ms"
    if report.get("gate") != "pass":
        return False, "eval report gate is not pass"
    return True, "eval report beats the baseline"


def _application(row: Mapping[str, Any]) -> str:
    """``applied``, ``not_applied``, or ``unknown`` when nothing was recorded."""
    recorded = any(
        key in row for key in ("executed_adjustment_id", "accepted", "executed_status")
    )
    if not recorded:
        return "unknown"
    accepted = row.get("accepted")
    executed = row.get("executed_adjustment_id")
    status = row.get("executed_status")
    if _blank(accepted) and _blank(executed) and _blank(status):
        return "unknown"
    accepted_text = str(accepted or "").strip().lower()
    status_text = str(status or "").strip().lower()
    if accepted_text in _ACCEPTED or status_text in _ACCEPTED:
        return "applied"
    if not _blank(executed):
        recommended = str(
            row.get("recommended_adjustment_id")
            or row.get("recommended_macro")
            or row.get("macro")
            or ""
        ).strip()
        if not recommended or str(executed).strip() == recommended:
            return "applied"
        return "not_applied"
    if accepted_text in _NOT_APPLIED or status_text in _NOT_APPLIED:
        return "not_applied"
    return "unknown"


def _outcome_value(row: Mapping[str, Any]) -> float | None:
    success = _success_text(row)
    if success is None:
        raw = row.get("outcome")
        if isinstance(raw, bool):
            return 1.0 if raw else 0.0
        if _blank(raw):
            return None
        text = str(raw).strip().lower()
        if text in ("1", "true", "yes", "success", "stop"):
            return 1.0
        if text in ("0", "false", "no", "fail"):
            return 0.0
        return None
    return 1.0 if success == "true" else 0.0


def _recommended(row: Mapping[str, Any]) -> bool:
    for key in ("recommended_adjustment_id", "recommended_macro", "macro"):
        if not _blank(row.get(key)) and str(row.get(key)).strip().lower() != "none":
            return True
    return False


def evaluate_adjustments(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 7, n_boot: int = 400,
) -> dict[str, Any]:
    """Recommended vs applied vs outcome, per side, with bootstrap CIs.

    Applied means executed or accepted. When application was not recorded the
    count is ``unknown``, not zero. A side with fewer than ``SMALL_N`` snaps
    carries a small-n warning.
    """
    rng = random.Random(seed)
    sides: dict[str, Any] = {}
    warnings: list[str] = []
    aliases = {"offense": ("offense", "o"), "defense": ("defense", "d")}
    for side in ("offense", "defense"):
        chosen = [row for row in rows if _side(row) in aliases[side]]
        recommended = sum(1 for row in chosen if _recommended(row))
        states = [_application(row) for row in chosen]
        recorded = [row for row, state in zip(chosen, states) if state != "unknown"]
        applied_rows = [row for row, state in zip(chosen, states) if state == "applied"]
        if not recorded:
            applied: Any = "unknown"
        else:
            applied = len(applied_rows)
        outcomes = [(row, _outcome_value(row)) for row in applied_rows]
        known = [(row, value) for row, value in outcomes if value is not None]
        if not known:
            outcome: Any = "unknown"
            ci: Any = "unknown"
        else:
            point = sum(value for _row, value in known) / len(known)
            matches = sorted({_match_id(row) for row, _value in known})
            grouped = {
                match: [value for row, value in known if _match_id(row) == match]
                for match in matches
            }
            samples = []
            for _ in range(n_boot):
                drawn = [rng.choice(matches) for _ in matches]
                bag = [value for match in drawn for value in grouped[match]]
                if bag:
                    samples.append(sum(bag) / len(bag))
            outcome = round(point, 6)
            ci = _ci(samples) or "unknown"
        n = len(chosen)
        warn = n < SMALL_N
        if warn:
            warnings.append(f"small-n warning: {side} n={n}")
        sides[side] = {
            "n": n,
            "recommended": recommended,
            "applied": applied,
            "application_unrecorded": sum(1 for state in states if state == "unknown"),
            "outcome": outcome,
            "ci95": ci,
            "small_n_warning": warn,
        }
    return {
        "definition": (
            "applied means executed or accepted; unknown means application was not recorded"
        ),
        "small_n": SMALL_N,
        "warnings": warnings,
        "sides": sides,
        "auto_promote": False,
    }


def load_jsonl(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def rows_from_db(db: Any) -> list[dict[str, Any]]:
    """Live snaps plus outcome columns. Does not invent rows."""
    from cfb_coach.madden.model.dataset import build_rows

    built = build_rows(db=db)
    extra: dict[str, dict[str, Any]] = {}
    try:
        outcome_cols = {
            str(row[1]) for row in db.conn.execute("PRAGMA table_info(ml_outcomes)")
        }
    except Exception:  # noqa: BLE001
        outcome_cols = set()
    if outcome_cols and "snap_id" in outcome_cols:
        try:
            for item in db.conn.execute("SELECT * FROM ml_outcomes ORDER BY id ASC"):
                payload = dict(item)
                snap_id = payload.get("snap_id")
                if snap_id is not None:
                    extra[str(snap_id)] = payload
        except Exception:  # noqa: BLE001
            extra = {}
    for row in built:
        snap_id = row.get("snap_id") or row.get("ml_snap_id")
        found = extra.get(str(snap_id)) if snap_id is not None else None
        if not found:
            continue
        for key in (
            "accepted", "executed_adjustment_id", "recommended_adjustment_id",
            "executed_status", "executed_verification",
        ):
            if key in found and key not in row:
                row[key] = found.get(key)
    return built
