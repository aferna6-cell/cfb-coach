"""Score pipeline rows against a hand-labeled sample.

A field the human could not read is ``null``. The pipeline is right on that
field only when it also leaves the field blank. A filled value against a
blank label is a false fill. Recall is scored only where the human read
something.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FIELDS = (
    "down",
    "distance",
    "quarter",
    "yardline",
    "formation",
    "play",
    "coverage_seen",
    "result",
)

_RESULT_ALIASES = {
    "gain": "+",
    "incomplete": "incomplete",
    "inc": "incomplete",
    "sack": "sack",
    "int": "int",
    "interception": "int",
    "fumble": "fumble",
    "td": "td",
    "touchdown": "td",
    "convert": "convert",
    "no gain": "+0",
    "no-gain": "+0",
}


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _norm_result(value: Any) -> str:
    text = _norm_text(value)
    if not text:
        return ""
    if text in _RESULT_ALIASES:
        return _RESULT_ALIASES[text]
    if text.startswith("+") or text.startswith("-") or text[:1].isdigit():
        return text.replace(" ", "")
    for alias, canon in _RESULT_ALIASES.items():
        if text.startswith(alias):
            return canon if canon != "+" else text
    return text


def _equal(field: str, pred: Any, hand: Any) -> bool:
    if field == "result":
        return _norm_result(pred) == _norm_result(hand)
    if field in {"down", "distance", "quarter", "yardline"}:
        return _num(pred) == _num(hand)
    return _norm_text(pred) == _norm_text(hand)


def _num(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return _norm_text(value)


def _blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def score(predictions: list[dict[str, Any]], labels: list[dict[str, Any]], *, slack_s: float = 8.0) -> dict[str, Any]:
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used: set[int] = set()
    for label in labels:
        t = float(label["t_start"])
        best_i = None
        best_d = slack_s + 1
        for i, pred in enumerate(predictions):
            if i in used or pred.get("t_start") is None:
                continue
            if label.get("video_id") and pred.get("video_id") and label["video_id"] != pred["video_id"]:
                continue
            dist = abs(float(pred["t_start"]) - t)
            if dist < best_d:
                best_i, best_d = i, dist
        if best_i is None or best_d > slack_s:
            pairs.append(({}, label))
        else:
            used.add(best_i)
            pairs.append((predictions[best_i], label))

    per_field: dict[str, Any] = {}
    for field in FIELDS:
        known = 0
        known_correct = 0
        filled = 0
        filled_correct = 0
        false_fills = 0
        abstain_ok = 0
        compared = 0
        for pred, label in pairs:
            if field not in label:
                continue
            compared += 1
            pval = pred.get(field, "")
            hval = label.get(field)
            hand_blank = _blank(hval)
            pred_blank = _blank(pval)
            match = _equal(field, pval, hval)
            if hand_blank and pred_blank:
                abstain_ok += 1
            elif hand_blank and not pred_blank:
                false_fills += 1
            elif not hand_blank:
                known += 1
                if match:
                    known_correct += 1
            if not pred_blank:
                filled += 1
                if match and not hand_blank:
                    filled_correct += 1
        per_field[field] = {
            "hand_known": known,
            "recall": _rate(known_correct, known),
            "recall_correct": known_correct,
            "precision": _rate(filled_correct, filled),
            "precision_correct": filled_correct,
            "filled": filled,
            "false_fills": false_fills,
            "abstain_when_unknown": abstain_ok,
            "labeled": compared,
        }
    return {
        "labeled_snaps": len(labels),
        "matched_predictions": len(used),
        "unmatched_labels": len(labels) - len(used) if len(pairs) else 0,
        "fields": per_field,
    }


def _rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(num / den, 3)


def score_files(pred_json: Path, hand_json: Path) -> dict[str, Any]:
    pred = json.loads(pred_json.read_text(encoding="utf-8"))
    hand = json.loads(hand_json.read_text(encoding="utf-8"))
    snaps = pred["snaps"] if isinstance(pred, dict) else pred
    labels = hand["snaps"] if isinstance(hand, dict) else hand
    # Hand file may cover one video; filter predictions to that id when set.
    video_id = hand.get("video_id") if isinstance(hand, dict) else None
    if video_id:
        snaps = [s for s in snaps if s.get("video_id") == video_id]
    report = score(snaps, labels)
    report["video_id"] = video_id or ""
    return report
