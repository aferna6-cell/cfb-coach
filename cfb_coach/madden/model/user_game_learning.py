"""Auditable, OFFLINE training from the last four logged human Madden games.

Runs against the owner's local Madden DB (never in GitHub). Read-only by
default; training writes new shadow artifacts, without enabling live models.
Unverified plays, missing outcomes, CPU games and recommendations are excluded.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

from cfb_coach.madden.model.defense_intelligence import verified_human_rows

SCHEMA = "madden.user_four_game_learning.v1"


def select_recent_human_games(
    rows: Sequence[Mapping[str, Any]], max_games: int = 4,
) -> dict[str, Any]:
    if max_games < 1:
        raise ValueError("max_games must be positive")
    sessions: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        if str(row.get("opponent_type") or "").lower() != "human":
            continue
        gid = str(row.get("game_id") or "").strip()
        oid = str(row.get("opponent_id") or "").strip()
        if not gid or not oid:
            continue
        entry = sessions.setdefault(gid, {"game_id": gid, "opponent_id": oid, "rows": []})
        if entry["opponent_id"] != oid:
            # Ambiguous cross-opponent identity: do not mix different opponents
            # into one reported game.
            entry["ambiguous"] = True
        entry["rows"].append(dict(row))
    games = [item for item in sessions.values() if not item.get("ambiguous")][-max_games:]
    chosen = [row for game in games for row in game["rows"]]
    verified = verified_human_rows(chosen)
    by_game: dict[str, dict[str, int]] = {}
    for game in games:
        relevant = [row for row in verified if row["game_id"] == game["game_id"]]
        by_game[game["game_id"]] = {
            "total_rows": len(game["rows"]),
            "verified_offense": sum(r["side"] == "offense" for r in relevant),
            "verified_defense": sum(r["side"] == "defense" for r in relevant),
            "unverified_or_unlabeled": len(game["rows"]) - len(relevant),
            "observed_opponent_concepts": sum(
                bool(r.get("concept_seen")) and r["side"] == "defense" for r in relevant
            ),
        }
    return {
        "schema": SCHEMA,
        "requested_games": max_games,
        "found_human_games": len(games),
        "selected_game_ids": [item["game_id"] for item in games],
        "rows_scanned": len(chosen),
        "verified_labeled_human_executions": len(verified),
        "verified_offense": sum(r["side"] == "offense" for r in verified),
        "verified_defense": sum(r["side"] == "defense" for r in verified),
        "games": by_game,
        "training_rows": verified,
        "notes": [
            "Game order uses the local database snap insertion order, not alphabetical IDs.",
            "No CPU games, invented snap identities, or assumed execution from recommendations.",
            "Post-snap concept_seen is retained only as a historical tendency.",
            "A low number of verified defensive snaps means a prior-driven defense model.",
        ],
    }


def model_diagnostics(report: Mapping[str, Any]) -> dict[str, Any]:
    total = int(report.get("verified_labeled_human_executions") or 0)
    defense = int(report.get("verified_defense") or 0)
    offense = int(report.get("verified_offense") or 0)
    return {
        k: v for k, v in report.items() if k != "training_rows"
    } | {
        "model_readiness": {
            "offense": "verified_rows_available" if offense else "no_verified_offense_rows",
            "defense": "experimental_small_sample" if defense else "no_verified_defensive_rows",
            "four_games_available": int(report.get("found_human_games") or 0) == 4,
            "strong_outcome_claims": False,
            "total_verified_n": total,
        }
    }


def train_shadow_models(
    rows: Sequence[Mapping[str, Any]], *,
    out_dir: str | Path,
    max_games: int = 4,
) -> dict[str, Any]:
    from cfb_coach.madden.model import defense_coordinator
    from cfb_coach.madden.model import experimental_model
    report = select_recent_human_games(rows, max_games=max_games)
    if not report["training_rows"]:
        return {
            **model_diagnostics(report), "trained": False,
            "reason": "no eligible verified human-game executions",
            "installed": False, "live_mode_changed": False,
        }
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    result = model_diagnostics(report)
    paths: dict[str, str] = {}
    if report["verified_offense"]:
        offense_rows = [r for r in report["training_rows"] if r["side"] == "offense"]
        offense_art = experimental_model.train_experimental(offense_rows, side="offense")
        paths["offense"] = str(experimental_model.save_artifact(
            offense_art, folder / "offense_four_human_games_shadow.json"
        ))
    if report["verified_defense"]:
        defense_rows = [r for r in report["training_rows"] if r["side"] == "defense"]
        def_art = defense_coordinator.train_defense(defense_rows)
        paths["defense"] = str(defense_coordinator.save_model(
            def_art, folder / "defense_four_human_games_shadow.json"
        ))
        result["opponent_concepts_observed"] = (
            def_art["opponent_tendencies"]["observed_post_snap_concepts"]
        )
    result.update({
        "trained": True, "artifacts": paths,
        "installed": False, "live_mode_changed": False,
        "counterfactual_claim": False,
        "note": "These are shadow artifacts. Explicit user activation and per-game validation required.",
    })
    return result
