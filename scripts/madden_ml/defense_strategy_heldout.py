"""Held-out check of the offline defensive evaluator on logged human games. Read-only.

For each human defensive snap, in time order, the evaluator sees only the
pre-snap context, earlier games (league family shares) and the earlier snaps
of the same game. It reports:

* threat calibration: log loss / top-1 of the opponent's actual concept family
  against the situation prior and a league-by-situation baseline
* legality and latency on the installed defensive book
* anonymized example assessments

It does not score defensive calls against outcomes: no human defensive snap
has a confirmed execution, so which call was run is unknown.

    python scripts/madden_ml/defense_strategy_heldout.py --db AUDIT_COPY.db --out DIR [--artifact PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cfb_coach.madden.model.defense_strategy_eval import (  # noqa: E402
    SITUATION_THREAT_PRIOR,
    THREAT_FAMILIES,
    assess_candidates,
    situation_key,
)
from cfb_coach.madden.model.strategic_context import context_from_snap  # noqa: E402
from cfb_coach.madden.situation import concept_family, parse_madden_situation  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "madden_ml"))
from opponent_prediction_eval import load_games  # noqa: E402

EPS = 1e-6
SEED = 7
N_BOOT = 2000
MIN_LEAGUE = 15


def load_defense_book(db: str) -> dict[str, list[str]]:
    """Installed defensive formations from a temporary copy, so the source is never opened for writing."""
    from cfb_coach.db import CoachDB
    from cfb_coach.madden import playbook

    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "book.db"
        shutil.copyfile(Path(db).expanduser(), copy)
        cdb = CoachDB(copy)
        try:
            books = playbook.load_books(cdb)
        finally:
            cdb.close()
    forms = (books.get("defense") or {}).get("formations") or {}
    return {str(f): [str(p) for p in plays] for f, plays in forms.items()}


def _league(earlier_games, key: str) -> dict[str, float] | None:
    counts: Counter = Counter()
    for game in earlier_games:
        for t, row in enumerate(game):
            fam = concept_family(row.get("concept_seen"))
            if fam not in THREAT_FAMILIES:
                continue
            if situation_key(context_from_snap(row, earlier=game[:t])) == key:
                counts[fam] += 1
    n = sum(counts.values())
    if n < MIN_LEAGUE:
        return None
    prior = SITUATION_THREAT_PRIOR[key]
    return {f: (counts[f] + 4.0 * prior[f]) / (n + 4.0) for f in THREAT_FAMILIES}


def _boot(by_game: dict[str, list[float]]) -> dict[str, Any]:
    games = sorted(by_game)
    flat = [d for g in games for d in by_game[g]]
    rng = random.Random(SEED)
    stats = sorted(
        (lambda v: sum(v) / len(v))([d for g in (games[rng.randrange(len(games))] for _ in games) for d in by_game[g]])
        for _ in range(N_BOOT))
    return {"mean_diff": sum(flat) / len(flat), "lo": stats[int(0.025 * N_BOOT)], "hi": stats[int(0.975 * N_BOOT)]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--artifact", default=None, help="defense_v1 artifact (read only)")
    args = ap.parse_args(argv)

    games = load_games(args.db)
    book = load_defense_book(args.db)
    artifact = None
    candidates = [{"formation": f, "play": p} for f, plays in book.items() for p in plays]
    if args.artifact:
        from cfb_coach.madden.model.defense_v1 import installed_candidates, load_artifact

        artifact = load_artifact(args.artifact)
        scored = {(c["formation"], c["play"]): c for c in installed_candidates(artifact, book)}
        candidates = [scored.get((c["formation"], c["play"]), c) for c in candidates]

    league_cache: dict[tuple[int, str], dict[str, float] | None] = {}
    rows_out, latencies, legal_counts, examples = [], [], [], []
    loss = {"evaluator": {}, "situation_prior": {}, "evaluator_no_game_history": {}}
    for gi, game in enumerate(games):
        for t, row in enumerate(game):
            ctx = context_from_snap(row, earlier=game[:t], legal_candidates=candidates)
            key = situation_key(ctx)
            if (gi, key) not in league_cache:
                league_cache[(gi, key)] = _league(games[:gi], key)
            league = league_cache[(gi, key)]
            sit = parse_madden_situation(str(row.get("situation_raw") or ""), default_side="defense")
            sit.side = "defense"
            a = assess_candidates(ctx, candidates, sit=sit, artifact=artifact, league_prior=league)
            latencies.append(a.latency_ms)
            legal_counts.append(sum(c.legal for c in a.candidates) / max(1, len(a.candidates)))
            fam = concept_family(row.get("concept_seen"))
            if fam in THREAT_FAMILIES:
                ctx0 = context_from_snap(row, earlier=[])
                a0 = assess_candidates(ctx0, candidates[:1], sit=sit, league_prior=league)
                g = f"G{gi + 1}"
                for name, dist in (("evaluator", a.threat), ("situation_prior", {
                        f: v / sum(SITUATION_THREAT_PRIOR[key].values()) for f, v in SITUATION_THREAT_PRIOR[key].items()}),
                        ("evaluator_no_game_history", a0.threat)):
                    loss[name].setdefault(g, []).append(-math.log(max(EPS, dist[fam])))
                rows_out.append({"game": g, "fam": fam, "top1": max(a.threat, key=a.threat.get) == fam,
                                 "game_plays": a.threat_sources["game_plays"], "key": key})
            if len(examples) < 4 and t >= 8 and t % 9 == 0:
                examples.append({
                    "game": f"G{gi + 1}", "snap_in_game": t + 1, "situation": f"{row.get('down')}&{row.get('distance')}",
                    "yards_to_goal": ctx.yards_to_goal.value, "situation_key": key,
                    "threat": {k: round(v, 3) for k, v in a.threat.items()},
                    "threat_sources": a.threat_sources, "uncertainty": a.uncertainty,
                    "top": [{"call": f"{c.formation} | {c.play}", "family": c.coverage_family, "front": c.front,
                             "legal": c.legal, "score": round(c.score, 3),
                             "conversion_risk": None if c.conversion_risk is None else round(c.conversion_risk, 3),
                             "components": [(x.name, round(x.value, 3), x.evidence) for x in c.components],
                             "versus_next": c.versus_next} for c in a.candidates[:3]],
                    "lowest_legal": next(f"{c.formation} | {c.play}" for c in reversed(a.candidates) if c.legal),
                    "illegal": [f"{c.formation} | {c.play}" for c in a.candidates if not c.legal][:3],
                    "actual_family_after_snap": fam,
                })

    def summary(name: str) -> dict[str, Any]:
        flat = [v for g in loss[name].values() for v in g]
        return {"log_loss": sum(flat) / len(flat), "n": len(flat)}

    diff = {g: [a - b for a, b in zip(loss["evaluator"][g], loss["situation_prior"][g])] for g in loss["evaluator"]}
    diff_hist = {g: [a - b for a, b in zip(loss["evaluator"][g], loss["evaluator_no_game_history"][g])]
                 for g in loss["evaluator"]}
    latencies.sort()
    report = {
        "games": len(games), "snaps": sum(len(g) for g in games), "labeled_family": len(rows_out),
        "book": {"formations": len(book), "plays": len(candidates),
                 "model_scored": sum(1 for c in candidates if c.get("family"))},
        "threat": {name: summary(name) for name in loss},
        "evaluator_vs_situation_prior": _boot(diff),
        "game_history_effect": _boot(diff_hist),
        "top1": sum(r["top1"] for r in rows_out) / len(rows_out),
        "top1_by_game_history": {
            b: (lambda s: {"n": len(s), "top1": sum(r["top1"] for r in s) / len(s) if s else None})(
                [r for r in rows_out if (r["game_plays"] < 5) == (b == "0-4")]) for b in ("0-4", "5+")},
        "legal_share_mean": sum(legal_counts) / len(legal_counts),
        "latency_ms": {"p50": latencies[len(latencies) // 2], "p95": latencies[int(0.95 * len(latencies))],
                       "max": latencies[-1]},
        "examples": examples,
        "not_evaluated": "call-versus-outcome: no human defensive snap has a confirmed execution",
    }
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    (out / "defense_strategy_heldout.json").write_text(json.dumps(report, indent=1, default=str))
    brief = {k: v for k, v in report.items() if k != "examples"}
    print(json.dumps(brief, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
