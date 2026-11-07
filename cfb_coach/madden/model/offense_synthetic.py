"""SYNTHETIC game generator for pipeline self-checks. NEVER evidence about real Madden games.

Plants a known coverage x route-shape effect so tests can verify the training pipeline can LEARN a
coverage-conditional effect from time-correct data, including one that CONTRADICTS football convention.
Each game has a persistent opponent coverage persona, so earlier coverage entries are genuinely informative
about later snaps (as with a real CPU opponent), while the coverage on the current snap is only known afterwards.
"""
from __future__ import annotations

import random
from typing import Any, Mapping

from cfb_coach.madden.model import dataset as dataset_mod
from cfb_coach.madden.model import offense_art_features as af
from cfb_coach.madden.model import offense_assignments as oa

PERSONAS = {
    "man": {"Cover 2 Man": 0.55, "Cover 1 Hole": 0.25, "Cover 1 Robber": 0.10, "Cover 3": 0.10},
    "zone": {"Cover 3": 0.45, "Cover 3 Match": 0.20, "Cover 2": 0.20, "Cover 4 Quarters": 0.15},
    "shell": {"Cover 4 Quarters": 0.40, "Cover 4 Palms": 0.30, "Cover 6": 0.15, "Cover 4 Drop": 0.15},
}
MAN = {"Cover 2 Man", "Cover 1 Hole", "Cover 1 Robber"}


def planted_success_prob(formation: str, play: str, coverage: str, *, truth: str, strength: float) -> float:
    """P(success) = 0.5 + shape x coverage effect on the shallow-route share (a feature with real spread across plays).

    'expected': shallow-heavy plays beat man coverage (football convention); 'contradict': they LOSE to it.
    """
    shallow = af.art_features(formation, play)["art_shallow_share"]
    sign = 1.0 if truth == "expected" else -1.0
    p = 0.5
    if coverage in MAN:
        p += sign * strength * (shallow - 0.62)
    else:
        p -= sign * strength * (shallow - 0.62) * 0.5
    return min(0.9, max(0.1, p))


def make_rows(*, n_games: int = 24, snaps: int = 45, seed: int = 7, truth: str = "expected",
              strength: float = 0.8, opponent: str = "cpu") -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    book = [(p["formation"], p["play"]) for p in oa.load_database()["plays"]
            if af.art_features(p["formation"], p["play"])["art_present"]]
    rows: list[dict[str, Any]] = []
    for g in range(n_games):
        persona = list(PERSONAS)[g % len(PERSONAS)]
        dist = PERSONAS[persona]
        labels, weights = list(dist), list(dist.values())
        for i in range(snaps):
            f, pl = rnd.choice(book)
            cov = rnd.choices(labels, weights)[0]
            down, distance = rnd.choice([1, 2, 3]), rnd.choice([3, 6, 10])
            ok = rnd.random() < planted_success_prob(f, pl, cov, truth=truth, strength=strength)
            raw = {"session_id": f"g{g:02d}", "ml_snap_id": f"g{g:02d}-{i:04d}", "opponent_id": opponent,
                   "opponent_type": "cpu", "side": "offense", "down": down, "distance": distance,
                   "yardline": rnd.randint(8, 92), "quarter": 1 + (i * 4) // snaps,
                   "result": f"+{distance + 3}" if ok else "incomplete", "formation": f, "play": pl,
                   "executed_status": "identified", "executed_verification": "verified",
                   "executed_formation": f, "executed_play": pl, "coverage_seen": cov}
            rows.append(dataset_mod._normalize_row(raw, provenance="synthetic_planted_signal"))
    return rows


def game_order(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    order: dict[str, int] = {}
    for r in rows:
        order.setdefault(str(r["game_id"]), len(order))
    return order
