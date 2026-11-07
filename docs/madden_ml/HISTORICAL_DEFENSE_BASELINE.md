# Madden 27 — historical opponent coverage baseline

**Research only.** This report uses a read-only database backup. Nothing runs in the live coach.

## Data provenance

- Our offense-side snaps: **1097** (opponents were on defense).
- Named or pressure-only historical labels: **872**.
- Named coverages: **758**; pressure-only: **114**.
- Our verified executed offensive snaps: **655**.
- Named coverage + our verified execution: **440**.
- Opponent call labels are stored in `coverage_seen` and are **not independently screen-verified** by this DB.
- We **do not** interpret `pressure` as Cover 0.

## Chronological whole-game coverage prediction test

Evaluated 630 historical named labels across 22 held-out sessions, with first 5 sessions used for training.
The target is agreement with stored labels, **not proven Madden post-play ground truth**.

| Baseline | Brier ↓ | Log loss ↓ | Top-1 ↑ |
|---|---:|---:|---:|
| uniform | 0.95 | 2.99573 | 0.0 |
| global | 0.87761 | 2.19954 | 0.23492 |
| contextual | 0.87975 | 2.20796 | 0.23175 |

### Stricter slice: our executed play verified

440 labels evaluated in 15 held-out sessions; opponent coverage source still unverified.

| Baseline | Brier ↓ | Log loss ↓ | Top-1 ↑ |
|---|---:|---:|---:|
| uniform | 0.95 | 2.99573 | 0.0 |
| global | 0.89862 | 2.27616 | 0.19318 |
| contextual | 0.89701 | 2.27078 | 0.18864 |

## Matchup associations

440 verified executed offensive snaps support at least one named coverage + evaluable outcome.
Found **5** named play/coverage pairs and **34** formation/coverage pairs with at least 5 examples.
This is observational, using standard 40/60/100 success; it is not a randomized counter-play experiment.
Small cells should not be used as evidence of a guaranteed play-beats-coverage relation.

## Next pilot requirements

1. Capture Madden's displayed **post-play opponent defensive call** with screenshot/frame locator and confidence. Store it separate from pre-snap predictions.
2. Replay the scorer before the snap using only prior games, plus current pre-snap alignment observations, with the revealed call hidden.
3. Evaluate the structural offense model (with the 91 route-art feature records) on held-out outcomes; do not train/evaluate on the same game.
4. For QB read advice, require player-identified routes; visual geometry alone cannot truthfully label 'post first, drag second' for a specific play.
5. Before CPU pilot, confirm no regression of 150ms latency, legal play selection, and no unapproved macro/action.