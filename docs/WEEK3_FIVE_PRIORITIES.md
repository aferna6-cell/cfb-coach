# Week 3 / all-experimental-games — five improvement priorities (DRAFT)

Base: `madden-ml/coordinators` at `8d10b0d`. **Do not merge or use in a competitive game before QA.**
This draft is isolated; `main`, the active branch, the installed offense artifact, and the franchise database are untouched by these code changes.

## 1. Third-and-short and goal line

- 3rd/4th and 1–3: runs, quick passes, play-action, and RPO receive close situational conversion *proxies*. None is forced or quota-selected.
- Optional per-concept/per-family conversion success buckets can be fit on **future explicitly supervised data**. Old offense artifacts load with empty buckets and the prior behavior.
- `tests/madden_ml/test_short_yardage_compete.py`, `test_week3_priorities.py` cover competing families and backward compatibility.

## 2. Joint action ranking and overrides

The first ranking scores `(formation, play)` using ML probabilities, diversity and small situation effects. The **joint** ranking then evaluates `(formation, play, plan)`, where a plan is no adjustment, a sourced legal hot route/protection, a compatible pair, or an installed and armed custom macro. It adds bounded strategy, matchup, opponent and action effects. This is not a second supervised model and scores are **not** measured counterfactual outcomes.

Previously, it could choose a different play even when its best action was `NO_ADJUSTMENT`, with no explicit margin rationale. The draft now:
- Writes `joint_override_audit` with initial anchor, unrestricted winner, final winner, score difference, and score components. No unexecuted alternative receives a success label.
- Suppresses **only** a no-adjustment play override with a proxy-score advantage under 0.025. Sourced adjustments and substantial advantages remain eligible.
- Preserves full legal playbook scoring, latency checks and the existing fallback.

Gate before promotion: offline replay against the **same situations** from all seven substantial empirical/prior games; report override counts and winner changes. Do not claim causal yardage or win-rate improvement for unexecuted alternatives.

Read-only historical triage (the archived top-12 comparison is **not** a complete rerun):

```bash
python3 scripts/madden_ml/review_joint_history.py "$DB"
```

For the provided October 20 backup, 280 empirical decisions preserve both stages; 276 changed plays, 271 with no action. Only 41 play-score pairs were retained, and 39 of those would meet the new 0.025 proxy margin. The unretained pairs are unknown; these figures are an audit of old scores, not a replay or performance uplift estimate.


## 3. Verified execution with one tap

In the minimal Madden live page, **Ran call ✓** explicitly sets `used_recommended`. **Changed play** reveals the execution fields. Merely seeing a call does not count. Untouched submissions remain unknown. Tapping `Ran call ✓` in a no-action situation also confirms no adjustment; it does **not** confirm an optional hot route or macro automatically. Fields reset after submit. Completed snaps continue to be recorded by the existing idempotent server path.

## 4. Current pre-snap evidence

- A known `m:ss` entered in the game-clock field is converted to current game-clock seconds **before** choosing the play. Earlier code stamped it after calling the model.
- Separate the **game clock** used for two-minute/late-game strategy from the **play clock** used to decide if pre-snap adjustments can fit. This prevents confusing <10 seconds of game time with a short play clock.
- Repeated/untyped clock values remain unknown, with null/suspect provenance saved by PR #81.
- The existing `live` check box indicates an observed alignment. Unchecked last-play labels remain a soft prior, never proof of current coverage. No post-snap labels enter the pre-snap decision.

## 5. Shadow challenger; no automatic replacement

New commands adapted from the separately tested PR #77:

```bash
cd ~/cfb-coach
git fetch origin
git switch madden-ml/experimental-five-priorities-20261020
DB="${CFB_COACH_MADDEN_DB:-$HOME/.cfb-coach/madden27.db}"
python3 -m cfb_coach ml train-offense --db "$DB" --seed 7 \
  --out "$HOME/.cfb-coach/challenger/offense.json"
python3 -m cfb_coach ml eval-offense --db "$DB" --seed 7 \
  --out "$HOME/.cfb-coach/challenger/eval" \
  --artifact "$HOME/.cfb-coach/challenger/offense.json"
```

**Important:** The trainer excludes unverified or missing executions, and excludes older/missing VOD labeling and unsupported source data. Insufficient real verified rows must yield **NO-GO**, never silently treat the wins as supervision. Held-out comparisons report log-loss vs situation-only and prior-driven models with bootstrap uncertainty and 150ms latency. `auto_promoted` is always false. Challenger output is kept outside the repo and **is not** a drop-in replacement for the current `madden-ml.experimental.offense.52.empirical` artifact.

## Local QA commands

```bash
python3 -m pytest -q tests/madden_ml/test_week3_priorities.py \
  tests/madden_ml/test_short_yardage_compete.py \
  tests/madden_ml/test_offense_shadow.py \
  tests/madden_ml/test_cpu_audit_fixes.py \
  tests/madden_ml/test_cpu_audit_fixes_2.py \
  tests/madden_ml/test_wildcard_dual_ml.py
```

A draft-only GitHub Actions workflow also runs these checks (plus situated-action tests). The live default model remains unchanged until tests and offline comparison pass. **Do not retrain with `ml experimental --retrain`** to enable any of these changes. Use an explicit backup and verify the installed game book before any CPU playtest.

## Known constraints

- The existing logs contain many recommendations, but no verified executions for the last two CPU wins. The challenger may have too little usable evidence for a promotion gate.
- Short-yard conversion cells are observational, not proof one unexecuted alternative would have succeeded.
- Confirming a recommended play does not assert that a separate action was applied.
- Existing CPU mode is offense-only. The unfinished defense model is not activated by this PR.
