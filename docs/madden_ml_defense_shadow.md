# Stage 3 — Defensive Shadow Advisor (Sprint 5)

Opponent-aware defensive AI coordinator foundation. **Shadow only** — never
controls the live defensive call. Experimental offensive ML is unchanged.

## What it does

On human-opponent defensive snaps:

1. Builds an opponent offense tendency model (concepts, runs, RPO, scramble)
   with evidence counts and uncertainty (one snap ≠ a tendency).
2. Ranks in-book defensive plays for the current situation.
3. Recommends at most one **armed and valid** Custom Adjustment when essential,
   after checking researched editor settings and pairwise conflicts.
4. Attaches the recommendation as `ml_defense_shadow` next to the heuristic call.
5. Logs after snap seal (`mode=shadow`, `final_pick` = heuristic).

CPU opponents remain offense-only. No defense shadow is produced for CPU.

## Commands

```bash
python -m cfb_coach ml defense-shadow-status
python -m cfb_coach ml defense-shadow-demo
python -m cfb_coach ml defense-shadow-report
python -m cfb_coach ml defense-shadow-report --game-id <id>
# Always refuses until readiness criteria below are met:
python -m cfb_coach ml defense-shadow-activate
```

## Live-call convention

Shadow format matches the fast pad:

`Formation — Play | Adj (only if essential) | User/key`

The **shown** live line remains the heuristic recommendation.

## Evidence layers (kept separate)

| Layer | Role |
| --- | --- |
| Researched football priors | Soft family boosts from Madden research |
| Observed opponent tendencies | Shrinkage counts; uncertain until repeated |
| Verified executed defensive actions | Future supervised signal (not yet) |
| Verified outcomes | Linked only to the shown (heuristic) call |
| Uncertain historical recommendations | Discounted; never treated as ground truth |

Do not claim causal improvement from an adjustment unless usage **and** outcomes
are verified for that adjustment.

## Custom Adjustment rules

- Only ids in the current eight-macro defensive loadout may be recommended.
- Every required Madden defensive editor field must have an explicit researched
  value or Default (`editor_fields.json` inventory). Pool membership alone is
  not enough.
- The adjustment must be compatible with the **selected** formation/play
  (researched base, shell, coverage family). Incompatible → recommend the play
  without an adjustment.
- Conflicting exclusive settings between two macros → do not stack.
- Experimental / benched macros (e.g. `HEAT`) are **ineligible** unless
  `CFB_COACH_ALLOW_EXPERIMENTAL_D_MACROS` is explicitly authorized.
- Never invent settings or activate an unarmed macro.

## Latency budget

The full shadow path (opponent model + ranking + CA validation) must finish
within **150 ms**. On timeout or exception the live heuristic call is preserved
and the fallback reason is recorded (`timeout during …` / `exception: …`).

## Observation readiness (shadow advisor for human opponents)

The shadow advisor is ready for **human-opponent observation** (log-only) when:

- Sprint 5.1 acceptance tests are green
- Suggested plays are always in the applied D book
- Suggested adjustments are armed, complete, and play-compatible (or withheld)
- Experimental macros never auto-recommend
- Verified-execution reporting distinguishes linked / verified / unknown
- 150 ms budget fallbacks preserve the heuristic
- Experimental offense + CPU offense-only remain unchanged

## Stage 3 activation readiness (NOT enabled)

Live control stays off (`ml_defense_shadow_control=off`). Before allowing
defensive ML to call plays in a future game, all of the following must hold:

1. **Data quality**
   - ≥ N verified executed defense snaps with outcomes (suggested N ≥ 40 across ≥ 3 games)
   - Historical unverified Franchise D snaps remain discounted priors only
   - Laptop `audit-history` confirms which games contribute
2. **Evaluation**
   - `defense-shadow-report` shows stable agree/disagree patterns
   - `verified_executions_linked` uses identified+verified only
   - Shadow never credited for unexecuted alternatives
   - Situation suites green: crossers, flood, RPO/run, scramble, short yardage,
     3rd-and-long, red zone, late game
3. **Safeguards**
   - 150 ms full-path budget with heuristic fallback
   - Armed-macro + complete editor settings + play↔adj compatibility gate
   - CPU offense-only unchanged
   - Experimental offensive mode unaffected
   - Explicit opt-in CLI (not this sprint’s refuse stub)
   - Undo / relog / restart attribution tested
4. **User confirmation**
   - Readiness checklist reviewed
   - Separate authorization to enable control (do not merge casually)

Until then: `python -m cfb_coach ml defense-shadow-activate` **refuses**.

## Isolation from experimental offense

- Defense shadow attaches only when `call.side` starts with `d`.
- Experimental ML (`maybe_apply_experimental`) remains offense-only.
- Sealed commit paths are separate (`commit_experimental_decision` vs
  `commit_defense_shadow`).
- Default heuristic offense path is unchanged.
