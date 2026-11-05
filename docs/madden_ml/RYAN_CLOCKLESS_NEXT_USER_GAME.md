# Nov 4 Ryan postgame audit and zero-extra-input requirements

This draft is stacked on PR #89, not on `main`. Do not merge without user approval and laptop tests. No user database or trained model artifact is committed.

## Source and attribution

Read-only audit of the Nov 4 `madden27.db` backup, session `698831a80afb434b`:
- Ryan played the **Bengals**. Human game, primary franchise, **loss 28–20**.
- Owner says the **offensive caller was heuristic**. No offensive snap has a linked ML decision ID. 34 offensive snaps, 33 marked identified/verified through self-attestation, 2 interceptions, 3 recorded offensive TDs.
- Defensive calls came from the **unverified logistic defense v2** (`logged_unverified_v1`, `madden.defense.logistic.v2`). 37 defensive snaps with linked decisions; all 37 defensive executed statuses are unknown/unverified. Do not treat them as verified supervised defensive labels.
- 30/37 defensive calls used `Dime 3-2 Odd`; 16 `Cover 3 Hard Flat`, 12 `Cover 3 Cloud`. Opponent concepts reported: eight Stretch, eight Inside Zone, five Mesh, two Four Verticals, two RPO bubble (25 named defensive snaps).
- No causal benefit is claimed for hypothetical alternative calls. PR #89's rolling limits, run-heavy guard, and opponent tendency weighting are **not yet proven in a live replay**. The local trained `defense_v1.json` is not included in Git; confirm it on the user's machine.

## Approved interaction contract (supersedes clock/showing suggestions)

1. **Clock is intentionally unused**. Missing game-clock fields are not a user-input defect. If clock is absent, never infer it or apply a clock-specific adjustment just from quarter/score.
2. **Never require a pre-snap `showing` field**. The owner enters ordinary situation/outcome and focuses on playing the game.
3. The coordinator may choose any *legal, in-book, verified, compatible* play and action. Never invent controller button sequences, macro settings, or pretend an action was installed/armed.
4. Without a current look, choose a base formation+play as usual. Offer at most one short, **explicitly conditional** note, e.g. `IF Cover 2: LB → C2; OTHERWISE run the base play`. This can be prioritized by historical observations, but must **not** describe the historical coverage as the current defense.
5. Conditional notes are **not active adjustments**, **not automatic executions**, **not source of verified training labels**. The existing recommendation and execution tracking remain unchanged. No new confirmation tap is required.
6. Keep the compact live formation/play prominent, with the optional IF note on a distinct second line; an ineligible/unverified action yields no hint.
7. Preserve defense-model fallback to the established selector. Do not switch experimental defense on by default. Require explicit human opt-in and a local JSON artifact; the prior offline defense model failed to beat the situation-only baseline.

## What this branch implements

- A bounded, no-input `IF look` advisory helper for **offensive Custom Adjustments already armed, confirmed, and paired with the final selected formation/play**.
- It ranks *conditional* hints by prior logged coverage frequency for that opponent, never assuming the current look.
- Live browser/terminal show the note separately; it does not modify `call.macro`/`call.adjustment`, recorded `pending_offense_action`, or outcome attestation.
- Synthetic unit tests for no-clock/no-showing usage, no false execution, history priority, unsupported macros, and existing active adjustments.

## Still required before next competitive human game

- Run `python3 -m pytest tests/madden_ml/test_conditional_look_actions.py -q` and existing relevant offense/live/defense gates on the user's checkout. GitHub Actions does not currently show a fresh run for this stacked branch.
- Inspect artifact `~/.cfb-coach/models/defense_v1.json`; confirm schema v2, installed defense-book coverage, legal candidate diversity and actual file presence.
- Replay a set of Ryan's 37 defensive situations **on the user's machine against the installed artifact**. Compare the original concentration to new calls; check short-yardage guard, run-heavy response, diversity, in-book legality, latency, fallback reasons, and no invented adjustments. This is an offline policy replay, not an outcome counterfactual.

  Read-only replay command on this stacked branch:

  ```bash
  python3 scripts/replay_madden_defense_session.py \
    --db "$HOME/.cfb-coach/madden27.db" \
    --artifact "$HOME/.cfb-coach/models/defense_v1.json" \
    --session 698831a80afb434b
  ```

  The replay deliberately uses only concepts observed **before** each simulated decision. It does not use full-game Ryan opponent priors, which would leak future events, and does not report hypothetical wins or stop rates.
- Decide separately whether defensive conditional controls can be safely offered without an observed live concept. Keep any such note optional, legal, and non-executed.
- PR #89 reports 1,077 tests passing, three skipped, and five failures reported as preexisting on the base. Reproduce/triage the five, not treat the report as a green release gate.
