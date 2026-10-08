# Milestone C — offensive ML control (Sprint 6A)

**Status:** experimental; not a competitively validated model.
This branch is stacked on the Sprint 4.2 execution-default/recovery branch.
It is not merged into main or the experimental-pilot base.

## Live browser: formation + play only

The large coaching line displays **only** the final selected formation and play:

`Gun Doubles Clamp Stack — Texas Y-Stutter Wheel`

The interface no longer puts reads, ML probabilities, heuristic comparisons,
macro instructions, or rationales into the main coaching display. Those stay in
controller state, SQLite decision logs, and postgame tools.

The outcome-entry controls, next situation, **Used recommended** default,
**Used different** correction, and undo remain accessible. Settings, playbook
confirmation, game log, and the end-game button are under the collapsed
**Game settings, log and finish game** control, so no information is lost.

## Controlled offensive pilot: CPU

CPU is offense-only, and its existing experimental ML behavior is unchanged.

```bash
python -m cfb_coach ml experimental --retrain
python -m cfb_coach play --game madden27 -o cpu
```

**Rollback:** `python -m cfb_coach ml heuristic` restores heuristic
play-calling for the following decision.

## Controlled offensive pilot: humans (explicit opt-in)

Previously, global `mode=experimental` could select offensive ML calls for
human opponents too. Now the human-opponent override is **off by default** even
if CPU experimental mode is enabled.

To experiment during one human-opponent live session, opt in explicitly:

```bash
python -m cfb_coach ml experimental --retrain
python -m cfb_coach play --game madden27 -o gavin --ml-human-control
```

The flag is **per invocation and not persistent**. Without it, the human-game
offense stays heuristic. The defense is unaffected by this flag; this branch
does **not** activate defensive ML. The current experimental offense has
in-book and situational-candidate guards and a 150 ms fallback; it is still an
early low-data model.

Human-pilot fail-safes:
- If experimental mode is not active, `play --ml-human-control` refuses to start
  as an ML pilot and explains how to enable the mode.
- If there is a timeout, model error, or illegal candidate, the call falls
  back to the existing heuristic.
- An explicit `python -m cfb_coach ml heuristic` switches subsequent
  calls back immediately; `Ctrl+C` stops the current play session.
- ML explanations and observed outcomes remain in the SQLite database for
  postgame analysis, not on the in-game display.
- Selecting **Used recommended** is the default for the next snap. Choose
  **Used different** and enter the actual play whenever you diverge.

Do not claim that one successful CPU game proves human-game effectiveness.
The single 35–10 CPU game included 51 ML calls but just one model/heuristic
disagreement. Sprint 4.2 enables evidence-backed recovery of that game's
outcome-linked plays; run the recovery preview and apply only after checking
the real laptop DB.

## Milestone C next increments

**Sprint 6B:** Add a separate offline/off-policy evaluation report, per-game
regret proxies *without* attributing counterfactual results, holdout splits
by game and opponent, calibration checks, CPU-versus-human separation,
minimum supervised evidence threshold for *promotion*, and rollback tests.

**Sprint 6C:** Add a canary / graduated control policy that keeps high
uncertainty and unsupported contexts on the heuristic. Compare a held-out
baseline and run multiple CPU games before claiming any advantage.

**Sprint 6D:** Human-game observational period, explicit controlled pilots,
verified execution capture, and opponent-specific adaptation from only
legitimately observed pre-snap tendencies. Defense remains separate.

Do not merge the stacked PRs or promote a model automatically without
explicit user approval. Keep rollout and rollback obvious.

## Sprint 6B: model-primary offensive calls and verified optional actions

With `ml experimental` enabled, **every eligible offensive snap** uses the
trained model's top-ranked in-book play, not the heuristic's near-tie preference.
Football legality constraints still filter the candidate list. The heuristic
only handles missing/invalid/timeout/error cases and explicit manual rollback.
The model is not yet demonstrated to be superior to the heuristic.

`python -m cfb_coach ml experimental --retrain` enables and retrains this
behavior. `play --game madden27 -o cpu` remains offense-only. Human offensive
games require the additional explicit `--ml-human-control` session flag.

The same model-selected play now flows through an experimental action policy:

- The optional action menu ranks already-armed, explicitly configured offense
  Custom Adjustments and researched hot routes/protection actions.
- Only model-selected formation/play matches are eligible. Coverage-dependent
  actions need an explicitly **live** pre-snap look or repeated observed tell.
  An isolated previous-snap look does not trigger them.
- The chosen optional action is accessible in a **collapsed** drawer beneath
  the large `Formation — Play` call, not printed alongside it.
- If you actually use the extra hot route or macro, check
  **I actually applied the optional hot route / Custom Adjustment** when logging
  the snap. The checkbox starts unchecked each snap.
- `Used recommended` still attests to the executed **play**, but it does
  **not** automatically attest to executing a separate action.
- `ml postgame-experimental` reports optional actions suggested and separately
  verified as applied. Model decision JSON also stores the action ranking and
  selection policy for later offline evaluation.
- Play-changing audibles are intentionally withheld until a separate executed
  play can be logged reliably after the audible.

**Honest limitation:** `offense_action_policy.research_prior.v1` is an
eligibility/score policy based on researched actions and play-model uncertainty.
It is **not** yet trained on the causal lift of route or macro changes. Future
work will use verified executed actions and outcomes for observational action
modeling, evaluate under opponent/game holdouts, and avoid counterfactual claims.

Keep the experimental model active only when you are deliberately testing it.
If calls become unreasonable, from another terminal use:
`python -m cfb_coach ml heuristic` (applies to subsequent calls).

### Development / rollout order

1. Sprint 4.2 recovery remains a separate explicit dry-run and backed-up
   apply on the laptop. No branch automatically mutates old Franchise games.
2. Sprint 6A compact Madden UI and human-game opt-in (PR #26).
3. Sprint 6B model-primary offense and optional action policy (stacked PR #27).
4. Future sprint: collect independently verified adjustment executions and
   outcome labels; train and evaluate action effects without inventing results;
   only then widen the action policy.
