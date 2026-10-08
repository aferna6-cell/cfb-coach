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
