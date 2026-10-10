# Sprint 16 — Defensive coordinator pilot and autonomous macro drafting

Branch: \`cursor/sprint16-defensive-coordinator-5f33\`
Base: Sprint 15A elite VOD validation, NOT \`main\`.

## Exact target architecture (defensive twin of the offensive coordinator)

1. **Pregame designer:** evaluate every available catalogued defensive formation and keep its full source-book play list; at most 15, using tactical coverage families and situational probes, roster and opponent evidence when available. Never claim an impossible cross-book formation union is installed.
2. **Context-aware call:** on human-vs-human possessions, rank **all actual installed defensive calls** using situation, score/clock/field zone, opponent concept tendencies when evidenced, empirical defensive stop outcomes, and coverage-family balance. Use a fast, defensive-only model and 150 ms fallback; do not reuse offensive success estimators blindly.
3. **Defensive game intelligence:** defensive personnel/front, run/pass fit, QB mobility, bunch/stack, motion, explosives, red-zone, two-minute and late-game situation. Pre-snap observations must be separated from post-snap facts, inferred tendencies and unknowns.
4. **Joint decision:** one formation + one coverage/pressure call + zero or one validated in-game adjustment package/macro. Compare doing nothing against modifications. Never contradict the selected installed play or recommend settings without verified buttons.
5. **Autonomous *design* of new defensive macros:** research-grounded synthesis from separately sourced editor settings, complete fields (explicit researched values or Default), conflict detection, versioned blueprints, test, approve and rollback. This is distinct from physically creating or enabling a macro in Madden; the Xbox game does not have an automation interface.
6. **Learning:** defensive verified stops, pressure outcomes, explosive concessions and matchup context; separate human opponents and sessions; improve both play policy and adjustment policy. No recommendations counted as executed or post-snap outcomes leaked into live.
7. **Evaluation:** full-playbook coverage, diversity, third-and-long, run defense, pressure versus mobile QB, red zone, busted coverage, play-to-adjustment legality, 150 ms p95, game-based holdouts; shadow-vs-active comparisons and honest outcome attribution.
8. **Defense video and expert knowledge:** independently reviewed elite defensive VODs, source rights checks and provenance, separate from offensive policy, admitted only after validation.

## Implemented in this first pilot

- New \`defense_coordinator.py\` scores every play of the **confirmed installed defensive book** with conservative contextual family prior and verified stop-rate estimates, anti-repeat and 150 ms fallback. Unknown names remain considered with low prior.
- New defensive formation portfolio selector (up to 15) from the entire catalog; staged/confirmed **manual Madden installation** without editing the active defensive book during preview.
- New \`defense_macro_lab.py\` composes genuinely new named defensive macro **drafts** from two distinct researched defensive macro recipes, resolves conflicts in the editor setting list and includes every setting/Default. Only source-backed values permitted; physical creation, base-play testing, explicit approval and a defensive slot remain required.
- Separate defensive model artifact and opt-in flag. CPU offense-only and offensive model mode are unchanged.
- Verified created defensive macros can be suggested by the live defensive caller only after explicit installation/verification, for a compatible formation/play and a *currently observed* offensive concept. No guessed pre-snap concept becomes a confirmed live trigger.
- New pytest regression tests; GitHub Actions runs on the branch.

**Limitations:** This is an opt-in, early experimental policy, not a validated competitive-defense model. No Xbox execution, no autonomous video understanding, no evidence of a learned personalized defensive result until your verified logs are present. The current first release's baseline defensive decision combines family priors and shrinkage stop rates; it does not yet model individual defender assignments or all route-match patterns. New synthesized macros are sourced *combinations*, not entirely unprecedented Madden control features. No hidden controller action is invented.

## Ubuntu: first inspect without changing the game

\`\`\`bash
cd ~/cfb-coach
git fetch origin
git switch cursor/sprint16-defensive-coordinator-5f33
git pull --ff-only
python3 -m cfb_coach ml backup-db
python3 -m cfb_coach ml defense-design --max-formations 15
python3 -m cfb_coach ml defense-macro-lab -o gavin
python3 -m cfb_coach ml defense-experimental
\`\`\`

**Do not activate the experimental selector until you've verified the baseline offensive CPU test**. Defensive live calls only occur in a human-opponent game: CPU remains offense-only by design.

### To test the currently installed defense with a human opponent

\`\`\`bash
python3 -m cfb_coach ml defense-experimental --retrain
python3 -m cfb_coach ml defense-experimental --enable
python3 -m cfb_coach play --game madden27 --opponent gavin --franchise lab
\`\`\`

If your opponent ID is not \`gavin\`, use the actual registered human opponent. Enter the current situation accurately, including live offensive alignment/indicators when genuinely observed. You operate the controller.

To immediately restore the established defensive caller:

\`\`\`bash
python3 -m cfb_coach ml defense-experimental --disable
\`\`\`

### For a new *physically installed* model-ranked defensive formation portfolio

\`\`\`bash
python3 -m cfb_coach ml defense-design --stage --max-formations 15
python3 -m cfb_coach ml defense-design --show
\`\`\`

Use the proposal to build the exact formations **and all their listed plays** in Madden. Only after verifying in the game:

\`\`\`bash
python3 -m cfb_coach ml defense-design \
  --confirm-installed YOUR_PROPOSAL_ID \
  --attest "I installed and tested every listed defensive formation and its complete plays inside Madden."
\`\`\`

### For a new model-composed defensive Custom Adjustment

\`\`\`bash
python3 -m cfb_coach ml defense-macro-lab -o gavin --stage
python3 -m cfb_coach ml defense-macro-lab -o gavin --show
\`\`\`

Make the exact editor settings in Madden **Defense > Custom Adjustments**, test on the listed legal base play, and arm an available defense slot. Only then approve (if full, explicitly retire an existing slot after changing it in Madden):

\`\`\`bash
python3 -m cfb_coach ml defense-macro-lab -o gavin \
  --verify ML-D-XXXXXXX \
  --attest "I created all settings in the Madden editor, tested the paired defensive play and armed the macro in an available defense slot."
\`\`\`

If revoked or unavailable in the game:

\`\`\`bash
python3 -m cfb_coach ml defense-macro-lab -o gavin --unverify ML-D-XXXXXXX
\`\`\`

No generated macro is automatically set as an armed Custom Adjustment, and drafting never edits your actual Madden controller/game.

## Next validation requirements before competitive use

- Real chronological human user-game defense logs with **verified executions**, not inferred recommendations.
- Full-playbook play counts, coverage families, 3rd-down stops, explosive passes, sack/pressure outcomes, 1st-down conversion allowed, goal-to-go stops.
- Action ablation: no adjustment vs one, verified macro versus baseline; no attribution without controlled evidence.
- Verify score/state in live input and that the observed-vs-inferred defensive data policy isn't accidentally conflated.
- Split training and evaluation by game/opponent. Avoid source/context leakage and proof-of-win claims from synthetic fixtures.
- Promote macro effectiveness only after repeated independent verified observations.

**Development posture:** separate branch/draft PR; not merged to main. You must deliberately opt in. Both new systems preserve the existing defensive caller as fallback.
