# Lions Ravens two-formation defense — scheme dataset v1

**Status: DATA ONLY. NOT live-enabled; NOT trained on defensive execution results.**

Authoritative dataset: [`datasets/madden27/lions_ravens_two_formation_defense_v1.json`](../../datasets/madden27/lions_ravens_two_formation_defense_v1.json).

This supersedes the earlier *three-formation* draft example in `civil_match_nickel_combined.json` **for the user's intended scheme**. That older file is retained for reference only. Do **not** use its Dime 2-3 match calls when implementing the final scheme.

## Installed setup: Baltimore Ravens defense

| Formation | Purpose | Calls | Required macro |
|---|---|---|---|
| **Nickel 2-4 Wide** | Default MATCH | Cover 4 Quarters, Cover 4 Palms, Cover 9 Show 2*, Cover 3 Match, Cover 2 Match, Cover 6 | No macro normally; optional Zone It on verified Quarters/Palms |
| **Nickel 2-4 Wide** | Civil Nickel STOCK | Cover 3 Cloud, Cover 2 Match, Tampa 2 Contain, Cover 9 Show 2 | **24 Base — always required** |
| **Nickel 2-4 Wide** | Stock spot-drop fallback | Cover 4 Zone | None |
| **Dime 2-3 Odd** | **Blitz only** | DB Fire 2 | **DB Fire — always required** |

*Owner t358u allowed Match-package Cover 9 Show 2 into the live book. The Guide still says generic `Cover 9`; confirm its actual matching/check behavior in game before treating a transferred match-guide USER assignment as verified.*

Names are grounded in public Ravens Madden 27 playbook listings, but are **not** a hash of the user's installed Xbox playbook. No other Dime formation or Nickel subformation may enter this restricted scheme.

## Required scoring and display contract

- Every candidate has unique `call_id`, even when Madden's base play name is identical (e.g. **Cover 2 Match** appears in two packages). The classifier cannot collapse the same base play from different packages because macro semantics differ.
- The **existing defensive ML scorer** chooses the call using down/distance, opponent formation/concept distribution, earlier closed snaps, risk, red-zone context and verified past outcomes. No arbitrary forced play sequencing.
- In **nickel_stock**, the selected call is invalid unless `24_BASE` is attached and the owner has confirmed that macro. In **dime_blitz**, the only allowed call is DB Fire 2 with `DB_FIRE` attached and confirmed. **Never silently omit a required macro.**
- **Dime is not a general passing-down alternative.** Only use it when ML explicitly selects a blitz, and not just because down and distance is long.
- In the red zone, `nickel_tampa2` remains a scored candidate with **24 Base**, not a mandatory rule; it is not permission to add Goal Line formations.
- **Zone It** is an OPTIONAL, separately scored macro for the verified Quarters/Palms match calls. It disables match behavior and plays assigned zones. It is not equivalent to *stock Cover 4 Zone* for every deep shell and must not inherit a now-invalid match USER responsibility. Stock Cover 4 Zone is a distinct call.
- The guide's original 18 matchup-specific user-role annotations are stored in `user_assignment_rules`. Display only if exact call, opponent offensive alignment, activated check and installed formation are verified. A guide-specific USER is not automatically the correct USER for a stock Nickel call that omits the guide's individual route/manning edits.
- The 10 more elaborate Nickel guide recipes remain `nickel_guide_examples_reference_only`, **not candidate variations**. The user requested stock calls and named macros only. Do not apply the guide's man-ups, narrow safeties, custom zones, spy, etc. without separate user instruction.
- If the correct USER cannot be verified, display `USER: UNVERIFIED` for rehearsal and mark the call **NOT LIVE READY**. Do not guess an LB/safety.
- Keep **one global coaching-adjustments profile** for the game. The original Match and Nickel profiles differ; user-confirmed compatibility of stock calls and both macros is required. Switching formations does not automatically change those global settings.
- Log **exact call ID, base play, formation, required macro, optional Zone It, verified actual USER and executed-versus-recommended status**. Do not train on unexecuted recommended actions.
- No new outcome-trained defensive model is in this dataset. Existing offense must stay untouched.

## Prior to game

1. Set the **Ravens defensive playbook** and identify/confirm **Nickel 2-4 Wide** and **Dime 2-3 Odd**, along with each exact base play.
2. Confirm the **24 Base** and **DB Fire** named macros physically exist, are configured, and are executed successfully on console.
3. Confirm the persistent coaching adjustment settings suit both packages, or disable conflicting candidates.
4. Verify offensive-alignment-specific Match guide USER roles **in Nickel 2-4 Wide**, plus user controls in pure stock Nickel and Dime blitz calls. A prior Dime USER guide does not prove its match checks/roles map to Nickel.
5. Compare **stock Cover 4 Zone** versus optional **Zone It** on Quarters/Palms. Note that Zone It may change the correct USER responsibility.
6. Use the current defensive scorer in **shadow mode**, then carry out verified CPU reps before treating this as a trained/validated defense.

## Tests

Run `pytest -q tests/test_lions_ravens_two_formation_defense.py` against the relevant GitHub branch. These are static catalog contract tests, not live model or console validation.

## Rights / provenance

This public repo contains only **short, structured, user-supplied factual play-name and role metadata**; no copied Civil.gg lesson text, diagrams, raw screenshots or videos. The full paid lessons remain with the provider. Confirm applicable terms before redistributing or using paid material for external training.
