# Defensive V2 — implementation, commands, and honest release gate

Updated 2026-11-09. Development PR #142 is stacked on foundation PR #139 and previous CPU2 PR #138; **DRAFT**, no main merge, no live data or model promotion. This is an opt-in test-franchise implementation, **not a certified game-ready defensive model**.

## Delivered in the current branch

- **Actual live routing** from Madden call site, not shadow-only: --defense-model v2, isolated DB and franchise, confirmed installed playbook hash, 150ms decision deadline, explicit EXPERIMENTAL model attribution and fallback to existing call.
- **Learned 15-formation defensive pregame proposals:** existing portfolio architect with V2 supervised outcome risk substituted for canned counter-table value. No fixed formation rotation; source-book provenance, full stock-play proposal and fingerprints. Does not install.
- **Scheme understanding:** source-backed and confidence-labeled coverage, shell, variants and pressure. V2 now learns cross-play associations from defensible structured feature axes and verified executions. The base catalog has not verified all eleven assignments. The separate optional private owner-verified 11-player ledger is supported, and otherwise all defender duties remain UNKNOWN.
- **Opponent beliefs:** strictly earlier, closed, correctly session-sequenced opponent concepts. Missing sequences never sneak into live tendency updates.
- **Outcome learning:** verified own defensive executions against CPU or human. Hierarchical expected offense-success, yards, TD, explosive, turnover risk plus conservative matchup shrinkage. The current approach is observational, NOT a demonstrated optimal defensive policy.
- **Individual adjustments:** verified Xbox controller actions are allowed only if physically confirmed by owner, legal/compatible, and there is sufficient exact verified execution data to score the changed package conservatively. Multiple compatible actions are allowed if supported; there are no invented macro presets. NO ADJUSTMENT wins by default until those conditions hold.
- **Offline postgame training:** writes a separate challenger and time-correct frozen-champion comparison; no automatic promotion.

## Test-franchise setup

All of these commands run from the repository checkout and require separate private files. Replace placeholders below.

The *test* DB must be a real separately configured Madden DB outside the default ~/.cfb-coach directory. **Do not put the test DB in the default primary game directory.** Verify the game_sessions.dynasty identity first. Using a separate DB file alone does not copy over all of the necessary installed-book configuration; your test DB must contain a confirmed applied defensive book.

~~~bash
# 1. Inspect the recorded franchise identifiers.
PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py inventory \
  --db /PATH/TEST_FRANCHISE/madden27.db

# 2. Audit verification and target coverage.
PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py audit \
  --db /PATH/TEST_FRANCHISE/madden27.db --franchise EXACT_DYNASTY

# 3. Train a research-only artifact; this is not a promotion.
PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pilot.py train \
  --db /PATH/TEST_FRANCHISE/madden27.db --franchise EXACT_DYNASTY \
  --out /tmp/defense_v2_test_champion.json

# 4. Model-proposed up-to-15 defensive formation book.
PYTHONPATH=. python3 scripts/madden_ml/defense_v2_pregame.py \
  --test-db /PATH/TEST_FRANCHISE/madden27.db \
  --franchise EXACT_DYNASTY --opponent CPU_OPPONENT_ID \
  --model /tmp/defense_v2_test_champion.json --max-formations 15 \
  --out /tmp/defense-v2-proposal.json
~~~

Physically create the proposed book in Madden. Confirm all formations/stock plays actually exist and the *applied* book recorded in CFB Coach matches the physical installation. The pregame output has TWO hashes: source-aware "fingerprint" and "runtime_installed_fingerprint". The live V2 caller requires the **runtime** one, calculated from the exact active defensive play inventory; the source-aware one is NOT interchangeable.

~~~bash
# 5. Read-only local config preflight.
PYTHONPATH=. python3 scripts/madden_ml/defense_v2_preflight.py \
  --test-db /PATH/TEST_FRANCHISE/madden27.db \
  --franchise EXACT_DYNASTY \
  --model /tmp/defense_v2_test_champion.json \
  --installed-fingerprint DEF-INST-EXAMPLE

# 6. Only once all local/playbook/loaded performance and model checks pass:
python3 -m cfb_coach play --game madden27 \
  --db /PATH/TEST_FRANCHISE/madden27.db \
  --opponent CPU_OPPONENT_ID --cpu-defense --no-macros \
  --defense-model v2 --defense-model-path /tmp/defense_v2_test_champion.json \
  --defense-test-db /PATH/TEST_FRANCHISE/madden27.db \
  --defense-franchise EXACT_DYNASTY \
  --defense-installed-fingerprint DEF-INST-EXAMPLE
~~~

**Do not copy the placeholder fingerprint.** It must match the actual applied book after installation and the owner's independent in-game verification. The preflight checks stored state and hash only; it cannot independently see the Xbox screen.

A new test franchise cannot initially have training data of its own. Until verified defensive executions exist, the V2 live route refuses the empty model; use the existing defensive caller while collecting **confirmed executed** plays and outcomes. Do not manufacture execution labels from recommendations.

## Individual Xbox controls (optional)

By default the model recommends NO ADJUSTMENT. The official Madden 27 control catalog identifies entry controls but does not establish that a particular Xbox submenu/target works on the owner's console. To allow an individual action, the owner physically verifies an exact sequence in their console and records the private file.

Example of the required *format only* (NOT a claim these buttons were physically tested):

~~~json
{
  "schema": "madden.defense.v2.xbox_confirmations.v1",
  "confirmed_actions": [
    {
      "action_id": "def.coverage_hub",
      "instance_id": "my_confirmed_press",
      "label": "Press coverage",
      "console": "Xbox",
      "physically_verified": true,
      "verified_on": "YYYY-MM-DD",
      "verified_buttons": "EXACT STEPS I ACTUALLY TESTED",
      "params": {"option": "Press"}
    }
  ]
}
~~~

Pass --defense-action-confirmations /PRIVATE_PATH/controls.json during the play command. This flag is optional; the model will still refuse all individual adjustments until the training DB has enough separately verified *executed-adjustment* outcomes for that exact combination and stock play/context. NO ADJUSTMENT competes each snap. Confirmed controller steps are never treated as evidence the move improves outcomes; empirical execution labels are evaluated separately. Arbitrary macro suggestions are not used by V2.

## First-party verified defensive assignments (optional)

A private JSON file may contain an EXACT formation/play key, source "owner_xbox_play_art", verified date, and exactly eleven unique slots, each with a physically verified role, responsibility and verified=true. Accepted schema: madden.defense.v2.verified_roles.v1. The live card then reports 11/11 owner-Xbox-verified assignments. Unknown or incomplete assignments remain 0/11 UNKNOWN and are never guessed. Pass --defense-verified-roles /PRIVATE_PATH/roles.json on the play command.

This metadata is useful for a grounded explanation. Its role descriptions are NOT yet automatically trainable per-defender transformed play-art; structural model features currently use source-backed shell/type/pressure/variant axes. Further first-party play-art annotations and validated player-level transforms are still needed.

## Postgame training

~~~bash
PYTHONPATH=. python3 scripts/madden_ml/defense_v2_postgame.py \
  --test-db /PATH/TEST_FRANCHISE/madden27.db \
  --franchise EXACT_DYNASTY \
  --champion /tmp/defense_v2_test_champion.json \
  --out-dir /tmp/new_defense_v2_challenger_unique
~~~

The candidate and comparison report are placed in a new folder. No model is promoted automatically. The challenger is evaluated on later whole games with only earlier games available for re-fitting and with game-clustered uncertainty. Predictive superiority does not imply causal improvement from recommending different calls.

## Tests and pending evidence

- GitHub Actions targeted contract covers CPU verified labels, execution-vs-observation distinctions, time-fenced opponent beliefs, direct live selection, installed hash, safe fallback, Xbox action gates, strict 11-defender roles, playbook ranking, postgame challenger and 150ms synthetic scorer checks.
- A 15-formation synthetic run previously scored 434 recognizable calls out of 452 indexed stock calls with a **95.469ms p95 rank-only time** over 20 runs on GitHub CI. This is NOT the full locally loaded UI/DB latency and cannot approve actual play.
- Run full tests/madden_ml; no failures may be ignored. Real owner database, model artifacts and console were not available to the GitHub coding environment. Real loaded performance, playbook confirmation, defender assignments and control validation remain outstanding.

**GAME READY for a new Defensive V2 test-franchise game: NO-GO pending the above evidence.** The existing caller remains usable for collecting defensive training snaps. The actual V2 live implementation exists on this draft branch, with no primary-franchise activation.

## Later: cross-side shared learning

Issue #140 tracks possible reuse of useful opponent defensive structures observed when our offense faces CPU/human teams. Those observations have different provenance from *our verified defensive executions*. They can inform shrunk feature priors, but must not be treated as proven effectiveness of our own defensive calls without controls for matchup and selection bias.
