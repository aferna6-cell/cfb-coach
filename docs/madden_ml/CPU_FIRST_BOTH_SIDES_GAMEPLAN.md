# First CPU game with BOTH experimental models: pregame checklist

**Status: proposal stage. NOT installed and NOT a game-ready GO.**

The repository's public Madden 27 source snapshot includes 11 offensive source books / 184 distinct formation names and 7 defensive source books / 65 distinct formation names. Real Madden editor availability and the user's installed private offensive artifact must be verified locally.

## Generate both books with the actual models

Use an **independent testing-franchise Madden database** (never the default primary database), checked out on PR #139 branch.

From repository root:

    export CFB_COACH_MADDEN_DB="/ABSOLUTE/TEST/FRANCHISE/madden27.db"
    python3 scripts/madden_ml/defense_v2_pilot.py inventory --db "$CFB_COACH_MADDEN_DB"
    python3 scripts/madden_ml/cpu2_audit_database.py --db "$CFB_COACH_MADDEN_DB" --out /tmp/cpu-offense-audit.json

Review the CPU2 audit to find the real latest structural offensive model path and hash. Then:

    python3 scripts/madden_ml/cpu_both_sides_pregame.py \
      --db "$CFB_COACH_MADDEN_DB" \
      --franchise "EXACT_RECORDED_DYNASTY" \
      --cpu-opponent cpu \
      --offense-artifact "/ABSOLUTE/PATH/TO/CURRENT_OFFENSIVE_ARTIFACT.json" \
      --allow-cold-start \
      --out-dir /tmp/cpu-first-both

Output files: offense_proposal.json (offensive model-selected full stock source formations and plays), defense_proposal.json (V2 model-selected up-to-15 full defensive formations and plays), defense_cold_start_model.json (zero verified defensive training rows, explicitly prior-only), and cpu_both_sides_pregame_report.json (status, model hash and stored-book comparison).

For a later game with already verified defensive execution data, use the existing defense V2 train command to generate an actual model, then replace --allow-cold-start with --defense-artifact /PATH/TO/TRAINED_DEFENSE.json.

The first-game cold start is a bounded, source-backed football-prior policy with Bayesian opponent beliefs. It is NOT a model trained on actual defensive executions and must never be presented as trained. After the first game the challenger can learn from truly executed verified defensive calls and confirmed individual adjustments.

## Install and verify

Physically build the exact proposed formations and full source-specific play lists in the Madden editor; record/attest them through the existing CFB Coach stage/mark-applied flow. The generator itself is read-only and does not install them. Then verify the lock against the saved proposals:

    python3 scripts/madden_ml/cpu_both_sides_pregame.py \
      --db "$CFB_COACH_MADDEN_DB" \
      --franchise "EXACT_RECORDED_DYNASTY" \
      --cpu-opponent cpu \
      --check-only \
      --out-dir /tmp/cpu-first-both

Both stored-book audit sides must report ready_for_first_live_test=true. This proves the recorded installed inventory matches the model proposals, NOT that Madden physically accepted every formation.

## Run preflight tests BEFORE kickoff

1. Run the targeted GitHub and local Madden test suite for offense and defensive V2, including dual book audit, source legality, opponent chronology, adjustments, and user reads.
2. Re-run the current route-aware offensive 100-call full 15-formation/installed-book benchmark on the owner's laptop with macros OFF. Do not reuse a pre-PR #138 benchmark; require <=150ms end-to-end p95, no silent heuristic fallbacks, and correct actual model attribution. Historical CPU Game 1 had 67/67 route-aware timeouts.
3. Benchmark the installed defense stock+adjustment+USER-read complete live path end-to-end; target <=150ms, no uninstalled calls, and no current/future snap leakage. The CI catalog preview and unit tests alone are NOT loaded-game latency evidence.
4. Check the resulting game session has exact franchise/opponent identity and sequential snap IDs when playing BOTH sides. Verify executed defensive play and adjustment are not inferred from what the model merely recommended.
5. Check offense and defense modes independently. The defensive flag does not enable the offensive experimental model. Confirm any broader CI failures have been reviewed (the previous wider suite reported 19 failures).
6. Declare GO only when both sides meet the local checks.

## Experimental game launch ONLY after GO

    export CFB_COACH_DEFENSE_V2_TEST_FRANCHISE="EXACT_RECORDED_DYNASTY"
    export CFB_COACH_DEFENSE_V2_ARTIFACT=/tmp/cpu-first-both/defense_cold_start_model.json
    python3 -m cfb_coach play --game madden27 -o cpu --cpu-defense --defense-model v2

This is an illustrative launch; check play --help for additional required arguments and confirm offense is in experimental mode on the **test** DB. For a trained defensive artifact use its path instead. Disable defensive V2 by removing --defense-model v2 in the next session. Nothing is promoted into the main franchise by these pregame scripts.

## Adjustment status

V2 can rank individual coverage hub changes including press/back off, inside/outside leverage, Smart Zone settings and pass commit when evidence favors them, plus a receiver-specific double team if a target is available. It does NOT yet independently make every defender hot route, line shift, blitz assignment, QB contain, safety alignment or saved custom macro. NO ADJUSTMENT always competes. The on-screen USER defender read prioritizes run fit vs crossers, vertical seams, bunch, etc without claiming exact unseen 11-slot assignments.

**Current gate: code prepared; owner-local model, editor, full-book latency and actual gameplay confirmation required.**


## Public-source defensive starter preview (actual V2 coldstart CI execution)

Before using private game data, the first-game zero-execution V2 portfolio selected **6 defensive formations and 81 stock plays**, not a fixed 15-form quota. The full named play inventory is saved at:

    docs/madden_ml/CPU1_DEFENSE_V2_COLDSTART_PREVIEW.json

| Formation | Stock book source | Catalogued plays |
|---|---|---:|
| Dime 2-3 Odd | Raiders | 24 |
| 3-4 Cub | Lions | 6 |
| Nickel 2-4 Dbl Mug | Vikings | 15 |
| 2-4-5 Over Wide | Raiders | 18 |
| 4-3 Even 6-1 | 49ers | 6 |
| 3-3-5 3 High Penny | Vikings | 12 |

This is a **provisional source-catalog preview** for a prior-only test franchise, not the final physically installed game book. Re-run the actual dual pregame script against the **correct latest offensive model artifact and actual test franchise database**; the resulting defense proposal is authoritative for that trial, and might differ as game evidence grows.

After installation, run the **read-only V2 core 100-call benchmark** against the exact locked installed defensive inventory:

    python3 scripts/madden_ml/bench_defense_v2_scoring.py \
      --db "$CFB_COACH_MADDEN_DB" \
      --model /tmp/cpu-first-both/defense_cold_start_model.json \
      --runs 100 \
      --out /tmp/cpu-first-both/defense_core_benchmark.json

This measures stock ranking + individual adjustments + USER reads, including first-call latency. It excludes live session DB lookups, UI work and decision logging, so it **cannot** alone satisfy the full <=150ms end-to-end gate. Benchmark the offensive full installed-book live path separately and verify neither side silently falls back.
