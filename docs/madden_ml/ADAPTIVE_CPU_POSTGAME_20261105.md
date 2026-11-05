# Madden 27 CPU — Nov 5, 2026 adaptive-offense work

**Status:** isolated draft, NOT merged or activated on the player's machine.
No game-clock entry, no `showing` entry, and no fixed formation-rotation schedule.

## Audit source of truth

- Latest primary CPU session: `3d9b477c10df4827`.
- Correct result **W 35–7**. The original game-over entry said 42–7;
  actual snap-log score was 35–7, with 41 completed offensive snaps,
  5 TD outcomes and 3 INT outcomes. No scoring play should be deleted.
- Guarded repair script `scripts/repair_madden_cpu_final_score.py` checks
  the session, game identity, snap-log score, 41 snaps/5 TD/3 INT, backs up
  SQLite, and corrects only final score and reconciliation metadata.
- A corrected **copy** of the uploaded backup was used for all offline work.
  The user's laptop SQLite has not been modified.

## Live policy changes

1. **Goal-to-go INT risks.** A small congestion-risk prior for developing
   routes inside the 5 plus a bounded, shrunken penalty for *verified*
   historical goal-line interceptions for that named play. An unverified
   old INT does NOT count as supervised execution. No play type is banned.
2. **Recent outcome adaptation.** Historical verified screen execution
   outcomes and consecutive same-play negative outcomes lower selection
   utility. Repeated HB Slip Screen failures across different formations
   count as evidence about that named play; no action is presumed executed
   without verification. Maximum combined outcome penalty is 0.105.
3. **Model-driven formation rotation.** Per-situation top-three legal play
   scores determine the strategic active pool. Formation entry uses a 0.013
   model-relative evidence band, staying uses 0.027; this hysteresis is not
   a timer, snap-count trigger, or round-robin. Challengers stay callable
   with only a tiny 0.012 soft pool penalty. The state is scoped to game
   session and opponent and rebuilt if the confirmed inventory changes.
   The physical installed Madden book is **never** mutated by this policy.
4. **Model-driven inventory expansion.** The offline designer can choose
   strong additional formations as reserves, using its empirical scores
   and at least three genuinely new play names. It does not have to
   fill 15; 15 is this designer's current installation cap, not a
   rotation schedule or quota. Every included formation carries all its
   source-book catalogued plays. Any physical change requires separate
   Madden editor installation and explicit proposal fingerprint attestation.
5. **Low-interaction UI.** The Madden compact browser hides the clock and
   live-look checkbox. No pre-snap `showing` input is required; prior
   play/coverage entry remains optional. No added taps.

## Real-data read-only results

The generated proposal using a **separately trained, not installed** 508-row
empirical model contained **15 formations / 324 plays**, versus the current
confirmed **5 formations / 93 plays**.

Chronological holdout from latest game:
- 472 earlier verified training examples, 36 latest labeled examples.
- Historical-only retrained model: Brier 0.23938, log loss 0.67180.
- Constant training-rate baseline: Brier 0.24221.
- Current 428-row artifact file was **not in the DB backup**. Its exact
  prediction weights cannot be compared here; do not call the new candidate
  better than the installed model or automatically promote it.
- New 508-row model JSON produced for offline review only, not installed.

Read-only **selection-stage** replay across 41 latest CPU situations:
- Applied book 93 plays: modeled active pool ranged 1–5 formations;
  0 illegal picks / replay exceptions, max selection-stage runtime 12.24 ms.
- Proposed book 324 plays: active pool ranged **2–11 formations**;
  entry/exit events were **event-driven**, never periodic;
  0 illegal picks / replay exceptions, max selection-stage runtime 34.77 ms.
- These are model-only candidate selection times, **not full end-to-end
  latency** or verified game outcomes. No alternative was actually played
  in the historical games. The offline proposed book is not live legal yet.

## Validation status

- Branch-scoped GitHub Actions performs 37 focused regression tests
  plus Python compilation (successful before final documentation updates).
- A broader Madden suite includes preexisting baseline failures, including
  a clock-recording assertion. Compare against the coordinator base
  before attributing unrelated failures to this branch.
- The local model-heldout evaluation uses observed executed-play labels,
  not claimed causal wins for a different recommendation.
- No model promotion, no branch merge, no applied book mutation.

## Safe actions on player's computer

From within the repo:
```bash
git fetch origin
git worktree add --detach ../cfb-coach-adaptive-review \
  origin/madden-ml/adaptive-cpu-playbook-20261105
cd ../cfb-coach-adaptive-review
export CFB_COACH_MADDEN_DB="$HOME/.cfb-coach/madden27.db"

# Preview targeted result correction, then apply AFTER inspecting:
python3 scripts/repair_madden_cpu_final_score.py --db "$CFB_COACH_MADDEN_DB"
python3 scripts/repair_madden_cpu_final_score.py --db "$CFB_COACH_MADDEN_DB" --apply

# Refit/evaluate offline only. Creates a candidate, never installs it:
python3 scripts/evaluate_madden_offense_candidate.py \
  --db "$CFB_COACH_MADDEN_DB" \
  --out-dir "$HOME/.cfb-coach/offline-eval-20261105"

# Model proposes up to 15 formations, using the installed empirical
# artifact by default. No changes are made for a preview:
python3 -m cfb_coach ml offense-design -o cpu \
  --max-formations 15 --db "$CFB_COACH_MADDEN_DB" --text

# Optionally reproduce a proposal using the newly trained candidate,
# still strictly a physical PLAYBOOK design, NOT an ML model install:
python3 -m cfb_coach ml offense-design -o cpu --max-formations 15 \
  --artifact "$HOME/.cfb-coach/offline-eval-20261105/offense_candidate_NOT_INSTALLED.json" \
  --db "$CFB_COACH_MADDEN_DB" --stage --text
```

The staged output prints a new `proposal_id` and the exact source for each
formation. Install ALL of the proposed formations/plays in Madden. ONLY AFTER
that physical editor work, use:
```bash
python3 -m cfb_coach ml offense-design --db "$CFB_COACH_MADDEN_DB" \
  --confirm-installed "<proposal_id>" \
  --attest "I installed every listed formation and all of its plays in the Madden editor."
```
No model ever self-attests a physical installation. Avoid running
`prep --mark-applied` as a substitute for this fingerprinted model-design
confirmation. After actual application, the model can choose from the
larger inventory and independently change its tactical active pool on
any call. All selected calls still must be physically available.

**Rollback:** stop using this draft branch and play on the previously
verified coordinator checkout. A physical custom-book rollback requires
reinstalling the old Madden book before using the designer's `--rollback-design`
attestation flow. The current experimental model file remains unchanged.
