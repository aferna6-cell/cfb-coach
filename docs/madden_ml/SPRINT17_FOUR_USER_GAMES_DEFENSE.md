# Sprint 17 — Defensive football intelligence, adaptive tendencies, macro invention and four user games

This builds on draft PR #39 / Sprint 16. Changes are contained in
`cursor/sprint17-defensive-intelligence-four-user-games`.
Nothing is merged into `main` and no game is secretly modified.

## What is implemented

- **Defensive knowledge:** explicit football principles for vertical shots,
  floods, crossers, bunch/stack, runs, RPOs and QB scrambling; conservative
  adjustments to defensive-family fit. These are general football priors, **not
  confirmed in-game route diagrams**.
- **Opponent adaptation:** reads `concept_seen` only from *verified executed*
  human-game defensive snaps, groups by specific opponent and down/distance
  context, shrinks sparse samples, and treats tendencies as historical priors,
  NEVER live knowledge of the next play.
- **Independent defensive learning:** fit coverage-family, context and
  particular defensive-call stop rates from verified HUMAN defensive executions.
  Older mixed-CPU/unverified artifacts are rejected. The offensive model remains
  separate.
- **Defensive situational adjustments:** compare doing nothing against
  researched Xbox adjustments with confirmed buttons/sources, and against
  model-created defensive Custom Adjustments that were physically installed and
  explicitly verified. Strong prior tendencies have a smaller adjustment
  influence than currently observed live concepts.
- **Defensive joint call:** rank all installed plays, then compare bounded
  play+adjustment plans among the top calls, within the existing 150 ms limit.
  The earlier defensive caller remains the safe fallback, and existing armed
  researched macros are preserved until action outcomes can be calibrated.
- **Original offensive macros:** stage two-action sourced offensive
  compositions (hot routes/protection) in the existing verification registry.
  No arbitrary controller inputs or claimed editor compatibility.
- **Original defensive macros:** existing Sprint 16 lab combines compatible
  researched settings, supplies a complete sheet including defaults and
  requires explicit in-game verification and an available defensive slot.
- **Four-human-game learning:** audit the last four human games with separate verified
  offense/defense counts. Verified actions support play-specific success. Even
  without verified actions, observed opponent yardage, conversions and explosive
  results can build conservative, opponent-level risk profiles WITHOUT blaming
  or crediting any unverified coverage or macro. All artifacts remain SHADOW.

## Ubuntu steps: inspect your four user games

```bash
cd ~/cfb-coach
git fetch origin
git switch cursor/sprint17-defensive-intelligence-four-user-games
git pull --ff-only
python3 -m cfb_coach ml find-db
python3 -m cfb_coach ml four-user-games --games 4
```

The DB is usually `/home/aidan/.cfb-coach/madden27.db`. Four historical
human games were reported in that local DB, but **the GitHub agent has no
direct copy**. The audit must run on the user's own Ubuntu laptop.

### Train on the verified evidence only (offline, never activates)

```bash
python3 -m cfb_coach ml four-user-games --games 4 --train
```

Check `verified_offense`, `verified_defense`,
`observed_opponent_concepts`, and the produced artifact paths.
No observed data must ever be reconstructed from a final score.
The existing separately confirmed offensive experimental model remains
unchanged unless you explicitly decide to install/test a new version.

**Known October 8 audit:** Gavin (65 snaps), James (71), Jaxon (54) and
Tiano (55) accounted for 245 logged human-game snaps, but **zero had verified
executions** at that time. The local DB may have changed since the audit.

When verified_defense=0, no coverage, play or macro-specific outcome model can
be learned. The opponent-level learner can still use independent observed
yardage, conversion and explosive results without attributing the outcome
to your selected defensive play. Inspect opponent_outcome_rows_learned; if
that count is also zero there is no usable outcome evidence.
Use ml confirm-execution with actual game evidence before claiming a
play was executed. Never bulk-certify recommendations.

## Generate YOUR OWN macro candidates on both sides

```bash
python3 -m cfb_coach ml macro-create --side both
python3 -m cfb_coach ml macro-create --side both --stage
```

This produces distinct, new blueprints from separately researched adjustment
primitives, not only a recommendation to use an existing preset macro.

For offense, build/test the listed Custom Adjustments in Madden and use
the existing `ml offense-design --verify-macro NAME` workflow to verify
and arm an exact draft (including explicit attestation).

For defense, build every editor setting and test the specified installed
base play, then use:

```bash
python3 -m cfb_coach ml defense-macro-lab -o gavin --show
python3 -m cfb_coach ml defense-macro-lab -o gavin \
  --verify ML-D-XXXXXXX \
  --attest "I created all listed defensive settings in Madden, tested the listed base play and physically armed a defense Custom Adjustment slot."
```

Neither preview nor staging changes the in-game controller. The generated
combination remains unverified until actual testing. Both sides are limited
by the game's active macro slot count.

## Defensive test — opt-in human game

```bash
python3 -m cfb_coach ml defense-experimental --retrain
python3 -m cfb_coach ml defense-experimental --enable
python3 -m cfb_coach play --game madden27 --opponent gavin --franchise lab
```

Important: `ml defense-experimental --retrain` fits **all** verified human
defensive training rows that exist in the Madden DB, while
`ml four-user-games --train` explicitly fits only the last four and leaves
those artifacts in shadow. The full training flow never fabricates missing
defensive data and rejects old mixed-verification policy files.

The current football intelligence requires accurately entered live information:
a post-snap observed concept from the last play is NEVER the actual current
concept, and should not be labeled live. The caller will rely on historic
opponent trends with bounded confidence when no live concept is visible.

To restore the legacy defensive caller immediately:

```bash
python3 -m cfb_coach ml defense-experimental --disable
```

## Evaluation required before trusting live competitive decisions

1. Run full CI, including the Sprint 16 regression and Sprint 17 tests.
2. Audit all four actual user games; confirm human and CPU populations remain
   disjoint, and no future game appears in a purported earlier fit.
3. Inspect offensive and defensive verified execution counts. No minimum
   success claim from mere game count.
4. Replay actual pre-snap situations with time-correct held-out evidence;
   do not use post-snap concepts as current observations.
5. Compare original caller vs new defensive caller, explain changed calls,
   action legality and 150 ms latency.
6. After a real human-opponent scrimmage, analyze conversion defense, explosive
   concessions, defensive stops, rush/contain decisions, and executed macros
   separately from recommendations.

This release is **an opt-in tactical pilot**, not automatic pre-snap vision or
autonomous Xbox gameplay. It does not assert that a synthesized macro is more
effective than the unmodified play, or that four sparse games prove superiority.
