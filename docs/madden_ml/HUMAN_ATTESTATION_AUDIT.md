# Human-game execution attestation: audit and training-weight experiment

Research only. The database is opened read-only, nothing is rewritten, no
evidence classification changes, and no model is promoted.

## The attestation

The owner stated (2026-11-06) that in games against human opponents they
consistently ran the play the model recommended. This is recorded as
`attestation_audit.RETROSPECTIVE_ATTESTATION`: a general statement about
formation and play, not per-snap verification. It does not cover hot routes,
adjustments or macros.

## Tiers (`cfb_coach/madden/model/attestation_audit.py`)

| Tier | Meaning |
|---|---|
| `independently_verified` | Per-snap confirmation (not the standing toggle). |
| `user_attested` | Displayed call, known outcome, no conflicting record. Source `standing_prospective` (standing toggle on at the time) or `retrospective_statement` (this statement). Confidence `high` when a decision record links the displayed final pick to the snap, `medium` when only the snap's call is known. |
| `excluded` | Missing call or outcome, a decision that disagrees with the displayed call, a mismatch flag, a recommendation not accepted, a different executed play, or the user marked the snap unknown. |

A recommendation never becomes an executed play unless the snap passes the
attested-tier checks, and the tier is always reported separately.

`python scripts/madden_ml/audit_human_attestation.py --db COPY.db --out DIR`

## Counts (7 human games, opponents anonymized)

| | Offense | Defense |
|---|---|---|
| Independently verified | 0 | 0 |
| Attested, standing toggle (already used by training) | 83 (50 high / 33 medium) | 0 |
| Attested, retrospective statement (new) | 153 (41 high / 112 medium) | 260 (82 high / 178 medium) |
| Excluded | 1 (marked unknown) | 0 |

No displayed call conflicts with its decision record. Four attested snaps
carried an adjustment that was never confirmed; the play is attested, the
adjustment is not. Each game falls entirely in one tier.

The statement could add **413** human plays (153 offense, 260 defense), all
at attested confidence, none verified.

## Does training on them help? (`scripts/madden_ml/attestation_weight_eval.py`)

Leave-one-human-game-out over 7 games; tuning inside the training games only.
Test labels are attested plays of the held-out game, so the test itself rests
on the attestation. Log-loss difference against the reference, 95% interval
from a game-clustered bootstrap; negative is better.

**Defense** (does the offense win the snap, logistic v2 design). Weight `w`
scales the play-specific part of the model on top of a situation-only model.

| Policy | Log loss | AUC | vs w0 |
|---|---|---|---|
| w0 (situation only, needs no execution evidence) | 0.664 | 0.57 | reference |
| w50 | 0.670 | 0.55 | +0.006 [-0.008, +0.019] |
| w100 | 0.687 | 0.49 | +0.023 [-0.003, +0.051] |
| training base rate | 0.692 | — | +0.028 [+0.002, +0.054] |

The current live defense artifact (`logged_unverified_v1`) is already trained
on these displayed calls, which is effectively w100. In-sample it scores
0.633, which is not a fair comparison because it has seen the test games.

**Offense, human games** (experimental model, CPU rows unchanged).

| Policy | Log loss | vs status quo |
|---|---|---|
| status quo (standing toggle at 1, retrospective only as context) | 0.704 | reference |
| w0 | 0.706 | +0.001 [-0.002, +0.007] |
| w50 | 0.703 | -0.001 [-0.003, +0.002] |
| w100 | 0.702 | -0.002 [-0.005, +0.002] |

Every human-offense policy is at or below a coin flip (0.693).

## Conclusion

- The attestation is enough to label 413 extra human plays as a separate
  `user_attested` tier, keeping the current classifications as they are.
- Using those plays' identities does not improve held-out predictions. On
  defense it tends to hurt (w100 is worse than situation-only), so the
  current defense model leans on play identity more than the evidence
  supports. On offense the effect is too small to tell apart from zero.
- Recommendation: keep attested plays as a separate tier, weight 0 for
  play-specific learning, until verified per-snap human plays exist to check
  them against. Nothing here is promoted.
