# Defense v1 live opt-in

Offline eval of this logistic model did not beat the situation baseline. The
policy name is `logged_unverified_v1`. It is not a verified defensive result
and it does not require verified snaps. The repo does not contain a trained
artifact or game rows. Put the JSON on the machine that will call the game.

Each defensive call is labeled `EXPERIMENTAL` when this file produced the
shown play, or `HEURISTIC` when the established caller was shown. A missing
file, invalid JSON, a candidate that is not in the installed defensive
playbook, a timeout, or any exception falls back to that heuristic call. The
reason is stored on the decision and printed as `fallback_reason`. The model
pick, the heuristic pick, and the call actually shown are stored together
(`ml_decisions.shadow_*`, `heuristic_*`, `final_*`, and `config_source` inside
`decision_json`) so the next retrain can compare them.

The offense coordinator is a separate mode. This flag does not enable it,
disable it, or change its rollback (`python -m cfb_coach ml heuristic`).

## Run it

```bash
python3 -m cfb_coach play --game madden27 -o ryan \
  --defense-model v1 \
  --defense-model-path ~/.cfb-coach/models/defense_v1.json \
  --no-vod-prior --terminal
```

`--defense-model-path` may be omitted. The default path is
`~/.cfb-coach/models/defense_v1.json`. `CFB_COACH_DEFENSE_V1` is the
environment override when the flag is omitted.

`--ml-defense-control` still requires
`python -m cfb_coach ml defense-experimental --enable --db <absolute>`.
If that installed coordinator file is missing or has zero verified snaps,
the same defense v1 file is used instead of the empty prior. A coordinator
file that already has verified snaps keeps the existing experimental caller
unless `--defense-model v1` is also passed.

## Roll back

Start the next session without `--defense-model v1` and without
`--ml-defense-control`. Nothing in this opt-in writes the coaching mode.
If defense experimental mode was enabled earlier:

```bash
python3 -m cfb_coach ml defense-experimental --disable --db "$CFB_COACH_MADDEN_DB"
```

## Artifact schema `madden.defense.logistic.v1`

```json
{
  "schema": "madden.defense.logistic.v1",
  "policy": "logged_unverified_v1",
  "target": "offense_success",
  "intercept": 0.0,
  "features": ["down", "distance", "distance_missing", "red_zone"],
  "coefficients": {"down": 0.0, "distance": 0.0, "distance_missing": 0.0, "red_zone": 0.0},
  "call_families": ["cover3", "man"],
  "packages": ["nickel", "dime"],
  "family_coefficients": {"cover3": -1.0, "man": 1.0},
  "package_coefficients": {"nickel": 0.0, "dime": 0.0},
  "candidates": [
    {"formation": "Nickel", "play": "Cover 3", "family": "cover3", "package": "nickel"}
  ]
}
```

Situation features: `down`, `distance`, `yardline`, `score_diff`, `quarter`,
`red_zone`, `goal_line`, `two_minute`, `short_yardage`, `long_yardage`.
A numeric feature may be paired with `<name>_missing` (1 when that number
was not supplied). `score_diff` is us minus them. `quarter` is read from the
situation extras.

The logit is intercept plus the situation dot product plus the family
coefficient plus the package coefficient. The pick is the candidate with the
lowest P(offense success). Ties break by formation, then play. If that
candidate is not in the locked defensive playbook, the snap is `HEURISTIC`.
The caller does not silently show the next legal candidate.
