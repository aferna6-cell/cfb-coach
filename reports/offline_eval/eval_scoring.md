# Coordinator decision-quality evaluation

These are decision-quality checks only. They are not effectiveness claims.

Display lines are copied from the harness so a reader can see them. Agreement is formation and play, not formatting.
Context rules are applied to the new decision. The old decision is the paired caller, not a second effectiveness score.

Share cap: 0.25

Latency budget: 150 ms

## Scoring rules

- `no_deep_shot_on_short_or_goal_line` (offense): On offense, a 3rd-or-4th-and-short (distance ≤ 2) or any goal-line short-yardage snap must not be a deep shot. Deep means cfb_coach.madden.catalog.is_deep (vert, flood, shot, seam, fade, and the other tokens that classifier already uses). Example: no deep shot on 3rd-and-1 or goal-line short yardage.
- `two_minute_trailing_not_clock_kill_run` (offense): On offense, a two-minute snap while trailing big (margin ≤ -14) must be clock-aware: the call fails when catalog.is_run is true or the play name is a kneel, victory, spike, or take-a-knee. A non-run passes. The rule does not run when score or clock was default-filled.
- `no_kneel_or_run_clock_when_trailing_late` (offense): On offense, while trailing big inside the two-minute window or the 4-min protect window, the call fails when the play name is a kneel, victory, spike, or take-a-knee. A normal run is not this rule; the two-minute rule covers clock-killing runs. Default-filled score or clock does not trigger it.
- `goal_line_short_not_dime` (defense): On defense, goal-line short yardage (distance ≤ 2) must not be a Dime formation. The installed heavier packages are 4-3 Over and the Nickel formations. Dime 3-2 Odd fails. This follows the existing note that goal line and true short yardage keep the heavy package.
- `trailing_late_no_prevent` (defense): On defense, while trailing big in the two-minute or 4-min protect window, a play named prevent or hail mary fails. Cover 4 Quarters is not this rule. Default-filled score or clock does not trigger it. The installed book has no prevent or hail-mary play name, so an in-book call passes and an invented name fails.

## Fixture inventory

- fixture_counts:

```json
{
  "by_source_side": {
    "observed": {
      "defense": 0,
      "offense": 161
    },
    "synthetic": {
      "defense": 36,
      "offense": 36
    }
  },
  "decision_rows": 771,
  "decision_tags": {
    "observed_situation,assigned_look": 483,
    "synthetic": 288
  },
  "decision_tags_by_side": {
    "observed_situation,assigned_look": {
      "defense": 0,
      "offense": 483
    },
    "synthetic": {
      "defense": 144,
      "offense": 144
    }
  },
  "identities": 233,
  "identity_tags": {
    "observed": 161,
    "synthetic": 72
  },
  "identity_tags_by_side": {
    "observed": {
      "defense": 0,
      "offense": 161
    },
    "synthetic": {
      "defense": 36,
      "offense": 36
    }
  }
}
```

- observed_csv: /workspace/tests/madden_ml/fixtures/item3/presnap_observed_cpu.csv
- observed_csv_present: True
- observed_source_rows: 161

## Summary counts

Decision rows scored: 771

### defense / all

- n: 144
- formation agree: 144/144
- play agree: 144/144
- in book: 144
- adjustment legal (armed macro or no adjustment): 144
- hard fail: 0
- adjustment id differs: 0
- context fail rows: 0
- book used: calls 25/114, formations 5/5
- call entropy bits: 4.482289237493325
- formation entropy bits: 2.1409968540705986
- top call share: 0.1111111111111111
- top formation share: 0.3611111111111111
- top share flag (cap 0.25): True
- rule no_deep_shot_on_short_or_goal_line: applies 0, passed 0, failed 0
- rule two_minute_trailing_not_clock_kill_run: applies 0, passed 0, failed 0
- rule no_kneel_or_run_clock_when_trailing_late: applies 0, passed 0, failed 0
- rule goal_line_short_not_dime: applies 8, passed 8, failed 0
- rule trailing_late_no_prevent: applies 36, passed 36, failed 0

### defense / observed

- n: 0
- formation agree: None
- play agree: None
- in book: 0
- adjustment legal (armed macro or no adjustment): 0
- hard fail: 0
- adjustment id differs: 0
- context fail rows: 0
- book used: calls 0/114, formations 0/5
- call entropy bits: None
- formation entropy bits: None
- top call share: None
- top formation share: None
- top share flag (cap 0.25): False
- rule no_deep_shot_on_short_or_goal_line: applies 0, passed 0, failed 0
- rule two_minute_trailing_not_clock_kill_run: applies 0, passed 0, failed 0
- rule no_kneel_or_run_clock_when_trailing_late: applies 0, passed 0, failed 0
- rule goal_line_short_not_dime: applies 0, passed 0, failed 0
- rule trailing_late_no_prevent: applies 0, passed 0, failed 0

### defense / synthetic

- n: 144
- formation agree: 144/144
- play agree: 144/144
- in book: 144
- adjustment legal (armed macro or no adjustment): 144
- hard fail: 0
- adjustment id differs: 0
- context fail rows: 0
- book used: calls 25/114, formations 5/5
- call entropy bits: 4.482289237493325
- formation entropy bits: 2.1409968540705986
- top call share: 0.1111111111111111
- top formation share: 0.3611111111111111
- top share flag (cap 0.25): True
- rule no_deep_shot_on_short_or_goal_line: applies 0, passed 0, failed 0
- rule two_minute_trailing_not_clock_kill_run: applies 0, passed 0, failed 0
- rule no_kneel_or_run_clock_when_trailing_late: applies 0, passed 0, failed 0
- rule goal_line_short_not_dime: applies 8, passed 8, failed 0
- rule trailing_late_no_prevent: applies 36, passed 36, failed 0

### offense / all

- n: 627
- formation agree: 627/627
- play agree: 627/627
- in book: 627
- adjustment legal (armed macro or no adjustment): 627
- hard fail: 0
- adjustment id differs: 0
- context fail rows: 0
- book used: calls 76/93, formations 5/5
- call entropy bits: 6.002164549146807
- formation entropy bits: 2.2036633995891357
- top call share: 0.04146730462519936
- top formation share: 0.29505582137161085
- top share flag (cap 0.25): True
- rule no_deep_shot_on_short_or_goal_line: applies 63, passed 63, failed 0
- rule two_minute_trailing_not_clock_kill_run: applies 24, passed 24, failed 0
- rule no_kneel_or_run_clock_when_trailing_late: applies 36, passed 36, failed 0
- rule goal_line_short_not_dime: applies 0, passed 0, failed 0
- rule trailing_late_no_prevent: applies 0, passed 0, failed 0

### offense / observed

- n: 483
- formation agree: 483/483
- play agree: 483/483
- in book: 483
- adjustment legal (armed macro or no adjustment): 483
- hard fail: 0
- adjustment id differs: 0
- context fail rows: 0
- book used: calls 73/93, formations 5/5
- call entropy bits: 5.981000225935749
- formation entropy bits: 2.224306313097711
- top call share: 0.037267080745341616
- top formation share: 0.2919254658385093
- top share flag (cap 0.25): True
- rule no_deep_shot_on_short_or_goal_line: applies 27, passed 27, failed 0
- rule two_minute_trailing_not_clock_kill_run: applies 0, passed 0, failed 0
- rule no_kneel_or_run_clock_when_trailing_late: applies 0, passed 0, failed 0
- rule goal_line_short_not_dime: applies 0, passed 0, failed 0
- rule trailing_late_no_prevent: applies 0, passed 0, failed 0

### offense / synthetic

- n: 144
- formation agree: 144/144
- play agree: 144/144
- in book: 144
- adjustment legal (armed macro or no adjustment): 144
- hard fail: 0
- adjustment id differs: 0
- context fail rows: 0
- book used: calls 27/93, formations 5/5
- call entropy bits: 4.627986806877674
- formation entropy bits: 2.0791262495325706
- top call share: 0.08333333333333333
- top formation share: 0.3055555555555556
- top share flag (cap 0.25): True
- rule no_deep_shot_on_short_or_goal_line: applies 36, passed 36, failed 0
- rule two_minute_trailing_not_clock_kill_run: applies 24, passed 24, failed 0
- rule no_kneel_or_run_clock_when_trailing_late: applies 36, passed 36, failed 0
- rule goal_line_short_not_dime: applies 0, passed 0, failed 0
- rule trailing_late_no_prevent: applies 0, passed 0, failed 0

## Opponent adaptation

A situation is adapted when it has at least three looks, every look matches on a same-seed rerun, and the new formation/play/adjustment is not the same on every look. A same-seed mismatch is seed noise and is not counted as adaptation.

- defense:observed: situations 0, adapted 0, unchanged 0, seed noise 0, fewer than 3 looks 0
- defense:synthetic: situations 36, adapted 0, unchanged 36, seed noise 0, fewer than 3 looks 0
- offense:observed: situations 161, adapted 0, unchanged 161, seed noise 0, fewer than 3 looks 0
- offense:synthetic: situations 36, adapted 0, unchanged 36, seed noise 0, fewer than 3 looks 0

## Latency

- rows: 771
- whole p50/p95/max: 4.238890000124229 / 4.914384000130667 / 8.925932000011016
- model p50/p95/max: 0.0 / 0.0 / 0.0
- over budget: 0
- fallback counts: {}

## Paired decisions

| row | side | tags | look sent | new formation | new play | new adjustment | new state | old formation | old play | old adjustment | formation agree | play agree | our legal | hard fail |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| syn-offense-001__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Inside Zone Split |  |  | Pistol Deuce Close | Inside Zone Split |  | True | True | True | False |
| syn-offense-001__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Inside Zone Split |  |  | Pistol Deuce Close | Inside Zone Split |  | True | True | True | False |
| syn-offense-001__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Deuce Close | Inside Zone Split |  |  | Pistol Deuce Close | Inside Zone Split |  | True | True | True | False |
| syn-offense-001__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Pistol Deuce Close | Inside Zone Split |  |  | Pistol Deuce Close | Inside Zone Split |  | True | True | True | False |
| syn-offense-002__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| syn-offense-002__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| syn-offense-002__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| syn-offense-002__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| syn-offense-003__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Spacing |  |  | Gun 5WR Tight | Spacing |  | True | True | True | False |
| syn-offense-003__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Spacing |  |  | Gun 5WR Tight | Spacing |  | True | True | True | False |
| syn-offense-003__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun 5WR Tight | Spacing |  |  | Gun 5WR Tight | Spacing |  | True | True | True | False |
| syn-offense-003__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Spacing |  |  | Gun 5WR Tight | Spacing |  | True | True | True | False |
| syn-offense-004__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Cluster Bunch |  |  | Gun 5WR Tight | Cluster Bunch |  | True | True | True | False |
| syn-offense-004__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Cluster Bunch |  |  | Gun 5WR Tight | Cluster Bunch |  | True | True | True | False |
| syn-offense-004__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun 5WR Tight | Cluster Bunch |  |  | Gun 5WR Tight | Cluster Bunch |  | True | True | True | False |
| syn-offense-004__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Cluster Bunch |  |  | Gun 5WR Tight | Cluster Bunch |  | True | True | True | False |
| syn-offense-005__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Pistol Trips | X Spot |  |  | Pistol Trips | X Spot |  | True | True | True | False |
| syn-offense-005__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | X Spot |  |  | Pistol Trips | X Spot |  | True | True | True | False |
| syn-offense-005__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | X Spot |  |  | Pistol Trips | X Spot |  | True | True | True | False |
| syn-offense-005__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | X Spot |  |  | Pistol Trips | X Spot |  | True | True | True | False |
| syn-offense-006__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Dive |  |  | Pistol Trips | HB Dive |  | True | True | True | False |
| syn-offense-006__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Dive |  |  | Pistol Trips | HB Dive |  | True | True | True | False |
| syn-offense-006__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | HB Dive |  |  | Pistol Trips | HB Dive |  | True | True | True | False |
| syn-offense-006__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | HB Dive |  |  | Pistol Trips | HB Dive |  | True | True | True | False |
| syn-offense-007__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Slants |  |  | Pistol Trips | Slants |  | True | True | True | False |
| syn-offense-007__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Slants |  |  | Pistol Trips | Slants |  | True | True | True | False |
| syn-offense-007__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | Slants |  |  | Pistol Trips | Slants |  | True | True | True | False |
| syn-offense-007__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | Slants |  |  | Pistol Trips | Slants |  | True | True | True | False |
| syn-offense-008__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| syn-offense-008__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| syn-offense-008__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| syn-offense-008__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| syn-offense-009__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| syn-offense-009__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| syn-offense-009__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| syn-offense-009__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| syn-offense-010__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| syn-offense-010__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| syn-offense-010__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| syn-offense-010__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| syn-offense-011__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| syn-offense-011__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| syn-offense-011__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| syn-offense-011__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| syn-offense-012__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| syn-offense-012__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| syn-offense-012__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| syn-offense-012__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| syn-offense-013__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mid Zone |  |  | Gun Trips X Nasty | Mid Zone |  | True | True | True | False |
| syn-offense-013__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mid Zone |  |  | Gun Trips X Nasty | Mid Zone |  | True | True | True | False |
| syn-offense-013__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mid Zone |  |  | Gun Trips X Nasty | Mid Zone |  | True | True | True | False |
| syn-offense-013__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mid Zone |  |  | Gun Trips X Nasty | Mid Zone |  | True | True | True | False |
| syn-offense-014__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| syn-offense-014__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| syn-offense-014__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| syn-offense-014__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| syn-offense-015__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| syn-offense-015__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| syn-offense-015__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| syn-offense-015__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| syn-offense-016__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake |  |  | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake |  | True | True | True | False |
| syn-offense-016__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake |  |  | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake |  | True | True | True | False |
| syn-offense-016__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake |  |  | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake |  | True | True | True | False |
| syn-offense-016__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake |  |  | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake |  | True | True | True | False |
| syn-offense-017__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| syn-offense-017__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| syn-offense-017__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| syn-offense-017__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| syn-offense-018__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| syn-offense-018__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| syn-offense-018__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| syn-offense-018__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| syn-offense-019__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| syn-offense-019__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| syn-offense-019__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| syn-offense-019__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| syn-offense-020__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| syn-offense-020__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| syn-offense-020__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| syn-offense-020__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| syn-offense-021__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| syn-offense-021__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| syn-offense-021__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| syn-offense-021__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| syn-offense-022__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| syn-offense-022__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| syn-offense-022__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| syn-offense-022__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| syn-offense-023__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| syn-offense-023__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| syn-offense-023__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| syn-offense-023__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| syn-offense-024__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh Post | RZ | armed | Gun 5WR Tight | Mesh Post | RZ | True | True | True | False |
| syn-offense-024__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh Post | RZ | armed | Gun 5WR Tight | Mesh Post | RZ | True | True | True | False |
| syn-offense-024__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun 5WR Tight | Mesh Post | RZ | armed | Gun 5WR Tight | Mesh Post | RZ | True | True | True | False |
| syn-offense-024__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Mesh Post | RZ | armed | Gun 5WR Tight | Mesh Post | RZ | True | True | True | False |
| syn-offense-025__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Dig |  |  | Gun Trips X Nasty | Hi Lo Dig |  | True | True | True | False |
| syn-offense-025__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Dig |  |  | Gun Trips X Nasty | Hi Lo Dig |  | True | True | True | False |
| syn-offense-025__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Hi Lo Dig |  |  | Gun Trips X Nasty | Hi Lo Dig |  | True | True | True | False |
| syn-offense-025__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Hi Lo Dig |  |  | Gun Trips X Nasty | Hi Lo Dig |  | True | True | True | False |
| syn-offense-026__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Quick Short Post |  |  | Gun Trips X Nasty | Quick Short Post |  | True | True | True | False |
| syn-offense-026__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Quick Short Post |  |  | Gun Trips X Nasty | Quick Short Post |  | True | True | True | False |
| syn-offense-026__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Quick Short Post |  |  | Gun Trips X Nasty | Quick Short Post |  | True | True | True | False |
| syn-offense-026__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Quick Short Post |  |  | Gun Trips X Nasty | Quick Short Post |  | True | True | True | False |
| syn-offense-027__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| syn-offense-027__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| syn-offense-027__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| syn-offense-027__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| syn-offense-028__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  | True | True | True | False |
| syn-offense-028__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  | True | True | True | False |
| syn-offense-028__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  | True | True | True | False |
| syn-offense-028__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  | True | True | True | False |
| syn-offense-029__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Triple Slants |  |  | Gun Trips X Nasty | Mtn Triple Slants |  | True | True | True | False |
| syn-offense-029__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Triple Slants |  |  | Gun Trips X Nasty | Mtn Triple Slants |  | True | True | True | False |
| syn-offense-029__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mtn Triple Slants |  |  | Gun Trips X Nasty | Mtn Triple Slants |  | True | True | True | False |
| syn-offense-029__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mtn Triple Slants |  |  | Gun Trips X Nasty | Mtn Triple Slants |  | True | True | True | False |
| syn-offense-030__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | PA Post Shot Choice |  |  | Gun Doubles Clamp Stack | PA Post Shot Choice |  | True | True | True | False |
| syn-offense-030__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | PA Post Shot Choice |  |  | Gun Doubles Clamp Stack | PA Post Shot Choice |  | True | True | True | False |
| syn-offense-030__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | PA Post Shot Choice |  |  | Gun Doubles Clamp Stack | PA Post Shot Choice |  | True | True | True | False |
| syn-offense-030__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | PA Post Shot Choice |  |  | Gun Doubles Clamp Stack | PA Post Shot Choice |  | True | True | True | False |
| syn-offense-031__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| syn-offense-031__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| syn-offense-031__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| syn-offense-031__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| syn-offense-032__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Zone WK |  |  | Pistol Trips | HB Zone WK |  | True | True | True | False |
| syn-offense-032__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Zone WK |  |  | Pistol Trips | HB Zone WK |  | True | True | True | False |
| syn-offense-032__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | HB Zone WK |  |  | Pistol Trips | HB Zone WK |  | True | True | True | False |
| syn-offense-032__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | HB Zone WK |  |  | Pistol Trips | HB Zone WK |  | True | True | True | False |
| syn-offense-033__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | armed | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | True | True | True | False |
| syn-offense-033__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | armed | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | True | True | True | False |
| syn-offense-033__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | armed | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | True | True | True | False |
| syn-offense-033__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | armed | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | True | True | True | False |
| syn-offense-034__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Spot Choice | RZ | armed | Gun Doubles Clamp Stack | Spot Choice | RZ | True | True | True | False |
| syn-offense-034__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Spot Choice | RZ | armed | Gun Doubles Clamp Stack | Spot Choice | RZ | True | True | True | False |
| syn-offense-034__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Spot Choice | RZ | armed | Gun Doubles Clamp Stack | Spot Choice | RZ | True | True | True | False |
| syn-offense-034__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Spot Choice | RZ | armed | Gun Doubles Clamp Stack | Spot Choice | RZ | True | True | True | False |
| syn-offense-035__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| syn-offense-035__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| syn-offense-035__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| syn-offense-035__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| syn-offense-036__null | offense | synthetic | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| syn-offense-036__empty | offense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| syn-offense-036__11-nickel | offense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| syn-offense-036__12-base | offense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| syn-defense-001__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | 1 Robber Press |  |  | 4-3 Over | 1 Robber Press |  | True | True | True | False |
| syn-defense-001__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | 1 Robber Press |  |  | 4-3 Over | 1 Robber Press |  | True | True | True | False |
| syn-defense-001__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | 1 Robber Press |  |  | 4-3 Over | 1 Robber Press |  | True | True | True | False |
| syn-defense-001__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | 1 Robber Press |  |  | 4-3 Over | 1 Robber Press |  | True | True | True | False |
| syn-defense-002__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 4 Quarters | SAFE DEEP | armed | Nickel Over | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-002__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 4 Quarters | SAFE DEEP | armed | Nickel Over | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-002__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 4 Quarters | SAFE DEEP | armed | Nickel Over | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-002__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 4 Quarters | SAFE DEEP | armed | Nickel Over | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-003__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-003__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-003__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-003__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-004__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Single Mug | Tampa 2 | SAFE DEEP | armed | Nickel Single Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-004__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Single Mug | Tampa 2 | SAFE DEEP | armed | Nickel Single Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-004__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Single Mug | Tampa 2 | SAFE DEEP | armed | Nickel Single Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-004__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Single Mug | Tampa 2 | SAFE DEEP | armed | Nickel Single Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-005__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | Cover 2 Man | SAFE DEEP | armed | 4-3 Over | Cover 2 Man | SAFE DEEP | True | True | True | False |
| syn-defense-005__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | Cover 2 Man | SAFE DEEP | armed | 4-3 Over | Cover 2 Man | SAFE DEEP | True | True | True | False |
| syn-defense-005__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | Cover 2 Man | SAFE DEEP | armed | 4-3 Over | Cover 2 Man | SAFE DEEP | True | True | True | False |
| syn-defense-005__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | Cover 2 Man | SAFE DEEP | armed | 4-3 Over | Cover 2 Man | SAFE DEEP | True | True | True | False |
| syn-defense-006__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | Cover 1 Hole |  |  | 4-3 Over | Cover 1 Hole |  | True | True | True | False |
| syn-defense-006__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | Cover 1 Hole |  |  | 4-3 Over | Cover 1 Hole |  | True | True | True | False |
| syn-defense-006__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | Cover 1 Hole |  |  | 4-3 Over | Cover 1 Hole |  | True | True | True | False |
| syn-defense-006__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | Cover 1 Hole |  |  | 4-3 Over | Cover 1 Hole |  | True | True | True | False |
| syn-defense-007__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Dime 3-2 Odd | Cover 4 Quarters | SAFE DEEP | armed | Dime 3-2 Odd | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-007__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Dime 3-2 Odd | Cover 4 Quarters | SAFE DEEP | armed | Dime 3-2 Odd | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-007__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Dime 3-2 Odd | Cover 4 Quarters | SAFE DEEP | armed | Dime 3-2 Odd | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-007__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Dime 3-2 Odd | Cover 4 Quarters | SAFE DEEP | armed | Dime 3-2 Odd | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-008__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 1 Hole | SAFE DEEP | armed | Nickel Over | Cover 1 Hole | SAFE DEEP | True | True | True | False |
| syn-defense-008__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 1 Hole | SAFE DEEP | armed | Nickel Over | Cover 1 Hole | SAFE DEEP | True | True | True | False |
| syn-defense-008__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 1 Hole | SAFE DEEP | armed | Nickel Over | Cover 1 Hole | SAFE DEEP | True | True | True | False |
| syn-defense-008__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 1 Hole | SAFE DEEP | armed | Nickel Over | Cover 1 Hole | SAFE DEEP | True | True | True | False |
| syn-defense-009__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Double Mug | Cover 3 Match | SAFE DEEP | armed | Nickel Double Mug | Cover 3 Match | SAFE DEEP | True | True | True | False |
| syn-defense-009__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Double Mug | Cover 3 Match | SAFE DEEP | armed | Nickel Double Mug | Cover 3 Match | SAFE DEEP | True | True | True | False |
| syn-defense-009__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Double Mug | Cover 3 Match | SAFE DEEP | armed | Nickel Double Mug | Cover 3 Match | SAFE DEEP | True | True | True | False |
| syn-defense-009__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Double Mug | Cover 3 Match | SAFE DEEP | armed | Nickel Double Mug | Cover 3 Match | SAFE DEEP | True | True | True | False |
| syn-defense-010__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 4 Quarters |  |  | Nickel Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-010__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 4 Quarters |  |  | Nickel Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-010__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 4 Quarters |  |  | Nickel Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-010__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 4 Quarters |  |  | Nickel Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-011__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | Free Fire | SAFE DEEP | armed | 4-3 Over | Free Fire | SAFE DEEP | True | True | True | False |
| syn-defense-011__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | Free Fire | SAFE DEEP | armed | 4-3 Over | Free Fire | SAFE DEEP | True | True | True | False |
| syn-defense-011__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | Free Fire | SAFE DEEP | armed | 4-3 Over | Free Fire | SAFE DEEP | True | True | True | False |
| syn-defense-011__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | Free Fire | SAFE DEEP | armed | 4-3 Over | Free Fire | SAFE DEEP | True | True | True | False |
| syn-defense-012__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | LB Sting 1 |  |  | 4-3 Over | LB Sting 1 |  | True | True | True | False |
| syn-defense-012__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | LB Sting 1 |  |  | 4-3 Over | LB Sting 1 |  | True | True | True | False |
| syn-defense-012__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | LB Sting 1 |  |  | 4-3 Over | LB Sting 1 |  | True | True | True | False |
| syn-defense-012__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | LB Sting 1 |  |  | 4-3 Over | LB Sting 1 |  | True | True | True | False |
| syn-defense-013__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 3 Match |  |  | Nickel Over | Cover 3 Match |  | True | True | True | False |
| syn-defense-013__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 3 Match |  |  | Nickel Over | Cover 3 Match |  | True | True | True | False |
| syn-defense-013__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 3 Match |  |  | Nickel Over | Cover 3 Match |  | True | True | True | False |
| syn-defense-013__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 3 Match |  |  | Nickel Over | Cover 3 Match |  | True | True | True | False |
| syn-defense-014__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Dime 3-2 Odd | Cover 3 Cloud | SAFE DEEP | armed | Dime 3-2 Odd | Cover 3 Cloud | SAFE DEEP | True | True | True | False |
| syn-defense-014__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Dime 3-2 Odd | Cover 3 Cloud | SAFE DEEP | armed | Dime 3-2 Odd | Cover 3 Cloud | SAFE DEEP | True | True | True | False |
| syn-defense-014__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Dime 3-2 Odd | Cover 3 Cloud | SAFE DEEP | armed | Dime 3-2 Odd | Cover 3 Cloud | SAFE DEEP | True | True | True | False |
| syn-defense-014__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Dime 3-2 Odd | Cover 3 Cloud | SAFE DEEP | armed | Dime 3-2 Odd | Cover 3 Cloud | SAFE DEEP | True | True | True | False |
| syn-defense-015__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Single Mug | Cover 1 Hole |  |  | Nickel Single Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-015__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Single Mug | Cover 1 Hole |  |  | Nickel Single Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-015__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Single Mug | Cover 1 Hole |  |  | Nickel Single Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-015__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Single Mug | Cover 1 Hole |  |  | Nickel Single Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-016__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Field Sim 3 | SAFE DEEP | armed | Nickel Over | Field Sim 3 | SAFE DEEP | True | True | True | False |
| syn-defense-016__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Field Sim 3 | SAFE DEEP | armed | Nickel Over | Field Sim 3 | SAFE DEEP | True | True | True | False |
| syn-defense-016__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Field Sim 3 | SAFE DEEP | armed | Nickel Over | Field Sim 3 | SAFE DEEP | True | True | True | False |
| syn-defense-016__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Field Sim 3 | SAFE DEEP | armed | Nickel Over | Field Sim 3 | SAFE DEEP | True | True | True | False |
| syn-defense-017__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 4 Quarters | SAFE DEEP | armed | Nickel Over | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-017__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 4 Quarters | SAFE DEEP | armed | Nickel Over | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-017__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 4 Quarters | SAFE DEEP | armed | Nickel Over | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-017__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 4 Quarters | SAFE DEEP | armed | Nickel Over | Cover 4 Quarters | SAFE DEEP | True | True | True | False |
| syn-defense-018__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 4 Quarters |  |  | Nickel Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-018__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 4 Quarters |  |  | Nickel Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-018__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 4 Quarters |  |  | Nickel Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-018__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 4 Quarters |  |  | Nickel Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-019__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Double Mug | Nickel Sim 2 | SAFE DEEP | armed | Nickel Double Mug | Nickel Sim 2 | SAFE DEEP | True | True | True | False |
| syn-defense-019__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Double Mug | Nickel Sim 2 | SAFE DEEP | armed | Nickel Double Mug | Nickel Sim 2 | SAFE DEEP | True | True | True | False |
| syn-defense-019__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Double Mug | Nickel Sim 2 | SAFE DEEP | armed | Nickel Double Mug | Nickel Sim 2 | SAFE DEEP | True | True | True | False |
| syn-defense-019__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Double Mug | Nickel Sim 2 | SAFE DEEP | armed | Nickel Double Mug | Nickel Sim 2 | SAFE DEEP | True | True | True | False |
| syn-defense-020__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Double Mug | Tampa 2 | SAFE DEEP | armed | Nickel Double Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-020__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Double Mug | Tampa 2 | SAFE DEEP | armed | Nickel Double Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-020__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Double Mug | Tampa 2 | SAFE DEEP | armed | Nickel Double Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-020__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Double Mug | Tampa 2 | SAFE DEEP | armed | Nickel Double Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-021__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Nickel Sim 2 | SAFE DEEP | armed | Nickel Over | Nickel Sim 2 | SAFE DEEP | True | True | True | False |
| syn-defense-021__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Nickel Sim 2 | SAFE DEEP | armed | Nickel Over | Nickel Sim 2 | SAFE DEEP | True | True | True | False |
| syn-defense-021__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Nickel Sim 2 | SAFE DEEP | armed | Nickel Over | Nickel Sim 2 | SAFE DEEP | True | True | True | False |
| syn-defense-021__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Nickel Sim 2 | SAFE DEEP | armed | Nickel Over | Nickel Sim 2 | SAFE DEEP | True | True | True | False |
| syn-defense-022__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 1 Hole |  |  | Nickel Over | Cover 1 Hole |  | True | True | True | False |
| syn-defense-022__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 1 Hole |  |  | Nickel Over | Cover 1 Hole |  | True | True | True | False |
| syn-defense-022__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 1 Hole |  |  | Nickel Over | Cover 1 Hole |  | True | True | True | False |
| syn-defense-022__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 1 Hole |  |  | Nickel Over | Cover 1 Hole |  | True | True | True | False |
| syn-defense-023__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | 1 Robber Press | SAFE DEEP | armed | 4-3 Over | 1 Robber Press | SAFE DEEP | True | True | True | False |
| syn-defense-023__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | 1 Robber Press | SAFE DEEP | armed | 4-3 Over | 1 Robber Press | SAFE DEEP | True | True | True | False |
| syn-defense-023__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | 1 Robber Press | SAFE DEEP | armed | 4-3 Over | 1 Robber Press | SAFE DEEP | True | True | True | False |
| syn-defense-023__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | 1 Robber Press | SAFE DEEP | armed | 4-3 Over | 1 Robber Press | SAFE DEEP | True | True | True | False |
| syn-defense-024__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | Cover 3 Cloud Str |  |  | 4-3 Over | Cover 3 Cloud Str |  | True | True | True | False |
| syn-defense-024__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | Cover 3 Cloud Str |  |  | 4-3 Over | Cover 3 Cloud Str |  | True | True | True | False |
| syn-defense-024__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | Cover 3 Cloud Str |  |  | 4-3 Over | Cover 3 Cloud Str |  | True | True | True | False |
| syn-defense-024__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | Cover 3 Cloud Str |  |  | 4-3 Over | Cover 3 Cloud Str |  | True | True | True | False |
| syn-defense-025__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Single Mug | Cover 4 Quarters |  |  | Nickel Single Mug | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-025__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Single Mug | Cover 4 Quarters |  |  | Nickel Single Mug | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-025__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Single Mug | Cover 4 Quarters |  |  | Nickel Single Mug | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-025__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Single Mug | Cover 4 Quarters |  |  | Nickel Single Mug | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-026__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Tampa 2 | SAFE DEEP | armed | Nickel Over | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-026__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Tampa 2 | SAFE DEEP | armed | Nickel Over | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-026__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Tampa 2 | SAFE DEEP | armed | Nickel Over | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-026__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Tampa 2 | SAFE DEEP | armed | Nickel Over | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-027__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-027__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-027__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-027__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-028__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Dime 3-2 Odd | 1 Double WR1 | SAFE DEEP | armed | Dime 3-2 Odd | 1 Double WR1 | SAFE DEEP | True | True | True | False |
| syn-defense-028__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Dime 3-2 Odd | 1 Double WR1 | SAFE DEEP | armed | Dime 3-2 Odd | 1 Double WR1 | SAFE DEEP | True | True | True | False |
| syn-defense-028__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Dime 3-2 Odd | 1 Double WR1 | SAFE DEEP | armed | Dime 3-2 Odd | 1 Double WR1 | SAFE DEEP | True | True | True | False |
| syn-defense-028__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Dime 3-2 Odd | 1 Double WR1 | SAFE DEEP | armed | Dime 3-2 Odd | 1 Double WR1 | SAFE DEEP | True | True | True | False |
| syn-defense-029__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Single Mug | Tampa 2 | SAFE DEEP | armed | Nickel Single Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-029__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Single Mug | Tampa 2 | SAFE DEEP | armed | Nickel Single Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-029__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Single Mug | Tampa 2 | SAFE DEEP | armed | Nickel Single Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-029__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Single Mug | Tampa 2 | SAFE DEEP | armed | Nickel Single Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-030__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 3 Sky |  |  | Nickel Over | Cover 3 Sky |  | True | True | True | False |
| syn-defense-030__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 3 Sky |  |  | Nickel Over | Cover 3 Sky |  | True | True | True | False |
| syn-defense-030__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 3 Sky |  |  | Nickel Over | Cover 3 Sky |  | True | True | True | False |
| syn-defense-030__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 3 Sky |  |  | Nickel Over | Cover 3 Sky |  | True | True | True | False |
| syn-defense-031__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Cover 6 | SAFE DEEP | armed | Nickel Over | Cover 6 | SAFE DEEP | True | True | True | False |
| syn-defense-031__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Cover 6 | SAFE DEEP | armed | Nickel Over | Cover 6 | SAFE DEEP | True | True | True | False |
| syn-defense-031__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Cover 6 | SAFE DEEP | armed | Nickel Over | Cover 6 | SAFE DEEP | True | True | True | False |
| syn-defense-031__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Cover 6 | SAFE DEEP | armed | Nickel Over | Cover 6 | SAFE DEEP | True | True | True | False |
| syn-defense-032__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Double Mug | Tampa 2 | SAFE DEEP | armed | Nickel Double Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-032__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Double Mug | Tampa 2 | SAFE DEEP | armed | Nickel Double Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-032__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Double Mug | Tampa 2 | SAFE DEEP | armed | Nickel Double Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-032__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Double Mug | Tampa 2 | SAFE DEEP | armed | Nickel Double Mug | Tampa 2 | SAFE DEEP | True | True | True | False |
| syn-defense-033__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Over | Field Sim 3 | SAFE DEEP | armed | Nickel Over | Field Sim 3 | SAFE DEEP | True | True | True | False |
| syn-defense-033__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Over | Field Sim 3 | SAFE DEEP | armed | Nickel Over | Field Sim 3 | SAFE DEEP | True | True | True | False |
| syn-defense-033__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Over | Field Sim 3 | SAFE DEEP | armed | Nickel Over | Field Sim 3 | SAFE DEEP | True | True | True | False |
| syn-defense-033__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Over | Field Sim 3 | SAFE DEEP | armed | Nickel Over | Field Sim 3 | SAFE DEEP | True | True | True | False |
| syn-defense-034__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | Cover 4 Quarters |  |  | 4-3 Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-034__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | Cover 4 Quarters |  |  | 4-3 Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-034__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | Cover 4 Quarters |  |  | 4-3 Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-034__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | Cover 4 Quarters |  |  | 4-3 Over | Cover 4 Quarters |  | True | True | True | False |
| syn-defense-035__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | 4-3 Over | Cover 1 Hole | SAFE DEEP | armed | 4-3 Over | Cover 1 Hole | SAFE DEEP | True | True | True | False |
| syn-defense-035__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | 4-3 Over | Cover 1 Hole | SAFE DEEP | armed | 4-3 Over | Cover 1 Hole | SAFE DEEP | True | True | True | False |
| syn-defense-035__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | 4-3 Over | Cover 1 Hole | SAFE DEEP | armed | 4-3 Over | Cover 1 Hole | SAFE DEEP | True | True | True | False |
| syn-defense-035__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | 4-3 Over | Cover 1 Hole | SAFE DEEP | armed | 4-3 Over | Cover 1 Hole | SAFE DEEP | True | True | True | False |
| syn-defense-036__null | defense | synthetic | {"opp_personnel": null, "opp_set": null} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-036__empty | defense | synthetic | {"opp_personnel": "empty", "opp_set": "empty"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-036__11-nickel | defense | synthetic | {"opp_personnel": "11", "opp_set": "nickel"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| syn-defense-036__12-base | defense | synthetic | {"opp_personnel": "12", "opp_set": "base"} | Nickel Double Mug | Cover 1 Hole |  |  | Nickel Double Mug | Cover 1 Hole |  | True | True | True | False |
| obs-0001__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0001__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0001__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0002__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0002__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0002__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0003__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Spot Choice |  |  | Gun Doubles Clamp Stack | Spot Choice |  | True | True | True | False |
| obs-0003__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Spot Choice |  |  | Gun Doubles Clamp Stack | Spot Choice |  | True | True | True | False |
| obs-0003__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Doubles Clamp Stack | Spot Choice |  |  | Gun Doubles Clamp Stack | Spot Choice |  | True | True | True | False |
| obs-0004__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| obs-0004__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| obs-0004__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| obs-0005__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0005__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0005__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0006__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0006__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0006__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0007__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0007__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0007__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0008__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0008__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0008__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0009__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | PA Boot Y Leak |  |  | Pistol Deuce Close | PA Boot Y Leak |  | True | True | True | False |
| obs-0009__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | PA Boot Y Leak |  |  | Pistol Deuce Close | PA Boot Y Leak |  | True | True | True | False |
| obs-0009__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Deuce Close | PA Boot Y Leak |  |  | Pistol Deuce Close | PA Boot Y Leak |  | True | True | True | False |
| obs-0010__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0010__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0010__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0011__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0011__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0011__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0012__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | X Spot | RZ | armed | Pistol Trips | X Spot | RZ | True | True | True | False |
| obs-0012__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | X Spot | RZ | armed | Pistol Trips | X Spot | RZ | True | True | True | False |
| obs-0012__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Pistol Trips | X Spot | RZ | armed | Pistol Trips | X Spot | RZ | True | True | True | False |
| obs-0013__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Draw |  |  | Pistol Trips | HB Draw |  | True | True | True | False |
| obs-0013__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Draw |  |  | Pistol Trips | HB Draw |  | True | True | True | False |
| obs-0013__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | HB Draw |  |  | Pistol Trips | HB Draw |  | True | True | True | False |
| obs-0014__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0014__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0014__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0015__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0015__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0015__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0016__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |
| obs-0016__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |
| obs-0016__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |
| obs-0017__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0017__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0017__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0018__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0018__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0018__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0019__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mesh Spot |  |  | Pistol Deuce Close | Mesh Spot |  | True | True | True | False |
| obs-0019__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mesh Spot |  |  | Pistol Deuce Close | Mesh Spot |  | True | True | True | False |
| obs-0019__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Deuce Close | Mesh Spot |  |  | Pistol Deuce Close | Mesh Spot |  | True | True | True | False |
| obs-0020__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0020__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0020__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0021__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0021__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0021__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0022__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0022__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0022__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0023__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Smash |  |  | Pistol Trips | Smash |  | True | True | True | False |
| obs-0023__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Smash |  |  | Pistol Trips | Smash |  | True | True | True | False |
| obs-0023__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Pistol Trips | Smash |  |  | Pistol Trips | Smash |  | True | True | True | False |
| obs-0024__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0024__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0024__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0025__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0025__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0025__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0026__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| obs-0026__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| obs-0026__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| obs-0027__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0027__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0027__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0028__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh Post |  |  | Gun 5WR Tight | Mesh Post |  | True | True | True | False |
| obs-0028__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh Post |  |  | Gun 5WR Tight | Mesh Post |  | True | True | True | False |
| obs-0028__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun 5WR Tight | Mesh Post |  |  | Gun 5WR Tight | Mesh Post |  | True | True | True | False |
| obs-0029__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Rub Smash |  |  | Gun Doubles Clamp Stack | Mtn Rub Smash |  | True | True | True | False |
| obs-0029__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Rub Smash |  |  | Gun Doubles Clamp Stack | Mtn Rub Smash |  | True | True | True | False |
| obs-0029__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Mtn Rub Smash |  |  | Gun Doubles Clamp Stack | Mtn Rub Smash |  | True | True | True | False |
| obs-0030__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0030__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0030__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0031__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | HB Mid Zone |  |  | Gun Doubles Clamp Stack | HB Mid Zone |  | True | True | True | False |
| obs-0031__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | HB Mid Zone |  |  | Gun Doubles Clamp Stack | HB Mid Zone |  | True | True | True | False |
| obs-0031__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | HB Mid Zone |  |  | Gun Doubles Clamp Stack | HB Mid Zone |  | True | True | True | False |
| obs-0032__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Inside Zone |  |  | Gun Trips X Nasty | Inside Zone |  | True | True | True | False |
| obs-0032__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Inside Zone |  |  | Gun Trips X Nasty | Inside Zone |  | True | True | True | False |
| obs-0032__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Inside Zone |  |  | Gun Trips X Nasty | Inside Zone |  | True | True | True | False |
| obs-0033__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0033__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0033__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0034__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh Post |  |  | Gun 5WR Tight | Mesh Post |  | True | True | True | False |
| obs-0034__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh Post |  |  | Gun 5WR Tight | Mesh Post |  | True | True | True | False |
| obs-0034__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun 5WR Tight | Mesh Post |  |  | Gun 5WR Tight | Mesh Post |  | True | True | True | False |
| obs-0035__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0035__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0035__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0036__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| obs-0036__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| obs-0036__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Pistol Trips | Inside Cross | RZ | armed | Pistol Trips | Inside Cross | RZ | True | True | True | False |
| obs-0037__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Spot Choice | RZ | armed | Gun Doubles Clamp Stack | Spot Choice | RZ | True | True | True | False |
| obs-0037__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Spot Choice | RZ | armed | Gun Doubles Clamp Stack | Spot Choice | RZ | True | True | True | False |
| obs-0037__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | Spot Choice | RZ | armed | Gun Doubles Clamp Stack | Spot Choice | RZ | True | True | True | False |
| obs-0038__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Salem Smash Under |  |  | Gun 5WR Tight | Salem Smash Under |  | True | True | True | False |
| obs-0038__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Salem Smash Under |  |  | Gun 5WR Tight | Salem Smash Under |  | True | True | True | False |
| obs-0038__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Salem Smash Under |  |  | Gun 5WR Tight | Salem Smash Under |  | True | True | True | False |
| obs-0039__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Read Option |  |  | Pistol Trips | Read Option |  | True | True | True | False |
| obs-0039__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Read Option |  |  | Pistol Trips | Read Option |  | True | True | True | False |
| obs-0039__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Trips | Read Option |  |  | Pistol Trips | Read Option |  | True | True | True | False |
| obs-0040__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  | True | True | True | False |
| obs-0040__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  | True | True | True | False |
| obs-0040__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Drive Choice |  | True | True | True | False |
| obs-0041__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0041__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0041__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0042__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Bench |  |  | Pistol Deuce Close | Bench |  | True | True | True | False |
| obs-0042__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Bench |  |  | Pistol Deuce Close | Bench |  | True | True | True | False |
| obs-0042__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Pistol Deuce Close | Bench |  |  | Pistol Deuce Close | Bench |  | True | True | True | False |
| obs-0043__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |
| obs-0043__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |
| obs-0043__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |
| obs-0044__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| obs-0044__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| obs-0044__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | armed | Gun Doubles Clamp Stack | Mtn Choice Gos Y-Shake | RZ | True | True | True | False |
| obs-0045__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0045__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0045__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0046__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0046__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0046__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Pistol Deuce Close | Mesh Spot | RZ | armed | Pistol Deuce Close | Mesh Spot | RZ | True | True | True | False |
| obs-0047__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0047__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0047__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0048__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0048__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0048__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0049__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | PA Boot Lt |  |  | Pistol Trips | PA Boot Lt |  | True | True | True | False |
| obs-0049__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | PA Boot Lt |  |  | Pistol Trips | PA Boot Lt |  | True | True | True | False |
| obs-0049__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | PA Boot Lt |  |  | Pistol Trips | PA Boot Lt |  | True | True | True | False |
| obs-0050__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Zl Spot Shake |  |  | Gun 5WR Tight | Zl Spot Shake |  | True | True | True | False |
| obs-0050__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Zl Spot Shake |  |  | Gun 5WR Tight | Zl Spot Shake |  | True | True | True | False |
| obs-0050__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Zl Spot Shake |  |  | Gun 5WR Tight | Zl Spot Shake |  | True | True | True | False |
| obs-0051__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Slip Screen |  |  | Pistol Trips | HB Slip Screen |  | True | True | True | False |
| obs-0051__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Slip Screen |  |  | Pistol Trips | HB Slip Screen |  | True | True | True | False |
| obs-0051__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Trips | HB Slip Screen |  |  | Pistol Trips | HB Slip Screen |  | True | True | True | False |
| obs-0052__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0052__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0052__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0053__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn FK Rub |  |  | Gun Doubles Clamp Stack | Mtn FK Rub |  | True | True | True | False |
| obs-0053__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn FK Rub |  |  | Gun Doubles Clamp Stack | Mtn FK Rub |  | True | True | True | False |
| obs-0053__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Mtn FK Rub |  |  | Gun Doubles Clamp Stack | Mtn FK Rub |  | True | True | True | False |
| obs-0054__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0054__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0054__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0055__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| obs-0055__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| obs-0055__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| obs-0056__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0056__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0056__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0057__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0057__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0057__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Trips | Slants | RZ | armed | Pistol Trips | Slants | RZ | True | True | True | False |
| obs-0058__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Salem Drag |  |  | Gun Doubles Clamp Stack | Mtn Salem Drag |  | True | True | True | False |
| obs-0058__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Salem Drag |  |  | Gun Doubles Clamp Stack | Mtn Salem Drag |  | True | True | True | False |
| obs-0058__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Doubles Clamp Stack | Mtn Salem Drag |  |  | Gun Doubles Clamp Stack | Mtn Salem Drag |  | True | True | True | False |
| obs-0059__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0059__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0059__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0060__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Spacing |  |  | Gun 5WR Tight | Spacing |  | True | True | True | False |
| obs-0060__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Spacing |  |  | Gun 5WR Tight | Spacing |  | True | True | True | False |
| obs-0060__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun 5WR Tight | Spacing |  |  | Gun 5WR Tight | Spacing |  | True | True | True | False |
| obs-0061__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0061__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0061__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0062__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0062__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0062__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0063__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Pin Pull Sweep |  |  | Pistol Trips | Pin Pull Sweep |  | True | True | True | False |
| obs-0063__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Pin Pull Sweep |  |  | Pistol Trips | Pin Pull Sweep |  | True | True | True | False |
| obs-0063__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Trips | Pin Pull Sweep |  |  | Pistol Trips | Pin Pull Sweep |  | True | True | True | False |
| obs-0064__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0064__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0064__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun 5WR Tight | Mtn Double Post |  |  | Gun 5WR Tight | Mtn Double Post |  | True | True | True | False |
| obs-0065__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0065__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0065__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0066__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| obs-0066__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| obs-0066__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| obs-0067__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0067__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0067__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0068__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0068__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0068__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0069__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | HB Mid Zone |  |  | Gun Doubles Clamp Stack | HB Mid Zone |  | True | True | True | False |
| obs-0069__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | HB Mid Zone |  |  | Gun Doubles Clamp Stack | HB Mid Zone |  | True | True | True | False |
| obs-0069__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Doubles Clamp Stack | HB Mid Zone |  |  | Gun Doubles Clamp Stack | HB Mid Zone |  | True | True | True | False |
| obs-0070__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Zone WK |  |  | Pistol Trips | HB Zone WK |  | True | True | True | False |
| obs-0070__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Zone WK |  |  | Pistol Trips | HB Zone WK |  | True | True | True | False |
| obs-0070__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Pistol Trips | HB Zone WK |  |  | Pistol Trips | HB Zone WK |  | True | True | True | False |
| obs-0071__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0071__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0071__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Pistol Deuce Close | Mtn Counter Lt |  |  | Pistol Deuce Close | Mtn Counter Lt |  | True | True | True | False |
| obs-0072__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0072__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0072__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0073__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0073__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0073__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | Verticals |  |  | Pistol Trips | Verticals |  | True | True | True | False |
| obs-0074__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| obs-0074__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| obs-0074__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mtn Triple Slants | RZ | armed | Gun Trips X Nasty | Mtn Triple Slants | RZ | True | True | True | False |
| obs-0075__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0075__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0075__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0076__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh Post | RZ | armed | Gun 5WR Tight | Mesh Post | RZ | True | True | True | False |
| obs-0076__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh Post | RZ | armed | Gun 5WR Tight | Mesh Post | RZ | True | True | True | False |
| obs-0076__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun 5WR Tight | Mesh Post | RZ | armed | Gun 5WR Tight | Mesh Post | RZ | True | True | True | False |
| obs-0077__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | armed | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | True | True | True | False |
| obs-0077__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | armed | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | True | True | True | False |
| obs-0077__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | armed | Gun Trips X Nasty | Mtn Spot Y Corner | RZ | True | True | True | False |
| obs-0078__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| obs-0078__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| obs-0078__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| obs-0079__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0079__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0079__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun 5WR Tight | Mesh | RZ | armed | Gun 5WR Tight | Mesh | RZ | True | True | True | False |
| obs-0080__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Drag Sluggo |  |  | Gun Trips X Nasty | Mtn Drag Sluggo |  | True | True | True | False |
| obs-0080__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Drag Sluggo |  |  | Gun Trips X Nasty | Mtn Drag Sluggo |  | True | True | True | False |
| obs-0080__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Mtn Drag Sluggo |  |  | Gun Trips X Nasty | Mtn Drag Sluggo |  | True | True | True | False |
| obs-0081__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Quick Fork Zig |  |  | Gun Trips X Nasty | Quick Fork Zig |  | True | True | True | False |
| obs-0081__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Quick Fork Zig |  |  | Gun Trips X Nasty | Quick Fork Zig |  | True | True | True | False |
| obs-0081__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Trips X Nasty | Quick Fork Zig |  |  | Gun Trips X Nasty | Quick Fork Zig |  | True | True | True | False |
| obs-0082__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mtn Zone Split |  |  | Pistol Deuce Close | Mtn Zone Split |  | True | True | True | False |
| obs-0082__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mtn Zone Split |  |  | Pistol Deuce Close | Mtn Zone Split |  | True | True | True | False |
| obs-0082__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Pistol Deuce Close | Mtn Zone Split |  |  | Pistol Deuce Close | Mtn Zone Split |  | True | True | True | False |
| obs-0083__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| obs-0083__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| obs-0083__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  |  | Gun Doubles Clamp Stack | Texas Y-Stutter Wheel |  | True | True | True | False |
| obs-0084__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Quick Short Post |  |  | Gun Trips X Nasty | Quick Short Post |  | True | True | True | False |
| obs-0084__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Quick Short Post |  |  | Gun Trips X Nasty | Quick Short Post |  | True | True | True | False |
| obs-0084__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Trips X Nasty | Quick Short Post |  |  | Gun Trips X Nasty | Quick Short Post |  | True | True | True | False |
| obs-0085__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Slot Fades |  |  | Gun Trips X Nasty | Mtn Slot Fades |  | True | True | True | False |
| obs-0085__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Slot Fades |  |  | Gun Trips X Nasty | Mtn Slot Fades |  | True | True | True | False |
| obs-0085__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mtn Slot Fades |  |  | Gun Trips X Nasty | Mtn Slot Fades |  | True | True | True | False |
| obs-0086__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0086__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0086__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Same Side Zone |  |  | Gun Doubles Clamp Stack | Same Side Zone |  | True | True | True | False |
| obs-0087__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Zl Spot Shake |  |  | Gun 5WR Tight | Zl Spot Shake |  | True | True | True | False |
| obs-0087__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Zl Spot Shake |  |  | Gun 5WR Tight | Zl Spot Shake |  | True | True | True | False |
| obs-0087__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun 5WR Tight | Zl Spot Shake |  |  | Gun 5WR Tight | Zl Spot Shake |  | True | True | True | False |
| obs-0088__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0088__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0088__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0089__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0089__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0089__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0090__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0090__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0090__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0091__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Cluster Bunch |  |  | Gun 5WR Tight | Cluster Bunch |  | True | True | True | False |
| obs-0091__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Cluster Bunch |  |  | Gun 5WR Tight | Cluster Bunch |  | True | True | True | False |
| obs-0091__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun 5WR Tight | Cluster Bunch |  |  | Gun 5WR Tight | Cluster Bunch |  | True | True | True | False |
| obs-0092__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Rub Smash |  |  | Gun Doubles Clamp Stack | Mtn Rub Smash |  | True | True | True | False |
| obs-0092__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Rub Smash |  |  | Gun Doubles Clamp Stack | Mtn Rub Smash |  | True | True | True | False |
| obs-0092__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Mtn Rub Smash |  |  | Gun Doubles Clamp Stack | Mtn Rub Smash |  | True | True | True | False |
| obs-0093__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Slot Fades |  |  | Gun Trips X Nasty | Mtn Slot Fades |  | True | True | True | False |
| obs-0093__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Slot Fades |  |  | Gun Trips X Nasty | Mtn Slot Fades |  | True | True | True | False |
| obs-0093__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Trips X Nasty | Mtn Slot Fades |  |  | Gun Trips X Nasty | Mtn Slot Fades |  | True | True | True | False |
| obs-0094__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh |  |  | Gun 5WR Tight | Mesh |  | True | True | True | False |
| obs-0094__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh |  |  | Gun 5WR Tight | Mesh |  | True | True | True | False |
| obs-0094__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun 5WR Tight | Mesh |  |  | Gun 5WR Tight | Mesh |  | True | True | True | False |
| obs-0095__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0095__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0095__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | armed | Gun Doubles Clamp Stack | Mtn Shuffle Mesh Rail | RZ | True | True | True | False |
| obs-0096__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | L Spot | RZ | armed | Gun 5WR Tight | L Spot | RZ | True | True | True | False |
| obs-0096__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | L Spot | RZ | armed | Gun 5WR Tight | L Spot | RZ | True | True | True | False |
| obs-0096__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun 5WR Tight | L Spot | RZ | armed | Gun 5WR Tight | L Spot | RZ | True | True | True | False |
| obs-0097__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| obs-0097__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| obs-0097__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Mtn Stick Wheel |  |  | Gun Trips X Nasty | Mtn Stick Wheel |  | True | True | True | False |
| obs-0098__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | PA Boot Lt |  |  | Pistol Trips | PA Boot Lt |  | True | True | True | False |
| obs-0098__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | PA Boot Lt |  |  | Pistol Trips | PA Boot Lt |  | True | True | True | False |
| obs-0098__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | PA Boot Lt |  |  | Pistol Trips | PA Boot Lt |  | True | True | True | False |
| obs-0099__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Stretch |  |  | Pistol Trips | HB Stretch |  | True | True | True | False |
| obs-0099__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Stretch |  |  | Pistol Trips | HB Stretch |  | True | True | True | False |
| obs-0099__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Trips | HB Stretch |  |  | Pistol Trips | HB Stretch |  | True | True | True | False |
| obs-0100__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0100__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0100__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Trips X Nasty | WR Smash |  |  | Gun Trips X Nasty | WR Smash |  | True | True | True | False |
| obs-0101__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| obs-0101__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| obs-0101__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| obs-0102__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0102__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0102__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0103__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0103__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0103__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0104__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Off Tackle |  |  | Gun Trips X Nasty | HB Off Tackle |  | True | True | True | False |
| obs-0104__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Off Tackle |  |  | Gun Trips X Nasty | HB Off Tackle |  | True | True | True | False |
| obs-0104__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | HB Off Tackle |  |  | Gun Trips X Nasty | HB Off Tackle |  | True | True | True | False |
| obs-0105__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0105__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0105__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0106__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0106__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0106__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0107__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Counter Y |  |  | Pistol Deuce Close | Counter Y |  | True | True | True | False |
| obs-0107__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Counter Y |  |  | Pistol Deuce Close | Counter Y |  | True | True | True | False |
| obs-0107__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Pistol Deuce Close | Counter Y |  |  | Pistol Deuce Close | Counter Y |  | True | True | True | False |
| obs-0108__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0108__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0108__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0109__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0109__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0109__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0110__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| obs-0110__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| obs-0110__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Mills Double Out |  |  | Gun 5WR Tight | Mills Double Out |  | True | True | True | False |
| obs-0111__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Y Ohio Return |  |  | Gun Doubles Clamp Stack | Y Ohio Return |  | True | True | True | False |
| obs-0111__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Y Ohio Return |  |  | Gun Doubles Clamp Stack | Y Ohio Return |  | True | True | True | False |
| obs-0111__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Doubles Clamp Stack | Y Ohio Return |  |  | Gun Doubles Clamp Stack | Y Ohio Return |  | True | True | True | False |
| obs-0112__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Pin Pull Sweep |  |  | Pistol Trips | Pin Pull Sweep |  | True | True | True | False |
| obs-0112__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Pin Pull Sweep |  |  | Pistol Trips | Pin Pull Sweep |  | True | True | True | False |
| obs-0112__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Pistol Trips | Pin Pull Sweep |  |  | Pistol Trips | Pin Pull Sweep |  | True | True | True | False |
| obs-0113__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Hitch HB Rail |  |  | Gun Doubles Clamp Stack | Hitch HB Rail |  | True | True | True | False |
| obs-0113__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Hitch HB Rail |  |  | Gun Doubles Clamp Stack | Hitch HB Rail |  | True | True | True | False |
| obs-0113__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Hitch HB Rail |  |  | Gun Doubles Clamp Stack | Hitch HB Rail |  | True | True | True | False |
| obs-0114__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0114__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0114__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0115__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Slip Screen |  |  | Pistol Trips | HB Slip Screen |  | True | True | True | False |
| obs-0115__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Slip Screen |  |  | Pistol Trips | HB Slip Screen |  | True | True | True | False |
| obs-0115__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | HB Slip Screen |  |  | Pistol Trips | HB Slip Screen |  | True | True | True | False |
| obs-0116__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return Bench |  |  | Gun Doubles Clamp Stack | Return Bench |  | True | True | True | False |
| obs-0116__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return Bench |  |  | Gun Doubles Clamp Stack | Return Bench |  | True | True | True | False |
| obs-0116__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Return Bench |  |  | Gun Doubles Clamp Stack | Return Bench |  | True | True | True | False |
| obs-0117__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0117__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0117__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0118__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Inside Zone |  |  | Gun Trips X Nasty | Inside Zone |  | True | True | True | False |
| obs-0118__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Inside Zone |  |  | Gun Trips X Nasty | Inside Zone |  | True | True | True | False |
| obs-0118__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Trips X Nasty | Inside Zone |  |  | Gun Trips X Nasty | Inside Zone |  | True | True | True | False |
| obs-0119__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0119__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0119__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0120__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return Bench |  |  | Gun Doubles Clamp Stack | Return Bench |  | True | True | True | False |
| obs-0120__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return Bench |  |  | Gun Doubles Clamp Stack | Return Bench |  | True | True | True | False |
| obs-0120__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Doubles Clamp Stack | Return Bench |  |  | Gun Doubles Clamp Stack | Return Bench |  | True | True | True | False |
| obs-0121__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Stretch |  |  | Pistol Trips | HB Stretch |  | True | True | True | False |
| obs-0121__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Stretch |  |  | Pistol Trips | HB Stretch |  | True | True | True | False |
| obs-0121__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | HB Stretch |  |  | Pistol Trips | HB Stretch |  | True | True | True | False |
| obs-0122__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Zl Spot Shake | RZ | armed | Gun 5WR Tight | Zl Spot Shake | RZ | True | True | True | False |
| obs-0122__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Zl Spot Shake | RZ | armed | Gun 5WR Tight | Zl Spot Shake | RZ | True | True | True | False |
| obs-0122__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Zl Spot Shake | RZ | armed | Gun 5WR Tight | Zl Spot Shake | RZ | True | True | True | False |
| obs-0123__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0123__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0123__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0124__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn HB Swing |  |  | Gun Trips X Nasty | Mtn HB Swing |  | True | True | True | False |
| obs-0124__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn HB Swing |  |  | Gun Trips X Nasty | Mtn HB Swing |  | True | True | True | False |
| obs-0124__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Trips X Nasty | Mtn HB Swing |  |  | Gun Trips X Nasty | Mtn HB Swing |  | True | True | True | False |
| obs-0125__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0125__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0125__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0126__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0126__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0126__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0127__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0127__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0127__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | RPO Alert Out |  |  | Gun Doubles Clamp Stack | RPO Alert Out |  | True | True | True | False |
| obs-0128__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| obs-0128__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| obs-0128__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| obs-0129__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0129__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0129__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0130__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0130__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0130__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Doubles Clamp Stack | Return RPO Zone Alert |  |  | Gun Doubles Clamp Stack | Return RPO Zone Alert |  | True | True | True | False |
| obs-0131__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| obs-0131__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| obs-0131__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun 5WR Tight | Clear Deep |  |  | Gun 5WR Tight | Clear Deep |  | True | True | True | False |
| obs-0132__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Off Tackle |  |  | Gun Trips X Nasty | HB Off Tackle |  | True | True | True | False |
| obs-0132__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Off Tackle |  |  | Gun Trips X Nasty | HB Off Tackle |  | True | True | True | False |
| obs-0132__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Trips X Nasty | HB Off Tackle |  |  | Gun Trips X Nasty | HB Off Tackle |  | True | True | True | False |
| obs-0133__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0133__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0133__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | HB Mid Draw |  |  | Gun Trips X Nasty | HB Mid Draw |  | True | True | True | False |
| obs-0134__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0134__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0134__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Trips X Nasty | Verticals X Shallow |  |  | Gun Trips X Nasty | Verticals X Shallow |  | True | True | True | False |
| obs-0135__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0135__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0135__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0136__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0136__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0136__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Trips X Nasty | Hi Lo Cross |  |  | Gun Trips X Nasty | Hi Lo Cross |  | True | True | True | False |
| obs-0137__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| obs-0137__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| obs-0137__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Trips X Nasty | Hi Lo Cross | RZ | armed | Gun Trips X Nasty | Hi Lo Cross | RZ | True | True | True | False |
| obs-0138__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0138__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0138__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Trips X Nasty | Mtn Stick Wheel | RZ | armed | Gun Trips X Nasty | Mtn Stick Wheel | RZ | True | True | True | False |
| obs-0139__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0139__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0139__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Deuce Close | HB Dive |  |  | Pistol Deuce Close | HB Dive |  | True | True | True | False |
| obs-0140__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mesh |  |  | Gun 5WR Tight | Mesh |  | True | True | True | False |
| obs-0140__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mesh |  |  | Gun 5WR Tight | Mesh |  | True | True | True | False |
| obs-0140__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun 5WR Tight | Mesh |  |  | Gun 5WR Tight | Mesh |  | True | True | True | False |
| obs-0141__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Spot Choice |  |  | Gun Doubles Clamp Stack | Spot Choice |  | True | True | True | False |
| obs-0141__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Spot Choice |  |  | Gun Doubles Clamp Stack | Spot Choice |  | True | True | True | False |
| obs-0141__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Doubles Clamp Stack | Spot Choice |  |  | Gun Doubles Clamp Stack | Spot Choice |  | True | True | True | False |
| obs-0142__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mid Zone |  |  | Gun Trips X Nasty | Mid Zone |  | True | True | True | False |
| obs-0142__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mid Zone |  |  | Gun Trips X Nasty | Mid Zone |  | True | True | True | False |
| obs-0142__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Trips X Nasty | Mid Zone |  |  | Gun Trips X Nasty | Mid Zone |  | True | True | True | False |
| obs-0143__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Shuffle Verts Smash |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Verts Smash |  | True | True | True | False |
| obs-0143__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Shuffle Verts Smash |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Verts Smash |  | True | True | True | False |
| obs-0143__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | Mtn Shuffle Verts Smash |  |  | Gun Doubles Clamp Stack | Mtn Shuffle Verts Smash |  | True | True | True | False |
| obs-0144__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Counter Y |  |  | Gun Doubles Clamp Stack | Counter Y |  | True | True | True | False |
| obs-0144__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Counter Y |  |  | Gun Doubles Clamp Stack | Counter Y |  | True | True | True | False |
| obs-0144__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Doubles Clamp Stack | Counter Y |  |  | Gun Doubles Clamp Stack | Counter Y |  | True | True | True | False |
| obs-0145__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0145__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0145__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Pistol Trips | HB Counter |  |  | Pistol Trips | HB Counter |  | True | True | True | False |
| obs-0146__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Mtn Counter Y |  |  | Gun Doubles Clamp Stack | Mtn Counter Y |  | True | True | True | False |
| obs-0146__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Mtn Counter Y |  |  | Gun Doubles Clamp Stack | Mtn Counter Y |  | True | True | True | False |
| obs-0146__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Mtn Counter Y |  |  | Gun Doubles Clamp Stack | Mtn Counter Y |  | True | True | True | False |
| obs-0147__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0147__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0147__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Doubles Clamp Stack | Return PA Z Over |  |  | Gun Doubles Clamp Stack | Return PA Z Over |  | True | True | True | False |
| obs-0148__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0148__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0148__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Trips X Nasty | HB Slip Screen |  |  | Gun Trips X Nasty | HB Slip Screen |  | True | True | True | False |
| obs-0149__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | Inside Cross |  |  | Pistol Trips | Inside Cross |  | True | True | True | False |
| obs-0149__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | Inside Cross |  |  | Pistol Trips | Inside Cross |  | True | True | True | False |
| obs-0149__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Pistol Trips | Inside Cross |  |  | Pistol Trips | Inside Cross |  | True | True | True | False |
| obs-0150__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Spot Y Corner |  |  | Gun Trips X Nasty | Mtn Spot Y Corner |  | True | True | True | False |
| obs-0150__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Spot Y Corner |  |  | Gun Trips X Nasty | Mtn Spot Y Corner |  | True | True | True | False |
| obs-0150__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Gun Trips X Nasty | Mtn Spot Y Corner |  |  | Gun Trips X Nasty | Mtn Spot Y Corner |  | True | True | True | False |
| obs-0151__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0151__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0151__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Doubles Clamp Stack | HB Slip Screen |  |  | Gun Doubles Clamp Stack | HB Slip Screen |  | True | True | True | False |
| obs-0152__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0152__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0152__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Gun Doubles Clamp Stack | Portland Ywheel |  |  | Gun Doubles Clamp Stack | Portland Ywheel |  | True | True | True | False |
| obs-0153__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Mtn Spot Y Corner |  |  | Gun Trips X Nasty | Mtn Spot Y Corner |  | True | True | True | False |
| obs-0153__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Mtn Spot Y Corner |  |  | Gun Trips X Nasty | Mtn Spot Y Corner |  | True | True | True | False |
| obs-0153__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun Trips X Nasty | Mtn Spot Y Corner |  |  | Gun Trips X Nasty | Mtn Spot Y Corner |  | True | True | True | False |
| obs-0154__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0154__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0154__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Gun Doubles Clamp Stack | Inside Zone |  |  | Gun Doubles Clamp Stack | Inside Zone |  | True | True | True | False |
| obs-0155__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Doubles Clamp Stack | PA Post Shot Choice |  |  | Gun Doubles Clamp Stack | PA Post Shot Choice |  | True | True | True | False |
| obs-0155__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Doubles Clamp Stack | PA Post Shot Choice |  |  | Gun Doubles Clamp Stack | PA Post Shot Choice |  | True | True | True | False |
| obs-0155__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Doubles Clamp Stack | PA Post Shot Choice |  |  | Gun Doubles Clamp Stack | PA Post Shot Choice |  | True | True | True | False |
| obs-0156__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Y Lead Read Option |  |  | Pistol Deuce Close | Y Lead Read Option |  | True | True | True | False |
| obs-0156__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Y Lead Read Option |  |  | Pistol Deuce Close | Y Lead Read Option |  | True | True | True | False |
| obs-0156__13-goal_line | offense | observed_situation,assigned_look | {"opp_personnel": "13", "opp_set": "goal_line"} | Pistol Deuce Close | Y Lead Read Option |  |  | Pistol Deuce Close | Y Lead Read Option |  | True | True | True | False |
| obs-0157__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Quick Fork Zig |  |  | Gun Trips X Nasty | Quick Fork Zig |  | True | True | True | False |
| obs-0157__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Quick Fork Zig |  |  | Gun Trips X Nasty | Quick Fork Zig |  | True | True | True | False |
| obs-0157__11-nickel | offense | observed_situation,assigned_look | {"opp_personnel": "11", "opp_set": "nickel"} | Gun Trips X Nasty | Quick Fork Zig |  |  | Gun Trips X Nasty | Quick Fork Zig |  | True | True | True | False |
| obs-0158__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Trips | PA Boot |  |  | Pistol Trips | PA Boot |  | True | True | True | False |
| obs-0158__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Trips | PA Boot |  |  | Pistol Trips | PA Boot |  | True | True | True | False |
| obs-0158__12-base | offense | observed_situation,assigned_look | {"opp_personnel": "12", "opp_set": "base"} | Pistol Trips | PA Boot |  |  | Pistol Trips | PA Boot |  | True | True | True | False |
| obs-0159__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0159__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0159__21-dime | offense | observed_situation,assigned_look | {"opp_personnel": "21", "opp_set": "dime"} | Gun 5WR Tight | Mtn Post Swirl |  |  | Gun 5WR Tight | Mtn Post Swirl |  | True | True | True | False |
| obs-0160__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Pistol Deuce Close | Mtn Zone Split |  |  | Pistol Deuce Close | Mtn Zone Split |  | True | True | True | False |
| obs-0160__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Pistol Deuce Close | Mtn Zone Split |  |  | Pistol Deuce Close | Mtn Zone Split |  | True | True | True | False |
| obs-0160__10-dollar | offense | observed_situation,assigned_look | {"opp_personnel": "10", "opp_set": "dollar"} | Pistol Deuce Close | Mtn Zone Split |  |  | Pistol Deuce Close | Mtn Zone Split |  | True | True | True | False |
| obs-0161__null | offense | observed_situation,assigned_look | {"opp_personnel": null, "opp_set": null} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |
| obs-0161__empty | offense | observed_situation,assigned_look | {"opp_personnel": "empty", "opp_set": "empty"} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |
| obs-0161__22-quarter | offense | observed_situation,assigned_look | {"opp_personnel": "22", "opp_set": "quarter"} | Gun Trips X Nasty | Switch HB Wheel |  |  | Gun Trips X Nasty | Switch HB Wheel |  | True | True | True | False |

## Measured run

These lines are the run that produced this file. Counts above are the scorer's. Nothing here is an effectiveness claim.

- branch: `madden-ml/eval-run-55-56`
- code head at the start of the run: `eddde60aae45b30aeae90c93f0ac3a72901725c9`
- base `madden-ml/coordinators`: `60cb88eb936503d0334616681f6de43c263dfb3e`
- PR 55 head: `004a541117f50e59fe1c0e2d3dc0f9f7173fb43a`
- PR 56 head: `e8d566075b7bbdb5428071764a29be50cfa4af85`
- scored mode: `heuristic`
- new-mode counts: `{"heuristic": 771}`
- scored_on counts: `{"unknown": 771}`

### Hybrid probe

One fresh-database call, not included in the decision rows.

```json
{
  "counted_as_hybrid": false,
  "fallback_reason": "circuit_open",
  "formation_and_play_match_old": true,
  "included_in_scored_rows": false,
  "mode_downgrade_reason": "hybrid refused: scored_on is unknown",
  "model_eligible": false,
  "model_version": null,
  "new_formation": "Pistol Deuce Close",
  "new_mode": "hybrid",
  "new_play": "Inside Zone Split",
  "old_formation": "Pistol Deuce Close",
  "old_play": "Inside Zone Split",
  "requested_mode": "hybrid",
  "rule": "Hybrid counts only when this fresh-database probe has scored_on reviewed, a model version, and no fallback. Otherwise the scored grid stays on the requested mode.",
  "scored_on": "unknown",
  "seed": 2943045967,
  "side": "offense",
  "situation": "2&2 my 12 q2 clock 10:00 score 17-20"
}
```

### Situation and look check

```json
{
  "bound_empty": 233,
  "bound_null": 233,
  "null_and_empty_crossed": 0,
  "sent_empty_pairs": 233,
  "sent_null_pairs": 233
}
```

### Installed book used for the hard-fail check

```json
{
  "active_playbook_json": {
    "defense": {
      "formations": [
        "Dime 3-2 Odd",
        "Nickel Double Mug",
        "Nickel Single Mug",
        "4-3 Over",
        "Nickel Over"
      ],
      "mode": "custom",
      "name": "PRIMARY META D (custom)",
      "source_book": "49ers"
    },
    "offense": {
      "formations": [
        "Gun Doubles Clamp Stack",
        "Gun 5WR Tight",
        "Gun Trips X Nasty",
        "Pistol Deuce Close",
        "Pistol Trips"
      ],
      "mode": "custom",
      "name": "PRIMARY META O (custom)",
      "source_book": "Buccaneers"
    }
  },
  "plays_by_formation": {
    "defense": {
      "4-3 Over": [
        "1 Hole Press",
        "1 Robber Press",
        "Cover 1 Contain",
        "Cover 1 Hole",
        "Cover 2 Hard Flat",
        "Cover 2 Man",
        "Cover 3 Cloud Str",
        "Cover 3 Sky",
        "Cover 4 Press",
        "Cover 4 Quarters",
        "Cover 6",
        "DBL Safety Blitz",
        "Free Fire",
        "LB Sting 1",
        "OLB Fire Man",
        "Sam Edge 3",
        "Tampa 2",
        "Thunder Smoke"
      ],
      "Dime 3-2 Odd": [
        "1 Double WR1",
        "1 Double WR2",
        "Cover 1 Robber Press",
        "Cover 2 Man",
        "Cover 3 Buzz Match WK",
        "Cover 3 Cloud",
        "Cover 3 Hard Flat",
        "Cover 3 Lock",
        "Cover 4 Drop",
        "Cover 4 Palms",
        "Cover 4 Quarters",
        "Cover 6",
        "Cover 6 Willie",
        "DB Fire 2",
        "DBL Safety Go",
        "Double Bracket",
        "Edge Blitz 1",
        "Edge Blitz 3",
        "FS Zone Blitz",
        "Hot Blitz 3",
        "Okie Eagle 3",
        "Okie Roll 2",
        "Overload 3 Seam",
        "SS Zone Blitz",
        "Tampa 2",
        "Tampa 2 Drop",
        "Zero Blitz"
      ],
      "Nickel Double Mug": [
        "2 Invert Hard Flat",
        "Blitz Loop 3",
        "Coffee House 3 Seam",
        "Cover 1 Hole",
        "Cover 2 Invert",
        "Cover 3 Match",
        "Cover 3 Sky",
        "Field Sim 3",
        "LB Blitz 3",
        "Mid Blitz 0",
        "Nickel 2 Trap",
        "Nickel Overload 3",
        "Nickel Sim 2",
        "SS Blitz 3",
        "Tampa 2"
      ],
      "Nickel Over": [
        "1 Double Slot",
        "1 Double TE",
        "1 Double WR1",
        "1 Double WR2",
        "1 Robber Press",
        "Bracket Switch Willie",
        "Cov 3 Buzz Match",
        "Cover 1 Contain",
        "Cover 1 Hole",
        "Cover 2 Invert",
        "Cover 2 Man",
        "Cover 2 Roll",
        "Cover 3 Buzz Match",
        "Cover 3 Cloud",
        "Cover 3 Match",
        "Cover 3 Sky",
        "Cover 4 Drop Field",
        "Cover 4 Palms",
        "Cover 4 Quarters",
        "Cover 6",
        "Cover 6 Willie",
        "Cover 9 Show 2",
        "DBL Safety Blitz",
        "Double Bracket",
        "Double Bracket Switch",
        "Edge Blitz 0",
        "Edge Blitz 3",
        "Field Sim 3",
        "FS Will Blitz 0",
        "LB Blitz 1",
        "Nickel 2 Trap",
        "Nickel Blitz 3",
        "Nickel Sim 2",
        "Over Storm Brave",
        "Overload 3 Seam",
        "Overload Blitz",
        "Roll 23 Cloud",
        "Roll Cover 2 Invert",
        "Roll Cover 3 Cloud",
        "Show SS Sim 2",
        "Sim Pressure 3",
        "Tampa 2"
      ],
      "Nickel Single Mug": [
        "1 LB Dog",
        "Blitz Tex 3 Sim 3",
        "Cover 1 Hole",
        "Cover 2 Invert",
        "Cover 3",
        "Cover 3 Buzz Match",
        "Cover 4 Quarters",
        "DB Blitz Tex 3",
        "DT Mike Loop 3",
        "Nickel 2 Trap",
        "Nickel Blitz 3",
        "Tampa 2"
      ]
    },
    "offense": {
      "Gun 5WR Tight": [
        "Clear Deep",
        "Cluster Bunch",
        "L Spot",
        "Mesh",
        "Mesh Post",
        "Mills Double Out",
        "Mtn Double Post",
        "Mtn Fork Salem",
        "Mtn Post Swirl",
        "QB Draw",
        "Salem Smash Under",
        "Slot Dig Wide Corner",
        "Smash Dagger",
        "Spacing",
        "Zl Spot Shake"
      ],
      "Gun Doubles Clamp Stack": [
        "Counter Y",
        "Curl Ohio",
        "Flood Seam",
        "HB Mid Zone",
        "HB Slip Screen",
        "Hitch HB Rail",
        "Inside Zone",
        "Mtn Choice Gos Y-Shake",
        "Mtn Counter Y",
        "Mtn FK Rub",
        "Mtn Flood HB Sneak",
        "Mtn Rub Smash",
        "Mtn Salem Drag",
        "Mtn Shuffle Drive Choice",
        "Mtn Shuffle Mesh Rail",
        "Mtn Shuffle Verts Smash",
        "PA Post Shot Choice",
        "Portland Ywheel",
        "Return Bench",
        "Return PA Z Over",
        "Return RPO Zone Alert",
        "RPO Alert Out",
        "Same Side Zone",
        "Spot Choice",
        "Texas Smash",
        "Texas Y-Stutter Wheel",
        "Y Ohio Return"
      ],
      "Gun Trips X Nasty": [
        "HB Mid Draw",
        "HB Off Tackle",
        "HB Slip Screen",
        "Hi Lo Cross",
        "Hi Lo Dig",
        "Inside Zone",
        "Mid Zone",
        "Mtn Drag Sluggo",
        "Mtn HB Swing",
        "Mtn Slot Fades",
        "Mtn Spot Y Corner",
        "Mtn Stick Wheel",
        "Mtn Triple Slants",
        "PA X Cross Shot",
        "Quick Fork Zig",
        "Quick Short Post",
        "Quick Unders",
        "Slot Fade H Wheel",
        "Switch HB Wheel",
        "Verticals X Shallow",
        "WR Smash"
      ],
      "Pistol Deuce Close": [
        "Bench",
        "Counter Y",
        "HB Dive",
        "HB Stretch",
        "Inside Zone Split",
        "Mesh Spot",
        "Mtn Counter Lt",
        "Mtn PA Flood",
        "Mtn Zone Split",
        "PA Boot Y Leak",
        "PA Z Over",
        "Y Lead Read Option"
      ],
      "Pistol Trips": [
        "HB Counter",
        "HB Dive",
        "HB Draw",
        "HB Slip Screen",
        "HB Stretch",
        "HB Zone WK",
        "Inside Cross",
        "PA Boot",
        "PA Boot Lt",
        "PA HB Flat",
        "Pin Pull Sweep",
        "Read Option",
        "RPO Alert Bubble",
        "Slants",
        "Smash",
        "Strong Power",
        "Verticals",
        "X Spot"
      ]
    }
  }
}
```

### Real call differences

A real call difference is a scored row whose new formation or new play is not the old caller formation or play. Adjustment-only rows are counted separately. Display lines are not compared.

- real call differences (formation or play): 0
- signature differences (formation, play, or adjustment): 0
- adjustment-only differences: 0
- same-seed mismatches: 0

No paired examples: no scored row had a different formation or play.

### Adapter

Git merges of PR 55 and PR 56 had no conflicts. eval_scoring.evaluate passes the situation dict into adapt_run_situation. eval_harness.run_situation parses a situation line, and its look normalizer stringifies a mapping, so NULL and empty would not stay distinct. On this branch only, adapt_run_situation passes the row's raw line and one look token: both fields missing is NULL, the empty/empty pair is empty, and any other pair passes the set token when it is present. The requested personnel/set pair remains on opp_look_sent.

### Latency pass

This is `eval_harness.run_cli` over every scored decision row, 50 passes each, same raw line, side, look token, and seed. The latency section above is the scorer's first call per row, not this pass.

```json
{
  "budget_ms": 150,
  "cli_exit": 0,
  "fail_reasons": [],
  "fallback_counts": {},
  "model_ms": {
    "max": 0.0,
    "p50": 0.0,
    "p95": 0.0
  },
  "pass": true,
  "passes": 50,
  "reproduced": true,
  "samples": 38550,
  "situations": 771,
  "whole_call_ms": {
    "max": 26.69617699984883,
    "p50": 4.2675889999372885,
    "p95": 4.799972999990132
  }
}
```

- whole p50/p95/max: 4.2675889999372885 / 4.799972999990132 / 26.69617699984883
- model p50/p95/max: 0.0 / 0.0 / 0.0
- fallback counts: {}
- 150 ms budget (harness pass flag, model time): pass
