# Madden 27 adaptive offensive coordinator

Sprint 11 extends the existing experimental model-primary offense. It does not
create a second playcaller. Defense, CFB behavior, the 150 ms rollback guard,
and heuristic emergency fallback remain unchanged.

## Autonomy boundary

The coordinator may:

- evaluate the complete supported formation catalog and propose a portfolio;
- score every situation-eligible play in the confirmed installed book;
- jointly choose a formation, play, and legal adjustment plan;
- propose new one- or two-primitive Custom Adjustment blueprints;
- learn a bounded observational action tie-break only from verified execution,
  explicit applied/unchanged confirmation, and multiple games.

It may not:

- edit Madden, silently apply a proposed playbook, or infer installation;
- verify or arm its own macro;
- treat a previous coverage as the current coverage;
- train from a recommendation, unverified result, or post-snap fact that was
  unavailable when the decision was made;
- control a human-opponent offense without the per-session opt-in.

## Pregame-to-postgame workflow (Ubuntu)

Use an isolated/test database, not a live Franchise database.

```bash
cd /path/to/cfb-coach
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

# Train and explicitly enable the CPU experimental offense.
.venv/bin/python -m cfb_coach ml experimental --retrain

# Preview the model's whole-formation portfolio in the browser.
.venv/bin/python -m cfb_coach ml offense-design -o cpu --max-formations 12

# Stage only. This does not change the applied book.
.venv/bin/python -m cfb_coach ml offense-design -o cpu --max-formations 12 --stage

# After physically building every formation/play in Madden, use the proposal id.
.venv/bin/python -m cfb_coach ml offense-design -o cpu \
  --confirm-installed PROPOSAL_ID \
  --attest "I installed and checked every proposed formation and play inside Madden."

# Generate and stage source-backed macro ideas. They remain drafts.
.venv/bin/python -m cfb_coach ml offense-macro-lab --limit 16 --stage

# Verify settings without arming a slot.
.venv/bin/python -m cfb_coach ml offense-design -o cpu \
  --verify-macro-config MACRO_NAME \
  --attest "I checked every listed setting and confirmed this configuration is supported in Madden."

# Only after occupying a real slot, attest that it is armed.
.venv/bin/python -m cfb_coach ml offense-design -o cpu \
  --verify-macro MACRO_NAME \
  --attest "I built this Custom Adjustment with the sourced settings and armed it in Madden."

# Audit installed plays, action states, and readiness.
.venv/bin/python -m cfb_coach ml offense-inventory -o cpu
.venv/bin/python -m cfb_coach ml offense-actions -o cpu

# Run CPU offense-only live calling.
.venv/bin/python -m cfb_coach play --game madden27 -o cpu

# Postgame reporting and action evidence remain provenance-gated.
.venv/bin/python -m cfb_coach ml postgame-experimental --game-id GAME_ID
.venv/bin/python -m cfb_coach ml offense-action-learn --train --details
.venv/bin/python -m cfb_coach ml offense-action-learn --promote

# Immediate rollback for subsequent calls.
.venv/bin/python -m cfb_coach ml heuristic
```

## Decision behavior

The live baseline remains the learned play success probability. An explicit
football proxy then accounts for down/distance, conversion distance, field
zone, known score/clock objectives, and the credibility of pre-snap evidence.
It is labeled as a proxy—not expected points or win probability.

The joint search constructs at least `NO_ADJUSTMENT` for every eligible play.
It may also construct sourced single actions, compatible two-action manual
plans, and verified-and-armed macros. Conflicting receiver targets, multiple
protection states, unverifiable controls, incompatible formation/play pairs,
draft macros, and unarmed verified macros are rejected. Adjustment influence
is bounded so sparse action evidence cannot overwhelm the learned play model.

Historical opponent observations are stored as tendencies with sample size and
confidence. They may alter a small strategy prior but never become a fabricated
current-snap coverage observation.

## Capability status

- Implemented: complete-formation pregame evaluation, staged install/rollback,
  situation evaluator, complete-book joint candidates, compatible two-action
  plans, game-session tendency memory, compact live calls, explicit execution
  provenance, conservative action learning, deterministic regression tests.
- Shadow/opt-in: experimental offense and observational action-score learning.
- Unverified until user action: proposed playbook installation, suggested
  audibles, generated macro settings, macro editor compatibility and armed slot.
- Future: causal adjustment-effect estimates, supported audible/motion execution
  provenance, and validated expected-points/win-probability heads.

The two reported blowout CPU victories are not fixtures in this repository.
Without the user's local logs and verified execution/outcome records, they
cannot be audited or represented as regression evidence. Sprint 11 tests use
explicitly synthetic scenarios.
