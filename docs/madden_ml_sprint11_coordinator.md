# Sprint 11 — Adaptive AI Offensive Coordinator

Quick reference. Full designer/action history: `docs/madden_ml_offense_designer.md`.

## Goal

Before the game the model chooses a coherent formation portfolio. During the
game it chooses legal `(formation, play, adjustment_plan)` tuples with football
awareness and evolving opponent evidence. After the game, verified outcomes
inform future strategy. The model—not a static heuristic menu—is the primary
offensive decision-maker when experimental mode is enabled.

## Ubuntu CPU test commands

```bash
cd /path/to/cfb-coach
python3 -m pip install -e '.[dev]'

# Unit / integration (deterministic)
PYTHONPATH=. python3 -m pytest tests/madden_ml/test_sprint11_adaptive_coordinator.py -q
PYTHONPATH=. python3 -m pytest tests/madden_ml -q

# Designer dry-run (no DB mutation of applied book)
PYTHONPATH=. python3 -m cfb_coach ml offense-design -o cpu --text --max-formations 5

# Macro lab drafts only
PYTHONPATH=. python3 -m cfb_coach ml offense-macro-lab --compose

# Experimental enable + heuristic rollback
PYTHONPATH=. python3 -m cfb_coach ml experimental
PYTHONPATH=. python3 -m cfb_coach ml heuristic
```

## Live UX

Primary line remains:

`FORMATION — PLAY`

Executable adjustment instructions (if any) appear immediately below.
Detailed probabilities, joint candidates and diagnostics stay collapsed /
postgame.

Latency budget: **150 ms** full path; timeout → heuristic fallback.

## Provenance rules (non-negotiable)

- Suggested ≠ executed play
- Suggested ≠ applied adjustment
- Last-snap / tendency coverage ≠ current coverage certainty
- Draft macro ≠ verified ≠ armed
- Staged playbook ≠ applied playbook
EOF