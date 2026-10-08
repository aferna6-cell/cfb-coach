# Madden ML Experimental Pilot (Sprint 4)

Opt-in experimental offense selection. **Not** a validated competitive model.
Disabled by default. Do not merge without authorization.

## Enable / disable

```bash
# Train (or retrain) a prior-driven / empirical-light artifact and enable:
python -m cfb_coach ml experimental

# Immediately restore the heuristic coach:
python -m cfb_coach ml heuristic
# or:
python -m cfb_coach ml experimental --off
```

## Historical DB audit (never invents games)

```bash
python -m cfb_coach ml find-db
python -m cfb_coach ml backup-db
python -m cfb_coach ml audit-history
python -m cfb_coach ml inspect
```

Retrospective confirmation (requires user evidence — never auto-elevates):

```bash
python -m cfb_coach ml confirm-execution \
  --snap-id '<ml_snap_id>' \
  --formation 'Gun Bunch' \
  --play 'Mesh' \
  --evidence 'recording at 12:04 — confirmed Mesh'
```

## Laptop preflight (real Franchise DB)

The agent environment has no Franchise history. On the laptop that logged games:

```bash
python -m cfb_coach ml find-db
python -m cfb_coach ml backup-db
python -m cfb_coach ml audit-history
python -m cfb_coach ml inspect
python -m cfb_coach ml experimental-preflight
# equivalent stepwise:
python -m cfb_coach ml train-experimental --seed 7 --install
# then explicit opt-in (preflight leaves mode=heuristic):
python -m cfb_coach ml experimental --retrain
```

`experimental-preflight` prints `supervised_count`, `discounted_prior_count`,
`knowledge_version`, `model_version`, and `active_artifact_path`.

**Do not claim the eight historical games trained the model** until the laptop
audit shows those Franchise sessions contributed supervised rows
(`n_supervised > 0` from that DB). An empty/agent DB trains prior-driven only.

## Train / compare experimental model

```bash
python -m cfb_coach ml train-experimental --seed 7 --install
python -m cfb_coach ml postgame-experimental
```

## Research refresh (candidate only)

```bash
# Standalone (also works once research-only PR lands on main):
PYTHONPATH=. python scripts/madden_research_refresh.py
PYTHONPATH=. python scripts/madden_research_refresh.py --dry-run
# Experimental branch ml CLI:
python -m cfb_coach ml research-refresh
```

Scheduled workflow file: `.github/workflows/madden-research-daily.yml`  
**Not operational for unattended daily runs** until installed on the default
branch (`main`) and a successful Actions run is confirmed. Deploy path that
does **not** merge experimental gameplay: research-only PR #22 /
`docs/madden_research_deploy.md`.

Candidates land under `research/candidates/`. Active five formations and armed
Custom Adjustments are **never** overwritten without user confirmation.

## Live play with experimental ML

```bash
python -m cfb_coach ml experimental
python -m cfb_coach play --game madden27 --opponent <cpu_id>
```

The HTML pad shows the heuristic pick alongside the ML pick, evidence quality,
and falls back to heuristic on timeout/error/illegal candidates.
