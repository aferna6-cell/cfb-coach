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

### First experimental game execution recovery (Sprint 4.2)

HTML now defaults **What I ran** to `used_recommended` and restores that after
each submit (never resets to `unknown`). Fully specified `used_different`
executions verify correctly.

One-time recovery for game `141d4a16b4184ee7` (Lions CPU experimental game).
Dry-run by default; requires explicit attestation + `--apply`. Backs up via
the SQLite backup API. Never invents unlinked outcomes; never touches other
Franchise games.

```bash
# Preview (no writes):
python -m cfb_coach ml recover-game-execution \
  --attest 'I followed the final displayed recommendation on every play in game 141d4a16b4184ee7, including Texas Y-Stutter Wheel on snap 0010'

# Apply on the laptop DB only (after reviewing the preview):
python -m cfb_coach ml recover-game-execution \
  --attest 'I followed the final displayed recommendation on every play in game 141d4a16b4184ee7, including Texas Y-Stutter Wheel on snap 0010' \
  --apply

# Verify + retrain:
python -m cfb_coach ml postgame-experimental --game-id 141d4a16b4184ee7
python -m cfb_coach ml inspect
python -m cfb_coach ml train-experimental --seed 7 --install
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
