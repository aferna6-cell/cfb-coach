# Deploying Madden daily research without merging experimental gameplay

## Status

Scheduled GitHub Actions workflows run **only from the default branch** (`main`).
Having `.github/workflows/madden-research-daily.yml` on `madden-ml/experimental-pilot`
does **not** enable unattended daily refresh.

Do **not** claim daily research is operational until:

1. The workflow (plus `cfb_coach/madden/research_refresh.py` and
   `scripts/research_refresh_gha_outputs.py`) is on `main`, and
2. A `workflow_dispatch` (or schedule) run succeeds on that branch.

## Recommended deploy path (no experimental gameplay merge)

Open a **research-only** PR to `main` that contains only:

- `.github/workflows/madden-research-daily.yml`
- `cfb_coach/madden/research_refresh.py`
- `scripts/madden_research_refresh.py` (standalone entry — main has no `ml` CLI)
- `scripts/research_refresh_gha_outputs.py`
- `docs/madden_research_deploy.md` (this file)
- optional: `research/candidates/.gitkeep`

Example (from a clean checkout of `main`):

```bash
git fetch origin main madden-ml/experimental-pilot
git checkout -b cursor/madden-research-workflow-38d5 origin/main
git checkout origin/madden-ml/experimental-pilot -- \
  .github/workflows/madden-research-daily.yml \
  cfb_coach/madden/research_refresh.py \
  scripts/madden_research_refresh.py \
  scripts/research_refresh_gha_outputs.py \
  docs/madden_research_deploy.md \
  research/candidates/.gitkeep
# Ensure research_refresh imports stay valid on main (ai_research already exists).
git commit -m "Add Madden daily research candidate workflow (no gameplay changes)"
git push -u origin cursor/madden-research-workflow-38d5
# Open PR → main; merge after review
```

Then on `main`:

1. Actions → **madden-research-daily** → **Run workflow**
2. Confirm the job writes `research/candidates/*` and opens a review PR when meaningful
3. Confirm `research/madden27.json` was **not** overwritten

## On-demand (any branch)

```bash
# Works on main (no experimental gameplay required):
PYTHONPATH=. python scripts/madden_research_refresh.py
PYTHONPATH=. python scripts/madden_research_refresh.py --dry-run
python scripts/research_refresh_gha_outputs.py /tmp/research_refresh.json

# On experimental-pilot only (ml CLI present):
python -m cfb_coach ml research-refresh
```

Candidate updates never silently replace the active five formations or armed
Custom Adjustments.
