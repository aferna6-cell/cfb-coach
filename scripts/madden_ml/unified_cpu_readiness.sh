#!/usr/bin/env bash
# Owner-machine entry point: steps 1-5 (stops before physical installation). See docs/madden_ml/unified_v1_first_cpu_game/README.md
set -euo pipefail
SRC="${1:?usage: unified_cpu_readiness.sh <source-db> [workdir]}"
W="${2:-$HOME/unified_v1_cpu1}"
export PYTHONPATH="${PYTHONPATH:-.}"
python3 -m cfb_coach.madden.unified local-prep --source-db "$SRC" --workdir "$W" --opponent cpu
echo
echo "NEXT (physical): build the books in Madden per $W/prep/INSTALL_GUIDE.md, check them, then run confirm-install for each side and local-finish."
