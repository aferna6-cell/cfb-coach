import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_eval_script_refuses_the_live_directory_and_needs_a_copy(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "scripts/madden_ml/football_priors_eval.py"), "--db", str(Path.home() / ".cfb-coach" / "madden27.db"), "--out", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "COPY" in r.stderr
