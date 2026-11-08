"""Check actual live imports in a clean interpreter, independent of test order."""
import subprocess
import sys
import textwrap


def test_live_make_call_and_install_lock_never_import_full_catalog(tmp_path):
    program = r"""
import builtins
import json
import random
import sys
from types import SimpleNamespace
from pathlib import Path

forbidden = {
    "cfb_coach.madden.model.offense_score_cache",
    "cfb_coach.madden.model.offense_catalog_index",
    "cfb_coach.madden.model.offense_portfolio",
}
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    targets = {name} | {name + "." + item for item in fromlist or ()}
    assert not targets & forbidden, "Live imported full-universe machinery: " + repr(targets & forbidden)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import

from cfb_coach.db import CoachDB
from cfb_coach.live_server import LivePlayController
from cfb_coach.madden import playbook
from cfb_coach.madden.playcaller import make_call
from cfb_coach.madden.situation import Situation
from cfb_coach.madden.model.offense_adjustment_designer import validate_live_call
# validate_live_call lazily imports designer for the macro approval helper.
from cfb_coach.madden.model import offense_designer
assert not forbidden & sys.modules.keys()
db = CoachDB(Path(sys.argv[1]))
forms = {"Gun Trips X Nasty": ["Hi Lo Cross", "Inside Zone"]}
db.set_meta(playbook.META_KEY, json.dumps({"offense": {"formations": forms}}))
call = make_call(Situation(raw="1&10", side="offense", down=1, distance=10, yardline=35),
                 "cpu", db=db, rng=random.Random(1), live_macros=False)
assert call.play in forms[call.formation]
validate_live_call(db, SimpleNamespace(formation=call.formation, play=call.play), opponent_id="cpu")
try:
    validate_live_call(db, SimpleNamespace(formation=call.formation, play="Not Installed"), opponent_id="cpu")
except ValueError:
    pass
else:
    raise AssertionError("Installed lock was not enforced")
assert not forbidden & sys.modules.keys()
db.close()
"""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(program), str(tmp_path / "live.db")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
