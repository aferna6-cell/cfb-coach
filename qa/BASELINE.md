# Test baseline at 2def811

Recorded on branch `madden-ml/qa`, cut from `2def811b7d3d7a15f0aafc8ebdf271e07d2e711d` (`Use a trained VOD model as the primary call prior (#18)`). No files under `cfb_coach/` or `tests/` were modified.

## Environment

| Item | Value |
| --- | --- |
| Absolute repo path | `/workspace` |
| Commit | `2def811b7d3d7a15f0aafc8ebdf271e07d2e711d` |
| Virtualenv | `/workspace/.venv` (`python3 -m venv /workspace/.venv`) |
| Python | `Python 3.12.3` — `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]` |
| Python executable | `/workspace/.venv/bin/python` |
| pip | `pip 26.2.1 from /workspace/.venv/lib/python3.12/site-packages/pip (python 3.12)` |
| pytest | `pytest 9.1.1` |
| Install | `/workspace/.venv/bin/python -m pip install -e '.[dev]'` (dev extra is pytest only) |

`python3` in this venv is `/workspace/.venv/bin/python3`, a symlink to `/usr/bin/python3`. `sys.prefix` is `/workspace/.venv`.

`numpy` is not installed. `cfb_coach/vod_success/glm.py` imports it at module level. The `dev` extra does not depend on numpy (`pyproject.toml` lists `dev = ["pytest"]`).

## Required pytest command

Working directory: `/workspace`. Venv activated so `python` is `/workspace/.venv/bin/python`.

```bash
PYTHONPATH=. python -m pytest tests -p no:cacheprovider -rA --durations=25 --junitxml=qa/baseline_2def811.xml
```

Run twice. `--durations=25` printed no duration table because no tests executed.

### Run 1

- Start: `2026-10-08T13:04:31.232532974Z`
- End: `2026-10-08T13:04:31.911017926Z`
- Wall-clock: `0.678485` seconds
- pytest's own line: `1 error in 0.52s`
- Exit code: `2`

### Run 2

- Start: `2026-10-08T13:06:13.682894325Z`
- End: `2026-10-08T13:06:14.056566926Z`
- Wall-clock: `0.373673` seconds
- pytest's own line: `1 error in 0.22s`
- Exit code: `2`
- JUnit file committed from this run: `qa/baseline_2def811.xml` (1331 bytes)

## Required pytest counts

Both runs printed `collected 320 items / 1 error` and then `Interrupted: 1 error during collection`. The session exited before any test ran.

| Count | Value |
| --- | --- |
| collected | 320 |
| passed | 0 |
| failed | 0 |
| skipped | 0 |
| errors | 1 |
| xfail | 0 |
| xpass | 0 |

JUnit attributes on the committed run-2 file: `tests="1"` `errors="1"` `failures="0"` `skipped="0"`. The XML contains only the collection error. The 320 collected tests are not in the file because they were not executed.

## Failure and error list (required pytest)

1. `tests/test_vod_success.py` — `ModuleNotFoundError: No module named 'numpy'` (`tests/test_vod_success.py` imports `cfb_coach.vod_success.features`, which imports `cfb_coach.vod_success.glm`, which imports numpy).

No other node failed or errored. `tests/test_watch_live_ux.py::TestDemoStillWorks::test_demo_once_exits_clean` was not executed.

## Known test: `test_demo_once_exits_clean`

Node id: `tests/test_watch_live_ux.py::TestDemoStillWorks::test_demo_once_exits_clean`

The test hard-codes `cwd="/workspace/cfb-coach"` and `PYTHONPATH=/workspace/cfb-coach` (`tests/test_watch_live_ux.py` lines 299–306).

It does **not** fail in the required pytest command above, because that command aborts during collection of `tests/test_vod_success.py` and never runs this test. It is therefore not a failure of that command, and it is not the only problem that command reports. The only error is the numpy collection error.

When the test is executed, it does fail, and the missing directory is the cause:

- `os.path.exists("/workspace/cfb-coach")` is `False`. The repo is `/workspace`.
- The same `subprocess.run` the test uses raises `FileNotFoundError: [Errno 2] No such file or directory: '/workspace/cfb-coach'` before the child process starts. Pytest reports that as a failure, not a collection error.

Isolated pytest (venv, `PYTHONPATH=.`):

```bash
PYTHONPATH=. python -m pytest tests/test_watch_live_ux.py::TestDemoStillWorks::test_demo_once_exits_clean -p no:cacheprovider -rA --tb=short
```

Result: `FAILED tests/test_watch_live_ux.py::TestDemoStillWorks::test_demo_once_exits_clean` — `FileNotFoundError: [Errno 2] No such file or directory: '/workspace/cfb-coach'` (`1 failed in 0.11s`, exit code 1).

The same argv with a real working directory succeeds and meets every assertion in the test:

- `cwd=/workspace`, `PYTHONPATH=/workspace`: return code 0, stdout contains `SCREEN CO-PILOT v1.9.4` and `CO-PILOT`, and does not contain `waiting for frames` or `Pipeline capture=`.
- `cwd=/workspace`, `PYTHONPATH=/workspace/cfb-coach` (path still missing): also return code 0 with the same stdout checks. The editable install still imports. The hardcoded `PYTHONPATH` is not what aborts the test. The missing `cwd` is.

So the hardcoded cwd is the cause of this test's failure, and fixing only that path is enough for these assertions to pass. It is not the only problem in the required pytest command (that command's only error is the numpy collection error, and this test never ran).

## unittest discover

README documents `PYTHONPATH=. python3 -m unittest discover -s tests`. This runner does not load `tests/conftest.py` and does not collect plain pytest functions. `tests/test_vod_success.py` defines seven `test_*` functions and no `unittest.TestCase`; those functions are not collected. The module still fails to import.

Command (venv `python3`, cwd `/workspace`):

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

- Start: `2026-10-08T13:06:57.126708395Z`
- End: `2026-10-08T13:07:30.488361788Z`
- Wall-clock: `33.361653` seconds
- unittest's own line: `Ran 321 tests in 33.145s`
- Summary: `FAILED (failures=4, errors=3)`
- Exit code: `1`
- `skipped` does not appear in the log. skipped = 0.
- passed = 321 − 4 − 3 = 314

| Count | Value |
| --- | --- |
| ran | 321 |
| passed | 314 |
| failures | 4 |
| errors | 3 |
| skipped | 0 |

Errors:

1. `test_madden27.TestMaddenPrep.test_user_prep_eight_custom_adjustments_and_html` — `AttributeError: 'str' object has no attribute 'get'` (`cfb_coach/prep_browser.py` `_render_meta_scout`, `pn.get` on a str).
2. `unittest.loader._FailedTest.test_vod_success` — `ModuleNotFoundError: No module named 'numpy'`.
3. `test_watch_live_ux.TestDemoStillWorks.test_demo_once_exits_clean` — `FileNotFoundError: [Errno 2] No such file or directory: '/workspace/cfb-coach'`.

Failures:

1. `test_cfb_playbook.TestPrepPage.test_minimal_prep_page_only_formations_audibles_macros` — `AssertionError: 'LIVE META RESEARCH DID NOT RUN' not found in` the prep HTML.
2. `test_madden27.TestCfbUnchanged.test_cfb_play_and_prep_still_cfb` — `AssertionError: 'cfb27-2026-09' not found in` the prep text (the text included a live `Daily AI research ... STALE` line).
3. `test_madden_parity.TestResearchPicksBooks.test_prep_uses_research_for_book_and_focus` — `AssertionError: 'Buccaneers' != 'Texans'`.
4. `test_meta_refresh.TestPrepPageRenders.test_prep_page_shows_freshness_sources_zone_plan_conflicts` — `AssertionError: 'LIVE — fetched this prep' not found in` the prep-details HTML.

`tests/conftest.py` sets `CFB_COACH_AI_RESEARCH=off`. unittest discover does not load that file. Re-running the AttributeError plus the four assertion failures with `CFB_COACH_AI_RESEARCH=off` passed: `Ran 5 tests in 1.421s` / `OK` (exit 0). Those five are not failures under pytest, which does load `conftest.py`.

Under this unittest command, `test_demo_once_exits_clean` is one of three errors and is not the only failure.

## Flaky check

The required pytest command was run a second time (run 2 above). JUnit outcomes were compared by `(classname, name, status)`.

No test result differed. Both runs recorded a single collection error on `tests/test_vod_success.py` with `ModuleNotFoundError: No module named 'numpy'`. Passed, failed, skipped, xfail, and xpass stayed 0. Exit code stayed 2.

The two required runs never executed the 320 collected tests, so this check does not measure per-test flakes inside the suite. It only shows the collection error is stable across the two invocations. Wall-clock and pytest's internal time differed (0.678s / 0.52s vs 0.374s / 0.22s); that is duration, not a result change.

## Supplemental pytest (not the required command)

The required command never reaches the demo test. To see whether that test is the only failure once collection continues, the suite was also run with one extra flag. This output is not the baseline counts above.

```bash
PYTHONPATH=. python -m pytest tests -p no:cacheprovider --continue-on-collection-errors -ra --tb=line --junitxml=/tmp/pytest_continue.xml
```

- Wall-clock: `31.858832` seconds (`2026-10-08T13:08:31.874029585Z` to `2026-10-08T13:09:03.732861409Z`)
- pytest's own line: `1 failed, 319 passed, 1 error in 31.69s`
- Exit code: `1`
- JUnit written to `/tmp/pytest_continue.xml` only (not committed)

| Count | Value |
| --- | --- |
| collected | 320 |
| passed | 319 |
| failed | 1 |
| skipped | 0 |
| errors | 1 |
| xfail | 0 |
| xpass | 0 |

1. `tests/test_watch_live_ux.py::TestDemoStillWorks::test_demo_once_exits_clean` — `FileNotFoundError: [Errno 2] No such file or directory: '/workspace/cfb-coach'`
2. `tests/test_vod_success.py` — `ModuleNotFoundError: No module named 'numpy'` (collection error)

In this supplemental run the demo test is the only failure. It is not the only problem: collection of `tests/test_vod_success.py` still errors. The four unittest assertion failures and the madden prep `AttributeError` passed here.
