# Fixes report (Mon Nov 16, 2026)

Hand-back for `docs/HANDOVER_FIXES.md`. All four fixes are committed, plus one extra fix found during the clean run.
The local clean run passed every check, and the model artifacts are byte-identical to the freeze. A session
interruption damaged the git repository mid-commit; it was restored and the lost work redone (section 7).

## 1. Final state

| Item | Value |
|---|---|
| Final commit | `93386c6` |
| Tests | 115 passed, 3 skipped |
| `ruff` | clean |
| Model artifacts | byte-identical to `hashes_frozen.txt`; the only new file this round is `fresh_test_summary.json` |
| GitHub | nothing pushed, no remote, branch still `master` (as the handover asked) |

| Commit | What |
|---|---|
| `0d2fbc0` | Fix 1: the Overview quotes the fresh test |
| `2c8215b` | Fix 2: Critical only while a failure call is active |
| `3d9f3e4` | Fix 3: `make setup` finds Python 3.12 or stops clearly |
| `d47fced` | Fix 4: CI triggers confirmed (no change needed) |
| `c3a04fd` | Extra fix: `make demo` no longer rewrites the scenario report |
| `93386c6` | README screenshots retaken |

## 2. Fix 1: the Overview strip quotes the fresh test

The strip now reads, built only from `model/artifacts/fresh_test_summary.json`:

> Fresh test, 32 simulated faults in scenarios never used for training, tuning or diagnosis: bearings 16 of 20 caught
> (80%), drive 6 of 12 caught (50%), 0 false alerts. 16 failure predictions were made and all came true within 72 h;
> half the faults got one. Simulated data only.

- **How the file was made:** exported with the frozen models by `score.py --frozen --fresh --export-summary`, which
  compares every figure with the committed `fresh_test_report.md` and writes nothing if one differs.
- **Export check:** it matched the report figure for figure (bearings 16 of 20, drive 6 of 12, 0 false alerts, calls 10
  of 10 and 6 of 6, median 14.2 h and 5.6 h). Neither `fresh_test_report.md` nor `summary.json` changed.
- **What reads it:** a new `GET /api/model/fresh-test`. The tour step now names the fresh test.
- **Old figures:** the test_01/02 numbers are no longer shown anywhere in the dashboard.
- **Tests:** the endpoint serves the file (404 without it); the committed export equals the committed report; the
  Overview reads the fresh-test endpoint, and its sentence contains no typed-in figures.

## 3. Fix 2: no "Critical" without a failure call

- **Where the rule lives:** `alert_label` in `model/alerting.py`, used in the one place alert labels are set,
  `_model_alert` in `backend/engine.py`. The frontend only displays that label; it never recomputes one.
- **The rule:** a model alert's label may be Critical only while a failure call is active on that asset (status
  `predicted_failure`); otherwise it is capped at High. The numeric severity, the thresholds and the status rule are
  unchanged. Data-quality alerts stay Medium.
- **Zone 107 at day 12 now:** "High" (failure risk score 87%), matching its Abnormal status on the map.
- **Checked over a full replay:** 0 Critical alerts on zones without a failure call, across 203 checks.
- **Legend line** on the Alerts tab: "Critical is shown only when a failure is predicted within 72 h."
- **Tests:** a score above the critical band with no call gives High; with a call gives Critical; a data-quality alert
  stays Medium.

## 4. Fix 3: `make setup`

- **Order it tries:** `python3.12` on the PATH (venv plus pip), else uv (`uv venv --python 3.12`, then `uv pip install`),
  else it stops. It checks the new `.venv` is 3.12 and never falls back to another version. `requirements.txt` is
  unchanged.

| Situation | Result |
|---|---|
| This machine (`python3.12` on the PATH) | finds it; 152 s in the first clone, 60 s in the second |
| Only uv on the PATH | builds a 3.12 environment in 70 s |
| Neither on the PATH | exit code 2, no `.venv` created, message below |

```
make setup: Python 3.12 is required, and neither python3.12 nor uv was found on the PATH.
  Install Python 3.12 (https://www.python.org/downloads/ or: sudo apt install python3.12 python3.12-venv),
  or install uv (https://docs.astral.sh/uv/getting-started/installation/), then run make setup again.
```

## 5. Fix 4: CI triggers

- **No edit was needed:** `.github/workflows/ci.yml` already says `on: [push, pull_request]` with no branch filter, so
  it runs for `master` and `main`.
- **Both jobs' commands, run locally in the clone:**

| Job | Commands | Result |
|---|---|---|
| Python | `pip install -r requirements.txt`, `ruff check .`, `pytest -q` | install OK; `ruff` clean; 114 passed, 3 skipped (at fix 4) |
| Node | `npm ci`, `npx tsc --noEmit`, `npx vite build` | all OK (local Node is v26; CI uses 20) |

- CI itself has not run, because there is no remote.

## 6. Local clean run

The full run was on a fresh clone at `d47fced`, after the four fixes.

| Check | Result |
|---|---|
| Committed `frontend/dist` against a fresh build (`make frontend` steps) | identical |
| `make setup` | OK, 152 s |
| `make demo` to first prediction | 114 s (100 s of it building the demo data) |
| `/api/health` | OK |
| Dashboard served from the committed `dist`, no Node | yes |
| Seven views (Overview, Alerts, Predictions, Track record, Zone details, Data health, Vision): console errors | 0 |
| Replay | ran to the end (315 s; speed 3 after a short look at the default speed) |
| Overview strip | the fresh-test text in section 2 |
| Critical on an Abnormal zone | none |
| Final demo state | 4 failures, 4 alerts open at failure, 0 false alarms, 3 calls, all true (unchanged) |
| `make test`, `ruff`, `git status` | passed and clean, but `make demo` left the tree dirty (section 7), so fixed and re-checked |

**Re-check on the final commit `93386c6`, in a new clone:**

| Check | Result |
|---|---|
| `make setup` | OK, 60 s |
| `make demo-data` | 7 s; "report not rewritten: only 1 of 28 scenarios are on disk"; clone stays clean |
| `make test` | 115 passed, 3 skipped |
| `ruff` | clean |
| `git status` | clean |

- **Artifacts:** the model files, `meta.json` and the reports match `hashes_frozen.txt`. Since the freeze tag, only
  `fresh_test_report.md` (step 2) and `fresh_test_summary.json` (this round) were added.
- **Screenshots retaken:** `docs/screenshots/overview.png` (75 KB), `loop_map_alerts.png` (77 KB), `alerts.png` (43 KB).
- **Clones removed.**

## 7. Things that looked wrong

1. **`make demo` wiped the scenario report.** It rewrote `docs/scenario_set_report.md` from the one demo scenario on
   disk, deleting 26 rows and leaving a reviewer's clone dirty. Fixed in `c3a04fd`: the report is rewritten only when
   every configured scenario is on disk (tested).
2. **The interrupted session damaged the repository.** It left 15 empty object files, the branch pointing at an empty
   commit, and NUL bytes in `CHANGELOG.md` where a new entry should have been. Recovery: backed up `.git`, moved the
   empty files aside rather than deleting them, reset `master` to the last good commit `d47fced`, cleaned the
   changelog, and redid the two lost commits. `git fsck` is clean, and a scan found no other damaged file.
3. **Two folders to delete when you are satisfied:** the backup at `/home/manny/conveyor_git_backup_101802` and the
   moved-aside files in `/home/manny/conveyor_git_backup_101802-empty-objects`. They were left in place.
4. **The first clone folder disappeared during the interruption.** It was not deleted on purpose, so the re-check was
   done on a new clone.
5. **Demo data was slow once.** Building it took 94.5 s in the first clone, against 3 to 7 s every other time. Not
   reproduced.
6. **Mistakes of mine, fixed:** a stop command matched its own shell and killed it twice; the first version of the
   report parser read the criteria table instead of the results.
7. **The console check is basic.** It uses Chrome's log output in headless mode, not a test framework (Playwright is not
   installed).

## 8. Still to do (the user's part)

- Add the GitHub remote and decide on `master` versus `main`.
- Commit the README.
- Push, check CI on the pushed commit, and run the README's commands literally in a clone of the remote.
