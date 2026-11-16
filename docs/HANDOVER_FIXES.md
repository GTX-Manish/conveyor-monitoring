# Handover for Claude Code: fixes before submission (Mon Nov 16, 2026, 14:05 IST)

Follows `docs/FINAL_DAY_REPORT.md`. Four small fixes, then one local clean run. Budget: about 90 minutes in total.
Why each decision was made is written next to it.

## 0. Scope

**Do:** the four fixes in sections 2 to 5 and the local clean run in section 6.

**Do not:**
- Touch GitHub: no remote, no push, no branch rename. The user does that later.
- Write or edit the README, or touch `ai_usage/`. The user does those.
- Change any model, feature or threshold, or re-tune anything. The models are frozen and the fresh test is scored.
- Start any vision work. Leave the Vision tab as it is.

## 1. Start state (5 minutes)

1. Record `git rev-parse HEAD` (expect `c18f1c1`), confirm the tree is clean, run the tests (expect 110 passed, 3 skipped)
   and `ruff`. Stop and report if anything fails.
2. Confirm `sha256sum` of the files in `model/artifacts/` equals `hashes_frozen.txt`. Why: every fix below must leave the
   models and thresholds byte-identical, and this is the baseline.

## 2. Fix 1: the Overview strip must quote the fresh test

**Problem:** the strip says "on 12 faults the model never saw: 10 caught (83%), ... 8 of 8 failure predictions within 72 h".
Those are `test_01` and `test_02`, which were used for diagnosis. The clean numbers are the fresh test.
**Why fix:** the dashboard must not claim more than the evaluation supports, and it must agree with what the user will write up.

1. Add a machine-readable copy of the fresh-test result: `model/artifacts/fresh_test_summary.json` (per asset: faults,
   alerts caught, false alerts, alert precision, alert recall, calls, calls true, call recall, median hours from call to
   failure; plus scenario names, seeds and the boosted drive rate).
2. Produce it by re-exporting with the frozen models (`--frozen --fresh`), not by typing numbers in. **The numbers must equal
   the committed `fresh_test_report.md` exactly** (bearings 16 of 20, 0 false; drive 6 of 12, 0 false; calls 10 of 10 and 6 of
   6). If any number differs, stop and report; do not commit. Why: this is an export of a finished result, not a second
   scoring run. Do not change `fresh_test_report.md` or `summary.json`.
3. Point the Overview strip (backend endpoint and frontend text) at this file. Suggested wording:
   "Fresh test, 32 simulated faults in scenarios never used for training, tuning or diagnosis: bearings 16 of 20 caught
   (80%), drive 6 of 12 caught (50%), 0 false alerts. 16 failure predictions were made and all came true within 72 h; half the
   faults got one. Simulated data only."
   The numbers must be read from the file, not hard-coded in the frontend.
4. Search the frontend and backend for any other place that shows the old test numbers as headline accuracy (walkthrough
   text, Track record, tooltips) and switch them the same way, or label them "earlier test, used for diagnosis".
5. Test: the endpoint returns the fresh-test numbers from the file; the strip text contains no number from the old test.

## 3. Fix 2: severity must not say "Critical" while the status says "Abnormal"

**Problem:** at day 12, Zone 107's alert shows "Critical" (failure risk score 87%) while the map shows "Abnormal", because the
severity comes from the raw score and a failure call needs the call threshold (0.99).
**Why fix:** two labels on one screen that disagree look like a bug, and this case is in a screenshot.

1. **Rule:** an alert's label may be `critical` only while a failure call is active on that asset (status
   `predicted_failure`). Otherwise the label is capped at `high`. The numeric severity, the thresholds and the status rule do
   not change. Data-quality alerts are not affected (fixed `medium`).
2. Apply it in **one place** in the backend, where the alert's severity label is set. Check that the frontend does not
   recompute a label from the numeric severity; if it does, make it use the backend's label.
3. Update the legend or tooltip in one line: "Critical is shown only when a failure is predicted within 72 h."
4. Tests: score above the critical band with no call gives `high`; with a call gives `critical`; a data-quality alert stays
   `medium`.
5. Replay the demo and confirm no alert shows `critical` on a zone whose status is not `predicted_failure`.

## 4. Fix 3: `make setup` must work, or fail clearly

**Problem:** `make setup` was not run in the dry run because it needs `python3.12` on the PATH. A reviewer without it fails on
the first command.

1. Make `make setup` find Python 3.12 in this order: `python3.12` on the PATH; else `uv venv --python 3.12` if `uv` is
   installed; else stop with a clear message that says what to install. No silent fallback to another version.
2. After creating `.venv`, check that its Python is 3.12 and stop if it is not.
3. Keep `requirements.txt` pinned as it is. Do not add tools a reviewer must install beyond Python 3.12 (or `uv`).
4. Run `make setup` for real in the clean run (section 6). If you can, also check the failure message by running with a PATH
   that has neither `python3.12` nor `uv`.

## 5. Fix 4: CI must run on either branch name

**Problem:** the local branch is `master`. If `.github/workflows/ci.yml` triggers only on `main`, CI will never run after the
user pushes.

1. Open `.github/workflows/ci.yml`. Make the triggers work for both names (`push` and `pull_request` with no branch filter,
   or with both `main` and `master`). **Do not rename the branch and do not add a remote.**
2. Run the workflow's own commands locally, exactly as written in the file (Python job and Node job), and confirm both pass.
   Report that CI itself could not be run.

## 6. Local clean run (after all four fixes are committed)

1. `make frontend`, confirm `frontend/dist` is committed and identical to a fresh build.
2. `git clone` the local repository into a **new empty folder**. Copy nothing from the working tree.
3. In the clone, run **`make setup`**, then `make demo`. Check: `/api/health` OK; dashboard served from the committed
   `frontend/dist` with no Node; all tabs load with no console errors; the replay runs to the end; the Overview strip shows the
   fresh-test numbers; no "Critical" on an "Abnormal" zone; the final demo state is unchanged (4 failures, 4 alerts open at
   failure, 0 false alarms, 3 calls, all true).
4. In the clone: `make test`, `ruff`, `git status` clean.
5. Retake the screenshots that changed into `docs/screenshots/` (at least `overview.png`, `loop_map_alerts.png`, `alerts.png`);
   keep each under about 500 KB. Commit them.
6. Confirm the model files and `meta.json` still equal `hashes_frozen.txt`. The only new file in `model/artifacts/` is
   `fresh_test_summary.json`.
7. Remove the clone.

## 7. Done when

- All tests pass (110 plus the new ones, 3 skipped) and `ruff` is clean.
- The four fixes are committed, one commit each, with a one-line `docs/CHANGELOG.md` entry each.
- The local clean run passes every check in section 6.
- Model artifacts are byte-identical to the freeze.

## 8. Hand back (one message, plain text)

1. Final commit hash, and the test count.
2. Fix 1: the exact strip text now shown, and confirmation that the export matched the report.
3. Fix 2: where the rule lives, and what Zone 107 shows at day 12 now.
4. Fix 3: what `make setup` does on this machine, and the failure message.
5. Fix 4: the trigger lines now in `ci.yml`, and the local results of both jobs.
6. Clean run: each check and its result, `make setup` and `make demo` times, screenshot file names.
7. Anything that looked wrong, even if you fixed it.

## 9. If time slips (cut order)

Fix 4's local run of the Node job, then the failure-message check in fix 3. **Never skip** fix 1, fix 2, the real `make setup`
run, or the artifact hash check.
