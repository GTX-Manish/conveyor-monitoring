# Handover for Claude Code: final day (Mon Nov 16, 2026)

Written 12:15 IST. We aim to submit by midnight tonight; Tuesday Nov 17 is buffer only (hard deadline 11:59 PM IST).
Read `docs/BUILD_PLAN.md` and `docs/PROJECT_HISTORY_AND_METRICS.md` first (copy both into `docs/` if they are not there). This
file is your job for today. Where it differs from `BUILD_PLAN.md`, this file wins.

Why each decision was made is written next to it. **(assumption)** means our own choice.

## 0. The job in one paragraph

Three things, **in this order**: (1) retrain the drive model once, on more drive events, and freeze everything; (2) score
brand-new test scenarios exactly once with the frozen models; (3) prove the submitted commit runs from a clean clone. The
order matters: a "fresh" test is only fresh if no model changes after it, and the clean run must check the commit we submit.

**Do not:** change the zone model or any feature; change thresholds after seeing any fresh-test result; change the demo seed
(1015); write the README (it is drafted separately; you hand back numbers, section 5); touch `ai_usage/` (the user's);
start any vision work.

## 1. Start state (do first, 10 minutes)

1. Record `git rev-parse HEAD`, confirm the working tree is clean, and run the full test suite (expect 107 passed, 3 skipped)
   and `ruff`. Stop and report if either fails.
2. Check free disk (`shutil.disk_usage`): stop with a message under 5 GB (a 40 GB dataset download may be using the disk).
3. Write `model/artifacts/hashes_before.txt`: the `sha256sum` of every file in `model/artifacts/`. Why: step 1 must leave the
   zone models byte-identical, and this proves it.

## 2. Step 1: drive model retrain (one attempt; budget about 60 minutes)

**Why:** the drive model trained on 8 events and catches 1 of 6 on validation and 1 of 2 on test. The cause is too few
events, so the fix is more events.

**Why new scenarios and not a higher rate in the existing ones:** the fault stream is shared, so changing the drive rate in
`train_01` to `train_08` would change the bearing faults too, and the zone model would change with them. Adding separate
scenarios leaves the zone model, all existing scenarios and the demo untouched.

1. **New scenarios** `train_drive_01` to `train_drive_06`: seeds 601 to 606, 30 days, split name `train_drive`, config =
   `configs/default.yaml` plus one override, `faults.drive_overheat.per_30d: 4`. Keep every other setting at default,
   including the random noise, drift, shared-noise and per-fault strength draws (so the drive model sees the full messiness
   range). Do not rename any config key. Register the split so the existing tests still pass (seeds unique, splits disjoint).
   Generate with `python -m simulator.scenario_set --only train_drive_01,...` and report the drive events actually placed per
   scenario from `truth/` (expect about 4 each, about 24 in total). If the scheduler cannot place 4 (min start 24 h, end
   margin 12 h, no overlap), say so and use what it places.
2. **Data:** the zone data builder (`scripts/make_model_data.py`) must use an **explicit list** of splits and must not pick up
   `train_drive_*`. The drive data builder uses `train` plus `train_drive`.
3. **Train** only the three drive LightGBM models (abnormal, 24 h classifier, hours to failure) with the existing settings and
   seed 1. **Do not retrain the zone models.**
4. **Retune** the drive alert threshold and the drive call threshold on the **six existing validation scenarios**, with the
   existing tuning rule, unchanged. Why: oversampling shifts the probabilities, and validation is where that is corrected.
5. **Keep or discard by this rule, written before you look:** keep the new drive model only if, on validation, drive alert
   recall is **at least 3 of 6** and drive false alerts are **at most 1**. Otherwise revert the drive artifacts
   (`git checkout`), keep the old ones, and say so in the changelog. Why: choosing between two models is what validation is for.
   Test and stress drive numbers are reported for information; they do not decide.
6. **No second attempt.** One retrain, one decision. No feature changes, no extra tuning rounds.
7. **Update** `model/artifacts/meta.json` (thresholds, training scenarios, a new version string), `eval_report.md` and
   `summary.json` (re-run the normal evaluation; the zone numbers must come out unchanged), and the drive rows of
   `robustness_report.md` only if that run takes under about 20 minutes (otherwise note that the drive robustness rows are
   from the previous drive model).
8. **Demo:** rebuild `data/demo_01` and its database with the final model and write down what the demo now shows (failures,
   alerts open at failure, calls true). Do not re-pick the demo seed; the demo is curated by rules and the README says so.

**Done when:**
- All tests pass (107 passed, 3 skipped) and `ruff` is clean.
- `sha256sum` of the zone artifacts equals `hashes_before.txt` (print the comparison).
- The new `model/artifacts/hashes_frozen.txt` is written.
- A changelog entry gives the keep/discard decision with the validation numbers.
- Commit, push, and tag **`frozen-before-fresh-test`**.

## 3. Step 2: fresh untouched test (budget about 45 minutes)

**Why:** test_01 and test_02 were looked at during the V7 and V8 diagnosis, and the V8 features were designed to fix a test_01
miss. They are no longer clean. New seeds, scored once, give a number we can defend.

1. **Write the criteria first** and commit them before generating any data (`model/artifacts/fresh_test_report.md`, top
   section "Criteria set before scoring"). **(assumption)** Bearings: alert recall at least 0.75 and alert precision at least
   0.90. Drive: alert recall at least 0.5. These are not gates; they are what we said in advance. We report the result
   whatever it is.
2. **Scenarios** `test_03` to `test_06`: seeds 501 to 504, 30 days, split name `test_fresh`, default config plus one override,
   `faults.drive_overheat.per_30d: 3` (about 12 drive events in total; at the default rate there would be about 4, too few to
   judge). The report must say the drive rate is boosted and why. All other draws stay random and at default. Expect about 20
   bearing events.
3. **Frozen means frozen:** nothing in `model/artifacts/` changes. No scenario named `test_fresh*` may appear in any training
   or tuning code path (check the explicit split lists).
4. **Score once** with `scripts/score.py` on split `test_fresh`. You may change `score.py` only to accept the new split name;
   the scoring logic must not change. If the script crashes, fix the crash and re-run the same command on the same frozen
   artifacts, and say in the report that a crash re-run happened.
5. **Report** (`model/artifacts/fresh_test_report.md`), in the same format as the "Precision and recall at a glance" tables:
   noise, drift and shared-noise levels per scenario; faults per type per scenario; bearings and drive: faults, alerts caught,
   false alerts, alert precision, alert recall, calls, calls true within 72 h, call precision, call recall; hours from call to
   failure; per-scenario catches; plus one line comparing with the old test (labelled "used for diagnosis in V7 and V8").
6. After scoring, check that the artifact hashes still equal `hashes_frozen.txt`, and commit, push.

**Done when:** the report exists, the criteria section is in an earlier commit than the results, hashes match, tests pass.

## 4. Step 3: clean run on the submitted commit (budget about 45 minutes; run when the README is committed)

**Why:** a working tree hides untracked files, ignored folders and absolute paths. Only a fresh clone proves a reviewer can run it.

1. Confirm local `HEAD` equals `origin/main` (push first). Wait for CI on that commit and report its status.
2. `git clone` the GitHub remote into a **new empty folder**. Do not copy `data/`, `.venv` or anything else from the working
   tree. Make a new Python 3.12 venv.
3. Run the **README's commands literally, copy-pasted**, in order (`make setup`, then `make demo`). If a command in the README is
   wrong, fix the README (tell the user) and re-run from the start.
4. During the replay, check:
   - `/api/health` is OK, and the dashboard loads from the committed `frontend/dist` (Node must not be needed).
   - All six tabs render, with no browser console errors (use the browser or Playwright, not just curl).
   - The replay runs to the end (use speed 3 to finish quickly after a short look at the default speed).
   - The final alerts and the Track record match what you wrote down in step 1 point 8.
5. Take **screenshots** for the README into `docs/screenshots/`: Overview, the loop map with alerts, Active anomalies, Model
   predictions, Data health, Track record. Keep them small (PNG, under about 500 KB each).
6. In the clone: `make test` (107 passed, 3 skipped), `ruff`, `git status` clean.
7. Hygiene checks, and report each result:
   - `git ls-files | grep '^data/'` is empty;
   - `du -sh .git` and the working-tree size are small;
   - `requirements.txt` has no `torch` or `ultralytics`;
   - `grep -rn -E '/home/|/mnt/|C:\\\\' --include='*.py' --include='*.yaml' --include='*.ts*' --include='Makefile' .` finds no
     absolute paths in code or config;
   - `frontend/dist` is committed and matches `make frontend` output;
   - `model/artifacts/` contains every file the backend loads.
8. Time `make demo` from the clone to the first prediction and put it in the report.

**Done when:** every item above passes, or each failure is reported with a fix and the run is repeated from item 2.

## 5. What to hand back (one message, plain text)

1. Step 1: drive events placed per scenario, the keep/discard decision with validation numbers, the zone hash comparison.
2. Step 2: the fresh-test tables (copy of the report's summary), compared with the criteria, plus any crash re-run.
3. Step 3: CI status, the checklist results, `make demo` time, screenshot file names.
4. The demo's final state (failures, alerts open at failure, calls true), and the final commit hash and tag name.
5. Anything that looked wrong, even if you fixed it.

## 6. If time slips (cut order)

1. Shrink the fresh test to `test_03` and `test_04` (about 10 bearing events; say so in the report).
2. Skip the drive robustness re-run.
3. **Never skip** the zone-hash check, the criteria-before-scoring commit, or the clean-clone run. Tuesday is buffer, so nothing
   below those is worth risking a broken submission.

## 7. Rules that still apply

- Determinism: one seed and config give identical tables. No per-minute Python loops. Do not rename keys in
  `configs/default.yaml`. Ground truth is read only by the four allowed scripts (the leakage test lists them).
- Look at every plot you make. Commit after each step with a one-line changelog entry. If a number or rule here is wrong or
  impossible, say so and propose a fix; do not work around it silently.
