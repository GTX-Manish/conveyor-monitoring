# Final day report (Mon Nov 16, 2026)

Hand-back for `docs/HANDOVER_FINAL_DAY.md`. Steps 1 and 2 are done and committed. Step 3 is only partly done, because
the repository has no GitHub remote and there is no README yet, so nothing is pushed and CI has not run.

| Item | Value |
|---|---|
| Final commit | `c18f1c1bf35e8125a5625937a44842ce3238dc70` |
| Tag | `frozen-before-fresh-test` (on `eec965c`, the end of step 1) |
| Branch | `master` (not `main`) |
| Remote | none configured |
| Start state | HEAD `c0109b3`, tree clean, 107 passed and 3 skipped, `ruff` clean, 1,000 GB free |
| End state | 110 passed, 3 skipped, `ruff` clean, tree clean |

Commits made today, in order: `eec965c` step 1, `fc1a6e6` criteria, `c8e941a` step 2, `1095754` a docstring fix,
`c18f1c1` step 3 (local part).

## 1. Step 1: drive model retrained and kept

### What was done

- **New scenarios:** `train_drive_01` to `train_drive_06` (seeds 601 to 606, 30 days, split `train_drive`). Config is the
  default plus `faults.drive_overheat.per_30d: 4`; noise, drift, shared noise and fault strength stay random.
  They are separate scenarios because the fault stream is shared: raising the drive rate inside `train_01` to
  `train_08` would also have changed their bearing faults.
- **Zone data builder:** `scripts/make_model_data.py` now builds drive-only tables for these scenarios (no zone tables).
- **Training:** only the three drive models, with the existing settings and seed 1, on `train` plus `train_drive`
  (58,478 training rows). The model version string is `lgbm-w10-seed1-drive2`.
- **Thresholds:** drive alert and call thresholds retuned on the six existing validation scenarios with the unchanged rule.
- **Guards added (tested):** `training_splits` makes the zone model train on `train` and nothing else, whatever is
  passed; a drive-only `save` writes only the drive files and leaves the zone files untouched; no `test_fresh`
  scenario appears in any training or tuning path.

### Drive events placed

| Scenario | Seed | Drive events | Bearing events |
|---|---|---|---|
| train_drive_01 | 601 | 4 | 5 |
| train_drive_02 | 602 | 4 | 5 |
| train_drive_03 | 603 | 4 | 5 |
| train_drive_04 | 604 | 4 | 5 |
| train_drive_05 | 605 | 4 | 5 |
| train_drive_06 | 606 | 4 | 5 |
| **Total** | | **24** | 30 |

The scheduler placed all 4 in every scenario.

### Keep or discard: the rule was written before looking

> Keep the new drive model only if, on validation, drive alert recall is at least 3 of 6 and drive false alerts are at
> most 1. Otherwise revert.

| Validation (6 scenarios, 6 drive faults) | Result |
|---|---|
| Drive alert threshold | 0.96 |
| Alerts caught | **4 of 6** |
| False alerts | **0** |
| Alert precision | 1.00 |
| Call threshold | 0.05 (4 of 4 calls true within 72 h) |
| **Decision** | **Keep** |

One attempt, one decision. No second tuning round.

### Zone models are byte-identical

| File | sha256 before | sha256 after | Same |
|---|---|---|---|
| `zone_abnormal.txt` | `24cc4cff8b60ca47eaf5...` | `24cc4cff8b60ca47eaf5...` | yes |
| `zone_classifier.txt` | `d151357e8ab5ef05e9ec...` | `d151357e8ab5ef05e9ec...` | yes |
| `zone_regressor.txt` | `2d283f33310e44a3f671...` | `2d283f33310e44a3f671...` | yes |

Full hashes are in `model/artifacts/hashes_before.txt` and `hashes_frozen.txt`. The frozen evaluation's zone section
is also identical to the previous evaluation, and the zone rows of the robustness report are identical (which also
shows the simulator and model are deterministic).

### Drive results before and after (alerts caught / false alerts, seed 1)

| Split | Faults | Before | Now |
|---|---|---|---|
| Validation | 6 | 1 / 0 | **4 / 0** |
| Test (old test_01 and test_02) | 2 | 1 / 0 | **2 / 0** |
| Stress | 2 | 2 / 0 | 2 / 0 |

Test and stress were reported for information; they did not decide. Drive failure calls now: validation 4 of 4 true
(call recall 0.67), test 2 of 2, stress 2 of 2.

### Drive robustness (15 cases, caught of 2, false alerts)

The run took about 10 minutes, under the 20-minute limit, so the drive rows were re-run.

| Case | Before | Now |
|---|---|---|
| Faults 0.2 to 0.4 of normal strength | 1 caught, 2 false | 0 caught, 0 false |
| Very late, sharp growth | 1 | 2 |
| Noise fixed at 8x | 0 | 2 |
| Noise fixed at 12x | 1 | 2 |
| Shared noise fixed at 6x | 1 | 2 |
| The other 10 cases | unchanged | unchanged |

No case got worse, and no case has a false drive alert now.

### Demo (`demo_01`, seed 1015, not re-picked), final model

| | Value |
|---|---|
| Failures | 4 |
| Alert open at failure | 4 (the drive is now warned, 1.6 h before it fails) |
| Missed | 0 |
| False alarms | 0 |
| Failure calls | 3, all true within 72 h |

| Part | Alert lead time | Call lead time |
|---|---|---|
| Z107 | 55.4 h | 31.7 h |
| Z064 | 11.9 h | 4.6 h |
| Z065 | 1.0 h | no call |
| Drive | 1.6 h | 1.6 h |

Before step 1 the demo warned 3 of 4 and missed the drive.

## 2. Step 2: fresh untouched test

### Process

- The criteria were written and committed first (`fc1a6e6`, file `model/artifacts/fresh_test_report.md`), before any
  fresh scenario existed.
- Scenarios `test_03` to `test_06` (seeds 501 to 504, split `test_fresh`, default settings plus
  `faults.drive_overheat.per_30d: 3`). The drive rate is boosted because at the default rate there would be about 4
  drive events, too few to judge.
- Scored **once**, with `python -m scripts.score --frozen --fresh`. The scoring logic is unchanged (the same
  `score` and `score_calls` functions). **No crash re-run happened.**
- Artifact hashes equal `hashes_frozen.txt` before and after scoring.

### The fresh scenarios

| Scenario | Seed | Noise x | Drift x | Shared x | Bearing faults | Drive faults |
|---|---|---|---|---|---|---|
| test_03 | 501 | 2.60 | 1.86 | 2.92 | 5 | 3 |
| test_04 | 502 | 2.82 | 1.48 | 2.89 | 5 | 3 |
| test_05 | 503 | 3.80 | 2.37 | 2.54 | 5 | 3 |
| test_06 | 504 | 2.58 | 1.44 | 0.45 | 5 | 3 |

### Results against the criteria set in advance

| Asset | Faults | Alerts caught | False alerts | Alert precision | Alert recall | Criterion | |
|---|---|---|---|---|---|---|---|
| Bearings | 20 | 16 | 0 | 1.00 | 0.80 | recall at least 0.75, precision at least 0.90 | met |
| Drive | 12 | 6 | 0 | 1.00 | 0.50 | recall at least 0.50 | **met exactly** |

### Failure calls (72 h window)

| Asset | Calls | Came true | Call precision | Call recall | Median hours from call to failure |
|---|---|---|---|---|---|
| Bearings | 10 | 10 | 1.00 | 0.50 (10 of 20) | 14.2 |
| Drive | 6 | 6 | 1.00 | 0.50 (6 of 12) | 5.6 |

Hours from call to failure, bearings: 0.6, 3.7, 4.8, 4.8, 13.1, 15.3, 18.2, 22.1, 35.4, 35.9. Drive: 1.4, 1.8, 5.4,
5.8, 10.0, 17.3.

### Per scenario

| Scenario | Bearings caught | Bearing false alerts | Drive caught | Drive false alerts |
|---|---|---|---|---|
| test_03 | 3 of 5 | 0 | 1 of 3 | 0 |
| test_04 | 5 of 5 | 0 | 2 of 3 | 0 |
| test_05 | 3 of 5 | 0 | 1 of 3 | 0 |
| test_06 | 5 of 5 | 0 | 2 of 3 | 0 |

### Against the old test scenarios

`test_01` and `test_02` give bearings 8 of 10 caught and drive 2 of 2 caught, both with 0 false alerts. They were used
for diagnosis in V7 and V8, so the fresh numbers are the ones to quote.

### Caveats

- 20 bearing and 12 drive faults. One drive fault moves drive recall by 8 points.
- The drive rate is boosted (3 per 30 days, default 1).
- Failure calls miss about half the faults: weak faults raise no call.
- The drive result sits exactly on its bar.
- The criteria were our own choice, not gates. The result is reported as it came out.

## 3. Step 3: clean run on the submitted commit (partly done)

### Not done, and why

| Item | Status |
|---|---|
| Confirm local HEAD equals `origin/main`, push, wait for CI | **Not possible**: no remote, and the branch is `master` |
| Clone the GitHub remote and run the README's commands literally (`make setup`, `make demo`) | **Not possible**: no remote and no README |
| `make test` and `ruff` in that clone | Not done; run in the working tree instead (110 passed, 3 skipped, clean) |

### Done: a local dry run from a fresh clone

The clone was made from the local repository into a new folder, with a new Python 3.12 venv. It was removed afterwards.

| Check | Result |
|---|---|
| Clone | OK, no tracked files under `data/` |
| Venv and install | `uv pip install -r requirements.txt`, 11 s with a warm cache. **`make setup` itself was not run** (it needs `python3.12` on the PATH) |
| Build the demo data from its seed | `make demo-data`, 13 s |
| Start the demo (`scripts.demo`) | OK |
| **Time to first prediction** | **20 s** after the server and replay started |
| `/api/health` | OK, model version `lgbm-w10-seed1-drive2` |
| Dashboard source | served from the committed `frontend/dist`; Node was not needed |
| Replay | ran to the end |
| Six tabs plus the Zone details view | all seven loaded in headless Chrome, **no console error lines** (Overview, Alerts, Predictions, Track record, Zone details, Data health, Vision) |

The console check used Chrome's logging output from headless mode, not a test framework (Playwright is not installed).

### Hygiene checks (working tree)

| Check | Result |
|---|---|
| `git ls-files \| grep '^data/'` | empty |
| `.git` size / tracked files size | 16 MB / 7.2 MB |
| `requirements.txt` | no `torch`, no `ultralytics` |
| Absolute paths in `.py`, `.yaml`, `.ts`, `.tsx`, `Makefile` (outside tests and docs) | none |
| `frontend/dist` committed and equal to a fresh `vite build` | identical |
| `model/artifacts/` has every file the backend loads | yes (6 model files and `meta.json`) |
| Artifacts changed since the freeze tag | only `fresh_test_report.md` was added |

### Screenshots for the README

In `docs/screenshots/`, each under 80 KB:

| File | Shows |
|---|---|
| `overview.png` | Overview at the end of the replay |
| `loop_map_alerts.png` | the loop map with an open alert (day 12, Zone 107) |
| `alerts.png` | Active anomalies, with "What the model saw" |
| `predictions.png` | Model predictions |
| `data_health.png` | Data health |
| `track_record.png` | Track record at the end of the replay |

## 4. Things that looked wrong, or worth knowing

1. **Severity and status disagree.** At day 12, Zone 107's alert shows "Critical" with a failure risk score of 87%, but
   the map shows the zone as "Abnormal", not "Predicted failure". Severity comes from the raw score; the failure call
   needs 99%. **Not changed.**
2. **The Overview strip wording.** It says "on 12 faults the model never saw: 10 caught (83%), ... 8 of 8 failure
   predictions within 72 h". Those are the old test scenarios. The model never trained on them, but they were used
   for diagnosis. The text comes from `summary.json`, which is frozen, so it was left. The fresh test is the number to
   quote.
3. **A slip of mine.** I committed step 2 (`c8e941a`) while one test was failing: a docstring in `model/train.py`
   mentioned `test_fresh` and tripped my new check that no fresh scenario appears in training code. I fixed the
   docstring in `1095754`. The tag, and every commit from `1095754` on, pass all tests.
4. **`score.py` retrains by default.** Run as the handover describes, step 1's re-evaluation would have retrained and
   overwritten the zone models, and step 2 would have retrained too. I added `--frozen` (scores the saved models and
   thresholds, trains nothing; it recomputes the threshold sweeps and refuses to run if they differ from `meta.json`)
   and `--fresh`. This went beyond "only accept the new split name", but the scoring logic itself is unchanged.
5. **Small differences from the handover's commands:** `scenario_set --only` takes space-separated names, not a comma
   list; the handover's expected test count of 107 became 110 because of the new tests.
6. **Weak spots, unchanged:** failure-call recall is about 0.50; bearing recall falls under strong drift; the drive
   result sits exactly on its bar, from 12 events.
7. **Everything is simulated.** None of these numbers show the model works on a real belt.

## 5. What is needed to finish step 3

1. A GitHub remote for this repository, and a decision on renaming `master` to `main`.
2. The README (it is drafted separately), committed to the repo.

Then: push, wait for CI on that commit, clone the GitHub remote into a new empty folder, run the README's commands
literally, repeat the checks above, run `make test` and `ruff` in the clone, and time `make demo`.
