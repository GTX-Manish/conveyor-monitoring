# Project history and metrics: everything done so far

Covers the whole project from the first commit (Nov 14) to commit `88bfbfc` (Nov 16): the simulator, the model, the
backend, the dashboard, and the eight rounds of changes that made the data and the checks harder. Every round has
its **metrics table**.

Every number here was shown during the work. Tables marked **(log)** come from the development log and were not
saved as their own commit; the rest are in the evaluation report committed with that change
(`model/artifacts/eval_report.md` and `robustness_report.md` at the commit named in each section).

## 1. The short story

1. We built a simulator of a 6 km conveyor belt (120 sensor zones and a drive), with faults, hard cases and messy
   data, and a LightGBM model that predicts bearing and drive failures from it, plus a backend, a replayer and a
   dashboard.
2. The first evaluation looked perfect (every fault caught, no false alerts). That was suspicious, and it was.
3. Each later round removed something that had made the problem too easy: a loose check of "predicted failure",
   identical faults, fixed noise, no sensor drift, a validation set that did not cover the hard cases.
4. As the problem got harder, the numbers fell, and they should have. The current state is **8 of 10 bearing
   faults caught on the held-out test with no false alerts**, a weak drive model, and a model that stays quiet
   instead of guessing when drift hides a weak fault.
5. All data comes from our own simulator. None of these numbers show the model works on a real belt.

## 2. How to read the metrics

| Term | Meaning |
|---|---|
| **Fault** | One real failure event in a scenario (a bearing failing in one zone, or the drive overheating). |
| **Alert** | The model's warning on an asset: opens after 2 windows (20 min) above the alert threshold, closes after 6 windows below it, held open while the failure risk score is above 50%. |
| **Alert recall** | Faults with an alert opened between the fault's start and the failure, divided by all faults. |
| **False alert** | An alert that opened while the part was healthy. |
| **Alert precision** | Faults caught divided by (faults caught plus false alerts). Several alerts on one fault count once. |
| **Alert open at failure** | Faults where an alert was still open in the last window before the part failed. |
| **Failure call** | A "predicted failure": an alert is open and the failure risk score reaches its own threshold. Claims the part will fail within a window: 24 h until round V3, 72 h from V3 on. |
| **Call precision** | Calls followed by a real failure within the window, divided by calls made. |
| **Call recall** | Faults that got such a call in time, divided by all faults. |
| **Splits** | Train (8 scenarios), validation (thresholds are tuned here), test (2 scenarios, 10 bearing and 2 drive faults, scored once), stress (1 harder scenario, 8 bearing and 2 drive faults). Validation grew in V7. |
| **Small samples** | Test has 10 bearing faults and 2 drive faults. One fault moves recall by 10 points. One training seed unless a table says three. |

Plain accuracy is not reported: more than 99% of windows are normal, so a model that never alerts would score above
99%.

## 3. What was built, in order

| # | Step | Commit | What it is | Tests after |
|---|---|---|---|---|
| 0 | Stage 0 | `7e4efaa` | Repo, project docs | 0 |
| 1 | Stage 1 | `29c6a8e` | Healthy baseline of 120 zones, public/truth data split and leakage guards | |
| 2 | Stage 2 | `abd4c6d` | Belt coordinates, equipment faults (idler bearing, drive overheating), repair stops | |
| 2b | Belt damage removed | `57271de` | Removed because it only made sense with vision | |
| 3 | Stage 3 | `bb361f3` | Hard negatives (heavy load, heat wave, dust, start-up) and ground-truth labels | |
| 4 | Stage 4 | `e0e6279` | Messy data: missing readings, sensor outages, stuck sensors | 66 (63 pass, 3 skipped) |
| 5 | Build report, audit | `685fcaf`, `a2f16ad` | `SIMULATOR_BUILD_REPORT.md`, `SIMPLICITY_AUDIT.md`, two small fixes | 63 |
| 6 | Stage 6 | `9a8a0ec` | Scenario set (13 scenarios of 30 days plus a 14-day demo), demo seed finder, Python 3.12 | 74 |
| 7 | Stage 7 | `fb98505` | Features, LightGBM models, per-event evaluation, thresholds (round V1) | |
| 8 | Stages 8 and 9 | `c6872bb`, `c1fee0c` | Backend (ingest, data quality, live scoring), replayer, dashboard, Makefile, CI | 95 |
| 9 | Dashboard simplified | `39f4a62` | Six tabs and an 11-step walkthrough | |
| 10 | Alerts say why | `e7791ef` | Alerts list what the model saw, in plain sentences | 96 |
| 11 | Check predictions | `7cfb327` | Strict call check, repair records, Track record tab, alert hold (round V2) | 100 |
| 12 | 72 h and accuracy | `557b6fd` | 72-hour call window, precision and recall reporting, accuracy on Overview (round V3) | 100 |
| 13 | Random faults | `7067894` | Round V4 | 103 |
| 14 | Random noise | `1796161` | Round V5 | 104 |
| 15 | Drift and shared noise | `d92fc50` | Round V6 | 107 |
| 16 | Wide validation and new features | `88bfbfc` | Rounds V7 and V8 | 107 (3 skipped) |

The simulator's own build, with its sanity-target tables for stages 0 to 4, is in `docs/SIMULATOR_BUILD_REPORT.md`.
This document starts the metrics at the first model evaluation.

## 4. Scenario set

| Item | Value |
|---|---|
| Belt | 6 km loop, 4 m/s, 120 zones of 50 m in 12 sections, 18 h running and 6 h stopped per day, 1-minute readings |
| Signals | Per zone: vibration RMS, vibration kurtosis, bearing temperature, acoustic level, belt offset. Drive: motor current, motor and gearbox temperature, drive and belt speed, slip, tension, throughput |
| Faults | Idler bearing (about 5 per 30 days) and drive overheating (about 1 per 30 days) |
| One 30-day scenario | 5,184,000 zone rows, about 120 MB, about 5 s to generate on an idle machine, peak memory about 746 to 775 MB |
| Splits at the start (V1) | train 8, validation 2, test 2, stress 1, demo 1 (14 days) |
| Splits now | train 8, **validation 6**, test 2, stress 1, demo 1 |
| Fault events | train 40 bearing and 8 drive; validation 30 and 6 (was 10 and 2); test 10 and 2; stress 8 and 2 |

## 5. Round V1: first evaluation (Stage 7), identical faults

**Simulator at this point:** every bearing fault had the same strength, the same growth shape and the same mix of
signals; sensor noise was one fixed level; no drift; no shared noise. Only the zone, timing and duration of a fault
varied.

**Model:** 72 zone features (shared with live scoring), six LightGBM models (an "abnormal" model, a "fails within
24 h" classifier, an hours-to-failure regressor, for the zones and for the drive). Thresholds tuned on validation for
at most 1 to 2 false alerts in 20.

### V1a: first threshold rule (log), commit before `fb98505`

The rule took the threshold with the longest warning time (alert threshold 0.05). Validation could not tell
thresholds apart (all were perfect), and the lowest one broke on noisier data.

| Split | Faults | Caught | False alerts | Precision | Median lead time (h) |
|---|---|---|---|---|---|
| Validation | 10 | 10 | 0 | 1.00 | 158.7 |
| Test | 10 | 10 | 0 | 1.00 | 121.0 |
| **Stress** | 8 | 8 | **62** | **0.11** | 138.4 |

Stress false alerts against the alert threshold, with validation false alerts at 0 for every threshold: 0.05 gave
62, 0.10 gave 46, 0.30 gave 24, 0.50 gave 16, 0.90 gave 12, 0.99 gave 1.

### V1b: cautious rule, commit `fb98505` (report `eval_report.md` at that commit)

The rule changed to: among thresholds that meet the target and detect the most events, take the highest that keeps
75% of the best lead time. Alert threshold 0.999 (zones), 0.75 (drive). **This change was made after seeing the
stress result, so stress and test are not clean held-out numbers for it.**

Bearing alerts (the model) and the two baselines:

| Method | Split | Faults | Caught | False alerts | Median lead (h) |
|---|---|---|---|---|---|
| LightGBM alert model | Validation | 10 | 10 | 0 | 124.6 |
| LightGBM alert model | **Test** | 10 | 10 | 0 | **97.6** |
| LightGBM alert model | Stress | 8 | 8 | 0 | 116.4 |
| Fixed bearing-temperature limit | Validation | 10 | 8 | 0 | 11.2 |
| Fixed bearing-temperature limit | Test | 10 | 9 | 0 | 14.0 |
| Fixed bearing-temperature limit | Stress | 8 | 7 | 0 | 7.5 |
| Neighbour comparison | Validation | 10 | 10 | 0 | 76.5 |
| Neighbour comparison | Test | 10 | 10 | 0 | 63.2 |
| Neighbour comparison | Stress | 8 | 8 | 0 | 67.3 |

Drive alerts:

| Method | Split | Faults | Caught | False alerts | Median lead (h) |
|---|---|---|---|---|---|
| LightGBM alert model | Validation | 2 | 2 | 0 | 6.2 |
| LightGBM alert model | Test | 2 | 2 | 0 | 7.1 |
| LightGBM alert model | Stress | 2 | 2 | 0 | 9.2 |
| Fixed motor-temperature limit | Validation / Test / Stress | 2 each | 1 / 0 / 0 | 0 | 5.0 / – / – |

Other V1 figures: median error of the hours-to-failure estimate on advanced-stage windows, bearings 3.9 h
(validation), 3.3 h (test), 4.1 h (stress); drive 4.1, 1.6 and 2.4 h. Over three training seeds all bearing
detection rates were 1.00 and the test median lead was 102.8 h (range 97.6 to 107.2).

**What this said:** perfect scores everywhere, and a neighbour-comparison baseline that also caught every bearing
fault. The model's gain was warning time (98 h against 63 h on test), not detection.

## 6. Round V2: strict check of "predicted failure", and the alert hold (commit `7cfb327`)

**Why:** the user asked how accurate the model is and how a "predicted failure" is checked. The check was loose: a
call counted as right if it fired on a part that was degrading. It never tested that the part failed within 24 hours.

**Simulator:** no change to the physics. New repair-record feed in the replayer (a repair record is sent only after
a repair ends, so past predictions can be scored live) and a Track record tab.

**Model and scoring:** each failure call is scored on whether the part really failed within 24 h of it. The call
threshold is tuned on that rule (zones 0.97, was 0.35). **Alert hold:** an open alert no longer closes while the
failure risk score is above 50% (found because the demo's drive alert opened at 16:09 and closed at 17:19, three
hours before the drive failed, and the old score only checked that an alert had opened). The evaluation gained
"alert open at failure".

### Failure calls against a 24 h window (report at `7cfb327`, seed 1)

Bearings (zone model, call threshold 0.97):

| Split | Faults | Calls | Came true within 24 h | Early | False | Call precision | Call recall | Median hours from call to failure | Time-estimate error at call (h) |
|---|---|---|---|---|---|---|---|---|---|
| Validation | 10 | 10 | 9 | 1 | 0 | 0.90 | 0.90 (9 of 10) | 22.5 | 3.7 |
| **Test** | 10 | 10 | **5** | **5** | 0 | **0.50** | **0.50 (5 of 10)** | 24.1 | 3.9 |
| Stress | 8 | 8 | 8 | 0 | 0 | 1.00 | 1.00 (8 of 8) | 19.7 | 2.5 |

Drive (call threshold 0.05):

| Split | Faults | Calls | Came true | Early | False | Call precision | Call recall | Median hours from call to failure | Time-estimate error at call (h) |
|---|---|---|---|---|---|---|---|---|---|
| Validation | 2 | 3 | 3 | 0 | 0 | 1.00 | 1.00 (2 of 2) | 8.2 | 5.3 |
| Test | 2 | 2 | 2 | 0 | 0 | 1.00 | 1.00 (2 of 2) | 7.0 | 1.8 |
| Stress | 2 | 3 | 3 | 0 | 0 | 1.00 | 1.00 (2 of 2) | 7.1 | 2.7 |

Hours from call to failure, bearings, test, seed 1: 10.4, 15.8, 16.0, 23.8, 23.9, 24.3, 25.1, 25.9, 33.2, 35.1.
Over three training seeds, test calls that came true: 5, 7 and 7 of 10. No call was ever made on a healthy part.

### Window-level numbers for the 24 h classifier (same report)

| Asset | Split | Windows | Truly within 24 h of failure | Precision | Recall | Average precision |
|---|---|---|---|---|---|---|
| Bearings | Validation | 1,002,351 | 1,440 (0.14%) | 0.97 | 0.84 | 0.96 |
| Bearings | Test | 1,002,352 | 1,440 (0.14%) | 0.95 | 0.84 | 0.96 |
| Bearings | Stress | 501,136 | 1,152 (0.23%) | 0.99 | 0.65 | 0.97 |
| Drive | Validation | 8,328 | 186 (2.23%) | 0.06 | 0.50 | 0.42 |
| Drive | Test | 8,329 | 241 (2.89%) | 0.04 | 0.75 | 0.39 |
| Drive | Stress | 4,151 | 256 (6.17%) | 0.09 | 0.83 | 0.50 |

### Live check on the demo (log)

| | Failures | Alert open at failure | Missed | False alarms | Failure calls true within 24 h |
|---|---|---|---|---|---|
| Before the alert hold | 4 | 3 | 1 (drive alert had closed) | 0 | 2 of 4 |
| After the alert hold | 4 | 4 | 0 | 0 | 3 of 4 |

With the hold, "alert open at failure" is 10 of 10 bearings and 2 of 2 drives on test.

**What this said:** the "24 hour" claim was only half true on test (5 of 10), and one alert had been closing too
early. Neither showed in the first evaluation.

## 7. Round V3: 72-hour window, precision and recall reporting, accuracy on Overview (commit `557b6fd`)

**Why:** the user chose the safer wording, 72 hours, and asked for live accuracy on the Overview page and for
precision and recall to be reported so they could be balanced.

**Simulator:** none.

**Model and reporting:** the failure call now claims 72 h (one shared setting, `CALL_HORIZON_HOURS`); the call
threshold was retuned for it (zones 0.15 to 0.30 over three seeds, drive 0.05); the classifier itself still
predicts 24 h, so the dashboard calls its output a "failure risk score". The report gained a precision and recall
table, validation trade-off tables for both thresholds, and `summary.json`. Overview gained a live accuracy strip.

### Precision and recall at a glance (report at `557b6fd`, seed 1)

| Asset | Split | Faults | Alert precision | Alert recall | Call precision (72 h) | Call recall |
|---|---|---|---|---|---|---|
| Bearings | Validation | 10 | 1.00 (10 right, 0 false) | 1.00 (10 of 10) | 1.00 (10 of 10) | 1.00 (10 of 10) |
| Bearings | Test | 10 | 1.00 (10 right, 0 false) | 1.00 (10 of 10) | 1.00 (10 of 10) | 1.00 (10 of 10) |
| Bearings | Stress | 8 | 1.00 (8 right, 0 false) | 1.00 (8 of 8) | 1.00 (8 of 8) | 1.00 (8 of 8) |
| Drive | Validation | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |
| Drive | Test | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |
| Drive | Stress | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |

Window-level, same threshold, test: bearings precision 0.83, recall 0.92, average precision 0.96 (three seeds:
precision 0.83, 0.77, 0.81; recall 0.92, 0.95, 0.94); drive precision 0.04, recall 0.75. Hours from call to failure,
bearings, test: 25, 27, 28, 28, 31, 33, 38, 48, 48, 63. Demo: 4 of 4 failures with an alert open, 4 of 4 calls true.

**Caveat:** the 100% is the claim being loosened, not the model improving. Every test call had been within 36 hours,
and 72 h was chosen after seeing that.

## 8. Diagnosis D1: is 100% real, or overfitting? (no code change; log)

**Why:** the user asked how random the simulator is and whether the model was overfit.

**What is random and what is fixed:** the zone, time, duration, noise, per-zone offsets and operating conditions are
random in every scenario. Strength, growth shape and signal mix of a fault were fixed, so every fault had the same
signature. Train and test share nothing (different seeds, different layouts and fault schedules).

**What the model relied on (shipped zone "abnormal" model, share of total gain):**

| Feature | Share |
|---|---|
| Vibration kurtosis versus peer zones, 6 h average | 72.7% |
| Vibration kurtosis versus peer zones, 24 h average | 12.4% |
| Acoustic level versus peer zones, 24 h average | 11.8% |
| Bearing temperature versus peer zones, 24 h average | 2.5% |
| Context features (running share, position and so on) | 0.1% |

**Seen against unseen (zone model, 24 h classifier):**

| Data | Window precision | Window recall | Average precision |
|---|---|---|---|
| Training scenarios (seen) | 0.983 | 1.000 | 1.000 |
| Test scenarios (unseen) | 0.834 | 0.921 | 0.962 |

**Out-of-distribution test on the frozen model** (new 30-day scenarios, about 8 bearing and 2 drive faults each):

| Case | Bearings caught | Bearing false alerts | Bearing calls true (of calls) | Drive caught | Drive false alerts |
|---|---|---|---|---|---|
| Same strength, new seed | 8 of 8 | 0 | 8 (of 8) | 2 of 2 | 0 |
| Half strength | 8 of 8 | 0 | 6 (of 6) | **0 of 2** | 0 |
| Quarter strength | 8 of 8 | 0 | 3 (of 3) | **0 of 2** | 0 |
| Different growth shape (near linear) | 8 of 8 | 0 | **1 (of 8)**, 7 early | 2 of 2 | **5** |
| Noise x4, full strength | 8 of 8 | **51** | 8 (of 8) | 1 of 2 | 0 |
| Noise x4, half strength | 8 of 8 | **41** | 6 (of 6) | 2 of 2 | 5 |
| Noise x8, full strength | 8 of 8 | **591** | 7 (of 8) | 1 of 2 | 0 |

**What this said:** the model had learned the one fault recipe the simulator used. Bearing detection was robust to
weaker faults, but the failure call and its time estimate were tied to the growth curve, the drive model was
brittle, and alerts broke under heavy noise.

## 9. Round V4: random fault strength, shape and signal mix (commit `7067894`)

**Why:** the finding above.

**Simulator:** each fault draws its own **strength** (0.3 to 1.5 of the base amplitudes), **growth exponent**
(1.5 to 3.5) and a **signal mix** (0.5 to 1.5, separately per signal, so one fault shows more in heat and another
in noise). Ranges are config settings; (1, 1) with exponent 2.5 gives the old fixed fault. Draws come after all
placement draws, so fault schedules and the demo seed 1015 did not change. Labels follow each fault's own exponent.
Generated faults: bearing strength 0.35 to 1.48 (mean 0.87), exponent 1.53 to 3.48 (mean 2.48); drive strength 0.38
to 1.34, exponent 1.64 to 3.45.

**Model:** retrained on the new scenarios; no change to features.

### Precision and recall at a glance (seed 1)

| Asset | Split | Faults | Alert precision | Alert recall | Call precision (72 h) | Call recall |
|---|---|---|---|---|---|---|
| Bearings | Validation | 10 | 1.00 (10 right, 0 false) | 1.00 (10 of 10) | 1.00 (8 of 8) | 0.80 (8 of 10) |
| Bearings | **Test** | 10 | 1.00 (10 right, 0 false) | 1.00 (10 of 10) | 1.00 (7 of 7) | **0.70 (7 of 10)** |
| Bearings | Stress | 8 | **0.73 (8 right, 3 false)** | 1.00 (8 of 8) | 0.67 (4 of 6) | **0.50 (4 of 8)** |
| Drive | Validation | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |
| Drive | Test | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |
| Drive | Stress | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |

### Robustness table (report at `7067894`)

Frozen model, fresh 30-day scenarios. Columns: faults, caught, alert still open at failure, false alerts, alert
precision, calls, calls true within 72 h, early, false calls.

Bearings:

| Case | Faults | Caught | Open at failure | False alerts | Alert precision | Calls | True | Early | False calls |
|---|---|---|---|---|---|---|---|---|---|
| Like training (new seed) | 8 | 8 | 8 | 0 | 1.00 | 7 | 7 | 0 | 0 |
| Faults 0.5 to 0.75 of normal strength | 8 | 8 | 8 | 0 | 1.00 | 7 | 7 | 0 | 0 |
| Faults 0.2 to 0.4 of normal strength | 8 | 8 | 8 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Slow, nearly linear growth | 8 | 8 | 8 | 0 | 1.00 | 7 | 6 | 1 | 0 |
| Very late, sharp growth | 8 | 8 | 8 | 0 | 1.00 | 6 | 6 | 0 | 0 |
| Noise x2 | 8 | 8 | 8 | 3 | 0.73 | 7 | 6 | 1 | 0 |
| Noise x4 | 8 | 8 | 8 | 114 | 0.07 | 8 | 7 | 1 | 0 |
| Noise x8 | 8 | 8 | 8 | 990 | 0.01 | 7 | 6 | 1 | 0 |

Drive:

| Case | Faults | Caught | Open at failure | False alerts | Alert precision | Calls | True | Early | False calls |
|---|---|---|---|---|---|---|---|---|---|
| Like training (new seed) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Faults 0.5 to 0.75 of normal strength | 2 | 2 | 1 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Faults 0.2 to 0.4 of normal strength | 2 | 1 | 0 | 8 | 0.11 | 9 | 1 | 0 | 8 |
| Slow, nearly linear growth | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Very late, sharp growth | 2 | 1 | 1 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Noise x2 | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Noise x4 | 2 | 2 | 1 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Noise x8 | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |

**Demo:** 4 failures, 3 warned (recall 0.75), the weak drive fault missed; leads Z107 150.4 h, Z065 83.4 h, Z064
76.1 h.

**What this said:** failure calls dropped (7 of 10 on test) because weak faults never raise one. Alerts still
caught every bearing fault at 0.2 to 0.4 of normal strength with no false alerts, because the simulator's healthy
kurtosis noise is tiny. Alerts still broke when noise rose.

## 10. Round V5: random sensor noise per scenario (commit `1796161`)

**Why:** the user asked to make sensor noise much larger and random.

**Simulator:** each scenario draws a noise multiplier log-uniformly from 1x to 5x, on its own random stream (nothing
else changes; tested). The level is in `run_meta.json` and the scenario report.

| Scenario | Noise level (x) |
|---|---|
| train_01 to train_08 | 2.14, 2.79, 4.46, 3.59, 2.04, 4.65, 3.03, 1.21 |
| val_01, val_02 (the two old validation scenarios) | 1.71, 1.91 |
| test_01, test_02 | 1.93, 2.52 |
| stress_01 (2.0 x its own draw of 1.74) | 3.47 |
| demo_01 | 4.71 |

**Model:** retrained; no change to features.

### Precision and recall at a glance (seed 1)

| Asset | Split | Faults | Alert precision | Alert recall | Call precision (72 h) | Call recall |
|---|---|---|---|---|---|---|
| Bearings | Validation | 10 | 1.00 (10 right, 0 false) | 1.00 (10 of 10) | 0.90 (9 of 10) | 0.90 (9 of 10) |
| Bearings | **Test** | 10 | 1.00 (10 right, 0 false) | 1.00 (10 of 10) | 0.90 (9 of 10) | 0.90 (9 of 10) |
| Bearings | Stress | 8 | 1.00 (8 right, 0 false) | 1.00 (8 of 8) | 0.71 (5 of 7) | 0.62 (5 of 8) |
| Drive | Validation | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (3 of 3) | 1.00 (2 of 2) |
| Drive | Test | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |
| Drive | Stress | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |

### Robustness table (report at `1796161`)

Noise is now fixed exactly in the "fixed" cases, and drift and shared noise did not exist yet.

Bearings:

| Case | Faults | Caught | Open at failure | False alerts | Alert precision | Calls | True | Early | False calls |
|---|---|---|---|---|---|---|---|---|---|
| Like training (new seed, random noise 1 to 5x) | 8 | 8 | 8 | 0 | 1.00 | 8 | 6 | 2 | 0 |
| Faults 0.5 to 0.75 of normal strength | 8 | 8 | 8 | 0 | 1.00 | 8 | 8 | 0 | 0 |
| Faults 0.2 to 0.4 of normal strength | 8 | 8 | 8 | 0 | 1.00 | 6 | 6 | 0 | 0 |
| Slow, nearly linear growth | 8 | 8 | 8 | 0 | 1.00 | 7 | 4 | 3 | 0 |
| Very late, sharp growth | 8 | 8 | 8 | 0 | 1.00 | 8 | 8 | 0 | 0 |
| Noise fixed at 1x (quiet) | 8 | 8 | 8 | 0 | 1.00 | 8 | 7 | 1 | 0 |
| Noise fixed at 3x | 8 | 8 | 8 | 0 | 1.00 | 7 | 6 | 1 | 0 |
| Noise fixed at 5x (top of training range) | 8 | 8 | 8 | 0 | 1.00 | 8 | 6 | 2 | 0 |
| Noise fixed at 8x (beyond training range) | 8 | 8 | 8 | **23** | **0.26** | 8 | 7 | 1 | 0 |
| Noise fixed at 12x (far beyond) | 8 | 8 | 8 | **150** | **0.05** | 8 | 7 | 1 | 0 |

Drive:

| Case | Faults | Caught | Open at failure | False alerts | Alert precision | Calls | True | Early | False calls |
|---|---|---|---|---|---|---|---|---|---|
| Like training (new seed, random noise 1 to 5x) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Faults 0.5 to 0.75 of normal strength | 2 | 2 | 1 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Faults 0.2 to 0.4 of normal strength | 2 | 1 | 0 | 6 | 0.14 | 7 | 1 | 0 | 6 |
| Slow, nearly linear growth | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Very late, sharp growth | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Noise fixed at 1x (quiet) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Noise fixed at 3x | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Noise fixed at 5x (top of training range) | 2 | 2 | 1 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Noise fixed at 8x (beyond training range) | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Noise fixed at 12x (far beyond) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |

**Demo:** unchanged (3 of 4 warned, leads 150.4, 83.4 and 76.1 h), although its data was 4.7x noisier. Verified that
the demo data, model and database were all new.

**What this said:** after training on 1x to 5x noise the model handles that range with no false alerts, and still
catches every bearing fault at 12x. It breaks only beyond what it was trained on (8x: 23 false alerts). Noise
matters little because the model works from 6 h and 24 h averages, which smooth independent noise away (the noise in
an average of about 1,400 readings is roughly 1/37 of the noise in one reading).

## 11. Round V6: sensor drift and section-shared noise (commit `d92fc50`)

**Why:** averaging cannot remove drift or shared noise, and real sensors have both.

**Simulator (two new random effects, each drawn per scenario on its own stream):**
- **Drift:** each zone sensor wanders away from its true value and forgets itself over about 3.5 days. Long-run size
  at the base level: 1.5 C (temperature), 1.5 dB (acoustic), 4% (vibration), 0.2 (kurtosis), 2 mm (offset), times
  the scenario's level (0 to 3x).
- **Shared noise:** every zone in a splice section gets the same disturbance (a sunlit stretch reads warm, a noisy
  cable run), times the scenario's level (0 to 3x).
- The stress scenario is held at the top third of both ranges (it had first drawn almost none: drift 0.04x).

| Scenario | Noise x | Drift x | Shared x |
|---|---|---|---|
| train_01 to train_08 | 2.14, 2.79, 4.46, 3.59, 2.04, 4.65, 3.03, 1.21 | 1.58, 2.9, 2.58, 2.09, 0.15, 0.21, 2.46, 1.0 | 1.1, 0.49, 2.14, 0.31, 2.12, 2.84, 1.53, 0.43 |
| val_01, val_02 (random draws) | 1.71, 1.91 | 1.93, 0.11 | 0.19, 0.85 |
| test_01 | 1.93 | 2.67 | 1.72 |
| test_02 | 2.52 | 0.17 | 2.42 |
| stress_01 (held high) | 3.47 | 2.01 | 2.17 |
| demo_01 | 4.71 | 2.61 | 0.01 |

### Probe before retraining (log): the previous model on drift and shared noise

| Case (other messiness off) | Bearings caught | False alerts |
|---|---|---|
| Clean | 8 of 8 | 0 |
| Drift 3x | 7 of 8 | **693** |
| Drift 6x | 7 of 8 | **776** |
| Shared noise 3x | 8 of 8 | **394** |
| Shared noise 6x | 8 of 8 | **819** |
| All three at the top of their ranges | 7 of 8 | **1,134** |

This showed the new effects were a real test.

**Model:** retrained on the new data; no change to features.

### Precision and recall at a glance (seed 1)

| Asset | Split | Faults | Alert precision | Alert recall | Call precision (72 h) | Call recall |
|---|---|---|---|---|---|---|
| Bearings | Validation | 10 | 0.91 (10 right, 1 false) | 1.00 (10 of 10) | 1.00 (6 of 6) | 0.60 (6 of 10) |
| Bearings | **Test** | 10 | **0.38 (8 right, 13 false)** | **0.80 (8 of 10)** | 1.00 (6 of 6) | 0.60 (6 of 10) |
| Bearings | Stress | 8 | **0.58 (7 right, 5 false)** | 0.88 (7 of 8) | 1.00 (8 of 8) | 0.88 (7 of 8) |
| Drive | Validation | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |
| Drive | Test | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |
| Drive | Stress | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |

Alert thresholds tuned on validation: zones 0.96, 0.94 and 0.97 over three seeds (validation precision 0.91); failure
call 0.99, 0.99 and 0.96.

### Robustness table (report at `d92fc50`)

Bearings:

| Case | Faults | Caught | Open at failure | False alerts | Alert precision | Calls | True | Early | False calls |
|---|---|---|---|---|---|---|---|---|---|
| Like training (everything random) | 8 | 7 | 7 | 0 | 1.00 | 6 | 6 | 0 | 0 |
| Faults 0.5 to 0.75 of normal strength | 8 | 7 | 7 | 0 | 1.00 | 3 | 3 | 0 | 0 |
| Faults 0.2 to 0.4 of normal strength | 8 | 2 | 2 | 5 | 0.29 | 0 | 0 | 0 | 0 |
| Slow, nearly linear growth | 8 | 8 | 7 | 6 | 0.57 | 7 | 5 | 2 | 0 |
| Very late, sharp growth | 8 | 6 | 6 | 0 | 1.00 | 5 | 5 | 0 | 0 |
| Clean: no noise change, no drift, no shared noise | 8 | 8 | 8 | 0 | 1.00 | 5 | 4 | 1 | 0 |
| Noise fixed at 5x (top of training range) | 8 | 8 | 8 | 0 | 1.00 | 7 | 7 | 0 | 0 |
| Noise fixed at 8x (beyond training range) | 8 | 8 | 8 | 0 | 1.00 | 7 | 7 | 0 | 0 |
| Noise fixed at 12x (far beyond) | 8 | 8 | 7 | 1 | 0.89 | 4 | 4 | 0 | 0 |
| Drift fixed at 3x (top of training range) | 8 | 4 | 4 | **9** | **0.31** | 2 | 2 | 0 | 0 |
| Drift fixed at 6x (beyond training range) | 8 | 4 | 3 | **98** | **0.04** | 13 | 2 | 1 | 10 |
| Drift fixed at 10x (far beyond) | 8 | 4 | 3 | **250** | **0.02** | 51 | 5 | 1 | 45 |
| Shared noise fixed at 3x (top of training range) | 8 | 6 | 6 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Shared noise fixed at 6x (beyond training range) | 8 | 8 | 8 | 2 | 0.80 | 8 | 8 | 0 | 0 |
| All three at the top of the training range | 8 | 5 | 5 | **29** | **0.15** | 3 | 3 | 0 | 0 |

Drive:

| Case | Faults | Caught | Open at failure | False alerts | Alert precision | Calls | True | Early | False calls |
|---|---|---|---|---|---|---|---|---|---|
| Like training (everything random) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Faults 0.5 to 0.75 of normal strength | 2 | 2 | 1 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Faults 0.2 to 0.4 of normal strength | 2 | 1 | 0 | 5 | 0.17 | 6 | 1 | 0 | 5 |
| Slow, nearly linear growth | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Very late, sharp growth | 2 | 1 | 1 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Clean: no noise change, no drift, no shared noise | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Noise fixed at 5x (top of training range) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Noise fixed at 8x (beyond training range) | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |
| Noise fixed at 12x (far beyond) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Drift fixed at 3x (top of training range) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Drift fixed at 6x (beyond training range) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Drift fixed at 10x (far beyond) | 2 | 2 | 2 | 5 | 0.29 | 7 | 2 | 0 | 5 |
| Shared noise fixed at 3x (top of training range) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| Shared noise fixed at 6x (beyond training range) | 2 | 1 | 1 | 0 | 1.00 | 1 | 1 | 0 | 0 |
| All three at the top of the training range | 2 | 2 | 2 | 0 | 1.00 | 2 | 2 | 0 | 0 |

**Demo:** 3 of 4 failures warned, the weak drive fault missed; leads shortened to Z107 61.7 h, Z065 26.5 h, Z064
15.9 h (from 150.4, 83.4, 76.1 h).

**What this said, and why:** **the model no longer met the target.** Validation looked fine (precision 0.91) but its
two scenarios had happened to draw low shared noise (0.19x and 0.85x), while test_01 drew drift 2.67x and shared
noise 1.72x and test_02 shared noise 2.42x. The alert threshold was tuned on validation only, so it was too low for
the noisier test data. No threshold was changed after seeing test.

## 12. Round V7: wide validation alone (log; no separate commit)

**Why:** the user chose to fix the validation set first.

**Simulator:** none to the physics. The two random validation scenarios were replaced by six scenarios at fixed
levels, so the thresholds are tuned on the whole range the other splits draw from. Test is untouched. The threshold
candidates were extended above 0.999 (up to 0.99999).

| Validation scenario | Noise x | Drift x | Shared x | Faults (bearing / drive) |
|---|---|---|---|---|
| val_01 (quiet) | 1.0 | 0 | 0 | 5 / 1 |
| val_02 (mixed) | 3.0 | 1.5 | 1.5 | 5 / 1 |
| val_03 (noisy) | 5.0 | 0 | 0 | 5 / 1 |
| val_04 (drifting) | 2.0 | 3.0 | 0 | 5 / 1 |
| val_05 (shared noise) | 2.0 | 0 | 3.0 | 5 / 1 |
| val_06 (all high) | 4.0 | 3.0 | 3.0 | 5 / 1 |

**Model:** retrained and the thresholds re-tuned on the six scenarios (zone alert threshold 0.997, call threshold
0.4; drive alert 0.95, call 0.05); no change to features.

### Precision and recall at a glance (seed 1)

| Asset | Split | Faults | Alert precision | Alert recall | Call precision (72 h) | Call recall |
|---|---|---|---|---|---|---|
| Bearings | Validation | 30 | 0.96 (22 right, 1 false) | 0.73 (22 of 30) | 0.92 (23 of 25) | 0.67 (20 of 30) |
| Bearings | **Test** | 10 | **1.00 (6 right, 0 false)** | **0.60 (6 of 10)** | 1.00 (6 of 6) | 0.60 (6 of 10) |
| Bearings | Stress | 8 | 1.00 (7 right, 0 false) | 0.88 (7 of 8) | 1.00 (7 of 7) | 0.88 (7 of 8) |
| Drive | Validation | 6 | 1.00 (1 right, 0 false) | **0.17 (1 of 6)** | 1.00 (1 of 1) | 0.17 (1 of 6) |
| Drive | Test | 2 | 1.00 (1 right, 0 false) | 0.50 (1 of 2) | 1.00 (1 of 1) | 0.50 (1 of 2) |
| Drive | Stress | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |

### Where the misses were (log): per-scenario diagnosis at the new thresholds

Bearings (alert threshold 0.997). The scores are the model's abnormal score: the median over advanced-stage windows
of the failing parts, and the 99.9th percentile over healthy windows.

| Scenario | Messiness | Caught | False | Advanced median | Healthy 99.9th percentile |
|---|---|---|---|---|---|
| val_01 | quiet | 3 of 5 | 0 | 0.935 | 0.044 |
| val_02 | noise 3, drift 1.5, shared 1.5 | 3 of 5 | 0 | 0.576 | 0.834 |
| val_03 | noise 5 | 4 of 5 | 0 | 0.987 | 0.048 |
| val_04 | noise 2, drift 3 | 4 of 5 | 0 | **0.173** | **0.980** |
| val_05 | noise 2, shared 3 | 4 of 5 | 0 | 0.729 | 0.162 |
| val_06 | noise 4, drift 3, shared 3 | 4 of 5 | 1 | 0.693 | 0.978 |
| test_01 | drift 2.7, shared 1.7 | 3 of 5 | 0 | **0.058** | **0.959** |
| test_02 | drift 0.2, shared 2.4 | 3 of 5 | 0 | 0.912 | 0.216 |

Drive (alert threshold 0.95): caught 0 of 1 in val_01, 1 of 1 in val_02, 0 of 1 in val_03, val_04, val_05 and val_06,
0 of 1 in test_01 and 1 of 1 in test_02. No scenario had a false drive alert.

**What this said:** wide validation fixed precision (13 false alerts on test became 0) at the cost of recall (8 of
10 became 6 of 10). With drift at 1.5x to 3x, healthy zones score 0.96 to 0.98 while real faults score as low as
0.06 to 0.17, so no threshold separates them. A robustness run for this round was stopped, so there is no table.

## 13. Round V8: spread-normalised and cross-signal agreement features (commit `88bfbfc`, current state)

**Why:** the diagnosis above. Drift lifts every deviation; a fault differs from drift in two ways: it stands out
against the other zones, and it moves several signals together.

**Simulator:** none.

**Model (`model/features.py`):** two new feature groups, 72 zone features in all (up from 57):
- each zone's 6 h and 24 h deviation in units of the spread of all zones in the same window (median and MAD),
  which removes the scenario-wide noise and drift level;
- a count of how many of heat, noise, impact spikes and vibration agree, plus the minimum and mean of their scaled
  deviations (a drifting sensor moves one signal on its own).
Training and live scoring use the same function (a test checks it). Thresholds re-tuned on the six validation
scenarios: zone alert 0.997 (validation precision 1.00, 26 of 30), call 0.99 (17 of 18 calls true); drive alert 0.95,
call 0.05.

### Precision and recall at a glance (seed 1)

| Asset | Split | Faults | Alert precision | Alert recall | Call precision (72 h) | Call recall |
|---|---|---|---|---|---|---|
| Bearings | Validation | 30 | 1.00 (26 right, 0 false) | 0.87 (26 of 30) | 0.94 (17 of 18) | 0.57 (17 of 30) |
| Bearings | **Test** | 10 | **1.00 (8 right, 0 false)** | **0.80 (8 of 10)** | 1.00 (6 of 6) | 0.60 (6 of 10) |
| Bearings | Stress | 8 | 1.00 (7 right, 0 false) | 0.88 (7 of 8) | 1.00 (4 of 4) | 0.50 (4 of 8) |
| Drive | Validation | 6 | 1.00 (1 right, 0 false) | 0.17 (1 of 6) | 1.00 (1 of 1) | 0.17 (1 of 6) |
| Drive | Test | 2 | 1.00 (1 right, 0 false) | 0.50 (1 of 2) | 1.00 (1 of 1) | 0.50 (1 of 2) |
| Drive | Stress | 2 | 1.00 (2 right, 0 false) | 1.00 (2 of 2) | 1.00 (2 of 2) | 1.00 (2 of 2) |

### Robustness table (report at `88bfbfc`)

Bearings:

| Case | Faults | Caught | Open at failure | False alerts | Alert precision | Calls | True within 72 h |
|---|---|---|---|---|---|---|---|
| Like training (everything random) | 8 | 6 | 6 | 0 | 1.00 | 6 | 6 |
| Faults 0.5 to 0.75 of normal strength | 8 | 7 | 7 | 0 | 1.00 | 3 | 3 |
| Faults 0.2 to 0.4 of normal strength | 8 | 1 | 0 | 0 | 1.00 | 0 | 0 |
| Slow, nearly linear growth | 8 | 7 | 6 | 0 | 1.00 | 3 | 2 |
| Very late, sharp growth | 8 | 8 | 8 | 0 | 1.00 | 5 | 5 |
| Clean: no noise change, no drift, no shared noise | 8 | 8 | 8 | 0 | 1.00 | 7 | 7 |
| Noise fixed at 5x (top of training range) | 8 | 8 | 8 | 0 | 1.00 | 8 | 8 |
| Noise fixed at 8x (beyond training range) | 8 | 8 | 8 | 0 | 1.00 | 7 | 6 |
| Noise fixed at 12x (far beyond) | 8 | 8 | 8 | 1 | 0.89 | 5 | 5 |
| Drift fixed at 3x (top of training range) | 8 | **3** | 3 | 0 | 1.00 | 1 | 1 |
| Drift fixed at 6x (beyond training range) | 8 | 2 | 2 | 0 | 1.00 | 2 | 2 |
| Drift fixed at 10x (far beyond) | 8 | 0 | 0 | 1 | 0.00 | 0 | 0 |
| Shared noise fixed at 3x (top of training range) | 8 | 8 | 8 | 1 | 0.89 | 3 | 3 |
| Shared noise fixed at 6x (beyond training range) | 8 | 8 | 8 | 11 | 0.42 | 8 | 8 |
| All three at the top of the training range | 8 | **3** | 3 | 1 | 0.75 | 3 | 3 |

Drive:

| Case | Caught (of 2) | False alerts | Alert precision |
|---|---|---|---|
| Like training (everything random) | 0 | 0 | – |
| Faults 0.5 to 0.75 of normal strength | 2 | 0 | 1.00 |
| Faults 0.2 to 0.4 of normal strength | 1 | 2 | 0.33 |
| Slow, nearly linear growth | 2 | 0 | 1.00 |
| Very late, sharp growth | 1 | 0 | 1.00 |
| Clean: no noise change, no drift, no shared noise | 1 | 0 | 1.00 |
| Noise fixed at 5x (top of training range) | 1 | 0 | 1.00 |
| Noise fixed at 8x (beyond training range) | 0 | 0 | – |
| Noise fixed at 12x (far beyond) | 1 | 0 | 1.00 |
| Drift fixed at 3x (top of training range) | 1 | 0 | 1.00 |
| Drift fixed at 6x (beyond training range) | 1 | 0 | 1.00 |
| Drift fixed at 10x (far beyond) | 2 | 0 | 1.00 |
| Shared noise fixed at 3x (top of training range) | 1 | 0 | 1.00 |
| Shared noise fixed at 6x (beyond training range) | 1 | 0 | 1.00 |
| All three at the top of the training range | 2 | 0 | 1.00 |

**What this said:** the new features recovered recall without bringing the false alerts back (test 6 of 10 became
8 of 10, validation 22 of 30 became 26 of 30, still no false alerts), and removed almost all false alerts under
drift (drift 3x: 9 became 0; all three at the top of their ranges: 29 became 1). **The cost:** under strong drift
the model stays quiet and misses weak faults (drift 3x catches 3 of 8). The drive model did not change.

## 14. Side by side: the held-out test and stress results after each round

Bearings, alerts and 72 h failure calls (the call window was 24 h in V2).

**Test (10 bearing faults):**

| Round | Alerts caught | False alerts | Alert precision | Call precision | Call recall |
|---|---|---|---|---|---|
| V1b first evaluation, identical faults | 10 | 0 | 1.00 | n/a (24 h classifier at 0.35 caught 10) | n/a |
| V2 strict 24 h check | 10 | 0 | 1.00 | 0.50 (5 of 10) | 0.50 |
| V3 72 h window | 10 | 0 | 1.00 | 1.00 (10 of 10) | 1.00 |
| V4 random faults | 10 | 0 | 1.00 | 1.00 (7 of 7) | 0.70 |
| V5 random noise | 10 | 0 | 1.00 | 0.90 (9 of 10) | 0.90 |
| V6 drift and shared noise | 8 | **13** | **0.38** | 1.00 (6 of 6) | 0.60 |
| V7 wide validation alone | 6 | 0 | 1.00 | 1.00 (6 of 6) | 0.60 |
| **V8 new features (current)** | **8** | **0** | **1.00** | 1.00 (6 of 6) | 0.60 |

**Stress (8 bearing faults):**

| Round | Alerts caught | False alerts | Alert precision | Call precision | Call recall |
|---|---|---|---|---|---|
| V1a first threshold rule | 8 | **62** | **0.11** | n/a | n/a |
| V1b cautious rule | 8 | 0 | 1.00 | n/a | n/a |
| V2 strict 24 h check | 8 | 0 | 1.00 | 1.00 (8 of 8) | 1.00 |
| V3 72 h window | 8 | 0 | 1.00 | 1.00 (8 of 8) | 1.00 |
| V4 random faults | 8 | 3 | 0.73 | 0.67 (4 of 6) | 0.50 |
| V5 random noise | 8 | 0 | 1.00 | 0.71 (5 of 7) | 0.62 |
| V6 drift and shared noise | 7 | 5 | 0.58 | 1.00 (8 of 8) | 0.88 |
| V7 wide validation alone | 7 | 0 | 1.00 | 1.00 (7 of 7) | 0.88 |
| **V8 new features (current)** | **7** | **0** | **1.00** | 1.00 (4 of 4) | 0.50 |

**Validation (bearings; 10 faults until V6, 30 from V7):**

| Round | Faults | Alerts caught | False alerts | Alert precision | Call precision | Call recall |
|---|---|---|---|---|---|---|
| V2 | 10 | 10 | 0 | 1.00 | 0.90 | 0.90 |
| V3 | 10 | 10 | 0 | 1.00 | 1.00 | 1.00 |
| V4 | 10 | 10 | 0 | 1.00 | 1.00 (8 of 8) | 0.80 |
| V5 | 10 | 10 | 0 | 1.00 | 0.90 | 0.90 |
| V6 | 10 | 10 | 1 | 0.91 | 1.00 (6 of 6) | 0.60 |
| V7 | 30 | 22 | 1 | 0.96 | 0.92 (23 of 25) | 0.67 |
| **V8** | 30 | 26 | 0 | 1.00 | 0.94 (17 of 18) | 0.57 |

**Drive, test (2 faults):** 2 of 2 caught with no false alerts through V6; **1 of 2 in V7 and V8.** Drive validation
(6 faults from V7): 1 of 6 in both V7 and V8. Drive stress: 2 of 2 throughout.

## 15. Test suite and checks

| Stage | Tests passed (skipped) |
|---|---|
| End of simulator stage 4 | 63 (3) |
| After stage 6 | 74 (3) |
| After stages 8 and 9 | 95 (3) |
| After "alerts say why" | 96 (3) |
| After repair records and Track record | 100 (3) |
| V4 random faults | 103 (3) |
| V5 random noise | 104 (3) |
| V6 drift and shared noise | 107 (3) |
| V8 features (current) | 107 (3) |

The three skipped tests are intentional (they need data that does not exist yet). `ruff` is clean. CI runs the Python
tests and the dashboard build.

Tests added with these rounds include: a failure call is scored on what happened next; per-fault strength, shape and
mix are random, reproducible and validated; noise level, drift and shared noise are random per scenario and change
nothing else; shared noise is identical within a section; the repair feed reads only four columns and releases a
record only after the repair ends; live features equal training features.

## 16. Known problems and open items

| Item | State |
|---|---|
| **Drive model is weak** | Trained on 8 events. Catches 1 of 6 on validation and 1 of 2 on test; fails at low strength. Suggested fix: more drive events in the training scenarios |
| **Recall under strong drift** | Drift at 3x catches 3 of 8 bearing faults; the model stays quiet instead of guessing |
| **Call recall is low** | 0.60 on test: weak faults do not raise a failure call |
| **Small samples** | 10 bearing and 2 drive faults on test; one training seed for the latest rounds |
| **The model learns the simulator** | Bearing detection leans on kurtosis and acoustic deviation from peer zones. A real belt would be noisier and its faults would not all follow one family of curves |
| **Decisions made after seeing results** | The 72 h window and the cautious threshold rule were chosen after seeing test and stress. They loosen the claim; they are not tested claims |
| **Live track record rests on 4 failures** | One curated demo scenario |
| **Vision** | Placeholders only (empty tables, endpoints that say "not enabled", a Vision tab); the dataset is still downloading |
| **Not built** | The precomputed-predictions fallback; the README |
| **Not retuned** | No threshold was changed after seeing test in V4 to V8 |
