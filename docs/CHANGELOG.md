# Changelog

- **Stage 0**: repo initialised with `.gitignore` (`data/`, caches, `.venv/`) and the project docs.
- **Stage 1**: healthy baseline (config, layout, conditions, healthy signals, generator, CLI, tests, sanity plot).
  Each scenario is split into `public/` (the only folder pipeline code may read, through `simulator.io.load_public`)
  and `truth/` (everything else). Every public table is checked against a column allowlist before it is written. Two
  changes from the spec, both to stop leakage. First, the per-zone random biases moved to `truth/layout_truth.parquet`,
  because they are hidden simulator parameters. Second, `config_used.yaml` and `run_meta.json` moved to `truth/`,
  because from Stage 2 on they carry fault rates and counts. `public/manifest.json` replaces them for the pipeline.
- **Stage 2**: belt coordinates and the three faults (idler bearing, belt damage, drive overheat), with repair stops.
  `truth/faults.parquet` holds the events; `truth/conditions.parquet` gains `belt_distance_m`. Choices the spec left
  open or that differ from it:
  - Belt travel D(t) is integrated from the clean `belt_speed_ms` signal (generated first), so the public belt speed
    is consistent with the true belt position.
  - A failure drawn inside a stop moves to the next running minute, because an asset fails while it is working.
  - Temperature effects pass through the same thermal lag as the healthy signal, so they do not vanish the instant
    the belt stops. Every effect is scaled by speed, so a stopped belt shows no fault signature.
  - Event counts for scenarios that are not a multiple of 30 days use the fractional part as a probability.
  - Two events on the drive keep at least 24 h apart (`faults.same_asset_gap_hours`).
- **Belt damage removed** (requested after Stage 2): the belt cut / splice defect fault is gone from the config, the
  scheduler, the effects, the tests and the plot. Version 1 now has two faults: idler bearing and drive overheat.
  Belt coordinates and `belt_distance_m` stay, because image capture events (Stage 5) still need them. Fault
  schedules for a given seed changed, because the belt-damage draws no longer happen. `SIMULATOR_SPEC.md` and
  `PROJECT_OVERVIEW.md` still describe the fault and have not been edited.
- **Docs updated for the belt-damage removal**: `SIMULATOR_SPEC.md` and `PROJECT_OVERVIEW.md` now describe two sensor
  faults, vision-only belt damage through "visual defects" at belt coordinates (Stage 5), and a 2 s capture interval
  so every belt point is seen on every lap (the earlier 30 s value saw a defect on about 8% of laps).
- **Stage 3**: hard negatives and labels. Heavy-load shifts, heat waves and dusty spells edit the operating conditions
  before the physics runs; the start-up transient multiplies vibration for 10 minutes after every start, including
  restarts after a repair. `truth/hard_negatives.parquet` and `truth/labels.parquet` are new. Choices of ours:
  - `hours_to_failure` and `fails_within_24h` are set only while a fault is developing (from onset). Before onset
    nothing in the data can predict the failure, so a positive label there would only be noise.
  - `labels.parquet` also carries `severity` and `fault_id`, and counts start-up transients as hard-negative periods.
  - A heat wave fades in and out over 3 h (`hard_negatives.heat_wave.edge_hours`) instead of stepping.
  - A heavy-load shift holds load at its drawn level, so load is flat for the length of the event.
  - Categorical columns are built from integer codes, which cut a 30-day generation to about 2 s.
- **Stage 4**: messiness. About 1% of readings are dropped at random per signal (zone and belt-wide), whole-suite
  outages silence one zone for 1 to 6 h, and stuck sensors freeze one signal in one zone for 2 to 48 h. All of it is
  injected after the physics, the fault effects and the labels, so ground truth is unchanged (tested). Events go to
  `truth/sensor_faults.parquet`; the measured missing rate goes to `truth/run_meta.json`. Choices of ours:
  - Injection order is stuck, then outage, then random loss, so a stuck sensor still loses packets.
  - A stuck sensor holds the last value it reported before the freeze; `sensor_faults.parquet` records it.
  - Rows are never dropped: a missing reading is a NaN in its row, and `timestamp` and `run_state` are always present.
  - Sensor faults may overlap each other and equipment faults; nothing prevents a stuck sensor in a failing zone.
- **Simplicity audit** (`docs/SIMPLICITY_AUDIT.md`): removed a wall-clock assertion that failed on a busy machine, and
  tidied stage leftovers (unused `Conditions.run_state`, optional `Scenario` fields, an unneeded `__all__`). Output is
  unchanged, except that an empty `faults` table now has timestamp-typed time columns.
- **Build plan** (`docs/BUILD_PLAN.md`) added, with two corrections from the user: vision is deferred with placeholders
  (not dropped), and thresholds target at most 1 to 2 false alerts in 20 (not 1 false alarm per zone per month).
- **Python 3.12**: the environment was rebuilt on Python 3.12.16 (it was 3.14.4) so LightGBM wheels install; versions
  are pinned in `requirements.txt` and `.python-version`. All earlier tests pass unchanged.
- **Stage 6**: scenario set. `python -m simulator.scenario_set` builds 13 thirty-day scenarios (8 train, 2 validation,
  2 test, 1 stress) and the 14-day demo from `configs/scenarios/*.yaml`, one process per scenario, skipping any whose
  config hash is unchanged. `scripts/find_demo_seed.py` picked seed 1015 for the demo from schedules alone. Outputs:
  `data/scenario_set.json`, `docs/scenario_set_report.md`, `docs/stage6_scenario_set.png`. Notes:
  - **`--workers 2` is allowed but slower here.** Measured: about 5 s per scenario with one worker, about 115 s each
    with two. The default is one worker; the full set took about 7 minutes.
  - Peak memory is about 775 MB per scenario, against the 3 GB limit.
  - Three demo rules were made stricter than the plan so the demo shows them: the stuck sensor needs 2 h of running
    time, the outage must come after day 1 (after the history preload), and both sit on a healthy zone.
  - `SIMULATOR_SPEC.md` carries a status note listing what the build plan supersedes.
- **Stage 7**: model data, models, evaluation. `model/features.py` (one module for training and live scoring) turns
  1-minute readings into 10-minute windows and then into 57 zone and 34 drive features. `python -m
  scripts.make_model_data`, `python -m model.train` and `python -m scripts.score --seeds 1 2 3` build the tables,
  the six LightGBM models and `model/artifacts/eval_report.md`. Results with frozen thresholds: test 10 of 10
  idler faults and 2 of 2 drive faults detected with no false alerts; stress 8 of 8 and 2 of 2 with none. Notes:
  - **Threshold target changed (user):** at most 1 to 2 false alerts in 20, not "1 false alarm per zone per month".
  - **Threshold rule changed after seeing the stress result.** The first rule took the threshold with the longest
    lead time (0.05) and gave 62 false alerts on stress. Validation cannot separate thresholds (all are perfect
    there), so the rule now takes the most cautious threshold that keeps 75% of the best lead time (0.999). The
    stress and test numbers were seen before this change, so they are not untouched held-out numbers for it.
  - **Lead time is days, not hours** (median about 100 h on test). The plan expected hours. The simulator's faults
    grow smoothly, so 24-hour averages of acoustic level and kurtosis reveal them early. A real bearing would not
    be this kind; the README must say so.
  - **The neighbour-comparison baseline also detects every idler fault** with no false alerts; the model's gain is
    lead time (98 h against 63 h on test), not detection. The fixed temperature limit misses 1 to 2 in 10.
  - Features compare each zone with the median of the zones on its run, so weather, load and dust cancel without
    being measured. A signal that repeats exactly within a window is dropped as stuck.
  - Alerts hold while the belt is stopped (nothing can be learned about a bearing that is not turning).
  - Normal zone windows are thinned to 20% for training (memory); all other windows are kept.
  - LightGBM's native interface is used, so scikit-learn is not a dependency.
- **Stage 8**: backend and replayer. `backend/engine.py` validates each batch (reject, de-duplicate, range-check),
  stores it, advances a watermark so later steps see minutes in order, tracks sensor health (stuck after 15
  identical running values, no data after 5 silent minutes) and scores every 10-minute window with the same
  feature code as training. `replayer/run.py` posts one batch per simulated minute over HTTP and can inject
  duplicates, out-of-range values, unknown zones and out-of-order batches. `python -m scripts.demo` runs both.
  A full 14-day replay raised alerts for all three bearing faults and the drive fault, and the ingest counts
  equal the replayer's injected counts exactly. Departures from the build plan, with reasons:
  - **No queue and worker thread.** Each batch is processed in the request under one lock. A worker thread was
    built first and was slower; the lock still gives one writer, and the waiting sender is the back-pressure.
  - **`synchronous=OFF`** (the plan says NORMAL). On a busy disk, syncing cut ingest from about 135 to 24 batches
    per second. The database is deleted and rebuilt on every start, so durability buys nothing here.
  - **Top replay speed on the development machine is about 2 simulated hours per second**, not 3. The setting
    allows 3; the replayer simply runs behind. The default of 1 holds.
  - **The precomputed-predictions fallback is not built** (it is second in the plan's cut order).
  - Per-window deviations moved from stage 2 to stage 1 of `model/features.py` so live scoring computes them once
    per window. Model tables are bit-identical before and after, so the trained models did not change.
  - Data-quality alerts are labelled `medium` directly: the plan gives them severity 0.5, which its own bands
    ("under 0.5 medium") would call `high`.
  - A time to failure is shown only when the failure call is made, so the dashboard never shows a time next to
    "not predicted yet".
  - Vision placeholders: empty `captures` and `vision_detections` tables and `/api/vision/*` endpoints that
    return `enabled: false`.
- **Stage 9**: dashboard (`frontend/`, React 18, Vite, TypeScript, Tailwind, TanStack Query, Recharts). Racetrack
  map of 120 zones and the drive in four statuses (each with a glyph or hatch, not colour alone), drill-down by
  section and zone with signal charts and model output, tabs for active anomalies (with the top 3 reasons), model
  predictions, data health and a Vision tab that says it is not enabled. The built `frontend/dist` is committed
  and served by the backend. Deep links (`?zone=107`, `?drive=1`, `?tab=health`) open a view directly.
  Screenshots are in `docs/screenshots/`.
- **Stage 10 (partly)**: `Makefile`, GitHub Actions CI (`ruff`, `pytest`, dashboard type-check and build), `ruff`
  clean. The README is still to be written.
- **Dashboard simplified** (user feedback: the single page was overwhelming). The page is now six top-level tabs,
  one question each: Overview (is anything wrong?), Alerts, Predictions, Zone details, Data health, Vision. Only
  one is on screen at a time. Overview shows a one-line headline, the belt map and a short "needs attention" list.
  Zone details plots one thing at a time (model output or one signal) chosen with buttons, and has a zone picker.
  Alert reasons sit behind a "Why?" toggle. Labels use plain words ("chance of failing within 24 h"). A guided
  walkthrough of 11 steps with Previous and Next opens on the first visit and from the "Tour" button.
  Deep links: `?tab=alerts`, `?zone=107`, `?tour=4`, `?tour=0`.
- `scripts/demo.py` refuses to start when something is already listening on its port (it used to feed a second
  replay into the running backend) and takes `--port`.
- **Alerts say what the model saw** (user feedback). `model/explain.py` turns the model's strongest features into
  sentences, for example "Bearing temperature is 9.8 °C higher than the other zones on its run (6 h average)".
  Every model alert's message ends with "What the model saw: ..." and its evidence carries the same sentences as
  `reasons`. The dashboard shows them as a list on the alert card and in Zone details, no longer behind a toggle.
  Context features (share of time running, position) are skipped: they are not symptoms.
- **Checking whether predictions come true** (user question: "what is the accuracy, and how do we know a predicted
  failure was real?"). Two gaps were found and closed:
  - **Offline.** A failure call used to count as right whenever it fired on a degrading part. `score_calls` in
    `model/evaluate.py` now scores each call on whether the part failed within 24 h of it, and the failure-call
    threshold is tuned on that (zone 0.97, was 0.35). **Result, seed 1: validation 9 of 10 calls came true within
    24 h, stress 8 of 8, but test only 5 of 10.** The other 5 test calls were 0.3 to 11 h too early (the part
    failed 24.3 to 35.1 h after the call); none was on a healthy part. Over three seeds test is 5 to 7 of 10. The
    90% target is met on validation and **not on test**; the threshold was not retuned after seeing test.
  - **Live.** The replayer now posts a repair record when each repair ends (`replayer/work_orders.py`: asset, fault
    kind, failure time, repair end; never the fault's start). The backend stores them in `work_orders` and writes
    one row per failure to `outcomes`. The dashboard has a **Track record** tab. This is a fifth reader of the
    answer key, added on the user's decision; tests check that it reads only those four columns, releases nothing
    before the repair ends, and that the model code never touches repair records.
  - **A defect the live check exposed.** In the demo the drive alert opened at 16:09 and closed at 17:19, three
    hours before the drive failed, because the drive's abnormal score wobbles around its threshold. The offline
    score had only checked that an alert opened before the failure. Fixed in two ways: an open alert no longer
    closes while the probability of failing within 24 h is above 50% (`HOLD_P_FAIL`; this can keep an alert open
    but never open one), and the evaluation now reports "alert still open at failure". With the fix that is 10 of
    10 and 2 of 2 on test, and the demo shows 4 of 4 failures with an alert open and no false alarms.
  - Window-level precision, recall and average precision for the 24-hour classifier are in the report; plain
    accuracy is not, because over 99% of windows are negative.
  - The drive's failure-call threshold tuned to 0.05, the lowest candidate: drive faults last under 48 h, so any
    call inside an open alert is within 24 h. It carries little information and rests on 2 validation events.
- **Failure call now claims 72 hours** (user's decision, the safer side; one shared setting,
  `CALL_HORIZON_HOURS` in `model/alerting.py`). On the previous 24 h rule only 5 of 10 test calls came true; all 10
  were within 36 h. 72 was chosen after seeing those test results, so it loosens the claim; it is not the model
  improving. The failure-call threshold was retuned on validation for the 72 h rule (zone 0.15 to 0.3 across three
  seeds). The classifier itself still predicts "within 24 h"; the dashboard calls its output a "failure risk score".
- **Precision and recall reported** (`model/artifacts/eval_report.md`, new "at a glance" table, plus validation
  trade-off tables for both thresholds so they can be balanced without touching test). Seed 1, held-out test:
  alerts 12 of 12 faults caught, 0 false; failure calls 12 of 12 within 72 h. At window level (24 h classifier) zone
  precision is 0.83 and recall 0.92 on test. Drive window-level precision is low at its chosen threshold (0.04).
- **Live accuracy on Overview.** `GET /api/outcomes` now returns precision and recall; `GET /api/model/evaluation`
  serves the offline test figures (`model/artifacts/summary.json`). The Overview strip shows recall, precision and
  failure predictions that came true so far in the run, how many cases they rest on, and the offline figures.
- **Faults are now random per fault** (user feedback: near-perfect scores came from every fault being identical).
  Each fault draws its own **strength** (0.3 to 1.5 of the base amplitudes), **growth exponent** (1.5 to 3.5) and a
  **mix** (0.5 to 1.5, separately per signal, so one fault shows more in heat and another in noise); ranges are in
  `faults.strength`, `faults.signal_mix`, `faults.exponent`. Ranges of (1, 1) with exponent (2.5, 2.5) give the old
  fixed fault, which the simulator tests still use. The draws come after all placement draws, so fault schedules
  are unchanged and demo seed 1015 still passes every rule. Stage labels follow each fault's own exponent.
  The whole set was regenerated, and the models retrained and re-evaluated (three seeds).
  - **Result, held-out test, seed 1:** alerts still catch 10 of 10 bearing faults and 2 of 2 drive faults with no
    false alerts. **Failure calls dropped:** 7 of 10 bearing faults got a call (recall 0.70, precision 1.00); the
    weak faults never raise one. Stress: 3 false alerts (alert precision 0.73) and failure-call recall 0.50.
  - **The demo now shows a miss:** the drive fault in the demo is weak and the model never alerts on it (3 of 4
    failures warned, recall 0.75).
  - **`scripts/robustness.py`** scores the frozen model on 8 deliberately different scenarios
    (`model/artifacts/robustness_report.md`). Bearing alerts catch every fault even at 0.2 to 0.4 of normal
    strength, but false alerts explode with noise (3 at x2, 114 at x4, 990 at x8) because the thresholds were tuned
    at normal noise. The drive model misses faults at low strength and raises 8 false alerts at 0.2 to 0.4.
  - **Why detection is still so good:** the model's signal is the vibration kurtosis deviation from peer zones,
    and the simulator's healthy kurtosis noise is tiny (sd 0.12), so even a weak fault stands out. That is a
    property of the simulator, not evidence the model would work on a real belt.
- **Random sensor noise per scenario** (user decision). Each scenario draws a noise multiplier log-uniformly from
  `signals.noise_random` (1 to 5), applied on top of `signals.noise_scale`; (1, 1) switches it off. It uses its own
  random stream (`noise`, appended to `STREAMS`), so layout, conditions and fault schedules are unchanged (tested).
  The effective level is in each scenario's `run_meta.json` and in `docs/scenario_set_report.md` (1.2x to 4.7x across
  the set; test_01 1.9x, test_02 2.5x, demo 4.7x, stress 3.5x). The stage 1 sanity targets keep the base noise
  through the test fixture. Regenerated, retrained and re-evaluated.
  - **Result, held-out test, seed 1:** bearing alerts 10 of 10 with 0 false alerts; failure calls 9 of 10 true
    (precision 0.90, recall 0.90); drive alerts 2 of 2, calls 2 of 2. Stress: alerts 8 of 8 with 0 false alerts,
    failure calls 5 of 7 true, recall 0.62.
  - **Robustness (`scripts/robustness.py`):** with noise fixed at 1x, 3x and 5x (the training range) bearing alerts
    catch 8 of 8 with 0 false alerts. **Outside the range it breaks:** 23 false alerts at 8x and 150 at 12x. The
    drive model still misses faults at 0.2 to 0.4 of normal strength and raises 6 false alerts there.
  - **Why noise matters so little inside the range:** the model works from 6 h and 24 h averages of the deviation
    from peer zones, which average independent noise away (sd falls by about 1/sqrt(1440)). The simulator's noise
    is independent between readings and zones. Real sensors drift, share noise and lose calibration, which
    averaging does not remove, so this result does not carry over to a real belt.
  - The demo scenario came out at 4.7x noise and its outcomes are unchanged (3 of 4 failures warned, the weak drive
    fault missed); verified that the demo data, model and database are all the new ones.
- **Sensor drift and section-shared noise** (user decision). Two kinds of sensor messiness that averaging does not
  remove, each drawn per scenario on its own random stream (`drift`, appended to `STREAMS`; nothing else moves):
  - **Drift** (`signals.drift_*`, level `drift_random` 0 to 3): each zone sensor wanders away from its true value over
    days (forgets itself over about 3.5 days; long-run sd 1.5 C temperature, 1.5 dB acoustic, 4% vibration, 0.2
    kurtosis, 2 mm offset, times the scenario's level).
  - **Shared noise** (`signals.shared_*`, level `shared_random` 0 to 3): every zone in a splice section gets the same
    disturbance (a sunlit stretch reads warm, a noisy cable run, a dusty section), so a section cannot be compared
    with its neighbours to rule it out.
  - The levels are in `run_meta.json` and `docs/scenario_set_report.md` (columns "Drift x" and "Shared x"). The
    stress scenario is held at the top third of both ranges; the other splits draw freely. Tests that assume base
    noise pin the levels with a `QUIET` setting.
  - Probe before retraining: the model trained without them raised 394 to 1,134 false alerts on scenarios with
    drift or shared noise at the top of the range, so this is a real test, not a cosmetic one.
  - **Result after retraining, held-out test, seed 1. The model no longer meets the target.** Bearing alerts: 8 of 10
    faults caught, **13 false alerts, precision 0.38**; failure calls 6 of 6 true but only 6 of 10 faults called.
    Stress: 7 of 8 caught, 5 false alerts, precision 0.58. Drive alerts still 2 of 2 with no false alerts.
    Validation looked fine (10 of 10, 1 false, precision 0.91) because its two scenarios happened to draw low shared
    noise (0.19 and 0.85) and low or middling drift, while test_01 drew drift 2.67 and shared 1.72 and test_02 shared
    2.42. The alert threshold was tuned on validation only, so it is too low for the noisier test scenarios. No
    threshold was changed after seeing test.
  - **Robustness** (`model/artifacts/robustness_report.md`, fixed levels, other kinds switched off): drift 3x, the top
    of the training range, gives 4 of 8 bearing faults caught and 9 false alerts; drift 6x, 98 false alerts; all
    three at the top of their ranges, 5 of 8 caught and 29 false alerts. Faults at 0.2 to 0.4 of normal strength:
    2 of 8 caught. Noise up to 12x is now handled (0 to 1 false alerts); the model learned to cope with noise but
    not with drift.
  - **Demo:** 3 of 4 failures warned (the weak drive fault is still missed), but lead times shortened (Z107 62 h,
    Z065 27 h, Z064 16 h, against 150, 83 and 76 h before).
  - **What is not fixed:** nothing here makes the model drift-robust and validation does not span the conditions the
    test scenarios draw. Options: more validation scenarios that cover the range, features that separate a slow
    sensor wander from a growing fault, or per-sensor baselines. This is the next decision.
- **Wider validation, then drift-robust features** (user: "make sure the model predicts the data pretty well").
  - **Validation is now stratified, not random:** 6 scenarios (`val_01` to `val_06`, seeds 201 to 206) at fixed
    levels of (noise, drift, shared noise) = quiet, (3, 1.5, 1.5), (5, 0, 0), (2, 3, 0), (2, 0, 3), (4, 3, 3), so the
    thresholds are tuned on the whole range the other splits draw from. 30 bearing and 6 drive events instead of 10
    and 2. Test is untouched. The threshold candidates were extended above 0.999 (up to 0.99999).
  - **Wide validation alone fixed precision and cost recall.** Held-out test bearing alerts: 0 false alerts
    (was 13) but only 6 of 10 caught (was 8). Diagnosis: with drift at 1.5x to 3x, healthy windows' abnormal scores
    reach 0.96 to 0.98 while real faults' fall to 0.06 to 0.17, so no threshold separates them.
  - **New features** (`model/features.py`): each zone's 6 h and 24 h deviation in units of the spread of all zones
    in the same window (robust median and MAD; removes the scenario-wide noise and drift level), and a count of how
    many of heat, noise, impact spikes and vibration agree (a drifting sensor moves one signal on its own, a fault
    moves several). 72 zone features, up from 57. Training and live use the same function (test still passes).
  - **Result, held-out test, seed 1 (one seed to save time):** bearing alerts **8 of 10 caught, 0 false alerts**
    (recall 0.80, precision 1.00); stress 7 of 8, 0 false; validation 26 of 30, 0 false. Failure calls are
    precise (6 of 6 on test) but rare: recall 0.57 on validation, 0.60 on test, 0.50 on stress.
  - **Robustness** (`model/artifacts/robustness_report.md`): false alerts are gone almost everywhere (drift 3x: 0, all
    three at the top of their ranges: 1, against 9 and 29 before). **The cost is recall under drift:** drift at 3x
    catches 3 of 8 bearing faults, all three at the top of their ranges 3 of 8, and faults at 0.2 to 0.4 of normal
    strength 1 of 8. The model now stays quiet when it cannot tell a weak fault from drift.
  - **Drive model unchanged and weak:** 1 of 6 on validation, 1 of 2 on test (2 of 2 on stress). It trained on 8
    events; at the thresholds tuned on validation it rarely fires.
- **Final day, step 1: drive model retrained on more drive events (kept).** Six new scenarios `train_drive_01` to
  `_06` (seeds 601 to 606, split `train_drive`, `faults.drive_overheat.per_30d: 4`, everything else default including
  random noise, drift, shared noise and fault strengths) placed 4 drive events each, 24 in total. They are separate
  scenarios because the fault stream is shared: raising the drive rate in `train_01` to `train_08` would also have
  changed their bearing faults. `scripts/make_model_data.py` builds drive-only tables for them (no zone tables).
  `model/train.py` gained explicit training splits (`training_splits`: the zone model trains on `train` only, tested),
  `--groups drive --extra-drive-splits train_drive` and a partial `save` that does not touch the other group's files
  (tested). Only the three drive models were trained, with the existing settings and seed 1 (model version
  `lgbm-w10-seed1-drive2`; 58,478 training rows), and the drive thresholds were retuned on the six existing
  validation scenarios with the unchanged rule.
  - **Decision rule, written before looking:** keep only if validation drive alert recall is at least 3 of 6 and
    drive false alerts at most 1. **Result: 4 of 6 caught, 0 false alerts, precision 1.00 (alert threshold 0.96, call
    threshold 0.05; 4 of 4 calls true). Kept.** One attempt, no second tuning round.
  - **Zone models byte-identical:** sha256 of `zone_abnormal.txt`, `zone_classifier.txt` and `zone_regressor.txt`
    equal `hashes_before.txt` (`hashes_frozen.txt` written after step 1). The frozen evaluation's zone section is
    identical to the previous one, and the zone rows of the robustness report are identical.
  - **`scripts/score.py --frozen`** scores the saved models and thresholds without training or saving anything
    (it recomputes the threshold sweeps from the saved models and refuses to run if they differ from `meta.json`).
    Without it, `score.py` retrains, which would have overwritten the zone models.
  - **Drive, before and after (alerts caught / false alerts):** validation 1 of 6 / 0 became 4 of 6 / 0; test 1 of 2
    became 2 of 2 / 0; stress 2 of 2 / 0 unchanged (alert open at failure 1 of 2). The drive's validation recall is
    0.67 and test recall 1.00, on 2 test events; test and stress did not decide.
  - **Drive robustness rows** (15 cases, caught of 2, false alerts): no case got worse; 0 false alerts everywhere (it
    was 2 for weak faults); noise 8x 0 to 2, noise 12x 1 to 2, very late growth 1 to 2, shared noise 6x 1 to 2, weak
    faults 1 to 0 (but 0 false alerts instead of 2).
  - **Demo (`demo_01`, seed 1015, not re-picked), final model:** 4 failures, 4 with an alert open at failure (the
    drive is now warned, 1.6 h before it failed), 0 missed, 0 false alarms; 3 failure calls, all true within 72 h
    (Z107 31.7 h before failure, Z064 4.6 h, drive 1.6 h). Z065 was alerted only 1.0 h before it failed and got no
    call. Before: 3 of 4 warned, the drive missed. Tests: 109 passed, 3 skipped.
- **Final day, step 2: fresh untouched test, scored once (all criteria met).** Criteria were committed first
  (`fc1a6e6`, before any scenario existed). Scenarios `test_03` to `test_06` (seeds 501 to 504, split `test_fresh`,
  default settings with `faults.drive_overheat.per_30d: 3`; noise 2.6x, 2.8x, 3.8x, 2.6x; drift 1.9x, 1.5x, 2.4x, 1.4x;
  shared 2.9x, 2.9x, 2.5x, 0.5x). Scored once with the frozen models (`score.py --frozen --fresh`); no crash
  re-run; artifact hashes equal `hashes_frozen.txt` before and after.
  - **Bearings:** 16 of 20 caught, 0 false alerts, alert precision 1.00, alert recall 0.80 (criteria: at least
    0.90 and 0.75, met). Failure calls: 10, all true within 72 h; call recall 0.50 (10 of 20), median 14.2 h from call
    to failure.
  - **Drive:** 6 of 12 caught, 0 false alerts, alert recall 0.50 (criterion: at least 0.50, **met exactly**). Calls: 6,
    all true; median 5.6 h from call to failure.
  - Per scenario, bearings caught: 3, 5, 3, 5 of 5; drive: 1, 2, 1, 2 of 3. The old test scenarios (8 of 10 bearings,
    2 of 2 drive) were used for diagnosis, so the fresh numbers are the ones to quote.
  - Caveats: 20 bearing and 12 drive faults; the drive rate is boosted to 3 per 30 days; failure calls miss about half
    the faults (weak faults raise no call); the drive result sits exactly on the bar.
- **Final day, step 3 (partly done): local clean-clone dry run and hygiene.** Not the handover's run: there is no GitHub
  remote and no README yet, so the clone was made from the local repository. Fresh clone, new Python 3.12 venv
  (`uv pip install -r requirements.txt`, 11 s with a warm cache; `make setup` itself was not run), `make demo-data` (13 s,
  seed 1015), `scripts.demo`: first prediction 20 s after the server and replay started, `/api/health` OK, the
  dashboard served from the committed `frontend/dist` (no Node), replay ran to the end, all seven views loaded in
  headless Chrome with no console error lines. Hygiene: no tracked files under `data/`, `.git` 16 MB, no `torch` or
  `ultralytics`, no absolute paths in code or config, `frontend/dist` identical to a fresh build, all files the
  backend loads are in `model/artifacts/`. Screenshots for the README in `docs/screenshots/`: `overview.png`,
  `loop_map_alerts.png`, `alerts.png`, `predictions.png`, `data_health.png`, `track_record.png` (each under 80 KB).
  Still to do when the remote and the README exist: CI on the pushed commit, the README's commands run literally in a
  fresh clone of the GitHub remote, and `make test` and `ruff` in that clone.
- **Fix 1: the Overview quotes the fresh test.** `model/artifacts/fresh_test_summary.json` is exported with the frozen
  models by `score.py --frozen --fresh --export-summary`, which checks every figure against the committed
  `fresh_test_report.md` and writes nothing if one differs (they matched: bearings 16 of 20, drive 6 of 12, 0 false
  alerts, calls 10 of 10 and 6 of 6). New `GET /api/model/fresh-test`; the strip sentence is built only from that file,
  and the tour step now names the fresh test. The old test_01/02 figures are no longer shown anywhere in the dashboard.
- **Fix 2: no "Critical" without a failure call.** `model/alerting.py` `alert_label` caps a model alert's label at `high`
  unless a failure call is active on that asset (status `predicted_failure`); used in the one place labels are set,
  `backend/engine.py` `_model_alert`. Numeric severity, thresholds and the status rule are unchanged; data-quality
  alerts stay `medium`. The Alerts tab says "Critical is shown only when a failure is predicted within 72 h."
