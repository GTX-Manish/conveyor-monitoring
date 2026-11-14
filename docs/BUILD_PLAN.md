# Build Plan: the one document Claude Code needs (2026-11-14, 17:45 IST)

Deadline: **Tue Nov 17, 11:59 PM IST**. We aim to submit **Mon Nov 16, midnight IST**; Tuesday is buffer only.
This file plus `SIMULATOR_SPEC.md` (already in the repo) and `SIMULATOR_BUILD_REPORT.md` are everything you need. Where this
file differs from `SIMULATOR_SPEC.md`, **this file wins** (section 3 lists the differences).

Why each decision was made is written next to it. **(assumption)** means our own choice; keep it in config.

---

## 1. What we are building

The Avartan Labs "AI Systems Engineer" take-home: a system that watches a 6 km mining conveyor loop and predicts failures.
The brief wants a design README, a backend, a prediction model, a vision component and a dashboard in one GitHub repo, with
sensor data only from a simulator we write. Graders weigh judgement, clear writing, build-versus-skip choices, messy input
and AI use.

**Scope decision (user, corrected 2026-11-14): vision is deferred, not dropped.** The image dataset is still downloading,
so the sensor pipeline and the data flowing to the dashboard are built first. The camera pipeline gets **placeholders
only** for now: empty `captures` and `vision_detections` tables, `GET /api/vision/*` endpoints that return empty lists
with `enabled: false`, and a Vision tab in the dashboard that says "not enabled in this run". Why: adding vision later is
then a data and model task with no schema or API change. No detector, images or capture events are built yet.

**The belt:** 6 km loop (3 km carry, 3 km return), 4 m/s (one lap = 25 min), 120 zones of 50 m with one sensor suite each,
12 sections (S1 to S12) of 10 zones, loading zones 1 to 2, unloading 59 to 60, 18 h running and 6 h stopped per day, 1-minute
steps. Per zone: vibration RMS, vibration kurtosis, bearing temperature, acoustic level, belt offset. For the drive: motor
current, motor and gearbox temperature, drive and belt speed, slip, tension, throughput.

**What a reviewer should see after `make demo`:** a racetrack map of the 120 zones coloured by status, faults developing,
the model warning with a location, a probability and a time, sensors going silent or freezing and being shown as "no data /
sensor suspect", and a data-health view of messy input being handled.

## 2. Ground rules

- **Hardware:** 7 GB RAM, 2 CPU cores, no GPU needed. One scenario in memory at a time; peak memory under 3 GB per generation
  process; at most 2 workers; LightGBM `n_jobs=2`. **Disk:** a 40 GB dataset download may be using the disk; check free
  space with `shutil.disk_usage` and stop with a clear message under 5 GB before writing data.
- **Python 3.12, first thing.** The current `.venv` is on 3.14.4. Run `uv python install 3.12`, rebuild with
  `uv venv --python 3.12`, reinstall, and **re-run all 66 tests before writing new code**. Why: 3.14 may lack wheels for
  LightGBM. Record it in `requirements.txt`, `.python-version` and `docs/CHANGELOG.md`. `make setup` refuses other minors.
- **Ground truth** is in each scenario's `truth/` folder. The live path (backend, replayer, frontend, feature and prediction
  code) must never read it. Exactly four scripts may read `truth/`: `simulator/scenario_set.py`, `scripts/find_demo_seed.py`,
  `scripts/score.py`, `scripts/make_model_data.py` (reads `labels.parquet` only, because training needs labels). `conditions.parquet`
  is never a model input.
- No per-minute Python loops over time; vectorise. Determinism: one seed and config give identical tables.
- Look at every plot you make before moving on. Commit after each stage; one line per stage in `docs/CHANGELOG.md`.
- **Do not rename keys in `configs/default.yaml`.** Why: scenario configs and 66 tests depend on them; a tidy-up has no grading value.
- If a number or rule here is wrong or impossible on the hardware, say so in the changelog and propose a fix; do not work around it silently.

## 3. What is already built, and what changed from `SIMULATOR_SPEC.md`

Stages 0 to 4 are built and tested (healthy baseline, equipment faults, hard negatives and labels, messiness). In the repo's
`SIMULATOR_SPEC.md`, mark these as superseded (one line each) so the file is not misleading:

| In the spec | Now |
|---|---|
| Section 0 "Nothing is built yet" | Stages 0 to 4 are built |
| "Python 3.11+" | Python 3.12 |
| Three faults, including belt damage (7.3, its config block, `BELT` asset, tension dip) | **Two faults: idler bearing and drive overheating.** Belt damage was removed (it only made sense with vision) |
| Section 9 image events, Stage 5, `image_events.parquet`, replay stream in the simulator | **Not built.** Replay is a separate process (section 7) |
| Section 6 belt coordinates | Keep the code, unused for now |
| Section 14 fault counts with belt events | About 40 idler and 8 drive events across training, 10 and 2 in validation, 10 and 2 in test, 8 and 2 in stress |

## 4. Stage 6: scenario set and demo scenario (tonight)

**Goal:** all scenarios the model needs, plus one curated demo, from one command, reproducibly, within the hardware limits.

| Name | Split | Seed | Days | Purpose |
|---|---|---|---|---|
| `train_01` to `train_08` | train | 101 to 108 | 30 | Fit the model |
| `val_01`, `val_02` | validation | 201, 202 | 30 | Tune thresholds |
| `test_01`, `test_02` | test | 301, 302 | 30 | Scored once at the end |
| `stress_01` | stress | 401 | 30 | Harder data, scored once at the end |
| `demo_01` | demo | from `find_demo_seed.py` | 14 | The live demo |

Why 12 scenarios of 30 days: fault events, not days, are what get scored, so more short scenarios give more independent events.

**Stress overrides (assumption; use the exact key names in `configs/default.yaml`):** `signals.noise_scale` 1.0 to 2.0;
`messiness.missing_rate` 0.01 to 0.03; `messiness.outages_per_30d` 6 to 12; `messiness.stuck_per_30d` 4 to 8;
`faults.idler_bearing.per_30d` 5 to 8; `faults.drive_overheat.per_30d` 1 to 2; `hard_negatives.heavy_load.per_30d` 4 to 6;
`hard_negatives.dusty.per_30d` 2 to 3. Why: shows how much the model leans on clean data and the default fault mix.

**Build:**
1. `configs/scenarios/<name>.yaml`, one per scenario (default config plus overrides). Commit them.
2. `python -m simulator.scenario_set` with `--only`, `--workers` (1 or 2; refuse above 2), `--force`, `--out data`.
   Idempotent: `manifest.json` stores a config hash; skip a scenario whose hash matches and whose files exist. Check disk first;
   record peak memory per scenario; fail if any exceeds 3 GB.
3. `data/scenario_set.json`: per scenario: name, split, seed, days, config hash, path, rows, public size in MB, seconds, peak
   MB, fault counts by type, hard-negative counts by type, sensor-fault counts by type, positive rate of `fails_within_24h`
   (counts taken from `truth/`).
4. `docs/scenario_set_report.md` (generated): the table, totals per split, the balance check.
5. `tools/plot_stage6.py` to `docs/stage6_scenario_set.png`: a timeline per scenario (fault onset to failure to repair, hard
   negatives, sensor faults as coloured bars).
6. The demo scenario.

**Balance check.** No seed used twice and no scenario in two splits: **fails the build**. Otherwise warnings: every scenario
has at least one hard negative of each type and one idler fault; validation and test each have at least 2 idler and 1 drive
event (if not, say so and suggest a new seed; never change seeds silently). Print the `fails_within_24h` positive rate per split.

**Demo scenario.** It should play a full story in about **5 minutes** of replay and the README must say it is curated.
`scripts/find_demo_seed.py` scans seeds 1000 to 3000 for a 14-day scenario using schedules only (no signal physics, so seconds):

| Rule | Why |
|---|---|
| An idler fault with onset on day 2 or later and failure between day 9 and 12 | A full progression is visible |
| Exactly one drive overheat, failure between day 5 and 13, at least 24 h from the idler failure | A second signature |
| A heat wave overlapping days 3 to 8 and ending before the idler fault's advanced stage | The model must stay calm while it is hot |
| A heavy-load shift on a night shift | A visible, harmless spike |
| A stuck sensor on a zone that is not the faulty one, starting between day 3 and 10 | Shows "sensor suspect" |
| An outage of at least 2 hours | A silent suite must not look healthy |

Take the first seed that passes (or the best match, saying which rule failed). Record the seed and rules in the report.

**Tests (`tests/test_stage6.py`, on a 2-day mini set; whole suite under about 30 s):** seeds unique and splits disjoint;
manifest has every field; a second run skips everything and `--force` rebuilds; stress overrides appear in that scenario's
`truth/config_used.yaml` and not in train; two builds of the same mini scenario give identical public tables; `--workers 3`
refused; demo seed script returns a seed or a clear "no match" with the failed rule. **Update `tests/test_leakage.py`:** scan
`backend/`, `replayer/`, `frontend/`, `common/`, `model/` and `scripts/` (skip folders that do not exist yet), allowing only the
four readers named in section 2, listed explicitly in the test.

**Finished when:** tests pass on Python 3.12; the full set builds in under 10 minutes on 2 CPUs and a second run does nothing;
the manifest, report and plot exist and the plot has been looked at; the demo scenario and its rules are recorded; the repo's
`SIMULATOR_SPEC.md` has the section 3 notes; changelog entry; commit.

## 5. Stage 7: model data, model, evaluation (Sunday morning)

**Windows.** Train on **10-minute windows**, not 1-minute rows. Why: 12 scenarios at 1 minute are about 62 M zone rows, which
does not fit in 7 GB with rolling features; at 10 minutes about 6 M rows, and the fault signals (days to weeks long) are as visible.

**`model/features.py`, one module, two stages, used by training and live mode** (different code in the two paths is the most
common reason a POC works offline and fails in the demo):
1. `window_aggregates`: per (asset, window) mean, min, max, standard deviation of each signal over valid readings; count of
   missing readings; flat-line flag; share of the window the belt was running. Zones use the five zone signals; `DRIVE` uses
   the belt-wide signals.
2. `rolling_features`: over the **window table**, not raw rows: 1 h, 6 h and 24 h rolling statistics; difference from the
   neighbouring zones; load- and ambient-corrected residuals using the belt-wide signals.

`scripts/make_model_data.py` builds `data/model/<scenario>.parquet` one scenario at a time. Labels per window come from
`labels.parquet`: `fails_within_24h` (true at window end), worst `stage` in the window, hours to failure.

**Models** (LightGBM; a zone model on all 120 zones and a separate drive model, because the drive has different signals and
very few events, which the README must say): (a) `abnormal` trained on the **advanced** stage versus normal only, because the
early stage (about the first 62% of a fault's life) is below the noise by design; (b) `fails_within_24h` classifier; (c) hours
to failure, trained on advanced-stage windows only. Handle class imbalance. Split by whole scenario, never by rows.

**Thresholds (user, corrected 2026-11-14):** tune the `abnormal` and `predicted_failure` thresholds on the validation
scenarios so that **at most 1 to 2 in every 20 model alerts are false** (alert precision of 90% or better; a false alarm
is an alert opened while the asset's true stage is `normal`), then freeze them for test. Among thresholds that meet this,
take the one that detects the most events. Why: the first draft's "1 false alarm per zone per month" is about 120 false
alarms a month against about 6 real faults, so 19 of 20 alerts would be false and nobody would trust the list.

**Explanation:** for each alert store the top 3 LightGBM contributions (`pred_contrib`) in `alerts.evidence`.

**Evaluation (`scripts/score.py`)**, per event, not per row: detection rate (an alert opened before the failure time), median
lead time, false alarms per zone per month, split out for hard-negative periods. Compare against two baselines (a fixed
bearing-temperature threshold; deviation from neighbouring zones), over three seeds, with mean and spread. Report the stress
and test scenarios once. Write `model/artifacts/eval_report.md`.

**Artifacts** (committed): `classifier.txt`, `abnormal.txt`, `regressor.txt` per model, and `meta.json` (features, window settings,
thresholds, training scenarios, seed, metrics, version string).

**Finished when:** the models beat both baselines on validation for the idler fault (if not, say so honestly in the report);
live and training features are identical for the same windows (test); artifacts are committed.

## 6. Stage 8: backend, replayer (Sunday afternoon)

**Processes.** `make demo` runs `scripts/demo.py`, which starts the backend, waits for `/api/health`, then starts the replayer,
and stops both on Ctrl-C. Startup deletes and recreates the SQLite file, so every run is fresh.
- **Backend:** FastAPI, one uvicorn worker, Pydantic v2, stdlib `sqlite3` in WAL mode, serves `/api` and the built frontend.
  One DB writer thread; all writes go through one queue (SQLite has one writer).
- **Replayer:** a separate process (`httpx`) reading `data/demo_01/public/` and POSTing one batch per simulated minute to
  `/api/ingest`. It first posts the first 24 simulated hours unthrottled (history preload), then replays at the speed set via
  `/api/replay`: **default 1 simulated hour per second, maximum 3** (higher is refused to protect the request rate). The
  14-day demo therefore plays 13 days in about **5 minutes**. Why a separate process over HTTP: the simulator-to-backend story
  is true end to end and the validation really runs.
- **Replayer-injected faults** (`--inject-faults`, on in `make demo`, off in tests; seeded; rates in `configs/app.yaml`,
  assumption): duplicate rows 0.05%; out-of-range values 0.02% of values; one row with an unknown zone per simulated day;
  every 200th batch swapped with the next. Totals go to `data/<scenario>/replayer_injected.json`. Why: the simulator only
  produces missing values, so otherwise validation would be exercised by tests only. The README says the replayer, not the
  simulator, adds them.

**Ingest validation (`POST /api/ingest`):** body `{"batch": [{ts, zone_id, run_state, <5 signals or null>}...], "belt": {...}}`.
Unknown `zone_id`, bad `ts` or `run_state`: row rejected and counted. Duplicate `(ts, zone_id)`: first wins, ignored, counted.
Out-of-range value (assumption: vibration RMS 0 to 50 mm/s, kurtosis 0 to 100, temperature -40 to 200 C, acoustic 0 to 140 dB,
offset -500 to 500 mm): set to `NULL`, counted. Null: stored as `NULL`, never filled in. Out-of-order batches are accepted.
A **watermark** (latest minute for which all 120 zones reported, or 2 minutes of grace) drives everything that needs time order.

**Data quality (config):** a signal is `stuck` after **15 identical values while the belt is running** (noisy floats essentially
never repeat; a real coarse sensor would need another rule, say so in the README); a suite is `no_data` after **5 minutes of all
five signals missing**; isolated missing readings raise nothing. State changes write `sensor_health` and `quality_events` and open or
close an alert (`sensor_outage` or `sensor_stuck`, severity 0.5, `medium`).

**Status of an asset:** `predicted_failure` if the model says so (wins); else **no data / sensor suspect** if any of its
sensors is `no_data` or `stuck`; else `abnormal` or `normal` from the model. Why: a silent or frozen sensor must never show
as normal, but real evidence of failure still wins.

**Live scoring.** When the watermark passes the end of a 10-minute window: aggregate that window (`window_aggregates`), append
it to an in-memory **window table of the last 144 windows (24 h) per asset** (seeded from the 24 h preload), run
`rolling_features` on it, score all 121 assets, write `predictions`, update alerts. Why in memory: at the default speed that is 6
windows per real second; re-reading 24 h of 1-minute rows each time would not fit on 2 cores. Keep a precomputed fallback
(predictions from the offline run) selectable by a flag, in case live scoring breaks.

**Alerts (assumption):** open after 2 consecutive windows above the `abnormal` threshold, close after 6 below. Type
`bearing_degradation` for a zone, `drive_overheating` for `DRIVE`. Severity `clip(0.6 p_fail + 0.4 (1 - min(eta_h, 24)/24), 0, 1)`
with an ETA, else `p_fail`; bands: under 0.25 low, under 0.5 medium, under 0.75 high, else critical. Why: a zone hours from
failure must look worse than one 20 h away.

**Asset ids:** `common/ids.py` maps simulator zone numbers `1..120` to asset ids `Z001..Z120` (and `DRIVE`) in one place;
model data, predictions and scoring import it and use no other conversion.

**SQLite schema** (`backend/schema.sql`, created at startup; pragmas `journal_mode=WAL`, `synchronous=NORMAL`,
`temp_store=MEMORY`, `cache_size=-65536`; timestamps are `INTEGER` UTC epoch seconds; `run_state` 0 stopped, 1 ramp, 2 running;
missing values `NULL`; ground truth never loaded):

```sql
CREATE TABLE zones (zone_id INTEGER PRIMARY KEY, section INTEGER NOT NULL, zone_in_section INTEGER NOT NULL,
  position_m REAL NOT NULL, run TEXT NOT NULL, is_loading INTEGER NOT NULL, is_unloading INTEGER NOT NULL,
  incline_deg REAL, curve_radius_m REAL);
CREATE TABLE readings (ts INTEGER NOT NULL, zone_id INTEGER NOT NULL, run_state INTEGER NOT NULL,
  vibration_rms_mm_s REAL, vibration_kurtosis REAL, bearing_temp_c REAL, acoustic_db REAL, belt_offset_mm REAL,
  PRIMARY KEY (ts, zone_id)) WITHOUT ROWID;
CREATE INDEX readings_zone_ts ON readings(zone_id, ts);
CREATE TABLE belt_readings (ts INTEGER PRIMARY KEY, run_state INTEGER NOT NULL, throughput_tph REAL, motor_current_a REAL,
  motor_temp_c REAL, gearbox_temp_c REAL, drive_speed_ms REAL, belt_speed_ms REAL, slip_pct REAL, tension_kn REAL);
CREATE TABLE sensor_health (zone_id INTEGER NOT NULL, signal TEXT NOT NULL, state TEXT NOT NULL,  -- 'ok'|'no_data'|'stuck'
  since_ts INTEGER NOT NULL, last_seen_ts INTEGER, last_value REAL, PRIMARY KEY (zone_id, signal)) WITHOUT ROWID;
CREATE TABLE quality_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, zone_id INTEGER NOT NULL, signal TEXT,
  type TEXT NOT NULL,  -- 'outage'|'stuck'
  start_ts INTEGER NOT NULL, end_ts INTEGER);
CREATE TABLE predictions (ts INTEGER NOT NULL, asset TEXT NOT NULL,  -- 'Z001'..'Z120' or 'DRIVE'
  p_abnormal REAL NOT NULL, p_fail_24h REAL NOT NULL, est_hours_to_failure REAL,
  status TEXT NOT NULL,  -- 'normal'|'abnormal'|'predicted_failure'
  model_version TEXT NOT NULL, PRIMARY KEY (asset, ts)) WITHOUT ROWID;
CREATE TABLE alerts (alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
  type TEXT NOT NULL,    -- 'bearing_degradation'|'drive_overheating'|'sensor_stuck'|'sensor_outage'
  source TEXT NOT NULL,  -- 'model'|'data_quality'
  asset TEXT NOT NULL, zone_id INTEGER, position_m REAL,
  severity REAL NOT NULL, severity_label TEXT NOT NULL, predicted_failure_ts INTEGER, eta_hours REAL,
  message TEXT NOT NULL, evidence TEXT,  -- JSON: {"top_features":[{"name":..,"contribution":..} x3],"p_fail_24h":..}
  state TEXT NOT NULL,   -- 'open'|'closed'
  created_ts INTEGER NOT NULL, updated_ts INTEGER NOT NULL, closed_ts INTEGER);
CREATE UNIQUE INDEX alerts_one_open ON alerts(type, asset) WHERE state = 'open';
CREATE TABLE ingest_log (batch_id INTEGER PRIMARY KEY AUTOINCREMENT, received_wall_ts INTEGER NOT NULL, sim_ts INTEGER NOT NULL,
  n_rows INTEGER NOT NULL, n_null_values INTEGER NOT NULL, n_rejected_rows INTEGER NOT NULL,
  n_out_of_range INTEGER NOT NULL, n_duplicates INTEGER NOT NULL);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- keys: scenario, replay_state, replay_position_ts, replay_speed, model_version, thresholds (JSON), schema_version
```

One open alert per (type, asset) stops the same fault filling the list. Features are not stored; they are recomputed by
`features.py`, so training and live cannot drift apart.

**API (JSON, no auth, CORS open to the Vite dev server only):**

| Endpoint | Returns or does |
|---|---|
| `GET /api/health` | `{ok, scenario, replay_state, db_rows, model_version}` |
| `GET /api/layout` | zones and sections |
| `GET /api/status` | 120 zones and `DRIVE`: status, p_abnormal, p_fail_24h, est hours, data quality, per-signal sensor state, current simulated time |
| `GET /api/zones/{id}/series?signal=&from=&to=` | one signal, at most 500 points (bucket means) |
| `GET /api/belt/series?signal=&from=&to=` | same for a drive signal |
| `GET /api/alerts?state=open` | type, asset, location, severity and label, eta, message, evidence |
| `GET /api/predictions?asset=&from=&to=` | probabilities and estimated hours over time |
| `GET /api/sensors/health` | stuck and silent sensors with since time |
| `GET /api/ingest/stats` | totals and recent batches from `ingest_log` |
| `GET /api/replay`, `POST /api/replay` | position, speed, state; play, pause, speed |
| `POST /api/ingest` | a batch of readings |

**Finished when:** a full demo replay runs to the end; the database's `ingest_log` totals equal `replayer_injected.json`; the
status rule, alert lifecycle and data-quality rules pass their tests (section 9).

## 7. Stage 9: frontend (Sunday evening)

React 18, Vite, TypeScript, Tailwind, TanStack Query polling every 2 s, Recharts, plain SVG for the map (no map library). The
built `frontend/dist` is **committed** and served by FastAPI, so a reviewer needs only Python; `make dev` runs Vite with a proxy
to port 8000. One page, no router:
- **Top bar:** simulated time, play, pause, speed (1x and 3x), ingest health.
- **Racetrack map:** carry zones 1 to 60 along the top, left to right; return zones 61 to 120 along the bottom, right to left;
  12 sections labelled and separated. Zones coloured by the four statuses (normal, abnormal, predicted failure, no data /
  sensor suspect), colour-blind-safe, with a legend.
- **Drill-down:** click a section to highlight and list its 10 zones; click a zone to see its five signals (Recharts), sensor
  health, predictions over time and open alerts.
- **Tabs:** Active anomalies (type, location, severity, eta, "why" from the top 3 features); Model predictions (what, where,
  when, sorted by risk); Data health (ingest counts including injected faults, stuck and silent sensors).
- **Cut first if behind:** per-zone charts, filters, polish. The map, anomalies, predictions and data-health tabs are required.

## 8. Stage 10: integration and delivery (Monday)

`make demo` works from a clean clone; README (design decisions with reasons, what we built and skipped, simulation honesty,
evaluation results, how to run, screenshots, what we would do differently); CI; `ai_usage/` is the user's and nobody else touches it.

**Repo layout:** `common/ids.py`, `simulator/`, `model/` (with `artifacts/`), `backend/`, `replayer/`, `frontend/`, `scripts/`,
`tests/`, `tools/`, `configs/` (`default.yaml`, `scenarios/`, `app.yaml`), `docs/`, `requirements.txt`, `.python-version`,
`Makefile`, `README.md`, `.github/workflows/ci.yml`, `ai_usage/`. `data/` is git-ignored and regenerated.

**Makefile:** `setup` (venv on 3.12, install), `demo-data` (build `data/demo_01`), `demo` (data if needed, then
`scripts/demo.py`; open http://localhost:8000), `data` (full scenario set), `features`, `train`, `score`, `frontend` (build into
`frontend/dist`), `dev`, `test`. The README also gives plain-Python lines for machines without `make`.

**CI (GitHub Actions):** a Python 3.12 job (`pip install -r requirements.txt`, `pytest` on mini scenarios) and a Node 20 job
(`npm ci`, `tsc --noEmit`, `vite build`). Lint with `ruff`; no `mypy`. Logging with stdlib `logging`; `GET /api/health`.

**Committed:** code, configs, trained models, built frontend. **Regenerated by `make demo`:** data from seeds.

## 9. Tests required

Ingest (rejected, duplicate, out-of-range, null, out-of-order; `ingest_log` totals equal `replayer_injected.json`); ids (round
trip for all 121 assets; scenario labels map onto database asset ids with no gaps); features (live equals training); data
quality (stuck after 15 identical running values and clears; `no_data` after 5 silent minutes; one missing reading raises
nothing); alerts (one open per type and asset; debounce; severity bands); status rule (predicted failure beats sensor suspect
beats the model's abnormal or normal); leakage scan (section 4); the 66 existing simulator tests.

## 10. Schedule (IST)

| When | Work |
|---|---|
| Sat evening | Python 3.12 rebuild and test run; copy this file into the repo's `docs/`; **Stage 6** |
| Sun morning | **Stage 7**: model data, features, baselines, train, evaluate, thresholds |
| Sun afternoon | **Stage 8**: backend, replayer, data quality, live scoring, API |
| Sun evening | **Stage 9**: frontend |
| Mon | **Stage 10**: integration, `make demo` from a clean clone, README, screenshots; the user submits before midnight |
| Tue | Buffer only |

**Cut order if behind:** dashboard extras first, then the precomputed-fallback flag, then the three-seed repeat (one seed, said honestly).

## 11. What the README must say plainly

- The model learns our simulator's rules; a good score does not prove it works on a real belt.
- Vision status at submission: what is built and what is a placeholder, said plainly.
- Only about 8 drive-overheating events in training and 2 in test, so the drive model is weak evidence.
- The demo scenario is curated; the test scenarios are not.
- The replayer, not the simulator, injects duplicates, out-of-range values and out-of-order batches.
- Early bearing faults are faint by design in the simulator (research shows they barely show in temperature or vibration RMS), so
  the model targets the advanced stage; lead time is hours, not days.
- Stuck-sensor detection by exact repeats assumes noisy floats; a coarse real sensor would need another rule.
- YAML-held numbers (fault amplitudes, thresholds, severity weights) are assumptions informed by a few papers.
