# Conveyor Simulator: Build Report

Status as of 2026-11-14. Covers everything built so far (Stages 0 to 4 of the simulator) and every decision made on
the way. Project deadline: Nov 17, 2026, 11:59 PM IST.

Related files: [`PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md) (the whole project),
[`SIMULATOR_SPEC.md`](../SIMULATOR_SPEC.md) (the build spec), [`CHANGELOG.md`](CHANGELOG.md) (one entry per stage).

---

## 1. Summary

The simulator generates sensor data for a 6 km looping mining conveyor: 120 zones reporting five signals each every
minute, plus eight belt-wide drive signals. It is the only allowed source of sensor data for the take-home.

**Built and tested (Stages 0 to 4):**

| Stage | What it adds | Commit |
|---|---|---|
| 0 | Git repository, `.gitignore` | `7e4efaa` |
| 1 | Healthy baseline: layout, operating conditions, healthy signals, generator, CLI | `29c6a8e` |
| 2 | Belt coordinates, equipment faults, repair stops | `abd4c6d` |
| (change) | Belt damage fault removed | `57271de` |
| 3 | Hard negatives and ground-truth labels; spec and overview updated | `bb361f3` |
| 4 | Messiness: missing readings, outages, stuck sensors | `e0e6279` |

**Not built yet:** Stage 5 (camera capture events, visual defects, replay stream), Stage 6 (the scenario set), and
everything outside the simulator (backend, model, vision, dashboard, README, `ai_usage/`).

**Numbers for a default 30-day run (seed 1):**

| Item | Value |
|---|---|
| Zone rows | 5,184,000 (43,200 minutes x 120 zones) |
| Generation time | about 2 s; about 5 s including writing files |
| Public data size | 116 MB |
| Equipment faults | 5 idler bearing, 1 drive overheat |
| Hard negatives | 4 heavy-load shifts, 1 heat wave, 2 dusty spells, 35 start-up transients |
| Sensor faults | 6 outages, 4 stuck sensors, 1.00% random missing readings |
| Label rows | 5,227,200 (43,200 minutes x 121 assets) |
| Positive rate of `fails_within_24h` | 0.165% of label rows |

**Tests:** 66 tests in 5 files; 63 pass and 3 are skipped on purpose (section 11). The suite runs in about 10 s.

**Code size:** about 1,200 lines in `simulator/`, about 2,200 including tests and plot tools.

---

## 2. How to run it

The machine had no `pip` and no `venv` module, so the environment was built with `uv` (installed to `~/.local/bin`).
Dependencies are in `requirements.txt` and live in `.venv/`, which is git-ignored.

```bash
.venv/bin/python -m simulator.run --days 30 --seed 1     # writes data/dev_seed1_30d/
.venv/bin/python -m pytest -q                            # the test suite
.venv/bin/python -m tools.plot_stage1                    # sanity plots, one per stage (1 to 4)
```

CLI options: `--config <yaml>`, `--days`, `--seed`, `--scenario <name>`, `--out <folder>` (default `data`).
`configs/default.yaml` holds every default value.

---

## 3. What the simulator produces

### 3.1 The belt

| Property | Value |
|---|---|
| Length | 6 km loop: 3 km carry side, 3 km return side |
| Speed | 4 m/s, so one lap takes 1500 s = 25 min |
| Zones | 120 zones of 50 m; zones 1 to 60 carry, 61 to 120 return |
| Sections | 12 sections of 500 m (10 zones each) |
| Loading / unloading | zones 1 and 2 / zones 59 and 60 |
| Duty cycle | 18 h running from 05:00, 6 h stopped, with 5-minute start and stop ramps |

### 3.2 Signals

Per zone, per minute: `vibration_rms_mm_s`, `vibration_kurtosis`, `bearing_temp_c`, `acoustic_db`, `belt_offset_mm`.

Belt-wide, per minute: `throughput_tph`, `motor_current_a`, `motor_temp_c`, `gearbox_temp_c`, `drive_speed_ms`,
`belt_speed_ms`, `slip_pct`, `tension_kn`.

### 3.3 Files on disk

Each scenario is written to `data/<scenario>/`, split into two folders.

**`public/`: the only folder the pipeline (backend, model, dashboard) may read.**

| File | Contents |
|---|---|
| `zones.parquet` | one row per (timestamp, zone): timestamp, section, zone numbers, position, run (carry/return), run state, the five signals |
| `belt_wide.parquet` | one row per timestamp: run state and the eight drive signals |
| `layout.parquet` | one row per zone: numbering, position, run, loading/unloading flags, curve flag and radius, incline |
| `manifest.json` | scenario name, start time, step, number of timestamps and zones, file list |

**`truth/`: ground truth and simulator internals. The pipeline must never read it.**

| File | Contents |
|---|---|
| `faults.parquet` | equipment fault events: id, type, asset, onset, failure, repair end, durations |
| `labels.parquet` | per (timestamp, asset): stage, severity, hours to failure, fails within 24 h, fault id, hard-negative flag |
| `hard_negatives.parquet` | normal events that look like trouble: id, type, start, end, zones, value |
| `sensor_faults.parquet` | outages and stuck sensors: id, type, zone, signal, start, end, stuck value |
| `conditions.parquet` | per minute: speed, load, ambient temperature, dust, cumulative belt distance |
| `layout_truth.parquet` | the hidden per-zone random biases |
| `config_used.yaml` | the exact config of the run |
| `run_meta.json` | seed, row counts, fault counts, measured missing rate, timing |

---

## 4. How a scenario is generated

The order matters and is fixed. Each step uses its own random stream.

1. **Random streams.** `SeedSequence(seed).spawn(6)` gives six independent generators: layout, conditions, faults,
   hard negatives, signals, messiness. Changing one part does not reshuffle another.
2. **Layout.** One row per zone, plus the hidden biases.
3. **Operating conditions.** Run state, speed, load, ambient temperature, dust.
4. **Hard negatives.** Scheduled, then applied as edits to the conditions.
5. **Equipment faults.** Scheduled; each failure is moved to a running minute.
6. **Repair stops.** The belt is stopped for each repair window; conditions are recomputed.
7. **Healthy signals.** Zone signals and belt-wide signals from the conditions, including the start-up transient.
8. **Belt distance.** Cumulative travel from the clean belt speed.
9. **Fault effects.** Added on top of the healthy signals.
10. **Labels.** Built from the clean fault schedule and the hard-negative windows.
11. **Messiness.** Stuck sensors, outages and random missing readings, applied to the public signals only.
12. **Write.** Public tables are checked against a column allowlist, then everything is written.

Why this order: repair stops change the run state, so they must exist before the physics. Labels are built before
messiness, so data defects can never alter ground truth.

Stage 5 will add image capture events between steps 10 and 11.

---

## 5. Protection against data leakage

"Leakage" here means ground truth, or a hidden simulator parameter, reaching data the model can read. A model that
reads a leaked label scores well for the wrong reason. Six guards are in place.

| Guard | What it does | Where |
|---|---|---|
| Folder split | `public/` and `truth/` are separate folders | `simulator/io.py` |
| Single loader | `load_public()` reads `public/` only and works with `truth/` deleted | `simulator/io.py` |
| Column allowlist | every public table must match an exact column list before it is written or loaded | `check_public_columns` |
| Forbidden words | a public column name containing `fault`, `label`, `stage`, `severity`, `failure`, `truth`, `bias`, `ambient`, `dust`, `load` and similar raises `LeakageError` | `simulator/io.py` |
| Label timing | labels are built before messiness; a test proves a messy run and a clean run have identical ground truth | `tests/test_stage4.py` |
| Source scan | a test scans `backend/`, `model/`, `vision/`, `dashboard/` for any reference to truth files | `tests/test_leakage.py` |

The source scan passes trivially today because those folders do not exist yet. It starts to matter when the backend
is written.

### Three changes from the spec, made to stop leakage

| Spec said | Built instead | Why |
|---|---|---|
| Per-zone random biases in `layout.parquet` | Moved to `truth/layout_truth.parquet` | They are the simulator's hidden parameters; a model could read each zone's temperature bias directly |
| `config_used.yaml` and `run_meta.json` readable by the pipeline | Moved to `truth/`; the pipeline gets `manifest.json` | From Stage 2 they carry fault rates and per-scenario fault counts |
| `fault_id_in_view` and `damage_severity` inside `image_events.parquet` | Planned for Stage 5: ground-truth columns go to a separate truth file | A file the pipeline reads should not carry ground-truth columns at all |

### What is deliberately public

- **`run_state`** (stopped, ramp, running). A real plant knows whether its belt is moving. A repair stop looks like any
  other stop: there is no "repair" value.
- **Layout facts** an engineer would know: positions, incline, curves, loading zones.
- **`belt_speed_ms`**. It integrates to the true belt position, which Stage 5 relies on.

---

## 6. Stage 1: healthy baseline

### 6.1 Layout (`simulator/layout.py`)

- **Terrain.** One random number per carry zone, smoothed (Gaussian, sigma 6 zones), mapped to -10 to +15 degrees. The
  return run is the carry run reversed and negated: a climb out is a descent back.
- **Curves.** Four curves of two zones each on the carry run, radius 300 to 800 m, mirrored onto the return run.
- **Hidden biases.** Per zone: temperature bias (sd 0.8 C), vibration scale (about 5%, times 1.25 in loading zones),
  lateral offset bias (sd 3 mm). These make zones differ from each other the way real installations do.

### 6.2 Operating conditions (`simulator/conditions.py`)

| Condition | How it is built |
|---|---|
| Run state | 18 h window from 05:00 with 5-minute ramps at each end |
| Speed | the ramp factor times a small slow jitter (sd 0.4%) |
| Load | shift mean (0.80 morning, 0.85 afternoon, 0.65 night) plus slow noise, clipped to 0.30 to 1.0, times speed |
| Ambient | 30 C mean, 10 C daily swing peaking at 15:00, small seasonal term, slow noise |
| Dust | 0.30 base, plus 0.15 when running, plus slow noise |

Two helpers do the time behaviour without loops: **AR(1) noise** (slowly wandering noise with a set standard
deviation) and a **first-order lag** (a thermal mass: temperature chases its target slowly).

### 6.3 Healthy signals (`simulator/healthy.py`)

| Signal | Driven by | Why it matters |
|---|---|---|
| Bearing temperature | ambient + speed + load, through a 20-minute lag | Ambient and load are the confounders the model must cancel |
| Vibration RMS | speed and load, raised by dust | Steps up when the belt runs; near 0.03 mm/s when stopped |
| Vibration kurtosis | about 3 with small noise | A healthy bearing sits near 3 |
| Acoustic level | 42 dB stopped, about 59 dB running | Noise grows with dust |
| Lateral offset | per-zone bias plus slow wander, doubled on curves | Mistracking shows here |
| Motor current | speed, load and how much the carry run climbs | |
| Motor / gearbox temperature | ambient plus current squared, through 25 / 40 minute lags | |
| Belt speed, slip | drive speed minus about 0.3% slip | |
| Tension | 110 kN stopped, about 156 kN running | |

Return-side idlers carry only the empty belt, so load heats them at 15% of the carry-side rate.

`signals.noise_scale` multiplies every noise term, so the stress scenario can double noise without new code.

### 6.4 Measured against the spec's targets (healthy 14-day run, seed 1)

| Check | Target | Measured |
|---|---|---|
| Zone rows | 2,419,200 | 2,419,200 |
| Running share of minutes | 0.74 to 0.76 | 0.749 |
| Bearing temperature, running | 20 to 50 C | 20.6 to 54.5 C |
| Carry vs return temperature | carry a few degrees hotter | 43.7 vs 39.1 C |
| Vibration: carry / loading / return | about 2.0 / 2.7 / 1.8 mm/s | 2.17 / 2.56 / 1.78 |
| Acoustic, running | 58 to 60 dB | 59.2 carry, 56.5 return |
| Motor current, running | about 300 to 380 A | 249 to 374 A |
| Motor temperature peak | about 65 C | 64.6 C |

Two values sit slightly outside the rough targets: bearing temperature peaks at 54.5 C, and motor current dips to
249 A on the low-load night shift. Both follow directly from the spec's formulas, so the coefficients were left alone.

Plot: [`stage1_healthy.png`](stage1_healthy.png).

### 6.5 Config (`simulator/config.py`)

Every number is a field of a typed dataclass and can be set in YAML. A misspelled key raises `TypeError`; a layout
that does not divide evenly (for example a 130 m section with 50 m zones) raises `ValueError`.

---

## 7. Stage 2: belt coordinates and equipment faults

### 7.1 Belt coordinates (`simulator/belt_coords.py`)

The belt is a loop and things ride on it. A **belt coordinate** `c` in 0 to 6000 m is fixed to the belt material.

- `D(t)` is the cumulative distance the belt has travelled. It grows only while the belt moves.
- Ground position of belt point `c`: `(c + D(t)) mod 6000`.
- Belt coordinate under ground position `p`: `(p - D(t)) mod 6000`.

`D(t)` is stored in `truth/conditions.parquet` as `belt_distance_m`. Nothing uses it yet; Stage 5 needs it for the
cameras.

**Decision: `D(t)` is integrated from the clean `belt_speed_ms` signal, not the nominal 4 m/s.** The spec computes
distance before the signals exist. Using the generated belt speed means the public speed signal is consistent with the
true belt position, including the 0.3% slip.

### 7.2 The two faults (`simulator/faults.py`)

| Fault | Where | Rate | Onset to failure | Repair |
|---|---|---|---|---|
| Idler bearing wearing out | one zone | 5 per 30 days | 3 to 14 days | 2 h |
| Drive overheating | the drive | 1 per 30 days | 6 to 48 h | 2 h |

All rates and amplitudes are assumptions informed by a few papers, not measured values. They are in the config so
they can be changed and defended.

**Severity.** Each fault has a severity `s` from 0 to 1:
`s = ((t - onset) / (failure - onset)) ^ 2.5`. It stays at 1 during the repair window and is 0 otherwise. The
exponent makes it slow at first, then accelerating.

**Idler bearing effects, in the faulty zone only:**

| Signal | Effect at severity `s` |
|---|---|
| Bearing temperature | + 25 C x s^2 |
| Acoustic | + 10 dB x s^1.5 |
| Vibration RMS | x (1 + 0.5 x s^2) |
| Kurtosis | + 1.5 x s |
| Motor current (belt-wide) | + 1.5 A x s^3 |
| Motor and gearbox temperature | none: the drive cannot see an idler fault |

**Drive overheat effects:**

| Signal | Effect at severity `s` |
|---|---|
| Motor temperature | + 35 C x s^1.3 |
| Gearbox temperature | + 20 C x s^1.3 |
| Motor current | + 6 A x s^2 |
| Speed, throughput, every zone signal | none |

**Repair stops.** At failure the belt stops for the repair window, then restarts with a 5-minute ramp. Afterwards the
asset is healthy again.

**Scheduling rules** (tested over 12 seeds): no event starts in the first 24 h; every event, including its repair,
ends at least 12 h before the scenario ends; no two events overlap on one asset.

### 7.3 Decisions in Stage 2

| Decision | Why |
|---|---|
| A failure drawn inside a stop moves to the next running minute | An asset fails while it is working, not while the belt is parked |
| Every effect is scaled by speed | A stopped belt shows no fault signature: no heat, no noise, no vibration |
| Temperature effects pass through the same thermal lag as the healthy signal | Otherwise the extra heat would vanish the instant the belt stops |
| Idler faults in one scenario are on distinct zones | The simplest way to guarantee no overlap on one asset |
| Two drive events keep at least 24 h apart | Same guarantee for the drive (`faults.same_asset_gap_hours`) |
| Fractional event counts use the fraction as a probability | A 14-day run expects 2.33 idler faults: it gets 2, and a third with probability 0.33 |
| The idler fault adds motor current after drive temperatures are computed | So the drive temperatures are bit-for-bit unchanged, as the spec requires |

### 7.4 How it was verified

Each test generates the same scenario twice: once normally, and once with the fault's amplitudes set to zero. The
random draws and the schedule are identical, so any difference between the two is the fault effect and nothing else.

- **Idler fault:** only the faulty zone changes; drive temperatures are identical; motor current rises by at most
  1.5 A. At severity 0.1 or below, every effect is within about one noise standard deviation.
- **Drive fault:** motor temperature rises by more than 25 C and gearbox by more than 12 C; no zone signal changes;
  current stays inside its noise early on.
- **Repair:** the belt shows `stopped` for the whole window and signals match the no-fault run afterwards.

Plot: [`stage2_faults.png`](stage2_faults.png).

---

## 8. The belt damage decision

Stage 2 originally had a third fault: **belt tear or damage**. A defect at a point on the belt hit every zone once per
lap (a kurtosis spike, weaker on the return side) and dipped the tension near failure. It was built, tested against
the lap formula, and then removed.

**Why it was removed.** Its purpose was to link sensors to camera images. That link was weaker than it looked, because
public image datasets are unrelated photos of damage:

- **No severity progression.** The sensor fault grows smoothly over days; no dataset has images that match severity
  0.2 versus 0.9.
- **No repeat views of one defect.** The loop story needs the same defect seen lap after lap. We would have had to pin
  one image to each fault and re-show it.
- **No sensor context.** The plan was to pick any damaged image when the simulator said damage was in view. Sensors
  and images would have agreed only because we wired them to agree.

**What replaces it.** Belt damage is handled by vision alone. In Stage 5 the simulator will place a few **visual
defects** at fixed belt coordinates, with no effect on any sensor. Each is pinned to one dataset image and comes into
view once per lap at each camera, so the vision pipeline still has to merge repeat detections by belt coordinate and
ignore splice joints.

**What this costs.** Sensors and images are not linked. A defect the camera sees has no sensor signature. This is
recorded in the overview's "honest limits" and should be stated in the README.

**What was kept.** `simulator/belt_coords.py` and `belt_distance_m`, because the cameras need them.

**Side effect.** Fault schedules for a given seed changed, because the belt-damage random draws no longer happen.

---

## 9. Stage 3: hard negatives and labels

### 9.1 Hard negatives (`simulator/hard_negatives.py`)

Normal events that look like trouble. Without them a model learns "anything unusual means fault" and its scores
flatter it. They are labelled as not faults, in their own file.

| Event | Effect | Rate per 30 days | Duration |
|---|---|---|---|
| Heavy-load shift | load held at 0.95 to 1.0 | 4 | 6 to 12 h |
| Heat wave | ambient + 6 to 10 C | 1 | 1 to 3 days |
| Dusty spell | dust raised to 0.8 to 0.9 | 2 | 4 to 24 h |
| Start-up transient | vibration x 1.3 | every start | 10 min |

**Decision: the first three edit the operating conditions before the physics runs.** Nothing is patched onto the
signals. A heavy-load shift raises motor current and temperatures because the healthy formulas say so.

**The start-up transient** is a multiplier applied in `healthy.py` for the first 10 minutes after every start,
including restarts after a repair.

**A property worth knowing:** a heat wave warms every zone by exactly the same amount (tested). A neighbour
comparison cancels it completely, which is what the model's neighbour features rely on.

### 9.2 Labels (`simulator/labels.py`)

One row per (timestamp, asset). Assets are the 120 zones plus `DRIVE`.

| Column | Meaning |
|---|---|
| `stage` | `normal`; `early` (0 < s < 0.3); `advanced` (s >= 0.3, not yet failed); `failed` (failure to repair end) |
| `severity` | the fault's `s` |
| `hours_to_failure` | hours until this asset fails; 0 while failed; empty when healthy |
| `fails_within_24h` | a fault is developing and its failure is at most 24 h away |
| `fault_id` | the active fault, if any |
| `is_hard_negative_period` | a hard negative is active (belt-wide, so the same for every asset) |

### 9.3 Decisions in Stage 3

| Decision | Why |
|---|---|
| `hours_to_failure` and `fails_within_24h` are set only from onset | Before the fault exists, nothing in the data can predict it. A positive label there would only be noise. A drive fault that runs 6 h from onset to failure has only 6 h of positive labels, not 24 |
| `labels.parquet` also carries `severity` and `fault_id` | Useful for analysis and for scoring per event |
| Start-up transients count as hard-negative periods | They are the fourth hard negative in the spec |
| A heat wave fades in and out over 3 h | A step change in ambient temperature is not plausible |
| A heavy-load shift holds load at its drawn level | Simple; it makes load flat for the length of the event |

### 9.4 How it was verified

- Each hard-negative type is compared against an identical run without it, and checked for a visible, plausible
  change in the right signals.
- One idler fault's labels are checked value by value against the schedule: the 24-hour boundary to the minute, the
  early/advanced boundary, the failed window, and that every other asset is normal throughout.

Plot: [`stage3_hard_negatives_labels.png`](stage3_hard_negatives_labels.png).

---

## 10. Stage 4: messiness

Data defects (`simulator/messiness.py`). These are **sensor faults**, kept apart from equipment faults so each can be
scored on its own.

| Defect | What happens | Rate |
|---|---|---|
| Missing readings | a reading becomes empty (NaN) | about 1% of readings, per signal |
| Outage | all five signals of one zone are silent | 6 per 30 days, 1 to 6 h each |
| Stuck sensor | one signal in one zone freezes at its last value while the equipment keeps changing | 4 per 30 days, 2 to 48 h each |
| Noise | already in the baseline, scaled by dust and `signals.noise_scale` | |

The stuck sensor is the hard case: a flat line looks healthy, so the data-quality check has to notice it.

### 10.1 Decisions in Stage 4

| Decision | Why |
|---|---|
| Injected last, after labels | Ground truth cannot be altered by a data defect |
| Order: stuck, then outage, then random loss | A stuck sensor still loses the odd packet, as a real one would |
| Rows are never dropped | A missing reading is an empty value in its row; `timestamp` and `run_state` are always present |
| Belt-wide signals get random loss only | The spec defines outages and stuck periods for zone suites only |
| Overlaps are allowed | A stuck sensor in a failing zone is the hard case the spec wants |
| Clean signals are never written | Only the messy version reaches `public/` |
| The stuck value is recorded | `sensor_faults.parquet` has a `stuck_value` column |

### 10.2 Measured against the spec's targets (30 days, seed 1)

| Check | Target | Result |
|---|---|---|
| Missing rate | 0.5% to 1.5% | 1.00% |
| Outage and stuck counts | 6 and 4 | 6 and 4 |
| Durations | 1 to 6 h and 2 to 48 h | within range |
| Flat lines | exactly constant | exactly constant, equal to the last value before the freeze |
| `sensor_faults.parquet` | lists each one | one row per outage and stuck period |

Tests also confirm that every reading outside a defect is bit-for-bit equal to the clean run.

Plot: [`stage4_messiness.png`](stage4_messiness.png).

---

## 11. Tests

| File | Tests | Covers |
|---|---|---|
| `tests/test_stage1.py` | 16 | layout counts, row counts, physical sanity, determinism, config errors |
| `tests/test_leakage.py` | 7 | public/truth split, allowlist, loader, source scan |
| `tests/test_stage2.py` | 27 | scheduling rules over 12 seeds, severity, effects, repair stops, belt coordinates |
| `tests/test_stage3.py` | 7 | each hard negative, default counts, labels by hand |
| `tests/test_stage4.py` | 9 | missing rate, outages, stuck sensors, untouched ground truth |

The 3 skips are intentional. Stage 2 tests run once per fault type: the idler test skips for the drive fault, the
drive test skips for the idler fault, and the determinism re-run is limited to one fault type to keep the suite fast.

The main testing method is the **twin run**: generate the same seed twice with one thing switched off. Because each
concern has its own random stream, everything else is identical, and the difference is exactly the effect under test.

---

## 12. Where the build differs from the spec

Collected in one place. Each is also in the changelog.

| Topic | Spec | Built | Why |
|---|---|---|---|
| Output folders | all files in `data/<scenario>/` | `public/` and `truth/` | Leakage guard by folder |
| Per-zone biases | in `layout.parquet` | in `truth/` | Hidden simulator parameters |
| Config and run metadata | pipeline may read | in `truth/` | They carry fault rates and counts |
| Belt damage | third sensor fault | removed; vision-only visual defects | No honest sensor-to-image link (section 8) |
| Belt distance | computed before the signals | integrated from the clean belt speed | Consistent with the public speed signal |
| Failure time | any time | moved to a running minute | Assets fail while working |
| Temperature effects | added instantly | passed through the thermal lag | Physically consistent when the belt stops |
| Label start | not specified | from onset only | No signal exists before onset |
| Heat wave | offset | offset with 3 h fade in and out | No step change |
| Camera capture interval | 30 s | 2 s (in the spec, not built yet) | At 30 s a defect is seen on about 8% of laps |
| Python version | 3.11+ | 3.14.4 | What the machine has |

---

## 13. Known issues and open questions

- **Acoustic does not lead temperature in noise terms.** The spec says acoustic "appears earlier than temperature".
  At severity 0.3 the acoustic effect is about 1.4 noise standard deviations and the temperature effect about 7.5,
  because temperature noise is small (0.3 C). If acoustic should be the early flag, the amplitudes or noise need
  changing. Left as specified.
- **Severity exponents compound.** The progression exponent (2.5) multiplies each effect's own exponent, so idler
  temperature grows roughly as time to the power 5. Effects stay very small for most of a fault's life and rise
  steeply at the end.
- **Kurtosis is marginal at early severity.** At `s = 0.1` the idler kurtosis effect is 0.15 against noise of about
  0.147, right at the "one sigma" rule.
- **Heavy load is a weak hard negative on day shifts.** The level (0.95 to 1.0) is only a little above the normal
  day-shift load (about 0.85). It is clearest at night.
- **Positive labels are rare.** `fails_within_24h` is true for 0.165% of label rows in a default run. The model and
  its evaluation must be built for that imbalance.
- **Few drive events.** One per 30 days means about 8 across the training scenarios and 2 in test. The README must
  say so.
- **The hard stuck case is not forced.** A stuck temperature sensor in a failing zone can happen by chance; in seed 1
  it does not.
- **Sensors and images are not linked** (section 8).
- **Bearing temperature peaks at 54.5 C**, above the spec's 50 C rough target, and higher still in a heat wave.
- **One unexplained slow run.** A 30-day generation once took 88 s inside the test suite. It was not reproduced
  standalone (4.8 s under the profiler) and the cause was not found. That test now checks the schedule only, and
  categorical columns were made faster, which brought a 30-day generation to about 2 s.
- **The source-scan leakage test has nothing to scan yet.**

---

## 14. What is left

| Item | Notes |
|---|---|
| Stage 5: image events and replay | Visual defects, capture events with belt coordinates, splice joints, a stream that replays a scenario in timestamp order. The 2 s capture interval is finer than the one-minute step, so belt travel must be interpolated; it also means about 2.9 million capture rows per 30 days |
| Stage 6: scenario set | 12 scenarios of 30 days (8 train, 2 validation, 2 test) plus 1 stress scenario; about 1.6 GB, regenerated by one script |
| Backend | FastAPI + SQLite, data-quality check that produces the "no data / sensor suspect" status |
| Prediction model | LightGBM, whole-scenario split, two baselines |
| Vision | public dataset, detector, de-duplication by belt coordinate |
| Dashboard | Streamlit |
| README and `ai_usage/` | design decisions, dataset choice, honest limits, AI chat logs |

Plan in the overview: backend and model on Nov 15, vision and dashboard on Nov 16, README and submission on Nov 17.

---

## 15. File map

```
simulator/config.py           typed config, YAML load/dump, rejects typos and bad layouts
simulator/layout.py           per-zone layout (public) and hidden biases (truth)
simulator/conditions.py       run state, speed, load, ambient, dust; AR(1) and lag helpers
simulator/healthy.py          healthy zone and belt-wide signals
simulator/belt_coords.py      belt distance and coordinate conversion
simulator/faults.py           fault scheduling, repair stops, severity, effects
simulator/hard_negatives.py   hard-negative scheduling and condition edits
simulator/labels.py           ground-truth labels
simulator/messiness.py        missing readings, outages, stuck sensors
simulator/generate.py         seed -> tables, in the fixed order
simulator/io.py               public/truth writer, allowlist, load_public
simulator/run.py              CLI
tests/                        66 tests
tools/plot_stage1..4.py       sanity plots, written to docs/
configs/default.yaml          every default value
docs/CHANGELOG.md             one entry per stage
docs/*.png                    the four sanity plots
```
