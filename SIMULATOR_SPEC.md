# Simulator Build Spec (for Claude Code)

> **Status note (2026-11-14). `docs/BUILD_PLAN.md` wins where it differs from this file.**
> - Stages 0 to 4 and Stage 6 (scenario set) are built. Anything below that says nothing is built is superseded.
> - Python 3.12 (this file says 3.11+).
> - Two sensor faults only: idler bearing and drive overheating. Belt damage (section 7.3) is removed.
> - Section 9 (image events, Stage 5) is **not built yet**: vision is deferred while the image dataset downloads,
>   with placeholders in the backend and dashboard. Replay is a separate process, not part of the simulator.
> - Section 6 (belt coordinates): the code is kept, unused for now.
> - Section 14 fault counts: 40 idler and 8 drive events in training, 10 and 2 in validation, 10 and 2 in test,
>   8 and 2 in stress (`docs/scenario_set_report.md`).


Read `PROJECT_OVERVIEW.md` first for context. This file is the full build spec for the simulator. Build exactly what is
listed, in the order given in section 13, and stop at each "finished when" test.

Why each decision was made is written next to it. If a value is marked **(assumption)**, it is our own choice, not a
measured or published number; keep it in the config so it can be changed and defended in the README.

---

## 0. Current state

**Nothing is built yet.** Start from an empty repository and build everything in this spec, stage by stage (section 13).
Stage 0 is `git init`, a `.gitignore` (`data/`, `__pycache__/`, `.pytest_cache/`) and a first commit; then commit after
every stage. (An earlier draft of the baseline was written and thrown away; this spec is now the only source, so every
formula and value you need is below.)

Suggested layout:

```
simulator/config.py       dataclass config + YAML load/dump (typo-rejecting)
simulator/layout.py       one row per zone
simulator/conditions.py   run state, speed, load, ambient, dust; AR(1) and thermal-lag helpers
simulator/healthy.py      healthy zone signals and belt-wide drive signals
simulator/generate.py     seed -> tables
simulator/run.py          CLI
tests/                    pytest
tools/                    sanity-plot scripts (write PNGs to docs/)
configs/default.yaml      dumped defaults
```

## 1. Goal

Generate synthetic but physically plausible sensor data for a 6 km conveyor loop, with labelled faults, labelled
normal-but-suspicious events, and realistic data defects, so that a model can be trained and tested on it, and a
dashboard can be fed from it live.

The simulator must be: **reproducible** (one seed reproduces a run exactly), **fast** (generate in seconds, not
minutes), **configurable** (every number in YAML), and **honest** (ground truth is never mixed into pipeline data).

## 2. Tech stack

| Piece | Choice | Why |
|---|---|---|
| Python 3.11+ | language | The model, backend and dashboard are Python too |
| numpy, scipy | vectorised maths, `lfilter` for AR(1) and thermal lag | No per-minute Python loops; 14 days generate in about 0.5 s |
| pandas, pyarrow | tables, parquet output | Long-format tables; about 57 MB per 14 days |
| numpy `SeedSequence` | spawn one random stream per concern | Changing one part does not reshuffle another |
| dataclasses + PyYAML | config | Typed, rejects typos and bad layouts, one small file per scenario |
| argparse | CLI | `python -m simulator.run` |
| pytest, matplotlib | tests and sanity plots | Tests catch wrong numbers; plots catch numbers that are right but look wrong |

Not used: SimPy (our time steps are fixed), numba (already fast), Pydantic (belongs in the backend).

## 3. Model selection (what the simulator must support)

The simulator exists to feed these models, so its outputs and labels must fit them.

- **Sensor model: gradient boosting (LightGBM).** Why: trains in minutes on 2 CPUs, handles tabular features and rare
  faults well, easy to explain. A sequence model comparison is deferred (README: "what we would do differently").
- **Targets the labels must provide:**
  - status per asset: `normal`, `abnormal`, `predicted_failure` (the fourth dashboard status, `no_data_sensor_suspect`,
    comes from the data-quality check, not from the model);
  - hours to failure per asset;
  - a binary label "fails within the next 24 h" (the main score target; hours-to-failure is secondary).
- **Features the model will build** (not the simulator's job, but signals must allow them): rolling statistics over
  1, 6 and 24 hours, difference from neighbouring zones, load- and ambient-corrected residuals.
- **Evaluation the simulator must make possible:** split by whole scenario (different seeds, fault locations,
  timings), never by random rows; one extra stress scenario with more noise, more missing data and a different fault
  mix; two baselines (fixed temperature threshold, neighbour-deviation check).
- **Vision model** is a separate pipeline (public images); the simulator only produces capture events (section 9).

## 4. Layout (stage 1; 6 km)

| Parameter | Value | Why |
|---|---|---|
| `belt_length_m` | 6000 (3000 carry side + 3000 return side) | Confirmed: 6 km loop, 3 km per side |
| `belt_speed_ms` | 4.0 | Revolution time 6000 / 4 = 1500 s = 25 min |
| `zone_spacing_m` | 50 | One sensor suite per zone; 120 zones. 25 m (240 zones) is a later experiment, a one-line change |
| `section_length_m` | 500 | 12 numbered splice sections (S1 to S12) of 10 zones each |
| `unloading_position_m` | 3000 | Carry run is 0 to 3000 m; return run is 3000 to 6000 m |
| `loading_zone_length_m` | 100 | Zones 1 and 2 are loading zones; zones 59 and 60 are unloading zones |

Numbering: `global_zone` 1 to 120 along the loop; `splice_section` 1 to 12; `zone_in_section` 1 to 10.
`position_m` is the zone midpoint on the ground path. Zones 1 to 60 are the carry side, 61 to 120 the return side.

Terrain: smooth random incline on the carry run (-10 to +15 degrees), mirrored and negated on the return run (a climb
out is a descent back). About 4 curves of 2 zones each (radius 300 to 800 m), mirrored onto the return run. Per-zone
random biases: bearing temperature (sd 0.8 C), vibration scale (about 5%, times 1.25 in loading zones), lateral offset
(sd 3 mm).

**Section boundaries are fixed stretches of the ground path.** Visual defects (section 9) live on the belt, in its own
coordinate system (section 6), and pass through the sections.

## 5. Operating conditions and healthy signals (stage 1)

All numbers below are starting values in the config. Unless a research source is named they are **(assumption)**.

**Helpers.** *AR(1) noise*: `x_t = phi x_(t-1) + e_t`, `e` standard deviation `sd x sqrt(1 - phi^2)`, started from its
stationary distribution, so the marginal standard deviation is `sd` (use `scipy.signal.lfilter` along the time axis).
*First-order lag*: `y_t = y_(t-1) + a (target_t - y_(t-1))`, `a = 1 - exp(-step / tau)`, `y_0 = target_0`, also with
`lfilter`. The lag is a thermal mass: temperatures chase their target slowly.

### 5.1 Layout generation
- `position_m` is the zone midpoint. A zone is on the `carry` run if its midpoint is below `unloading_position_m`, else
  `return`.
- Terrain: draw one standard normal per carry zone, smooth with a Gaussian filter (sigma 6 zones), rescale to 0..1, map
  to -10..+15 degrees. Return-run incline is the carry incline reversed and negated (a climb out is a descent back).
- Curves: `max(2, round(n_carry_zones / 15))` = 4 curves, each 2 zones long, starting on distinct every-6th zone between
  zone 3 and zone 57, radius uniform 300 to 800 m; mirrored (reversed) onto the return run.
- Loading zones: zones starting below 100 m (zones 1 and 2). Unloading zones: zones starting in `[2900, 3000)` m
  (zones 59 and 60).
- Per-zone random biases: `temp_bias_c` normal(0, 0.8); `vib_scale` lognormal(sigma 0.05) times 1.25 in loading zones;
  `offset_bias_mm` normal(0, 3).

### 5.2 Operating conditions (shared by all sensors)
- **Duty cycle:** let `m` be minutes since 05:00 (mod 24 h). The belt is in its running window if `m < 1080` (18 h).
  `base = clip(min(m, 1080 - m) / 5, 0, 1)` inside the window (5-minute start and stop ramps), else 0.
  `run_state` is `stopped` if `base = 0`, `ramp` if `0 < base < 1`, `running` if `base = 1`.
  `speed_frac = base x (1 + AR1(phi 0.99, sd 0.004))`, and 0 when `base = 0`. About 75% of minutes have speed above 0.
- **Load:** shift mean by hour of day: 05 to 13 is 0.80, 13 to 21 is 0.85, 21 to 05 is 0.65.
  `load_raw = clip(mean + AR1(phi 0.995, sd 0.06), 0.30, 1.0)`; `load = load_raw x speed_frac`;
  `throughput_tph = 3000 x load`.
- **Ambient temperature (C):** `30 + 10 cos(2 pi (hour - 15) / 24) + 3 sin(2 pi day / 365) + AR1(phi 0.999, sd 0.5)`.
- **Dust (0 to 1):** `clip(0.30 + 0.15 x base + AR1(phi 0.999, sd 0.03), 0, 1)`.

### 5.3 Healthy zone signals, per zone per minute
`speed`, `load`, `ambient`, `dust` are the conditions above; `lf` is the load factor, **1.0 on the carry side and 0.15 on
the return side** (return idlers carry only the empty belt, so load barely heats them).

| Signal | Formula | Why |
|---|---|---|
| `bearing_temp_c` | `lag(target, tau 20 min) + normal(0, 0.3)`, with `target = ambient + speed x (5 + temp_bias_c) + 7 x load x lf` | Friction heat follows speed and load and lags like a thermal mass. Ambient and load are the confounders the model must learn to cancel. |
| `vibration_rms_mm_s` | `vib_scale x (speed x (0.6 + 0.9 speed) + 0.5 x load x lf) x (1 + 0.3 dust) x exp(normal(0, 0.04)) + 0.02 x abs(normal(0, 1))` | Vibration steps up with speed and load (Bristol 2017). The floor keeps a stopped belt near 0.03 mm/s. |
| `vibration_kurtosis` | `3.0 + normal(0, 1) x 0.12 x (1 + 0.5 dust)` | A healthy bearing is about 3 (general diagnostics knowledge). |
| `acoustic_db` | `42 + 14 speed + 4 x load x lf + normal(0, 1) x 1.2 x (1 + dust)` | Quiet when stopped, louder when running. |
| `belt_offset_mm` | `offset_bias_mm + AR1(phi 0.98, sd 1.5 mm) x (1 + 1.0 x is_curve) + normal(0, 0.5)` | Mistracking shows as offset; curves wander more. |

### 5.4 Healthy belt-wide signals, per minute
- `lift` = mean of `sin(incline)` over carry zones (uphill raises drive load).
- `motor_current_a = speed x 140 + 230 x load x (1 + 5 x lift) + normal(0, 4)` while running. Expect about 300 to 380 A.
- `ratio2 = (motor_current_a / 400)^2`.
- `motor_temp_c = lag(ambient + 30 x ratio2, tau 25 min) + normal(0, 0.3)`;
  `gearbox_temp_c = lag(ambient + 22 x ratio2, tau 40 min) + normal(0, 0.3)`.
- `drive_speed_ms = 4 x speed`; `slip = 0.0025 + 0.0007 x load + normal(0, 0.0003)`;
  `belt_speed_ms = drive_speed_ms x (1 - slip)`; `slip_pct = 100 x slip` when `speed > 0.05`, else NaN.
- `tension_kn = 110 + 10 x speed + 45 x load + normal(0, 1.5)`.
- Config field `signals.noise_scale` (default 1.0) multiplies every noise standard deviation above, so the stress
  scenario can double noise without new code.

### 5.5 Sanity targets (check by plot)
Running day: bearing temperature 20 to 50 C, rising with the afternoon and lagging load; carry side hotter than return
by a few degrees; vibration about 2 mm/s on carry and 2.7 on the loading zone, about 1.8 on return, near 0 when
stopped; acoustic about 58 to 60 dB running and about 42 stopped; motor temperature up to about 65 C at the afternoon
peak.

Known simplification: belt speed is almost constant while running (jitter only), so speed effects are small.

## 6. Belt coordinates (needed for image events)

The belt is a loop and things ride on it. Define a **belt coordinate** `c` in `[0, 6000)` metres, fixed to the belt
material. Reference point `c = 0` sits at ground position 0 at the start of the scenario.

- Cumulative distance travelled by the belt: `D(t) = cumsum(belt_speed(t) x step_s)`, which only grows while the belt
  moves. Compute once per scenario.
- Ground position of belt point `c` at time `t`: `p(t) = (c + D(t)) mod 6000`.
- Belt coordinate under a fixed ground position `p`: `c = (p - D(t)) mod 6000`.
- A belt point revisits the same ground position every 6000 m of travel, which is 25 min at full speed. Stops
  pause everything, so laps are counted in distance, not wall time.
- **Splice joints** ride the belt at `c = 0, 500, 1000, ...` (12 joints). They are landmarks the vision pipeline must
  not report as damage. Emit them in image events (section 9).

Why: this is the one structure that makes a belt defect repeat once per lap at every camera, which the brief hides and
the README must reason about.

## 7. Faults (version 1: two faults)

### 7.1 Common rules

Each fault event has: `fault_id`, `type`, location, `onset_time`, `failure_time`, `repair_end`, and a **severity**
`s(t)` in `[0, 1]`:

```
s(t) = ((t - onset) / (failure - onset)) ^ p     for onset <= t < failure   (p = 2.5: slow, then accelerating)
s(t) = 0                                          before onset and after repair_end
```

- At `failure_time` the asset fails: the belt **stops for a repair window** (`repair_hours`, see 7.2 to 7.4), then
  that asset is healthy again. This creates realistic downtime and a clean label boundary.
- Faults are scheduled **before** the physics runs (so repair stops shape the run state), and their effects are added
  **after** the healthy signals are generated. Order in section 12.
- Events must not overlap on the same asset, onset must be at least 24 h after scenario start, and failure plus repair
  must end at least 12 h before scenario end (so every event is complete, none is cut off).
- **Early effects must be small next to noise.** Research (TU Delft 2018, Bristol 2017) found early-stage bearing
  defects barely visible in temperature or vibration RMS. At `s = 0.1` no signal may move by more than about one noise
  standard deviation. Temperature against neighbours is the strongest late signal.
- Faults are placed with the seed's fault stream; amplitudes and exponents are config values.

### 7.2 Idler bearing degradation (roller wearing out)

- **Where:** one zone `z` (1 to 120). **Counts:** 5 per 30 days **(assumption)**.
- **Duration** onset to failure: uniform 3 to 14 days **(assumption; real ones run days to weeks)**. **Repair:** 2 h.
- **Effects** on zone `z` while running, each scaled by `speed` so a stopped belt shows no heat:

| Signal | Added effect at severity s | Why |
|---|---|---|
| `bearing_temp_c` | `+ 25 x s^2` (total, at failure) | TU Delft: heavily damaged rolls reached about 63 to 66 C against about 30 C intact; temperature is the strongest late signal |
| `acoustic_db` | `+ 10 x s^1.5` | Squeal or rumble from a failing roller; appears earlier than temperature |
| `vibration_rms_mm_s` | `x (1 + 0.5 x s^2)` | RMS rises late |
| `vibration_kurtosis` | `+ 1.5 x s` | Impulsiveness; kept small because the conveyor papers did not confirm it as an early flag |
| `belt_offset_mm` | none | |
| `motor_current_a` (belt-wide) | `+ 1.5 A x s^3` | A seizing idler drags the belt; small and late |

The order of appearance is set by the exponents and amplitudes, all in `faults.idler_bearing` config.
**The drive cannot see idler faults** (Bristol): do not touch `motor_temp_c` or `gearbox_temp_c`.

### 7.3 Belt tear or damage (removed)

Removed on 2026-11-14; it was built in Stage 2 and then taken out. It was a sensor fault at a point on the belt
(once-per-lap impacts at every zone, tension dip near failure), meant to link sensors to camera images.

Why removed: public image datasets are unrelated photos of damage. They have no severity progression, no repeat views
of one defect and no sensor context, so sensors and images would have agreed only because we wired them to agree.
Belt damage is now handled by vision alone, through the visual defects in section 9. There is no `BELT` asset.

### 7.4 Drive overheating (motor)

- **Where:** the drive (asset `DRIVE`). **Counts:** 1 per 30 days **(assumption)**.
- **Duration:** uniform 6 to 48 h **(assumption; hours to days)**. **Repair (trip and cool-down):** 2 h.
- **Effects** while running:

| Signal | Added effect | Why |
|---|---|---|
| `motor_temp_c` | `+ 35 x s^1.3` | Bristol: motor temperature rose 27 C in a lab rig with no change in speed or power |
| `gearbox_temp_c` | `+ 20 x s^1.3` | Gearbox rose 16 C in the same rig |
| `motor_current_a` | `+ 6 x s^2` | Late and small |

  Speed and throughput are unchanged: a fault that only shows in temperature is exactly the case that should not be
  caught by a throughput rule.

### 7.5 Dropped for now

Drive-pulley slip, mistracking and overload or jam are not in version 1. Their signals (`slip_pct`, `belt_offset_mm`,
`motor_current_a`, `throughput_tph`) already exist in the baseline, so they are cheap to add later.

## 8. Hard negatives (normal events that look like trouble)

Why: without them the model learns "anything unusual means fault", and its scores flatter it. They are labelled as
**not faults** in their own file and appear in **every scenario**.

| Event | Effect | Default per 30 days | Duration |
|---|---|---|---|
| Heavy-load shift | load raised to 0.95 to 1.0 belt-wide; bearing and motor temperatures and current rise through the normal physics | 4 | 6 to 12 h |
| Heat wave | ambient temperature offset `+6` to `+10` C | 1 (multi-day) | 1 to 3 days |
| Dusty spell | dust raised to 0.8 to 0.9, increasing vibration noise and acoustic noise | 2 | 4 to 24 h |
| Start-up transient | in the first 10 minutes after each start, vibration RMS x 1.3 and temperature jump already caused by the ramp | every start (automatic) | 10 min |

Implement load surges, heat waves and dust spells as edits to `conditions` **before** the physics (so the healthy model
produces their effects), not as additive patches. The start-up transient is a multiplicative bump in `healthy.py`.
Write each event to `hard_negatives.parquet`: `event_id`, `type`, `start_time`, `end_time`, `zones` (all, for these).

## 9. Image capture events (camera stations)

The simulator does not make images. It makes **capture events** that the vision pipeline will pair with images.

- Three stations on the carrying face at ground positions **50 m, 1500 m and 2950 m** (start at loading, middle, end at
  unloading). Config `images.station_positions_m`.
- **Visual defects.** Place `images.defects_per_30d` = **3** defects **(assumption)** at belt coordinates uniform in
  `[0, 6000)`, each with an `appear_time` (it exists from then until the scenario ends). They have **no effect on any
  sensor signal** and are not equipment faults: no severity, no failure, no repair stop. They exist so the vision
  pipeline has the same defect coming into view lap after lap. Ground truth goes to `visual_defects.parquet`:
  `defect_id`, `belt_coordinate_m`, `appear_time`.
- Field of view `images.fov_m` = **10 m** **(assumption)**. Each running station emits one capture every
  `images.capture_interval_s` = **2 s** of sim time **(assumption, to confirm in Stage 5)**. At 4 m/s the belt moves
  8 m between captures, so consecutive views overlap and every belt point is seen on every lap. The earlier value of
  30 s left 120 m between views, so a defect would have been seen on only about 8% of laps.
- Captures are finer than the one-minute sensor step, so belt travel is interpolated inside each minute.
- Per capture, record: `capture_id`, `timestamp`, `station`, `belt_coordinate_m` (centre of the view, from the formula
  in section 6), `fov_m`, `splice_in_view` (true if a splice joint coordinate is inside the view),
  `defect_id_in_view` (the visual defect's id if its coordinate is inside the view, else null).
- Why this design: a camera sees a short stretch, so a defect is caught in one or two frames per lap; the same
  coordinate reappears every lap. That is exactly the duplicate-detection and belt-coordinate problem the vision
  pipeline must solve. **Pairing:** each visual defect is pinned to one damaged image from the dataset, shown whenever
  `defect_id_in_view` is set; a clean image is shown otherwise.
- Output: `image_events.parquet`. Volume at 2 s is about 2.9 million rows per 30 days; if that is too heavy, write
  only the captures with a splice or a defect in view, plus a count of the rest.

## 10. Messiness (data defects)

Injected **after** physics and fault effects, so ground truth stays clean. Version 1 has four defects. Slow drift,
spikes and timing defects (duplicate timestamps, clock skew) are out of scope for now.

| Defect | Value | Why |
|---|---|---|
| Missing readings | about 1% of readings dropped at random per signal (NaN) | Normal packet loss |
| Outages | a whole suite (all five zone signals) silent for 1 to 6 h; about 6 per 30 days across the belt **(assumption)** | The dashboard must show "no data", not "normal" |
| Stuck sensor | one signal in one zone freezes at its last value for 2 to 48 h while the equipment keeps changing; about 4 per 30 days **(assumption)** | The hardest case: it looks healthy, so the data-quality check must notice a flat line |
| Noise | already in the baseline, scaled by dust; `signals.noise_scale` doubles it in the stress scenario | Shows how much the model leans on clean data |

Rules:
- Missing values and outages are `NaN`; they are never filled in the pipeline data.
- Messiness may overlap a fault on purpose (a stuck temperature sensor in a failing zone is a hard case).
- Every outage and stuck period is written to `sensor_faults.parquet`: `sensor_fault_id`, `type` (`outage` or `stuck`),
  `zone`, `signal` (null for outages), `start_time`, `end_time`. Random 1% missing readings are not listed
  individually; log only the rate in `run_meta.json`.
- Equipment faults and sensor faults are kept in separate ground-truth files so each can be scored on its own.

## 11. Outputs

Per scenario, in `data/<scenario_name>/`. As built, the folder is split in two: `public/` holds what the pipeline may
read and `truth/` holds everything else. `config_used.yaml`, `run_meta.json`, `conditions.parquet` and the per-zone
random biases are in `truth/`; the pipeline gets `public/manifest.json`. See `docs/CHANGELOG.md` for why.

| File | Contents | Pipeline may read? |
|---|---|---|
| `zones.parquet` | one row per (timestamp, zone): timestamp, splice_section, zone_in_section, global_zone, position_m, run, run_state, the five signals | yes (after messiness) |
| `belt_wide.parquet` | one row per timestamp: run_state, throughput, motor current, motor temp, gearbox temp, drive speed, belt speed, slip_pct, tension | yes (after messiness) |
| `image_events.parquet` | capture events (section 9) | yes, except `defect_id_in_view`, which is ground truth and is written to a separate truth file |
| `visual_defects.parquet` | visual defects (section 9) | **no (ground truth)** |
| `layout.parquet` | per-zone static table | yes |
| `faults.parquet` | `fault_id`, `type`, `asset` (zone number or `DRIVE`), `onset_time`, `failure_time`, `repair_end`, key parameters | **no (ground truth)** |
| `labels.parquet` | per (timestamp, asset): `stage` (`normal`, `early` for `0 < s < 0.3`, `advanced` for `0.3 <= s < 1`, `failed`), `hours_to_failure`, `fails_within_24h`, `is_hard_negative_period` | **no (ground truth)** |
| `hard_negatives.parquet` | section 8 | **no** |
| `sensor_faults.parquet` | section 10 | **no** |
| `conditions.parquet` | ambient, load, dust, speed | only for analysis, not as model input |
| `config_used.yaml`, `run_meta.json` | the exact config, seed, row counts, rates | yes |

Dashboard status mapping, for reference: `normal` means stage normal; `abnormal` means stage early or advanced when no
failure is predicted yet; `predicted_failure` means `fails_within_24h`; `no_data_sensor_suspect` is produced later by
the data-quality check, not by the simulator.

The pipeline must never read ground-truth files. Enforce this by folder layout in the backend code, and note it in the
README.

## 12. Generation order (important)

```
1. rng streams    = SeedSequence(seed).spawn(6): layout, conditions, faults, hard_negatives, signals, messiness
2. layout         = build_layout()
3. conditions     = build_conditions()                          (run state, speed, load, ambient, dust)
4. hard_negatives = schedule_hard_negatives(); apply to conditions (load surge, heat wave, dust spell)
5. faults         = schedule_faults()                            (events, placed with their repair windows)
6. conditions     = apply_repair_stops(conditions, faults)       (belt stops during repair; speed 0)
7. belt coords    = cumulative distance D(t)
8. healthy        = generate_healthy(conditions)                 (zones, belt-wide)
9. effects        = apply_fault_effects(zones, belt_wide, faults)
10. labels        = build_labels(faults, hard_negatives)         (from clean truth, before messiness)
11. image events  = build_image_events(conditions, visual_defects, D)
12. messiness     = inject_messiness(zones, belt_wide)           (missing, outages, stuck) -> sensor_faults
13. write files, config_used.yaml, run_meta.json
```

Why this order: repairs stop the belt, so they must exist before the physics; labels come from clean truth so they are
never affected by messiness.

## 13. Build order and acceptance tests

Commit after each stage.

**Stage 0: repo hygiene.** `git init`, `.gitignore`, first commit. Finished when `git log` shows a commit.

**Stage 1: healthy baseline.** Config (with `signals.noise_scale`), layout, conditions, healthy signals, generator, CLI,
tests and a sanity plot, using six random streams from the start (section 12). Finished when:
- a 14-day run at the defaults gives 2,419,200 zone rows, generates in a few seconds, and the plot matches the targets
  in 5.5;
- tests pass for: 120 zones, 12 sections of zones 1 to 10, 60 carry zones, revolution 1500 s; row count equals
  timestamps times zones; running share of minutes between 0.74 and 0.76; stopped-belt vibration below 0.15; bearings
  never more than 2 C below ambient; carry side hotter than return; neighbouring zones' temperatures correlate above
  0.95; healthy kurtosis mean within 0.05 of 3; motor current between 100 and 520 A and mean slip below 1%; same seed
  gives identical tables and a different seed does not; a bad layout (section length 130 m) raises `ValueError`; a
  misspelled config key raises `TypeError`.

**Stage 2: belt coordinates and faults.** New files `simulator/faults.py` (scheduling and effects),
`simulator/belt_coords.py`, config sections `faults`. Finished when:
- a seeded run with one idler fault shows the rising temperature, acoustic and vibration in that zone only, in the
  plot, with early effects below one noise sigma at `s = 0.1`;
- a drive fault raises motor and gearbox temperature and nothing else belt-wide at first;
- repair stops appear as `stopped` periods in the run state, and signals are healthy again afterwards;
- tests: events never overlap on one asset, never start in the first 24 h, always finish 12 h before the end; severity
  is 0 outside events, 1 at failure; the idler fault never changes drive temperatures.

**Stage 3: hard negatives and labels.** `simulator/hard_negatives.py`, `simulator/labels.py`. Finished when each
hard-negative type produces a visible, plausible change that is labelled as not a fault, and `labels.parquet` has the
correct `stage`, `hours_to_failure` and `fails_within_24h` for every fault (check one by hand).

**Stage 4: messiness.** `simulator/messiness.py`. Finished when measured missing rate is within 0.5 to 1.5%, outage and
stuck counts and durations match the config, flat lines are exactly constant in the data, and `sensor_faults.parquet`
lists each one.

**Stage 5: image events and replay.** `simulator/images.py`, `simulator/stream.py` (yields readings in timestamp order,
optionally with a speed factor, for live mode). Finished when a visual defect's coordinate shows up in
`defect_id_in_view` once per lap per station, and the stream replays a scenario in order.

**Stage 6: scenario set.** `configs/` and a script that writes the scenario set (section 14). Finished when all
scenarios generate and `run_meta.json` shows the fault counts per scenario.

## 14. Scenario set and volume

Row count is `days x 1440 x 120`. At 6 km that is 5.2 million rows per 30 days (about 120 MB as parquet) or 15.6 million
per 90 days (about 350 MB).

**Recommended set (assumption; changes the earlier 90-day plan, see why):** 12 scenarios of **30 days**, different
seeds: 8 train, 2 validation, 2 test, plus 1 stress scenario (noise_scale 2, missing 3%, different fault mix). Total
about 1.6 GB, kept out of git and regenerated by one script.

Why not 90-day scenarios: the 6 km belt doubled the volume, and 12 of them would be about 4 GB. Fault events are what
get scored, not days, and 12 shorter scenarios give more independent events per split than a few long ones. If you
prefer the earlier 90-day plan, only the `days` value and the fault counts change.

Fault counts at default rates per 30 days: 5 idler, 1 drive. Across 8 training scenarios that is about 40 idler and
8 drive events; across the 2 test scenarios about 10 and 2. The drive count is small, which the README must state.

Dev loop: 14 days, one seed, `python -m simulator.run --days 14 --seed 1`.

## 15. Config additions (put in `Config`, same typo-rejecting style)

```yaml
faults:
  idler_bearing: {per_30d: 5, min_days: 3, max_days: 14, repair_hours: 2,
                  temp_c: 25, temp_exp: 2.0, acoustic_db: 10, acoustic_exp: 1.5,
                  vib_gain: 0.5, vib_exp: 2.0, kurtosis_add: 1.5, current_a: 1.5, current_exp: 3.0}
  drive_overheat: {per_30d: 1, min_hours: 6, max_hours: 48, repair_hours: 2,
                   motor_temp_c: 35, gearbox_temp_c: 20, exp: 1.3, current_a: 6}
  progression_exponent: 2.5
  min_start_hours: 24
  min_end_margin_hours: 12
hard_negatives:
  heavy_load: {per_30d: 4, min_hours: 6, max_hours: 12, load: [0.95, 1.0]}
  heat_wave:  {per_30d: 1, min_days: 1, max_days: 3, offset_c: [6, 10]}
  dusty:      {per_30d: 2, min_hours: 4, max_hours: 24, dust: [0.8, 0.9]}
  startup_vib_gain: 1.3
  startup_minutes: 10
messiness:
  missing_rate: 0.01
  outages_per_30d: 6
  outage_hours: [1, 6]
  stuck_per_30d: 4
  stuck_hours: [2, 48]
images:
  station_positions_m: [50, 1500, 2950]
  capture_interval_s: 2
  fov_m: 10
  defects_per_30d: 3
```

## 16. Rules for the build

- **No per-minute Python loops** over time; vectorise over (time, zone). Per-event loops (a few dozen) are fine.
- **Determinism:** every random draw uses the stream for its concern; same seed and config must give identical tables
  (test it).
- **Config:** all numbers in YAML; unknown keys raise an error.
- **Ground truth is separate:** never put a fault flag or label in `zones.parquet` or `belt_wide.parquet`.
- **Plots:** after each stage, write a sanity plot to `docs/` and look at it before moving on.
- **Tests:** add tests as you go; the full suite must stay under about 30 seconds.
- **Do not** add drift, spikes, timing defects, slip, mistracking or jam in version 1.
- **Do** keep a short `docs/CHANGELOG.md`: what changed and why, one line per stage.

## 17. Open assumptions (change if wrong)

- Fault counts and fault effect amplitudes are our assumptions, informed by a few papers, not measured values.
- The scenario set uses 30-day scenarios instead of the earlier 90-day plan.
- Cameras image only the carrying face; the return face is a README note on what we skipped.
- Splice sections are fixed ground-path stretches; splice joints ride the belt as landmarks for vision.
- Belt damage is vision-only. Visual defects have no sensor signature, so sensors and images are not linked.
- Vibration stays as summary features (RMS, kurtosis), not raw waveforms.
