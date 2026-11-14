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
