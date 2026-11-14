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
  - A zone is "crossed" by belt damage when the defect passes the zone's start boundary. The spec's extra clause
    ("or it contains the defect") is dropped: it would hit a zone on several steps during slow ramps.
  - Temperature effects pass through the same thermal lag as the healthy signal, so they do not vanish the instant
    the belt stops. Every effect is scaled by speed, so a stopped belt shows no fault signature.
  - Event counts for scenarios that are not a multiple of 30 days use the fractional part as a probability.
  - Two events on the drive or on the belt keep at least 24 h apart (`faults.same_asset_gap_hours`).
