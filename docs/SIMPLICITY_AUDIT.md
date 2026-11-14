# Simplicity audit of the simulator (Stages 0 to 4)

Audited on 2026-11-14 at commit `685fcaf`. Nothing in the code was changed by this audit.

**Update, same day:** findings 4.1 and 4.2 have been applied. All tables from a default 30-day run are
identical before and after, and the suite passes (63 passed, 3 skipped). One intended difference: in a scenario
with no faults, the three time columns of the empty `faults` table are now typed as timestamps (they were
untyped floats).

## 1. Verdict

The simulator does not need a rewrite. Its overall shape (config, then conditions, then healthy signals, then
faults, labels and messiness, each in its own small file) is about as small as the spec allows. The code is
1,214 lines in `simulator/`, 762 in `tests/` and 259 in `tools/`.

There are eight places where it is more complicated than it needs to be. Fixing all of them would remove about
70 lines (roughly 6% of `simulator/`) and three ideas a reader has to hold in their head. None of them changes
the generated data.

Two findings matter more than the line count suggests:

- **A test fails when the machine is busy.** `test_row_count` asserts that generation takes under 15 seconds.
  It failed today at 22.5 seconds while another process was running. The planned GitHub Actions CI would hit
  this at random.
- **The config layout should be settled before Stage 6.** Stage 6 writes one config file per scenario. Any
  change to config key names is cheap today and tedious afterwards.

## 2. How the audit was done

I read every file in `simulator/`, `tests/` and `tools/` and asked three questions of each part:

1. Is this code used, and is it written once?
2. Does a shorter design produce exactly the same output?
3. Is the complication paying for something (speed, leakage safety, a spec requirement)?

Where a claim could be measured, I measured it on a default 30-day run with seed 1:

| Measurement | Result |
|---|---|
| Generation time, machine idle | 2.3 s |
| Generation time, two runs at once | 152 s |
| Peak memory of one 30-day run (generate and write) | 0.8 GB, on a machine with 5.9 GB |
| `zones.parquet` | 118.7 MB |
| `zones.parquet` without the four repeated layout columns | 118.1 MB |
| `labels.parquet` | 2.1 MB for 5,227,200 rows |
| Test suite, run alone | 63 passed, 3 skipped, 26 s |
| Test suite, run beside another generation | 1 failed (the timing assertion), 256 s |

The proposed changes below were reasoned through against the code. They were not prototyped.

## 3. Findings at a glance

| # | Finding | Lines saved | Risk | Recommendation |
|---|---|---|---|---|
| 1 | A wall-clock assertion makes the test suite flaky | 1 | none | Fix now |
| 2 | Leftovers from building stage by stage | about 12 | none | Fix now |
| 3 | Hard-negative config uses three classes and mixed units | about 20 | low, but changes YAML keys | Fix before Stage 6, or not at all |
| 4 | Conditions are derived three times through a rebuild mechanism | about 25 | low to medium | Optional |
| 5 | `_count` is written three times, `_place` twice | about 10 | low | Optional |
| 6 | Three config values look adjustable but are not | 0 to 5 | none | Document, do not fix |
| 7 | `generate()` mixes orchestration with table building | 0 | none | Optional |
| 8 | Test helpers are repeated across files | about 8 | none | Optional |

## 4. Findings in detail

### 4.1 A wall-clock assertion makes the test suite flaky (fix now)

`tests/test_stage1.py:29` asserts `generate_seconds < 15`. Generation takes 2.3 seconds on an idle machine, but
its speed depends heavily on what else is running. With one other generation running beside it, a 14-day run
took 22.5 seconds and the test failed.

This also explains the unexplained 88-second run noted in the build report. I did not prove that it had the
same cause, but two generations at once reproduced a slowdown of that size and larger.

Simpler: delete the assertion. Speed is already reported in `run_meta.json` and printed by the CLI. A test
should fail only when the code is wrong.

### 4.2 Leftovers from building stage by stage (fix now)

Each of these was correct when written and was not tidied when a later stage made it unnecessary.

| Where | Leftover | Simpler |
|---|---|---|
| `simulator/conditions.py:35` | `Conditions.run_state`, a string array built on every derive. Nothing reads it. `generate()` works the state out again from `base` | Remove the field |
| `simulator/generate.py:34-39` | Six `Scenario` fields default to `None` because they arrived in later stages. `generate()` always fills them | Make them required |
| `simulator/generate.py:1` | The docstring says "stage 1 parts only" | Update it |
| `simulator/config.py:275` | `__all__` lists 8 of about 20 names and nothing uses a star import | Remove it |
| `simulator/faults.py:152` | `faults_frame` fixes its date columns with `.astype(... if events else {})`. The other two frame builders use `pd.DatetimeIndex` | Use `pd.DatetimeIndex` in all three |
| `simulator/generate.py:73` | Every belt signal is copied before messiness. Only `throughput_tph` shares memory with the conditions | Copy that one array where it is created in `healthy.py` |

### 4.3 Hard-negative config uses three classes and mixed units (fix before Stage 6, or not at all)

`HeavyLoadCfg`, `HeatWaveCfg` and `DustyCfg` hold the same three things: a rate, a duration range and a level
range. They differ only in field names (`load`, `offset_c`, `dust`) and units (`min_hours` in two, `min_days`
in one). Because of that, `schedule_hard_negatives` needs a five-line table to turn them back into one shape
(`simulator/hard_negatives.py:51-55`). The fault configs mix days and hours in the same way.

`MessinessCfg` already uses the simpler form: `outage_hours: (1.0, 6.0)`.

Simpler: one class used three times.

```python
@dataclass
class WindowCfg:
    per_30d: float
    hours: tuple   # (shortest, longest)
    level: tuple   # (lowest, highest): load, ambient offset in C, or dust
```

The scheduling loop then becomes three lines with no conversion table. The same `hours: (lo, hi)` form fits
both fault configs.

Cost: the YAML keys change, so `configs/default.yaml` is re-dumped and about six test configs are edited. The
generated data does not change.

Timing: this is worth doing only before Stage 6 writes the scenario configs. After that, leave it.

### 4.4 Conditions are derived three times through a rebuild mechanism (optional)

Today the order is:

1. Build the conditions (random draws).
2. Schedule hard negatives, then rebuild the conditions with `Conditions.edited(...)`.
3. Schedule faults, apply repair stops, then rebuild again with `Conditions.with_base(...)`.

To make steps 2 and 3 possible without new random draws, `Conditions` carries five extra fields (`cfg`,
`speed_noise`, `load_raw`, `dust_noise`, `dust_floor`), two rebuild methods and a private `_derive` function.

None of the scheduling needs the random conditions. Hard negatives need only the number of steps. Repair stops
need only the duty-cycle factor `base`, which is fixed by the clock and has no randomness.

Simpler order:

1. Compute `base` from the clock.
2. Schedule faults and apply repair stops to `base`.
3. Schedule hard negatives as three plain arrays: a load floor, an ambient offset and a dust floor.
4. Build the conditions once, from the final `base` and those three arrays.

This removes `edited`, `with_base`, `_derive` and the five extra fields. The output should be identical,
because each concern draws from its own random stream and the draw order inside each stream does not change.
A checksum of the public tables before and after would confirm it.

Why it is only optional: it touches the generation order in `generate()`, which is the part most worth leaving
alone two days before the deadline. The current mechanism is correct and tested.

### 4.5 `_count` is written three times, `_place` twice (optional)

`_count` (turn a rate per 30 days into a whole number of events) is identical in `faults.py`,
`hard_negatives.py` and `messiness.py`. Only one copy carries the comment that explains why it always draws a
random number. `_place` exists in two versions that differ in their rules (faults need a start margin and an
end margin, hard negatives need non-overlap).

Simpler: move `_count` to one shared place. Leave the two `_place` functions apart. Merging them would need
four optional arguments and would be harder to read than the two short originals.

### 4.6 Three config values look adjustable but are not (document, do not fix)

| Value | Why it is not really adjustable |
|---|---|
| `step_seconds` | The config rejects anything except 60, and the code uses 1,440 minutes per day in nine places |
| `unloading_position_m` | Validation requires it to be exactly half of `belt_length_m` |
| `load_mean_05_13`, `load_mean_13_21`, `load_mean_21_05` | The shift hours 5, 13 and 21 are fixed in `conditions.py:95`. Changing `day_start_hour` does not move them |

Removing them would save a few lines and would break the YAML layout. A one-line comment on each is enough.

### 4.7 `generate()` mixes orchestration with table building (optional)

`generate()` is 70 lines. The first 28 are the generation order, which is the most important thing in the
file. The other 40 build the `zones` and `belt_wide` tables and the metadata. Moving the table building into
two small functions saves no lines but lets the generation order be read on one screen.

### 4.8 Test helpers are repeated across files (optional)

The helper that reshapes a zone column into a time-by-zone grid is defined three times
(`test_stage1.py:36`, `test_stage2.py:37`, `test_stage4.py:32`), and the plot scripts define it again. One
copy in `tests/conftest.py` would do.

## 5. Considered and kept

These looked like candidates for simplification. Each one is paying for something.

| Candidate | Why it stays |
|---|---|
| Drop the repeated layout columns from `zones.parquet` and join them from `layout.parquet` | Measured: they cost 0.6 MB of 118.7 MB. The spec lists them, and keeping them saves every reader a join |
| Drop `labels.parquet` and derive labels from `faults.parquet` when needed | Measured: 2.1 MB. One fixed table means the model and the evaluation cannot disagree on a label definition |
| Remove the forbidden-word check on public column names, since the exact allowlist already blocks any other column | The allowlist stops a stray column. The word check stops someone adding a bad name to the allowlist itself. It costs 12 lines |
| Replace the hand-written config loader with pydantic | The loader is 25 lines. Pydantic would add a dependency to save about 15 of them |
| Draw each failure time directly from running minutes, in place of drawing it freely and snapping it forward | Repair stops from other faults change which minutes are running, so a snap step is needed either way |
| Replace `scipy.signal.lfilter` with plain loops for the noise and thermal lag | Loops over 43,200 steps by 120 zones would be easier to read and far slower. This is why generation takes 2 seconds |
| Compute fault severity once and share it between effects and labels | It is computed twice for about six events. Sharing it would couple two modules to save microseconds |
| Merge the four plot scripts | They are development tools, not part of the product. They can be deleted once the README has its figures |
| `belt_coords.py`: two of its three functions are used only by a test | Stage 5 (image capture) needs both. The file is 22 lines |

## 6. Suggested order of work

1. **About 15 minutes, no risk:** findings 4.1 and 4.2. Run the tests.
2. **Before Stage 6 starts, about 30 minutes:** finding 4.3, if it is wanted at all. Compare a checksum of the
   public tables before and after. They must match.
3. **Only if time is left after the model, backend and dashboard:** findings 4.4, 4.5, 4.7 and 4.8.

How to check any of these: generate the default 30-day scenario before and after the change and compare the
public and truth tables with `DataFrame.equals`. Every change in this document is meant to leave them
identical.

## 7. What this audit did not cover

- Whether the physics is right. The build report lists those issues (for example, acoustic does not lead
  temperature in noise terms).
- Stage 5 and Stage 6, the model, the backend and the dashboard, which do not exist yet.
- The Python version. The plan pins 3.12 and this machine runs 3.14.
