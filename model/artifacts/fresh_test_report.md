# Fresh test report

## Criteria set before scoring

Written and committed before any `test_fresh` scenario was generated and before anything was scored. The models are
frozen at git tag `frozen-before-fresh-test`; nothing in `model/artifacts/` changes during this test
(`hashes_frozen.txt` is checked before and after).

**Why a fresh test:** `test_01` and `test_02` were looked at during the V7 and V8 diagnosis, and the V8 features were
designed to fix a test_01 miss, so they are no longer clean. New seeds, scored once, give a number we can defend.

**Scenarios:** `test_03` to `test_06`, seeds 501 to 504, 30 days, split `test_fresh`, default configuration plus one
override, `faults.drive_overheat.per_30d: 3` (about 12 drive events in total; at the default rate there would be about
4, too few to judge). All other draws (noise, drift, shared noise, fault strength, shape and signal mix) are random
and at default. About 20 bearing events are expected.

**Criteria** (our own choice, set in advance; they are not gates):

| Asset | Criterion |
|---|---|
| Bearings | alert recall at least 0.75 and alert precision at least 0.90 |
| Drive | alert recall at least 0.50 |

**Rules for this test:**
- Score once, with the frozen models and thresholds, with `python -m scripts.score --frozen --fresh`. The scoring
  logic is unchanged.
- Report the result whatever it is. No threshold, feature, model or scenario is changed after seeing it.
- If the script crashes, fix the crash, re-run the same command on the same frozen artifacts, and say so in the report.
- No scenario named `test_fresh*` may appear in any training or tuning code path.
