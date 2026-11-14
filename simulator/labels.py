"""Ground-truth labels per (timestamp, asset), built from the clean fault schedule before any messiness.

Assets are the zones ("1" to "120") plus "DRIVE".

Definitions (ours where the spec leaves room):
- stage: normal | early (0 < s < 0.3) | advanced (0.3 <= s, not yet failed) | failed (failure to repair end).
- hours_to_failure: hours until this asset's failure while a fault is developing (onset <= t < failure),
  0 while failed, NaN when the asset is healthy. It is never set before onset: before the fault exists
  there is nothing in the data to predict from, so a label there would only be noise.
- fails_within_24h: a fault is developing and its failure is at most 24 h away.
- is_hard_negative_period: a hard negative (heavy load, heat wave, dusty spell, start-up) is active.
"""
import numpy as np
import pandas as pd

from .config import Config
from .faults import severity

STAGES = ["normal", "early", "advanced", "failed"]


def build_labels(events, hard_negative_mask, timestamps, n_zones: int, cfg: Config) -> pd.DataFrame:
    T = len(timestamps)
    assets = [str(z) for z in range(1, n_zones + 1)] + ["DRIVE"]
    col = {a: i for i, a in enumerate(assets)}
    A = len(assets)

    stage = np.zeros((T, A), dtype=np.int8)
    sev = np.zeros((T, A), dtype=np.float32)
    htf = np.full((T, A), np.nan, dtype=np.float32)
    within = np.zeros((T, A), dtype=bool)
    fault_idx = np.full((T, A), -1, dtype=np.int16)

    for i, ev in enumerate(events):
        a = col[ev.asset]
        s = severity(ev, T, cfg.faults.progression_exponent)
        dev = slice(ev.onset, ev.failure)
        down = slice(ev.failure, ev.repair_end)
        sev[dev, a] = s[dev]
        sev[down, a] = 1.0
        stage[dev, a] = np.where(s[dev] <= 0, 0, np.where(s[dev] < cfg.labels.early_below, 1, 2))
        stage[down, a] = 3
        hours = (ev.failure - np.arange(ev.onset, ev.failure)) / 60.0
        htf[dev, a] = hours
        htf[down, a] = 0.0
        within[dev, a] = hours <= cfg.labels.horizon_hours
        fault_idx[ev.onset:ev.repair_end, a] = i

    return pd.DataFrame({
        "timestamp": np.repeat(pd.DatetimeIndex(timestamps).values, A),
        "asset": pd.Categorical.from_codes(np.tile(np.arange(A, dtype=np.int16), T), categories=assets),
        "stage": pd.Categorical.from_codes(stage.ravel(), categories=STAGES),
        "severity": sev.ravel(),
        "hours_to_failure": htf.ravel(),
        "fails_within_24h": within.ravel(),
        "fault_id": pd.Categorical.from_codes(fault_idx.ravel(), categories=[e.fault_id for e in events]),
        "is_hard_negative_period": np.repeat(np.asarray(hard_negative_mask, dtype=bool), A),
    })
