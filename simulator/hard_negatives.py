"""Hard negatives: normal events that look like trouble (SIMULATOR_SPEC.md section 8).

They are ground truth labelled as NOT faults. Load surges, heat waves and dusty spells edit the operating
conditions before the physics runs, so the healthy model produces their effects; nothing is patched onto
the signals. The start-up transient is applied in `healthy.py` from `Conditions.startup_mask`.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HardNegativesCfg

HEAVY_LOAD, HEAT_WAVE, DUSTY, STARTUP = "heavy_load", "heat_wave", "dusty", "startup_transient"


@dataclass
class HardNegative:
    event_id: str
    type: str
    start: int   # step indices, end exclusive
    end: int
    value: float  # load level, ambient offset (C) or dust level


def _count(rate_per_30d, days, rng):
    x = rate_per_30d * days / 30.0
    n = int(np.floor(x))
    return n + int(rng.random() < x - n)


def _place(n, lo_min, hi_min, n_steps, rng):
    """Up to n non-overlapping (start, end) windows that fit inside the scenario."""
    out = []
    for _ in range(n):
        for _try in range(50):
            dur = int(rng.uniform(lo_min, hi_min))
            if dur >= n_steps:
                continue
            start = int(rng.integers(0, n_steps - dur))
            if any(start < e and s < start + dur for s, e in out):
                continue
            out.append((start, start + dur))
            break
    return out


def schedule_hard_negatives(cfg: HardNegativesCfg, days: int, rng) -> list:
    n_steps = days * 1440
    raw = []
    for kind, c, lo, hi, levels in (
        (HEAVY_LOAD, cfg.heavy_load, cfg.heavy_load.min_hours * 60, cfg.heavy_load.max_hours * 60, cfg.heavy_load.load),
        (HEAT_WAVE, cfg.heat_wave, cfg.heat_wave.min_days * 1440, cfg.heat_wave.max_days * 1440, cfg.heat_wave.offset_c),
        (DUSTY, cfg.dusty, cfg.dusty.min_hours * 60, cfg.dusty.max_hours * 60, cfg.dusty.dust),
    ):
        for start, end in _place(_count(c.per_30d, days, rng), lo, hi, n_steps, rng):
            raw.append((kind, start, end, float(rng.uniform(*levels))))
    raw.sort(key=lambda r: (r[1], r[0]))
    return [HardNegative(f"H{i + 1:03d}", *r) for i, r in enumerate(raw)]


def apply_hard_negatives(cond, events, cfg: HardNegativesCfg):
    """Return conditions with load, ambient temperature and dust edited inside each event window."""
    load_raw = cond.load_raw.copy()
    ambient = cond.ambient_c.copy()
    dust_floor = np.zeros(len(load_raw))
    for ev in events:
        w = slice(ev.start, ev.end)
        if ev.type == HEAVY_LOAD:
            load_raw[w] = np.maximum(load_raw[w], ev.value)
        elif ev.type == HEAT_WAVE:
            n = ev.end - ev.start
            edge = max(int(min(cfg.heat_wave.edge_hours * 60, n / 2)), 1)
            k = np.arange(n)
            shape = np.clip(np.minimum(k + 1, n - k) / edge, 0.0, 1.0)  # fades in and out
            ambient[w] += ev.value * shape
        elif ev.type == DUSTY:
            dust_floor[w] = np.maximum(dust_floor[w], ev.value)
    return cond.edited(load_raw=load_raw, ambient_c=ambient, dust_floor=dust_floor)


def startup_events(cond, minutes: int) -> list:
    """One hard-negative event per belt start; computed after repair stops are in the run state."""
    n = len(cond.base)
    return [HardNegative(f"U{i + 1:03d}", STARTUP, int(s), int(min(s + minutes, n)), float("nan"))
            for i, s in enumerate(cond.start_indices())]


def hard_negative_mask(events, n_steps: int):
    mask = np.zeros(n_steps, dtype=bool)
    for ev in events:
        mask[ev.start:ev.end] = True
    return mask


def hard_negatives_frame(events, timestamps) -> pd.DataFrame:
    ts = pd.DatetimeIndex(timestamps)
    step = ts[1] - ts[0]
    return pd.DataFrame({
        "event_id": [e.event_id for e in events],
        "type": [e.type for e in events],
        "start_time": pd.DatetimeIndex([ts[e.start] for e in events]),
        "end_time": pd.DatetimeIndex([ts[e.end - 1] + step for e in events]),  # exclusive end
        "zones": ["all"] * len(events),
        "value": [e.value for e in events],
    })
