"""Data defects (SIMULATOR_SPEC.md section 10): missing readings, suite outages and stuck sensors.

Injected after the physics, the fault effects and the labels, so ground truth stays clean. These are
SENSOR faults, kept apart from equipment faults so each can be scored on its own. Order of injection:
stuck (freeze), then outage (silence), then random missing readings on top of both.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MessinessCfg

OUTAGE, STUCK = "outage", "stuck"


@dataclass
class SensorFault:
    sensor_fault_id: str
    type: str
    zone: int        # 1-based
    signal: str      # None for an outage (the whole suite is silent)
    start: int       # step indices, end exclusive
    end: int
    stuck_value: float = float("nan")


def _count(rate_per_30d, days, rng):
    x = rate_per_30d * days / 30.0
    n = int(np.floor(x))
    return n + int(rng.random() < x - n)


def schedule_sensor_faults(cfg: MessinessCfg, days: int, n_zones: int, signals, rng) -> list:
    n_steps = days * 1440
    raw = []
    for kind, rate, (lo_h, hi_h) in ((OUTAGE, cfg.outages_per_30d, cfg.outage_hours),
                                     (STUCK, cfg.stuck_per_30d, cfg.stuck_hours)):
        for _ in range(_count(rate, days, rng)):
            dur = int(rng.uniform(lo_h * 60, hi_h * 60))
            zone = int(rng.integers(1, n_zones + 1))
            signal = str(signals[int(rng.integers(len(signals)))]) if kind == STUCK else None
            if dur >= n_steps - 1:
                continue  # does not fit this scenario
            start = int(rng.integers(1, n_steps - dur))  # >= 1 so a stuck sensor has a last value to hold
            raw.append((kind, zone, signal, start, start + dur))
    raw.sort(key=lambda r: (r[3], r[0], r[1]))
    return [SensorFault(f"S{i + 1:03d}", *r) for i, r in enumerate(raw)]


def inject_messiness(zone_sig, belt_sig, events, cfg: MessinessCfg, rng) -> dict:
    """Apply defects to the signal arrays in place. Returns measured rates for run_meta.json."""
    for ev in events:
        if ev.type == STUCK:
            arr = zone_sig[ev.signal]
            ev.stuck_value = float(np.float32(arr[ev.start - 1, ev.zone - 1]))  # the last value it reported
            arr[ev.start:ev.end, ev.zone - 1] = ev.stuck_value
    for ev in events:
        if ev.type == OUTAGE:
            for arr in zone_sig.values():
                arr[ev.start:ev.end, ev.zone - 1] = np.nan

    dropped, total = 0, 0
    for group in (zone_sig, belt_sig):
        for arr in group.values():
            drop = rng.random(arr.shape) < cfg.missing_rate
            arr[drop] = np.nan
            dropped += int(drop.sum())
            total += drop.size
    zone_cells = sum(a.size for a in zone_sig.values())
    return {
        "missing_rate_config": cfg.missing_rate,
        "missing_rate_measured": dropped / total if total else 0.0,
        "zone_nan_share": float(sum(np.isnan(a).sum() for a in zone_sig.values()) / zone_cells),
        "sensor_fault_counts": {k: sum(e.type == k for e in events) for k in (OUTAGE, STUCK)},
    }


def sensor_faults_frame(events, timestamps) -> pd.DataFrame:
    ts = pd.DatetimeIndex(timestamps)
    step = ts[1] - ts[0]
    return pd.DataFrame({
        "sensor_fault_id": [e.sensor_fault_id for e in events],
        "type": [e.type for e in events],
        "zone": np.array([e.zone for e in events], dtype=np.int16),
        "signal": pd.array([e.signal for e in events], dtype="object"),
        "start_time": pd.DatetimeIndex([ts[e.start] for e in events]),
        "end_time": pd.DatetimeIndex([ts[e.end - 1] + step for e in events]),  # exclusive end
        "stuck_value": [e.stuck_value for e in events],
    })
