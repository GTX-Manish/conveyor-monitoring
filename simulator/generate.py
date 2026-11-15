"""Seed -> tables. Follows the generation order in SIMULATOR_SPEC.md section 12."""
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .belt_coords import belt_distance
from .conditions import build_conditions
from .config import Config
from .faults import DRIVE_OVERHEAT, IDLER, apply_fault_effects, apply_repair_stops, faults_frame, schedule_faults
from .hard_negatives import (
    DUSTY,
    HEAT_WAVE,
    HEAVY_LOAD,
    STARTUP,
    apply_hard_negatives,
    hard_negative_mask,
    hard_negatives_frame,
    schedule_hard_negatives,
    startup_events,
)
from .healthy import BELT_SIGNALS, ZONE_SIGNALS, generate_healthy_belt, generate_healthy_zones
from .labels import build_labels
from .layout import build_layout
from .messiness import inject_messiness, schedule_sensor_faults, sensor_faults_frame

RUN_STATES = ["stopped", "ramp", "running"]
# New streams are only ever appended: stream i depends on i alone, so existing streams keep their draws.
STREAMS = ["layout", "conditions", "faults", "hard_negatives", "signals", "messiness", "noise"]


@dataclass
class Scenario:
    # public: the pipeline may read these
    zones: pd.DataFrame
    belt_wide: pd.DataFrame
    layout: pd.DataFrame
    # truth / analysis only: the pipeline must never read these
    layout_truth: pd.DataFrame
    conditions: pd.DataFrame
    meta: dict
    faults: pd.DataFrame
    events: list               # FaultEvent objects behind `faults`
    hard_negatives: pd.DataFrame
    labels: pd.DataFrame
    sensor_faults: pd.DataFrame
    sensor_fault_events: list  # SensorFault objects behind `sensor_faults`


def make_streams(seed: int) -> dict:
    """One independent generator per concern, so changing one part does not reshuffle another."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(ss) for name, ss in zip(STREAMS, children)}


def generate(cfg: Config) -> Scenario:
    t0 = time.perf_counter()
    rng = make_streams(cfg.seed)
    n_steps = cfg.days * 1440
    # Per-scenario noise level: drawn first and on its own stream, so it changes nothing else.
    lo, hi = cfg.signals.noise_random
    noise_factor = float(np.exp(rng["noise"].uniform(np.log(lo), np.log(hi))))
    signals = replace(cfg.signals, noise_scale=cfg.signals.noise_scale * noise_factor)

    layout, layout_truth = build_layout(cfg.layout, rng["layout"])
    cond = build_conditions(cfg.conditions, cfg.start, n_steps, rng["conditions"])
    # Hard negatives edit the conditions before the physics, so the healthy model produces their effects.
    hard_negs = schedule_hard_negatives(cfg.hard_negatives, cfg.days, rng["hard_negatives"])
    cond = apply_hard_negatives(cond, hard_negs, cfg.hard_negatives)
    events = schedule_faults(cfg.faults, cfg.days, len(layout), rng["faults"])
    cond = cond.with_base(apply_repair_stops(events, cond.base, cfg.conditions.ramp_minutes))
    hn = cfg.hard_negatives
    zone_sig = generate_healthy_zones(signals, cond, layout, layout_truth, rng["signals"],
                                      startup_mask=cond.startup_mask(hn.startup_minutes),
                                      startup_vib_gain=hn.startup_vib_gain)
    belt_sig = generate_healthy_belt(signals, cond, layout, cfg.layout.belt_speed_ms, rng["signals"])
    # Belt travel comes from the clean belt speed, so the public belt_speed_ms integrates to the true D(t).
    D = belt_distance(belt_sig["belt_speed_ms"], cfg.step_seconds)
    apply_fault_effects(zone_sig, belt_sig, events, cond, cfg)
    # Labels come from the clean schedule, before any messiness.
    hard_negs = hard_negs + startup_events(cond, hn.startup_minutes)
    labels = build_labels(events, hard_negative_mask(hard_negs, n_steps), cond.timestamp, len(layout), cfg)

    # Messiness goes in last, on the public signals only: labels above are already fixed from clean truth.
    sensor_faults = schedule_sensor_faults(cfg.messiness, cfg.days, len(layout), ZONE_SIGNALS, rng["messiness"])
    mess = inject_messiness(zone_sig, belt_sig, sensor_faults, cfg.messiness, rng["messiness"])

    T, Z = n_steps, len(layout)
    # categoricals are built from integer codes: far faster than from millions of strings
    state_codes = np.where(cond.base <= 0, 0, np.where(cond.base < 1, 1, 2)).astype(np.int8)
    run_codes = (layout["run"].to_numpy() == "return").astype(np.int8)
    zones = pd.DataFrame({
        "timestamp": np.repeat(cond.timestamp.values, Z),
        "splice_section": np.tile(layout["splice_section"].to_numpy(), T),
        "zone_in_section": np.tile(layout["zone_in_section"].to_numpy(), T),
        "global_zone": np.tile(layout["global_zone"].to_numpy(), T),
        "position_m": np.tile(layout["position_m"].to_numpy(), T),
        "run": pd.Categorical.from_codes(np.tile(run_codes, T), categories=["carry", "return"]),
        "run_state": pd.Categorical.from_codes(np.repeat(state_codes, Z), categories=RUN_STATES),
    })
    for name in ZONE_SIGNALS:
        zones[name] = zone_sig[name].astype(np.float32).ravel()  # (T, Z) C-order -> time-major rows

    belt_wide = pd.DataFrame({
        "timestamp": cond.timestamp.values,
        "run_state": pd.Categorical.from_codes(state_codes, categories=RUN_STATES),
    })
    for name in BELT_SIGNALS:
        belt_wide[name] = belt_sig[name].astype(np.float32)

    meta = {
        "scenario": cfg.scenario, "seed": cfg.seed, "days": cfg.days, "start": cfg.start,
        "n_timestamps": T, "n_zones": Z, "zone_rows": len(zones), "belt_wide_rows": len(belt_wide),
        "running_share": float(np.mean(cond.speed > 0)),
        "noise_factor": noise_factor, "noise_scale_effective": signals.noise_scale,
        "fault_counts": {k: sum(e.type == k for e in events) for k in (IDLER, DRIVE_OVERHEAT)},
        "hard_negative_counts": {k: sum(e.type == k for e in hard_negs)
                                 for k in (HEAVY_LOAD, HEAT_WAVE, DUSTY, STARTUP)},
        **mess,
        "generate_seconds": round(time.perf_counter() - t0, 3),
    }
    conditions = cond.to_frame()
    conditions["belt_distance_m"] = D
    return Scenario(zones, belt_wide, layout, layout_truth, conditions, meta,
                    faults=faults_frame(events, cond.timestamp), events=events,
                    hard_negatives=hard_negatives_frame(hard_negs, cond.timestamp), labels=labels,
                    sensor_faults=sensor_faults_frame(sensor_faults, cond.timestamp),
                    sensor_fault_events=sensor_faults)
