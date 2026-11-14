"""Seed -> tables. Follows the generation order in SIMULATOR_SPEC.md section 12 (stage 1 parts only)."""
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conditions import build_conditions
from .config import Config
from .healthy import BELT_SIGNALS, ZONE_SIGNALS, generate_healthy_belt, generate_healthy_zones
from .layout import build_layout

STREAMS = ["layout", "conditions", "faults", "hard_negatives", "signals", "messiness"]


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


def make_streams(seed: int) -> dict:
    """One independent generator per concern, so changing one part does not reshuffle another."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(ss) for name, ss in zip(STREAMS, children)}


def generate(cfg: Config) -> Scenario:
    t0 = time.perf_counter()
    rng = make_streams(cfg.seed)
    n_steps = cfg.days * 1440

    layout, layout_truth = build_layout(cfg.layout, rng["layout"])
    cond = build_conditions(cfg.conditions, cfg.start, n_steps, rng["conditions"])
    # (stage 3: hard negatives edit conditions here; stage 2: faults + repair stops, belt coordinates)
    zone_sig = generate_healthy_zones(cfg.signals, cond, layout, layout_truth, rng["signals"])
    belt_sig = generate_healthy_belt(cfg.signals, cond, layout, cfg.layout.belt_speed_ms, rng["signals"])

    T, Z = n_steps, len(layout)
    zones = pd.DataFrame({
        "timestamp": np.repeat(cond.timestamp.values, Z),
        "splice_section": np.tile(layout["splice_section"].to_numpy(), T),
        "zone_in_section": np.tile(layout["zone_in_section"].to_numpy(), T),
        "global_zone": np.tile(layout["global_zone"].to_numpy(), T),
        "position_m": np.tile(layout["position_m"].to_numpy(), T),
        "run": pd.Categorical(np.tile(layout["run"].to_numpy(), T), categories=["carry", "return"]),
        "run_state": pd.Categorical(np.repeat(cond.run_state, Z), categories=["stopped", "ramp", "running"]),
    })
    for name in ZONE_SIGNALS:
        zones[name] = zone_sig[name].astype(np.float32).ravel()  # (T, Z) C-order -> time-major rows

    belt_wide = pd.DataFrame({
        "timestamp": cond.timestamp.values,
        "run_state": pd.Categorical(cond.run_state, categories=["stopped", "ramp", "running"]),
    })
    for name in BELT_SIGNALS:
        belt_wide[name] = belt_sig[name].astype(np.float32)

    meta = {
        "scenario": cfg.scenario, "seed": cfg.seed, "days": cfg.days, "start": cfg.start,
        "n_timestamps": T, "n_zones": Z, "zone_rows": len(zones), "belt_wide_rows": len(belt_wide),
        "running_share": float(np.mean(cond.speed > 0)),
        "generate_seconds": round(time.perf_counter() - t0, 3),
    }
    return Scenario(zones, belt_wide, layout, layout_truth, cond.to_frame(), meta)
