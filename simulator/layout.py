"""Per-zone layout. Static public facts and hidden per-zone biases are returned separately.

Leakage rule: the random biases (temp_bias_c, vib_scale, offset_bias_mm) are the simulator's hidden
parameters. They go in the truth table, never in the public layout the pipeline may read.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .config import LayoutCfg


def build_layout(cfg: LayoutCfg, rng: np.random.Generator):
    """Return (public, truth) DataFrames with one row per zone, ordered by global_zone."""
    n_zones = int(round(cfg.belt_length_m / cfg.zone_spacing_m))
    n_carry = int(round(cfg.unloading_position_m / cfg.zone_spacing_m))
    zones_per_section = int(round(cfg.section_length_m / cfg.zone_spacing_m))

    idx = np.arange(n_zones)
    global_zone = idx + 1
    start_m = idx * cfg.zone_spacing_m
    position_m = start_m + cfg.zone_spacing_m / 2
    carry = position_m < cfg.unloading_position_m

    # Terrain: smooth random incline on the carry run; the return run is reversed and negated.
    raw = gaussian_filter1d(rng.standard_normal(n_carry), cfg.terrain_sigma_zones, mode="nearest")
    span = raw.max() - raw.min()
    unit = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
    lo, hi = cfg.incline_deg
    incline_carry = lo + (hi - lo) * unit
    incline = np.concatenate([incline_carry, -incline_carry[::-1]])

    # Curves: distinct every-6th starting zones between zone 3 and zone 57 (1-based), mirrored.
    n_curves = max(2, int(round(n_carry / 15)))
    candidates = np.arange(3, n_carry - cfg.curve_zones, 6)
    starts = np.sort(rng.choice(candidates, size=min(n_curves, len(candidates)), replace=False))
    radius_carry = np.full(n_carry, np.nan)
    for s in starts:
        radius_carry[s - 1:s - 1 + cfg.curve_zones] = rng.uniform(*cfg.curve_radius_m)
    radius = np.concatenate([radius_carry, radius_carry[::-1]])
    is_curve = ~np.isnan(radius)

    is_loading = start_m < cfg.loading_zone_length_m
    is_unloading = (start_m >= cfg.unloading_position_m - cfg.loading_zone_length_m) & carry

    temp_bias_c = rng.normal(0.0, cfg.temp_bias_sd_c, n_zones)
    vib_scale = rng.lognormal(0.0, cfg.vib_scale_sigma, n_zones) * np.where(is_loading, cfg.loading_vib_factor, 1.0)
    offset_bias_mm = rng.normal(0.0, cfg.offset_bias_sd_mm, n_zones)

    public = pd.DataFrame({
        "global_zone": global_zone.astype(np.int16),
        "splice_section": (idx // zones_per_section + 1).astype(np.int16),
        "zone_in_section": (idx % zones_per_section + 1).astype(np.int16),
        "position_m": position_m,
        "run": np.where(carry, "carry", "return"),
        "is_loading_zone": is_loading,
        "is_unloading_zone": is_unloading,
        "is_curve": is_curve,
        "curve_radius_m": radius,
        "incline_deg": incline,
    })
    truth = pd.DataFrame({
        "global_zone": global_zone.astype(np.int16),
        "temp_bias_c": temp_bias_c,
        "vib_scale": vib_scale,
        "offset_bias_mm": offset_bias_mm,
    })
    return public, truth
