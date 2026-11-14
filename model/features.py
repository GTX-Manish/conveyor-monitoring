"""Features for the failure models. One module, two stages, used by BOTH training and live scoring.

Stage 1, `window_aggregates`: raw 1-minute readings -> one row per (asset, 10-minute window).
Stage 2, `rolling_features`: the window table -> model features (rolling 1 h, 6 h, 24 h statistics, deviation
from other zones, drive residuals).

Live scoring keeps the last 24 h of the window table in memory and calls stage 2 on it; training calls the same
two functions on a whole scenario. Nothing here knows about labels.

Design notes (why the features look like this):
- Every zone shares the same weather, load and speed, so a zone is compared with the median of the other zones
  on its run (carry or return) in the same window. That removes ambient temperature, load and dust without
  needing to measure them, and it is what keeps a heat wave from looking like 60 hot bearings.
- Fault signals only exist while the belt turns, so rolling statistics use running windows only.
- A sensor that repeats one value exactly is stuck; its readings are dropped (NaN) so a frozen value cannot fake
  a deviation. Missing readings are never filled in.
"""
import warnings

import numpy as np

from common.ids import BELT_SIGNALS, RUNNING, ZONE_SIGNALS

WINDOW_MIN = 10
HISTORY_WINDOWS = 144                      # 24 h of 10-minute windows
ROLLS = {"1h": 6, "6h": 36, "24h": 144}
RUNNING_SHARE = 0.9                        # a window counts as running when the belt ran this share of it
FLAT_MIN_VALUES = 5                        # identical values needed in a window to call a signal stuck
LOG_SIGNALS = {"vibration_rms_mm_s"}       # compared as a ratio, not a difference
DRIVE_TEMPS = ["motor_temp_c", "gearbox_temp_c"]
DRIVE_LEVELS = ["motor_current_a", "tension_kn", "slip_pct", "throughput_tph", "belt_speed_ms"]


# ---------------------------------------------------------------- stage 1
def window_aggregates(zone_vals: dict, belt_vals: dict, run_state, layout: dict) -> dict:
    """Aggregate complete 10-minute windows.

    zone_vals: signal -> (W, 10, Z) floats with NaN for missing; belt_vals: signal -> (W, 10);
    run_state: (W, 10) integer codes; layout: 'is_carry' as (Z,) booleans.
    Returns the window table: a dict of (W, Z) and (W,) arrays. Everything that depends on one window alone
    is computed here (including each zone's deviation from its peers), so live scoring computes it once per
    window and stage 2 only has to roll over the table.
    """
    out = {}
    is_carry = np.asarray(layout["is_carry"], dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN windows are expected (outages, stops)
        for sig in ZONE_SIGNALS:
            x = np.asarray(zone_vals[sig], dtype=np.float64)
            n_valid = (~np.isnan(x)).sum(axis=1)
            mn, mx = np.nanmin(x, axis=1), np.nanmax(x, axis=1)
            flat = (n_valid >= FLAT_MIN_VALUES) & (mx == mn)
            mean, std = np.nanmean(x, axis=1), np.nanstd(x, axis=1)
            for name, arr in (("mean", mean), ("min", mn), ("max", mx), ("std", std)):
                out[f"z_{sig}_{name}"] = np.where(flat, np.nan, arr).astype(np.float32)
            out[f"z_{sig}_nmiss"] = (x.shape[1] - n_valid).astype(np.float32)
            out[f"z_{sig}_flat"] = flat.astype(np.float32)
        for sig in BELT_SIGNALS:
            x = np.asarray(belt_vals[sig], dtype=np.float64)
            out[f"b_{sig}_mean"] = np.nanmean(x, axis=1).astype(np.float32)
            out[f"b_{sig}_max"] = np.nanmax(x, axis=1).astype(np.float32)
            out[f"b_{sig}_nmiss"] = np.isnan(x).sum(axis=1).astype(np.float32)
    out["run_share"] = (np.asarray(run_state) == RUNNING).mean(axis=1).astype(np.float32)

    # Deviation of each zone from the median of the zones on its run, in running windows only.
    running = (out["run_share"].astype(np.float64) >= RUNNING_SHARE)[:, None]
    for sig in ZONE_SIGNALS:
        mean = out[f"z_{sig}_mean"].astype(np.float64)
        peak = out[f"z_{sig}_max"].astype(np.float64)
        std = out[f"z_{sig}_std"].astype(np.float64)
        if sig in LOG_SIGNALS:
            mean, peak = np.log(np.maximum(mean, 1e-3)), np.log(np.maximum(peak, 1e-3))
        ref = _run_median(mean, is_carry)
        out[f"z_{sig}_dev"] = np.where(running, mean - ref, np.nan)
        out[f"z_{sig}_peak_dev"] = np.where(running, peak - _run_median(peak, is_carry), np.nan)
        out[f"z_{sig}_spread"] = np.where(running, std - _run_median(std, is_carry), np.nan)
        if sig == "bearing_temp_c":
            out["z_temp_reference"] = ref
    out["ambient_proxy"] = nan_median(out["z_bearing_temp_c_mean"].astype(np.float64))
    return out


def append_windows(table: dict, new: dict, keep: int = HISTORY_WINDOWS) -> dict:
    """Window table with `new` rows appended, trimmed to the last `keep` windows (live scoring)."""
    if not table:
        return {k: v[-keep:] for k, v in new.items()}
    return {k: np.concatenate([table[k], new[k]], axis=0)[-keep:] for k in new}


# ---------------------------------------------------------------- rolling helpers (trailing windows, NaN-aware)
def roll_mean(x, k: int):
    """Mean of the valid values in the last k rows (this row included); NaN when there are none."""
    valid = ~np.isnan(x)
    s = np.cumsum(np.where(valid, x, 0.0), axis=0, dtype=np.float64)
    n = np.cumsum(valid, axis=0, dtype=np.int64)
    s[k:], n[k:] = s[k:] - s[:-k], n[k:] - n[:-k]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def roll_max(x, k: int):
    """Maximum of the valid values in the last k rows; NaN when there are none."""
    filled = np.where(np.isnan(x), -np.inf, x)
    pad = np.full((k - 1,) + filled.shape[1:], -np.inf)
    view = np.lib.stride_tricks.sliding_window_view(np.concatenate([pad, filled], axis=0), k, axis=0)
    out = view.max(axis=-1)
    return np.where(np.isinf(out), np.nan, out)


def nan_median(x):
    """Median of the valid values in each row of a 2-D array; NaN for a row with none.

    Same result as np.nanmedian(x, axis=1), about ten times faster, which matters for live scoring.
    """
    s = np.sort(x, axis=1)                       # NaNs sort to the end
    n = (~np.isnan(x)).sum(axis=1)
    rows = np.arange(len(x))
    lo, hi = np.maximum((n - 1) // 2, 0), n // 2
    hi = np.minimum(hi, x.shape[1] - 1)
    return np.where(n > 0, 0.5 * (s[rows, lo] + s[rows, hi]), np.nan)


def _run_median(x, is_carry):
    """Per-window median over the zones on the same run. x: (W, Z) -> (W, Z) of each zone's reference."""
    ref = np.empty_like(x, dtype=np.float64)
    for mask in (is_carry, ~is_carry):
        ref[:, mask] = nan_median(x[:, mask])[:, None]
    return ref


def _neighbour_mean(x):
    """Mean of the two neighbouring zones on the loop (zone 120 is next to zone 1)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack([np.roll(x, 1, axis=1), np.roll(x, -1, axis=1)]), axis=0)


# ---------------------------------------------------------------- stage 2
def zone_feature_names() -> list:
    names = []
    for sig in ZONE_SIGNALS:
        names += [f"{sig}__{k}" for k in ("dev10", "dev1h", "dev6h", "dev24h", "rise1h", "rise6h", "nb6h",
                                          "peak1h", "spread1h")]
    return names + ["missing_share", "missing_share_6h", "any_flat", "run_share", "run_share_1h", "run_share_24h",
                    "is_carry", "is_loading", "is_unloading", "throughput", "belt_speed", "temp_reference"]


def drive_feature_names() -> list:
    names = []
    for sig in DRIVE_TEMPS:
        names += [f"{sig}__{k}" for k in ("dev10", "dev1h", "dev6h", "dev24h", "rise1h", "rise6h")]
    for sig in DRIVE_LEVELS:
        names += [f"{sig}__{k}" for k in ("mean10", "mean1h", "mean24h")]
    return names + ["motor_minus_gearbox_1h", "current_per_load_1h", "current_per_load_24h",
                    "run_share", "run_share_1h", "run_share_24h", "temp_reference"]


def rolling_features(table: dict, layout: dict) -> dict:
    """Model features for every window in the table.

    layout: 'is_carry', 'is_loading', 'is_unloading' as (Z,) booleans (public layout facts).
    Returns {'zone': (W, Z, F) float32, 'drive': (W, Fd) float32}; column order is `zone_feature_names()` and
    `drive_feature_names()`. A row is only as good as its history: callers use rows with 24 h behind them.
    """
    is_carry = np.asarray(layout["is_carry"], dtype=bool)
    run_share = table["run_share"].astype(np.float64)
    W, Z = len(run_share), len(is_carry)
    running = run_share >= RUNNING_SHARE
    cols = []

    for sig in ZONE_SIGNALS:
        dev = table[f"z_{sig}_dev"]                                   # deviation from the zone's peers
        d1, d6, d24 = (roll_mean(dev, k) for k in ROLLS.values())
        cols += [dev, d1, d6, d24, d1 - d24, d6 - d24, d6 - _neighbour_mean(d6),
                 roll_max(table[f"z_{sig}_peak_dev"], ROLLS["1h"]), roll_mean(table[f"z_{sig}_spread"], ROLLS["1h"])]

    n_missing = sum(table[f"z_{s}_nmiss"].astype(np.float64) for s in ZONE_SIGNALS) / (WINDOW_MIN * len(ZONE_SIGNALS))
    any_flat = np.maximum.reduce([table[f"z_{s}_flat"] for s in ZONE_SIGNALS]).astype(np.float64)
    wide = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64)[:, None], (W, Z))    # noqa: E731
    across = lambda v: np.broadcast_to(np.asarray(v, dtype=np.float64)[None, :], (W, Z))  # noqa: E731
    cols += [n_missing, roll_mean(n_missing, ROLLS["6h"]), any_flat,
             wide(run_share), wide(roll_mean(run_share, ROLLS["1h"])), wide(roll_mean(run_share, ROLLS["24h"])),
             across(is_carry), across(layout["is_loading"]), across(layout["is_unloading"]),
             wide(table["b_throughput_tph_mean"]), wide(table["b_belt_speed_ms_mean"]), table["z_temp_reference"]]
    zone = np.stack(cols, axis=-1).astype(np.float32)

    # Drive: its temperatures are compared with the median bearing temperature of all zones, which follows the
    # weather the same way, so a hot day is not mistaken for a hot motor.
    ambient_proxy = table["ambient_proxy"]
    dcols, temp_dev_1h = [], {}
    for sig in DRIVE_TEMPS:
        dev = np.where(running, table[f"b_{sig}_mean"].astype(np.float64) - ambient_proxy, np.nan)
        d1, d6, d24 = (roll_mean(dev, k) for k in ROLLS.values())
        temp_dev_1h[sig] = d1
        dcols += [dev, d1, d6, d24, d1 - d24, d6 - d24]
    level_1h = {}
    for sig in DRIVE_LEVELS:
        x = np.where(running, table[f"b_{sig}_mean"].astype(np.float64), np.nan)
        level_1h[sig] = roll_mean(x, ROLLS["1h"])
        dcols += [x, level_1h[sig], roll_mean(x, ROLLS["24h"])]
    with np.errstate(invalid="ignore", divide="ignore"):
        per_load = np.where(running, table["b_motor_current_a_mean"].astype(np.float64)
                            / np.maximum(table["b_throughput_tph_mean"].astype(np.float64), 1.0), np.nan)
    dcols += [temp_dev_1h["motor_temp_c"] - temp_dev_1h["gearbox_temp_c"],
              roll_mean(per_load, ROLLS["1h"]), roll_mean(per_load, ROLLS["24h"]),
              run_share, roll_mean(run_share, ROLLS["1h"]), roll_mean(run_share, ROLLS["24h"]), ambient_proxy]
    drive = np.stack(dcols, axis=-1).astype(np.float32)
    return {"zone": zone, "drive": drive}


# ---------------------------------------------------------------- shaping raw tables into windows
def to_windows(values, n_per_window: int = WINDOW_MIN):
    """(T, ...) -> (W, 10, ...), dropping an incomplete last window."""
    values = np.asarray(values)
    w = len(values) // n_per_window
    return values[:w * n_per_window].reshape((w, n_per_window) + values.shape[1:])
