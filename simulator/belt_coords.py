"""Belt coordinates: things that ride on the looping belt (SIMULATOR_SPEC.md section 6).

A belt coordinate c in [0, L) is fixed to the belt material; c = 0 sits at ground position 0 at the
scenario start. D[t] is the cumulative distance the belt has travelled by the END of step t, so it only
grows while the belt moves and laps are counted in distance, not wall time.
"""
import numpy as np


def belt_distance(belt_speed_ms, step_s: float):
    """Cumulative travel D[t] (m) at the end of each step, from the true (clean) belt speed."""
    return np.cumsum(np.asarray(belt_speed_ms, dtype=float) * step_s)


def ground_position(c, D, belt_length_m):
    """Ground position of belt point c at the end of each step."""
    return (c + D) % belt_length_m


def belt_coordinate(p, D, belt_length_m):
    """Belt coordinate under fixed ground position p at the end of each step."""
    return (p - D) % belt_length_m


def zone_crossings(c, D, t_idx, zone_spacing_m, n_zones):
    """Steps and zones where belt point c enters a zone (crosses its start boundary).

    For each step t in t_idx the point moves over the arc (p(t-1), p(t)]; every zone whose start boundary
    lies in that arc is "crossed" in step t. Each zone is therefore crossed exactly once per lap, and a
    stopped belt crosses nothing. Returns (rows, zones) as aligned integer arrays (zones are 0-based).
    """
    t_idx = np.asarray(t_idx)
    D = np.asarray(D, dtype=float)
    d_prev = np.where(t_idx > 0, D[np.maximum(t_idx - 1, 0)], 0.0)
    k_now = np.floor((c + D[t_idx]) / zone_spacing_m).astype(np.int64)
    k_prev = np.floor((c + d_prev) / zone_spacing_m).astype(np.int64)
    cnt = k_now - k_prev
    total = int(cnt.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    rows = np.repeat(t_idx, cnt)
    offs = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    zones = (np.repeat(k_prev + 1, cnt) + offs) % n_zones
    return rows.astype(np.int64), zones.astype(np.int64)
