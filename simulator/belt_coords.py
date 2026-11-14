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
