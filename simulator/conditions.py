"""Operating conditions shared by all sensors, plus AR(1) and thermal-lag helpers."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .config import ConditionsCfg


def ar1(rng, shape, phi, sd):
    """AR(1) along axis 0 with stationary marginal sd `sd`, started from its stationary distribution."""
    shape = (shape,) if np.isscalar(shape) else tuple(shape)
    if sd == 0:
        return np.zeros(shape)
    e = rng.normal(0.0, sd * np.sqrt(1.0 - phi ** 2), shape)
    x0 = rng.normal(0.0, sd, shape[1:])  # x_{-1}
    zi = (phi * x0)[None, ...]
    out, _ = lfilter([1.0], [1.0, -phi], e, axis=0, zi=zi)
    return out


def lag(target, tau_min, step_min=1.0):
    """First-order lag along axis 0, y_0 = target_0 (a thermal mass chasing its target)."""
    a = 1.0 - np.exp(-step_min / tau_min)
    target = np.asarray(target, dtype=float)
    zi = ((1.0 - a) * target[0])[None, ...]
    out, _ = lfilter([a], [1.0, -(1.0 - a)], target, axis=0, zi=zi)
    return out


@dataclass
class Conditions:
    timestamp: pd.DatetimeIndex
    run_state: np.ndarray      # 'stopped' | 'ramp' | 'running'
    base: np.ndarray           # duty-cycle ramp factor 0..1
    speed: np.ndarray          # speed fraction, 0 when stopped
    load: np.ndarray           # load_raw * speed
    ambient_c: np.ndarray
    dust: np.ndarray
    throughput_tph: np.ndarray

    def to_frame(self):
        return pd.DataFrame({"timestamp": self.timestamp, "speed": self.speed, "load": self.load,
                             "ambient_c": self.ambient_c, "dust": self.dust})


def build_conditions(cfg: ConditionsCfg, start: str, n_steps: int, rng: np.random.Generator) -> Conditions:
    ts = pd.date_range(start, periods=n_steps, freq="min")
    start_ts = pd.Timestamp(start)
    minute_of_day = (start_ts.hour * 60 + start_ts.minute + np.arange(n_steps)) % 1440
    hour = minute_of_day / 60.0
    day_index = start_ts.day_of_year + (start_ts.hour * 60 + start_ts.minute + np.arange(n_steps)) // 1440

    # Duty cycle
    window = cfg.run_hours * 60.0
    m = (minute_of_day - cfg.day_start_hour * 60.0) % 1440
    inside = m < window
    base = np.where(inside, np.clip(np.minimum(m, window - m) / cfg.ramp_minutes, 0.0, 1.0), 0.0)
    run_state = np.where(base <= 0, "stopped", np.where(base < 1, "ramp", "running"))
    speed = base * (1.0 + ar1(rng, n_steps, cfg.speed_ar_phi, cfg.speed_ar_sd))
    speed = np.where(base <= 0, 0.0, speed)

    # Load
    mean = np.where((hour >= 5) & (hour < 13), cfg.load_mean_05_13,
                    np.where((hour >= 13) & (hour < 21), cfg.load_mean_13_21, cfg.load_mean_21_05))
    load_raw = np.clip(mean + ar1(rng, n_steps, cfg.load_ar_phi, cfg.load_ar_sd), *cfg.load_clip)
    load = load_raw * speed

    # Ambient
    ambient = (cfg.ambient_mean_c
               + cfg.ambient_daily_amp_c * np.cos(2 * np.pi * (hour - cfg.ambient_peak_hour) / 24.0)
               + cfg.ambient_season_amp_c * np.sin(2 * np.pi * day_index / 365.0)
               + ar1(rng, n_steps, cfg.ambient_ar_phi, cfg.ambient_ar_sd))

    # Dust
    dust = np.clip(cfg.dust_base + cfg.dust_running_gain * base
                   + ar1(rng, n_steps, cfg.dust_ar_phi, cfg.dust_ar_sd), 0.0, 1.0)

    return Conditions(ts, run_state, base, speed, load, ambient, dust, cfg.max_throughput_tph * load)
