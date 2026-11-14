"""Healthy zone signals (T x Z arrays) and belt-wide drive signals (length T), fully vectorised."""
import numpy as np

from .conditions import Conditions, ar1, lag
from .config import SignalsCfg

ZONE_SIGNALS = ["vibration_rms_mm_s", "vibration_kurtosis", "bearing_temp_c", "acoustic_db", "belt_offset_mm"]
BELT_SIGNALS = ["throughput_tph", "motor_current_a", "motor_temp_c", "gearbox_temp_c",
                "drive_speed_ms", "belt_speed_ms", "slip_pct", "tension_kn"]


def generate_healthy_zones(cfg: SignalsCfg, cond: Conditions, layout_public, layout_truth, rng):
    """Return a dict signal -> (T, Z) float array."""
    ns = cfg.noise_scale
    T, Z = len(cond.speed), len(layout_public)
    speed = cond.speed[:, None]
    load = cond.load[:, None]
    ambient = cond.ambient_c[:, None]
    dust = cond.dust[:, None]
    lf = np.where(layout_public["run"].to_numpy() == "carry", 1.0, cfg.return_load_factor)[None, :]
    temp_bias = layout_truth["temp_bias_c"].to_numpy()[None, :]
    vib_scale = layout_truth["vib_scale"].to_numpy()[None, :]
    offset_bias = layout_truth["offset_bias_mm"].to_numpy()[None, :]
    is_curve = layout_public["is_curve"].to_numpy()[None, :].astype(float)

    # Draw order is fixed so a seed reproduces exactly.
    target = ambient + speed * (cfg.temp_speed_c + temp_bias) + cfg.temp_load_c * load * lf
    bearing = lag(target, cfg.temp_tau_min) + rng.normal(0.0, cfg.temp_noise_c * ns, (T, Z))

    vib = (vib_scale * (speed * (cfg.vib_speed_lin + cfg.vib_speed_quad * speed) + cfg.vib_load * load * lf)
           * (1.0 + cfg.vib_dust_gain * dust) * np.exp(rng.normal(0.0, cfg.vib_log_noise * ns, (T, Z)))
           + cfg.vib_floor * ns * np.abs(rng.standard_normal((T, Z))))

    kurt = cfg.kurt_base + rng.standard_normal((T, Z)) * cfg.kurt_noise * ns * (1.0 + cfg.kurt_dust_gain * dust)

    acoustic = (cfg.acoustic_base_db + cfg.acoustic_speed_db * speed + cfg.acoustic_load_db * load * lf
                + rng.standard_normal((T, Z)) * cfg.acoustic_noise_db * ns * (1.0 + dust))

    offset = (offset_bias + ar1(rng, (T, Z), cfg.offset_ar_phi, cfg.offset_ar_sd_mm * ns)
              * (1.0 + cfg.offset_curve_gain * is_curve)
              + rng.normal(0.0, cfg.offset_noise_mm * ns, (T, Z)))

    return {"vibration_rms_mm_s": vib, "vibration_kurtosis": kurt, "bearing_temp_c": bearing,
            "acoustic_db": acoustic, "belt_offset_mm": offset}


def generate_healthy_belt(cfg: SignalsCfg, cond: Conditions, layout_public, belt_speed_nominal_ms, rng):
    """Return a dict signal -> (T,) float array."""
    ns = cfg.noise_scale
    T = len(cond.speed)
    speed, load, ambient = cond.speed, cond.load, cond.ambient_c
    stopped = cond.base <= 0

    carry = layout_public["run"].to_numpy() == "carry"
    lift = float(np.mean(np.sin(np.radians(layout_public["incline_deg"].to_numpy()[carry]))))

    current = (speed * cfg.motor_speed_a + cfg.motor_load_a * load * (1.0 + cfg.motor_lift_gain * lift)
               + rng.normal(0.0, cfg.motor_noise_a * ns, T))
    current = np.where(stopped, 0.0, current)
    ratio2 = (current / cfg.motor_rated_a) ** 2
    motor_temp = (lag(ambient + cfg.motor_temp_rise_c * ratio2, cfg.motor_temp_tau_min)
                  + rng.normal(0.0, cfg.drive_temp_noise_c * ns, T))
    gearbox_temp = (lag(ambient + cfg.gearbox_temp_rise_c * ratio2, cfg.gearbox_temp_tau_min)
                    + rng.normal(0.0, cfg.drive_temp_noise_c * ns, T))

    drive_speed = belt_speed_nominal_ms * speed
    slip = cfg.slip_base + cfg.slip_load * load + rng.normal(0.0, cfg.slip_noise * ns, T)
    belt_speed = drive_speed * (1.0 - slip)
    # slip is undefined (not missing) when the belt is (nearly) stopped
    slip_pct = np.where(speed > 0.05, 100.0 * slip, np.nan)
    tension = (cfg.tension_base_kn + cfg.tension_speed_kn * speed + cfg.tension_load_kn * load
               + rng.normal(0.0, cfg.tension_noise_kn * ns, T))

    return {"throughput_tph": cond.throughput_tph, "motor_current_a": current, "motor_temp_c": motor_temp,
            "gearbox_temp_c": gearbox_temp, "drive_speed_ms": drive_speed, "belt_speed_ms": belt_speed,
            "slip_pct": slip_pct, "tension_kn": tension}
