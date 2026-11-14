"""Typed, typo-rejecting config. Every number in the simulator lives here (or in YAML dumped from here).

Values marked (assumption) in SIMULATOR_SPEC.md are our own choices, not measured numbers.
"""
import typing
from dataclasses import dataclass, field, fields, is_dataclass

import yaml


@dataclass
class LayoutCfg:
    belt_length_m: float = 6000.0
    belt_speed_ms: float = 4.0
    zone_spacing_m: float = 50.0
    section_length_m: float = 500.0
    unloading_position_m: float = 3000.0
    loading_zone_length_m: float = 100.0
    incline_deg: tuple = (-10.0, 15.0)
    terrain_sigma_zones: float = 6.0
    curve_radius_m: tuple = (300.0, 800.0)
    curve_zones: int = 2
    temp_bias_sd_c: float = 0.8
    vib_scale_sigma: float = 0.05
    loading_vib_factor: float = 1.25
    offset_bias_sd_mm: float = 3.0

    def __post_init__(self):
        def whole(a, b):
            return abs(a / b - round(a / b)) < 1e-9

        if self.zone_spacing_m <= 0 or self.belt_length_m <= 0:
            raise ValueError("lengths must be positive")
        if not whole(self.belt_length_m, self.section_length_m):
            raise ValueError("belt_length_m must be a whole number of sections")
        if not whole(self.section_length_m, self.zone_spacing_m):
            raise ValueError("section_length_m must be a whole number of zones")
        if not whole(self.unloading_position_m, self.zone_spacing_m):
            raise ValueError("unloading_position_m must fall on a zone boundary")
        if not 0 < self.unloading_position_m < self.belt_length_m:
            raise ValueError("unloading_position_m must lie inside the belt")
        if abs(self.unloading_position_m * 2 - self.belt_length_m) > 1e-9:
            raise ValueError("carry and return runs must be equal length")


@dataclass
class ConditionsCfg:
    day_start_hour: float = 5.0  # duty window opens at 05:00
    run_hours: float = 18.0
    ramp_minutes: float = 5.0
    speed_ar_phi: float = 0.99
    speed_ar_sd: float = 0.004
    load_mean_05_13: float = 0.80
    load_mean_13_21: float = 0.85
    load_mean_21_05: float = 0.65
    load_ar_phi: float = 0.995
    load_ar_sd: float = 0.06
    load_clip: tuple = (0.30, 1.0)
    max_throughput_tph: float = 3000.0
    ambient_mean_c: float = 30.0
    ambient_daily_amp_c: float = 10.0
    ambient_peak_hour: float = 15.0
    ambient_season_amp_c: float = 3.0
    ambient_ar_phi: float = 0.999
    ambient_ar_sd: float = 0.5
    dust_base: float = 0.30
    dust_running_gain: float = 0.15
    dust_ar_phi: float = 0.999
    dust_ar_sd: float = 0.03


@dataclass
class SignalsCfg:
    noise_scale: float = 1.0  # multiplies every noise sd below (stress scenario uses 2.0)
    return_load_factor: float = 0.15
    # bearing temperature
    temp_speed_c: float = 5.0
    temp_load_c: float = 7.0
    temp_tau_min: float = 20.0
    temp_noise_c: float = 0.3
    # vibration RMS
    vib_speed_lin: float = 0.6
    vib_speed_quad: float = 0.9
    vib_load: float = 0.5
    vib_dust_gain: float = 0.3
    vib_log_noise: float = 0.04
    vib_floor: float = 0.02
    # kurtosis
    kurt_base: float = 3.0
    kurt_noise: float = 0.12
    kurt_dust_gain: float = 0.5
    # acoustic
    acoustic_base_db: float = 42.0
    acoustic_speed_db: float = 14.0
    acoustic_load_db: float = 4.0
    acoustic_noise_db: float = 1.2
    # lateral offset
    offset_ar_phi: float = 0.98
    offset_ar_sd_mm: float = 1.5
    offset_curve_gain: float = 1.0
    offset_noise_mm: float = 0.5
    # drive
    motor_speed_a: float = 140.0
    motor_load_a: float = 230.0
    motor_lift_gain: float = 5.0
    motor_noise_a: float = 4.0
    motor_rated_a: float = 400.0
    motor_temp_rise_c: float = 30.0
    motor_temp_tau_min: float = 25.0
    gearbox_temp_rise_c: float = 22.0
    gearbox_temp_tau_min: float = 40.0
    drive_temp_noise_c: float = 0.3
    slip_base: float = 0.0025
    slip_load: float = 0.0007
    slip_noise: float = 0.0003
    tension_base_kn: float = 110.0
    tension_speed_kn: float = 10.0
    tension_load_kn: float = 45.0
    tension_noise_kn: float = 1.5


@dataclass
class IdlerBearingCfg:
    per_30d: float = 5.0
    min_days: float = 3.0
    max_days: float = 14.0
    repair_hours: float = 2.0
    temp_c: float = 25.0
    temp_exp: float = 2.0
    acoustic_db: float = 10.0
    acoustic_exp: float = 1.5
    vib_gain: float = 0.5
    vib_exp: float = 2.0
    kurtosis_add: float = 1.5
    current_a: float = 1.5
    current_exp: float = 3.0


@dataclass
class BeltDamageCfg:
    per_30d: float = 1.0
    min_days: float = 2.0
    max_days: float = 6.0
    repair_hours: float = 4.0
    kurtosis_add: float = 6.0
    vib_gain: float = 0.4
    acoustic_db: float = 4.0
    return_side_factor: float = 0.5
    tension_kn: float = 8.0
    tension_exp: float = 3.0


@dataclass
class DriveOverheatCfg:
    per_30d: float = 1.0
    min_hours: float = 6.0
    max_hours: float = 48.0
    repair_hours: float = 2.0
    motor_temp_c: float = 35.0
    gearbox_temp_c: float = 20.0
    exp: float = 1.3
    current_a: float = 6.0
    current_exp: float = 2.0


@dataclass
class FaultsCfg:
    idler_bearing: IdlerBearingCfg = field(default_factory=IdlerBearingCfg)
    belt_damage: BeltDamageCfg = field(default_factory=BeltDamageCfg)
    drive_overheat: DriveOverheatCfg = field(default_factory=DriveOverheatCfg)
    progression_exponent: float = 2.5
    min_start_hours: float = 24.0
    min_end_margin_hours: float = 12.0
    same_asset_gap_hours: float = 24.0  # (assumption) minimum gap between two events on DRIVE or BELT


@dataclass
class Config:
    scenario: str = "scenario"
    seed: int = 1
    days: int = 14
    start: str = "2026-01-01 00:00:00"
    step_seconds: int = 60
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    conditions: ConditionsCfg = field(default_factory=ConditionsCfg)
    signals: SignalsCfg = field(default_factory=SignalsCfg)
    faults: FaultsCfg = field(default_factory=FaultsCfg)

    def __post_init__(self):
        if self.step_seconds != 60:
            raise ValueError("only 60 s steps are supported in version 1")
        if self.days < 1:
            raise ValueError("days must be >= 1")


def _build(cls, data):
    data = dict(data or {})
    hints = typing.get_type_hints(cls)
    unknown = set(data) - set(hints)
    if unknown:
        raise TypeError(f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in data.items():
        if is_dataclass(hints[name]):
            value = _build(hints[name], value)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def _plain(obj):
    if is_dataclass(obj):
        return {f.name: _plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (tuple, list)):
        return [_plain(x) for x in obj]
    return obj


def config_from_dict(data) -> Config:
    return _build(Config, data)


def load_config(path) -> Config:
    with open(path) as fh:
        return config_from_dict(yaml.safe_load(fh))


def dump_config(cfg: Config, path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(_plain(cfg), fh, sort_keys=False)


def config_to_dict(cfg: Config) -> dict:
    return _plain(cfg)


__all__ = ["Config", "LayoutCfg", "ConditionsCfg", "SignalsCfg", "config_from_dict",
           "load_config", "dump_config", "config_to_dict"]
