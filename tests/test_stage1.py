import numpy as np
import pandas as pd
import pytest

from simulator.config import Config, LayoutCfg, config_from_dict
from simulator.generate import generate
from simulator.healthy import BELT_SIGNALS


def test_layout_counts(scn14):
    lay = scn14.layout
    assert len(lay) == 120
    assert lay["splice_section"].nunique() == 12
    for _, grp in lay.groupby("splice_section"):
        assert sorted(grp["zone_in_section"]) == list(range(1, 11))
    assert (lay["run"] == "carry").sum() == 60
    assert list(lay.loc[lay["is_loading_zone"], "global_zone"]) == [1, 2]
    assert list(lay.loc[lay["is_unloading_zone"], "global_zone"]) == [59, 60]
    assert lay["is_curve"].sum() == 16  # 4 curves x 2 zones, mirrored onto the return run


def test_revolution_time():
    c = LayoutCfg()
    assert c.belt_length_m / c.belt_speed_ms == 1500


def test_row_count(scn14, cfg14):
    assert len(scn14.zones) == cfg14.days * 1440 * 120 == 2_419_200
    assert len(scn14.belt_wide) == cfg14.days * 1440


def test_running_share(scn14):
    assert 0.74 <= scn14.meta["running_share"] <= 0.76


def _zone_matrix(scn, col):
    return scn.zones[col].to_numpy().reshape(-1, 120)


def test_stopped_vibration_low(scn14):
    vib = _zone_matrix(scn14, "vibration_rms_mm_s")
    stopped = scn14.belt_wide["run_state"].to_numpy() == "stopped"
    assert vib[stopped].max() < 0.15


def test_bearing_not_below_ambient(scn14):
    temp = _zone_matrix(scn14, "bearing_temp_c")
    ambient = scn14.conditions["ambient_c"].to_numpy()[:, None]
    assert (temp - ambient).min() > -2.0


def test_carry_hotter_than_return(scn14):
    temp = _zone_matrix(scn14, "bearing_temp_c")
    running = scn14.belt_wide["run_state"].to_numpy() == "running"
    assert temp[running, :60].mean() > temp[running, 60:].mean() + 1.0


def test_neighbour_correlation(scn14):
    temp = _zone_matrix(scn14, "bearing_temp_c")
    for z in range(119):
        assert np.corrcoef(temp[:, z], temp[:, z + 1])[0, 1] > 0.95


def test_kurtosis_mean(scn14):
    assert abs(scn14.zones["vibration_kurtosis"].mean() - 3.0) < 0.05


def test_motor_current_and_slip(scn14):
    bw = scn14.belt_wide
    running = bw["run_state"] == "running"
    cur = bw.loc[running, "motor_current_a"]
    assert cur.min() > 100 and cur.max() < 520
    assert bw["slip_pct"].mean() < 1.0


def test_sanity_targets(scn14):
    """Loose numeric versions of the plot targets in spec section 5.5."""
    running = scn14.belt_wide["run_state"].to_numpy() == "running"
    vib = _zone_matrix(scn14, "vibration_rms_mm_s")[running]
    ac = _zone_matrix(scn14, "acoustic_db")
    temp = _zone_matrix(scn14, "bearing_temp_c")[running]
    assert 1.6 < vib[:, 2:60].mean() < 2.4
    assert vib[:, :2].mean() > vib[:, 2:60].mean() * 1.15
    assert vib[:, 60:].mean() < vib[:, 2:60].mean()
    assert 56 < ac[running].mean() < 62
    assert 40 < ac[~running & (scn14.belt_wide["run_state"].to_numpy() == "stopped")].mean() < 44
    assert 15 < temp.min() and temp.max() < 55
    assert scn14.belt_wide["motor_temp_c"].max() < 75


def test_determinism():
    a = generate(Config(seed=7, days=2))
    b = generate(Config(seed=7, days=2))
    c = generate(Config(seed=8, days=2))
    assert a.zones.equals(b.zones)
    assert a.belt_wide.equals(b.belt_wide)
    assert a.layout.equals(b.layout)
    assert not a.zones["bearing_temp_c"].equals(c.zones["bearing_temp_c"])


def test_noise_scale_changes_only_noise():
    quiet = {"noise_random": [1, 1], "drift_random": [0, 0], "shared_random": [0, 0]}
    a = generate(config_from_dict({"seed": 3, "days": 1, "signals": quiet}))
    b = generate(config_from_dict({"seed": 3, "days": 1, "signals": {**quiet, "noise_scale": 2.0}}))
    assert a.zones["vibration_kurtosis"].isna().mean() < 0.02  # default messiness: about 1% missing
    ka, kb = a.zones["vibration_kurtosis"], b.zones["vibration_kurtosis"]
    assert 1.8 < kb.std() / ka.std() < 2.2
    pd.testing.assert_frame_equal(a.layout, b.layout)


def test_bad_layout_raises():
    with pytest.raises(ValueError):
        LayoutCfg(section_length_m=130)
    with pytest.raises(ValueError):
        config_from_dict({"layout": {"section_length_m": 130}})


def test_typo_raises():
    with pytest.raises(TypeError):
        config_from_dict({"seeed": 1})
    with pytest.raises(TypeError):
        config_from_dict({"signals": {"noise_scael": 2}})


def test_yaml_roundtrip(tmp_path):
    from simulator.config import dump_config, load_config
    cfg = config_from_dict({"seed": 5, "signals": {"noise_scale": 2.0}})
    dump_config(cfg, tmp_path / "c.yaml")
    assert load_config(tmp_path / "c.yaml") == cfg


def test_noise_level_is_random_per_scenario_and_changes_nothing_else():
    quiet = {"noise_random": [1, 1]}
    runs = {seed: generate(config_from_dict({"seed": seed, "days": 1})) for seed in (1, 2, 3, 4, 5, 6)}
    factors = [r.meta["noise_factor"] for r in runs.values()]
    assert all(1.0 <= f <= 5.0 for f in factors) and len({round(f, 4) for f in factors}) == len(factors)
    assert max(factors) / min(factors) > 1.5                                   # genuinely different noise levels
    again = generate(config_from_dict({"seed": 3, "days": 1}))
    assert again.meta["noise_factor"] == runs[3].meta["noise_factor"] and again.zones.equals(runs[3].zones)
    base = generate(config_from_dict({"seed": 3, "days": 1, "signals": quiet}))
    assert base.meta["noise_factor"] == 1.0 and base.meta["noise_scale_effective"] == 1.0
    # the noise draw has its own stream: layout, faults and conditions are untouched by it
    assert base.layout.equals(runs[3].layout) and base.layout_truth.equals(runs[3].layout_truth)
    drop = ["belt_distance_m"]     # belt travel integrates the belt speed, which carries slip noise
    assert base.conditions.drop(columns=drop).equals(runs[3].conditions.drop(columns=drop))
    assert base.faults.equals(runs[3].faults)
    assert not base.zones["acoustic_db"].equals(runs[3].zones["acoustic_db"])    # but the sensor noise differs
    # an explicit noise_scale multiplies the random factor
    scaled = generate(config_from_dict({"seed": 3, "days": 1, "signals": {"noise_scale": 2.0}}))
    assert scaled.meta["noise_scale_effective"] == pytest.approx(2.0 * runs[3].meta["noise_factor"])
    with pytest.raises(ValueError):
        config_from_dict({"signals": {"noise_random": [0, 2]}})


# ---- sensor drift and section-shared noise ------------------------------------------------------------------

QUIET = {"noise_random": [1, 1], "drift_random": [0, 0], "shared_random": [0, 0]}


def _gen(signals, seed=3, days=4):
    return generate(config_from_dict({"seed": seed, "days": days, "signals": {**QUIET, **signals},
                                      "faults": {"idler_bearing": {"per_30d": 0}, "drive_overheat": {"per_30d": 0}},
                                      "messiness": {"missing_rate": 0, "outages_per_30d": 0, "stuck_per_30d": 0}}))


def _extra(signals, col="bearing_temp_c"):
    """What drift and/or shared noise added to one signal: the messy run minus the same run without them."""
    base, messy = _gen({}), _gen(signals)
    d = messy.zones[col].to_numpy(dtype=float) - base.zones[col].to_numpy(dtype=float)
    return d.reshape(-1, 120)


def test_drift_is_slow_per_sensor_and_leaves_the_rest_alone():
    d = _extra({"drift_random": [2, 2]})
    assert d.std() > 1.0                                              # drift_temp_c 1.5 x 2, over 4 days
    hour_change = d[60:] - d[:-60]
    assert hour_change.std() < 0.2 * d.std()                          # slow: an hour moves it a small part of its range
    assert not np.allclose(d[:, 0], d[:, 1]) and not np.allclose(d[:, 0], d[:, 5])   # one wander per sensor
    base, messy = _gen({}), _gen({"drift_random": [2, 2]})
    assert base.layout.equals(messy.layout) and base.faults.equals(messy.faults)
    assert base.conditions.drop(columns="belt_distance_m").equals(messy.conditions.drop(columns="belt_distance_m"))
    for col in BELT_SIGNALS:                                          # drift is a zone-sensor effect only
        assert base.belt_wide[col].equals(messy.belt_wide[col])


def test_shared_noise_is_identical_within_a_section_and_independent_between_sections():
    d = _extra({"shared_random": [2, 2]})
    assert d.std() > 0.5
    for sec in range(12):
        block = d[:, sec * 10:(sec + 1) * 10]
        assert np.allclose(block, block[:, [0]], atol=1e-3)            # all 10 zones see the same disturbance (float32)
    assert abs(np.corrcoef(d[:, 0], d[:, 10])[0, 1]) < 0.5            # a different section sees a different one
    assert abs(np.corrcoef(d[:, 0], d[:, 60])[0, 1]) < 0.5


def test_drift_and_shared_levels_are_random_per_scenario_and_validated():
    runs = {seed: generate(config_from_dict({"seed": seed, "days": 1})) for seed in range(1, 7)}
    drifts = [r.meta["drift_scale_effective"] for r in runs.values()]
    shared = [r.meta["shared_scale_effective"] for r in runs.values()]
    assert all(0 <= v <= 3 for v in drifts + shared) and len(set(drifts)) == 6 and len(set(shared)) == 6
    assert max(drifts) - min(drifts) > 0.5 and max(shared) - min(shared) > 0.5
    off = _gen({})
    assert off.meta["drift_scale_effective"] == 0 and off.meta["shared_scale_effective"] == 0
    with pytest.raises(ValueError):
        config_from_dict({"signals": {"drift_random": [-1, 2]}})
    with pytest.raises(ValueError):
        config_from_dict({"signals": {"shared_random": [3, 1]}})
