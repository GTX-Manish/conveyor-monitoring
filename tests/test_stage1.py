import numpy as np
import pandas as pd
import pytest

from simulator.config import Config, LayoutCfg, config_from_dict
from simulator.generate import generate


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
    assert scn14.meta["generate_seconds"] < 15


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
    a = generate(Config(seed=3, days=1))
    b = generate(config_from_dict({"seed": 3, "days": 1, "signals": {"noise_scale": 2.0}}))
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
