"""Stage 4: data defects (missing readings, suite outages, stuck sensors)."""
import numpy as np
import pandas as pd
import pytest

from simulator.config import Config, config_from_dict
from simulator.generate import generate, make_streams
from simulator.healthy import BELT_SIGNALS, ZONE_SIGNALS
from simulator.messiness import OUTAGE, STUCK, schedule_sensor_faults, sensor_faults_frame

Z = 120
DAYS = 6
CLEAN = {"missing_rate": 0, "outages_per_30d": 0, "stuck_per_30d": 0}
# 15 x 6/30 = exactly 3 outages, 10 x 6/30 = exactly 2 stuck sensors
MESSY = {"missing_rate": 0.01, "outages_per_30d": 15, "stuck_per_30d": 10, "stuck_hours": [2, 24]}


def run(messiness, seed=4):
    return generate(config_from_dict({"seed": seed, "days": DAYS, "messiness": messiness}))


@pytest.fixture(scope="module")
def messy():
    return run(MESSY)


@pytest.fixture(scope="module")
def clean():
    return run(CLEAN)


def zmat(scn, col):
    return scn.zones[col].to_numpy().reshape(-1, Z)


def fault_cells(scn):
    """Boolean (T, Z) per signal: cells covered by an outage or a stuck period."""
    T = len(scn.belt_wide)
    cover = {s: np.zeros((T, Z), dtype=bool) for s in ZONE_SIGNALS}
    for ev in scn.sensor_fault_events:
        for s in (ZONE_SIGNALS if ev.type == OUTAGE else [ev.signal]):
            cover[s][ev.start:ev.end, ev.zone - 1] = True
    return cover


def test_schedule_counts_and_durations():
    cfg = Config(seed=3, days=30)
    events = schedule_sensor_faults(cfg.messiness, cfg.days, Z, ZONE_SIGNALS, make_streams(cfg.seed)["messiness"])
    assert [sum(e.type == k for e in events) for k in (OUTAGE, STUCK)] == [6, 4]
    for e in events:
        hours = (e.end - e.start) / 60
        lo, hi = (1, 6) if e.type == OUTAGE else (2, 48)
        assert lo - 0.02 <= hours <= hi
        assert 1 <= e.zone <= Z and 1 <= e.start < e.end <= cfg.days * 1440
        assert (e.signal is None) == (e.type == OUTAGE)
        assert e.type == OUTAGE or e.signal in ZONE_SIGNALS


def test_missing_rate(messy):
    cover = fault_cells(messy)
    for col in ZONE_SIGNALS:  # measured outside outage windows, so only random loss is counted
        rate = np.isnan(zmat(messy, col))[~cover[col]].mean()
        assert 0.005 <= rate <= 0.015, col
    running = messy.belt_wide["run_state"].to_numpy() == "running"
    for col in BELT_SIGNALS:  # slip_pct is undefined when stopped, so measure while running
        assert 0.005 <= messy.belt_wide[col].isna().to_numpy()[running].mean() <= 0.015, col
    assert 0.005 <= messy.meta["missing_rate_measured"] <= 0.015
    assert messy.meta["sensor_fault_counts"] == {OUTAGE: 3, STUCK: 2}


def test_rows_and_context_columns_are_never_dropped(messy, clean):
    assert len(messy.zones) == len(clean.zones) and len(messy.belt_wide) == len(clean.belt_wide)
    for col in ["timestamp", "global_zone", "splice_section", "zone_in_section", "position_m", "run", "run_state"]:
        assert messy.zones[col].equals(clean.zones[col])
    assert messy.belt_wide[["timestamp", "run_state"]].equals(clean.belt_wide[["timestamp", "run_state"]])


def test_outage_silences_the_whole_suite(messy):
    outages = [e for e in messy.sensor_fault_events if e.type == OUTAGE]
    assert len(outages) == 3
    for ev in outages:
        z = ev.zone - 1
        for col in ZONE_SIGNALS:
            m = zmat(messy, col)
            assert np.isnan(m[ev.start:ev.end, z]).all(), col
            assert np.isnan(m[ev.start:ev.end, (z + 1) % Z]).mean() < 0.1  # the neighbour keeps reporting


def test_stuck_sensor_is_exactly_flat(messy, clean):
    stuck = [e for e in messy.sensor_fault_events if e.type == STUCK]
    assert len(stuck) == 2
    outage_cover = np.zeros((len(messy.belt_wide), Z), dtype=bool)
    for ev in messy.sensor_fault_events:
        if ev.type == OUTAGE:
            outage_cover[ev.start:ev.end, ev.zone - 1] = True
    for ev in stuck:
        z = ev.zone - 1
        seen = zmat(messy, ev.signal)[ev.start:ev.end, z]
        truth = zmat(clean, ev.signal)[ev.start:ev.end, z]
        reported = seen[~np.isnan(seen)]
        assert len(reported) > 60
        assert (reported == reported[0]).all()                      # exactly constant
        assert reported[0] == np.float32(ev.stuck_value) == zmat(clean, ev.signal)[ev.start - 1, z]
        assert np.nanstd(truth) > 0                                 # the equipment kept changing
        for other in set(ZONE_SIGNALS) - {ev.signal}:               # the rest of the suite is not frozen
            o = zmat(messy, other)[ev.start:ev.end, z]
            o = o[~np.isnan(o) & ~outage_cover[ev.start:ev.end, z]]
            assert len(o) == 0 or np.unique(o).size > 10, other


def test_values_outside_defects_are_untouched(messy, clean):
    cover = fault_cells(messy)
    for col in ZONE_SIGNALS:
        m, c = zmat(messy, col), zmat(clean, col)
        keep = ~np.isnan(m) & ~cover[col]
        assert (m[keep] == c[keep]).all(), col
    for col in BELT_SIGNALS:
        m, c = messy.belt_wide[col].to_numpy(), clean.belt_wide[col].to_numpy()
        keep = ~np.isnan(m)
        assert (m[keep] == c[keep]).all(), col


def test_ground_truth_is_not_touched_by_messiness(messy, clean):
    assert messy.labels.equals(clean.labels)
    assert messy.faults.equals(clean.faults)
    assert messy.hard_negatives.equals(clean.hard_negatives)
    assert messy.conditions.equals(clean.conditions)


def test_sensor_faults_frame_and_truth_only(messy, tmp_path):
    from simulator.io import PUBLIC_COLUMNS, VISION_PUBLIC_COLUMNS, load_public, write_scenario
    sf = messy.sensor_faults
    assert list(sf.columns) == ["sensor_fault_id", "type", "zone", "signal", "start_time", "end_time", "stuck_value"]
    assert len(sf) == 5 and sf["sensor_fault_id"].is_unique
    assert sf.loc[sf["type"] == OUTAGE, "signal"].isna().all()
    assert sf.loc[sf["type"] == STUCK, "signal"].isin(ZONE_SIGNALS).all()
    ts = messy.belt_wide["timestamp"]
    for ev, (_, row) in zip(messy.sensor_fault_events, sf.iterrows()):
        assert row["start_time"] == ts[ev.start] and row["end_time"] == ts[ev.end - 1] + pd.Timedelta(minutes=1)

    root = write_scenario(messy, config_from_dict({"scenario": "s4"}), tmp_path)
    assert (root / "truth" / "sensor_faults.parquet").exists()
    allowed = {f"{n}.parquet" for n in [*PUBLIC_COLUMNS, *VISION_PUBLIC_COLUMNS]} | {"manifest.json"}
    assert {p.name for p in (root / "public").iterdir()} == allowed
    public = load_public(root)
    assert "missing_rate" not in str(public["manifest"]) and "sensor_fault" not in str(public["manifest"])
    # the public data really is the messy version: NaNs survive the parquet round trip
    assert public["zones"]["bearing_temp_c"].isna().sum() == messy.zones["bearing_temp_c"].isna().sum() > 0


def test_messy_run_is_deterministic(messy):
    again = run(MESSY)
    assert messy.zones.equals(again.zones) and messy.belt_wide.equals(again.belt_wide)
    assert messy.sensor_faults.equals(again.sensor_faults)
    assert sensor_faults_frame([], messy.belt_wide["timestamp"]).empty
