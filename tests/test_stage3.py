"""Stage 3: hard negatives (normal events that look like trouble) and ground-truth labels."""
import numpy as np
import pandas as pd
import pytest

from simulator.config import Config, config_from_dict
from simulator.generate import generate, make_streams
from simulator.hard_negatives import (DUSTY, HEAT_WAVE, HEAVY_LOAD, STARTUP, hard_negatives_frame,
                                      schedule_hard_negatives)

Z, A = 120, 121
NO_MESS = {"missing_rate": 0, "outages_per_30d": 0, "stuck_per_30d": 0}
NO_FAULTS = {"idler_bearing": {"per_30d": 0}, "drive_overheat": {"per_30d": 0}}
NO_HN = {HEAVY_LOAD: {"per_30d": 0}, HEAT_WAVE: {"per_30d": 0}, DUSTY: {"per_30d": 0}}
DAYS = 4
ONE = 30 / DAYS  # per_30d rate that gives exactly one event in DAYS days


def run(hard_negatives, faults=NO_FAULTS, seed=5, days=DAYS):
    return generate(config_from_dict({"seed": seed, "days": days, "faults": faults,
                                      "hard_negatives": hard_negatives, "messiness": NO_MESS}))


@pytest.fixture(scope="module")
def plain():
    return run(NO_HN)


def only(kind, **extra):
    return run({**NO_HN, kind: {"per_30d": ONE, **extra}})


def zmat(scn, col):
    return scn.zones[col].to_numpy().reshape(-1, Z).astype(float)


def window(scn, kind):
    hn = scn.hard_negatives
    rows = hn[hn["type"] == kind]
    assert len(rows) == 1
    ts = scn.belt_wide["timestamp"]
    inside = ((ts >= rows["start_time"].iloc[0]) & (ts < rows["end_time"].iloc[0])).to_numpy()
    running = scn.belt_wide["run_state"].to_numpy() == "running"
    return inside, inside & running


def assert_labelled_not_a_fault(scn, inside):
    lab = scn.labels
    assert (lab["stage"] == "normal").all() and not lab["fails_within_24h"].any()
    assert len(scn.faults) == 0
    flag = lab["is_hard_negative_period"].to_numpy().reshape(-1, A)
    assert flag[inside].all()
    assert (flag == flag[:, :1]).all()  # belt-wide: same for every asset


def test_heavy_load(plain):
    scn = only(HEAVY_LOAD)
    inside, run_in = window(scn, HEAVY_LOAD)
    assert run_in.sum() > 120
    load = scn.conditions["load"].to_numpy()
    speed = scn.conditions["speed"].to_numpy()
    assert (load[run_in] / speed[run_in]).min() >= 0.95 - 1e-9
    # effects arrive through the normal physics: more current, hotter drive, hotter carry bearings
    cur = scn.belt_wide["motor_current_a"].to_numpy() - plain.belt_wide["motor_current_a"].to_numpy()
    mot = scn.belt_wide["motor_temp_c"].to_numpy() - plain.belt_wide["motor_temp_c"].to_numpy()
    temp = zmat(scn, "bearing_temp_c") - zmat(plain, "bearing_temp_c")
    assert cur[run_in].mean() > 10
    assert mot[run_in].max() > 1.0
    assert temp[run_in][:, :60].max() > 0.5 > temp[run_in][:, 60:].max()  # return idlers barely notice
    before = np.flatnonzero(inside)[0]
    assert np.abs(cur[:before]).max() == 0 and np.abs(temp[:before]).max() == 0
    assert_labelled_not_a_fault(scn, inside)


def test_heat_wave(plain):
    scn = only(HEAT_WAVE, min_days=1, max_days=1.5)
    inside, _ = window(scn, HEAT_WAVE)
    amb = scn.conditions["ambient_c"].to_numpy() - plain.conditions["ambient_c"].to_numpy()
    assert 6 <= amb.max() <= 10 and amb[~inside].max() == 0
    assert np.abs(np.diff(amb)).max() < 0.1  # fades in and out, no step
    temp = zmat(scn, "bearing_temp_c") - zmat(plain, "bearing_temp_c")
    assert temp.max() > 5
    # every zone warms by the same amount, so a neighbour comparison cancels a heat wave exactly
    assert temp.std(axis=1).max() < 1e-3
    mot = scn.belt_wide["motor_temp_c"].to_numpy() - plain.belt_wide["motor_temp_c"].to_numpy()
    assert mot.max() > 5
    assert_labelled_not_a_fault(scn, inside)


def test_dusty_spell(plain):
    scn = only(DUSTY, min_hours=12, max_hours=24)
    inside, run_in = window(scn, DUSTY)
    assert run_in.sum() > 120
    dust = scn.conditions["dust"].to_numpy()
    assert 0.8 <= dust[inside].min() and dust[inside].max() <= 0.9 + 1e-9
    ac_s, ac_p = zmat(scn, "acoustic_db")[run_in], zmat(plain, "acoustic_db")[run_in]
    assert (ac_s - ac_s.mean(axis=0)).std() / (ac_p - ac_p.mean(axis=0)).std() > 1.1   # noisier acoustic
    assert zmat(scn, "vibration_rms_mm_s")[run_in].mean() / zmat(plain, "vibration_rms_mm_s")[run_in].mean() > 1.05
    k_s, k_p = zmat(scn, "vibration_kurtosis")[run_in], zmat(plain, "vibration_kurtosis")[run_in]
    assert k_s.std() / k_p.std() > 1.08
    assert np.allclose(zmat(scn, "bearing_temp_c"), zmat(plain, "bearing_temp_c"))  # dust does not heat anything
    assert_labelled_not_a_fault(scn, inside)


def test_startup_transient(plain):
    flat = run({**NO_HN, "startup_vib_gain": 1.0})
    ratio = zmat(plain, "vibration_rms_mm_s") / zmat(flat, "vibration_rms_mm_s")
    starts = plain.hard_negatives[plain.hard_negatives["type"] == STARTUP]
    state = plain.belt_wide["run_state"].to_numpy()
    moving = state != "stopped"
    n_starts = int((moving[1:] & ~moving[:-1]).sum())
    assert len(starts) == n_starts == DAYS
    ts = plain.belt_wide["timestamp"]
    mask = np.zeros(len(ts), dtype=bool)
    for s, e in zip(starts["start_time"], starts["end_time"]):
        assert (e - s).total_seconds() == 600
        mask |= ((ts >= s) & (ts < e)).to_numpy()
    assert mask.sum() == 10 * n_starts
    assert np.allclose(ratio[mask], 1.3, atol=1e-4) and np.allclose(ratio[~mask], 1.0, atol=1e-6)
    flag = plain.labels["is_hard_negative_period"].to_numpy().reshape(-1, A)[:, 0]
    assert (flag == mask).all()


def test_default_counts():
    """Schedule only (no signals), so a 30-day check stays fast."""
    cfg = Config(seed=2, days=30)
    events = schedule_hard_negatives(cfg.hard_negatives, cfg.days, make_streams(cfg.seed)["hard_negatives"])
    count = lambda k: sum(e.type == k for e in events)  # noqa: E731
    assert (count(HEAVY_LOAD), count(HEAT_WAVE), count(DUSTY)) == (4, 1, 2)
    hn = hard_negatives_frame(events, pd.date_range(cfg.start, periods=cfg.days * 1440, freq="min"))
    assert (hn["zones"] == "all").all() and (hn["end_time"] > hn["start_time"]).all()
    hours = (hn["end_time"] - hn["start_time"]).dt.total_seconds() / 3600
    for kind, lo, hi in ((HEAVY_LOAD, 6, 12), (HEAT_WAVE, 24, 72), (DUSTY, 4, 24)):
        h = hours[hn["type"] == kind]
        assert h.min() >= lo - 0.02 and h.max() <= hi
    for kind in (HEAVY_LOAD, HEAT_WAVE, DUSTY):  # events of one type never overlap
        spans = sorted((e.start, e.end) for e in events if e.type == kind)
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_labels_by_hand():
    """One idler fault, every label value checked against the schedule."""
    faults = {"idler_bearing": {"per_30d": 5, "min_days": 2, "max_days": 3}, "drive_overheat": {"per_30d": 0}}
    scn = run(NO_HN, faults=faults, seed=11, days=6)
    ev = scn.events[0]
    T = len(scn.belt_wide)
    lab = scn.labels
    assert len(lab) == T * A
    assert list(lab["asset"].cat.categories) == [str(z) for z in range(1, Z + 1)] + ["DRIVE"]
    a = ev.zone - 1
    grid = lambda c: lab[c].to_numpy().reshape(T, A)  # noqa: E731
    stage, htf, within, sev, fid = (grid(c) for c in
                                    ("stage", "hours_to_failure", "fails_within_24h", "severity", "fault_id"))

    # every other asset is normal throughout
    others = np.arange(A) != a
    assert (stage[:, others] == "normal").all() and not within[:, others].any()
    assert np.isnan(htf[:, others]).all() and (sev[:, others] == 0).all()

    f, o, r = ev.failure, ev.onset, ev.repair_end
    dur = f - o
    assert stage[o - 1, a] == "normal" and np.isnan(htf[o - 1, a])
    assert np.isclose(htf[o, a], dur / 60)
    assert np.isclose(htf[f - 60, a], 1.0) and within[f - 60, a]
    assert np.isclose(htf[f - 1440, a], 24.0) and within[f - 1440, a]
    assert not within[f - 1441, a] and np.isclose(htf[f - 1441, a], 24 + 1 / 60, atol=1e-4)
    assert within[:, a].sum() == 1440
    assert stage[f - 1, a] == "advanced" and stage[f, a] == "failed"
    assert (stage[f:r, a] == "failed").all() and (htf[f:r, a] == 0).all() and not within[f:r, a].any()
    assert stage[r, a] == "normal" and np.isnan(htf[r, a]) and sev[r, a] == 0

    # early / advanced boundary: s = ((t - onset) / dur) ** 2.5 crosses 0.3
    k = int(np.ceil(dur * 0.3 ** (1 / 2.5)))
    assert stage[o + k - 1, a] == "early" and stage[o + k, a] == "advanced"
    assert sev[o + k - 1, a] < 0.3 <= sev[o + k, a]
    assert (stage[o + 1:o + k, a] == "early").all() and (stage[o + k:f, a] == "advanced").all()
    assert set(lab.loc[lab["fault_id"].notna(), "fault_id"].astype(str)) == {ev.fault_id}
    assert lab["fault_id"].notna().sum() == r - o


def test_labels_and_hard_negatives_are_truth_only(tmp_path):
    from simulator.io import PUBLIC_COLUMNS, load_public, write_scenario
    scn = only(HEAVY_LOAD)
    root = write_scenario(scn, config_from_dict({"scenario": "s3"}), tmp_path)
    truth = {p.name for p in (root / "truth").iterdir()}
    assert {"labels.parquet", "hard_negatives.parquet", "faults.parquet"} <= truth
    public = {p.name for p in (root / "public").iterdir()}
    assert public == {f"{n}.parquet" for n in PUBLIC_COLUMNS} | {"manifest.json"}
    for name, df in load_public(root).items():
        if name != "manifest":
            assert not {"stage", "severity", "hours_to_failure", "fails_within_24h", "fault_id",
                        "is_hard_negative_period"} & set(df.columns)
