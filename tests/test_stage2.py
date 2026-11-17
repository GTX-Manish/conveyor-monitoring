"""Stage 2: belt coordinates, fault scheduling, repair stops and fault effects."""
import numpy as np
import pandas as pd
import pytest

from simulator.belt_coords import belt_coordinate, ground_position
from simulator.conditions import build_conditions
from simulator.config import Config, config_from_dict
from simulator.faults import DRIVE_OVERHEAT, IDLER, apply_repair_stops, schedule_faults, severity
from simulator.generate import generate, make_streams
from simulator.healthy import BELT_SIGNALS, ZONE_SIGNALS

Z = 120
NO_MESS = {"missing_rate": 0, "outages_per_30d": 0, "stuck_per_30d": 0}
ZERO = {
    IDLER: {"temp_c": 0, "acoustic_db": 0, "vib_gain": 0, "kurtosis_add": 0, "current_a": 0},
    DRIVE_OVERHEAT: {"motor_temp_c": 0, "gearbox_temp_c": 0, "current_a": 0},
}
FIXED = {"strength": (1.0, 1.0), "signal_mix": (1.0, 1.0), "exponent": (2.5, 2.5)}   # the old fixed fault
SHORT = {
    IDLER: {"min_days": 1, "max_days": 2},
    DRIVE_OVERHEAT: {"min_hours": 24, "max_hours": 48},
}


def one_fault(kind, zero=False, seed=11):
    """A 6-day scenario with exactly one fault of `kind`. `zero` gives the same schedule with no effect."""
    faults = {**FIXED, **{k: {"per_30d": 0} for k in ZERO}}
    faults[kind] = {"per_30d": 5, **SHORT[kind], **(ZERO[kind] if zero else {})}  # 5 x 6/30 = exactly 1
    return generate(config_from_dict({"seed": seed, "days": 6, "faults": faults, "messiness": NO_MESS}))


@pytest.fixture(scope="module", params=[IDLER, DRIVE_OVERHEAT])
def pair(request):
    return request.param, one_fault(request.param), one_fault(request.param, zero=True)


def zmat(scn, col):
    return scn.zones[col].to_numpy().reshape(-1, Z).astype(float)


def sev(scn):
    return severity(scn.events[0], len(scn.belt_wide))


# ---- scheduling -------------------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(12))
def test_schedule_rules(seed):
    cfg = Config(seed=seed, days=30)
    n = cfg.days * 1440
    rng = make_streams(seed)
    cond = build_conditions(cfg.conditions, cfg.start, n, rng["conditions"])
    events = schedule_faults(cfg.faults, cfg.days, Z, rng["faults"])
    base = apply_repair_stops(events, cond.base, cfg.conditions.ramp_minutes)

    assert [sum(e.type == k for e in events) for k in (IDLER, DRIVE_OVERHEAT)] == [5, 1]
    by_asset = {}
    for e in events:
        assert e.onset >= 24 * 60                       # never in the first 24 h
        assert e.repair_end <= n - 12 * 60              # complete 12 h before the end
        assert e.onset < e.failure < e.repair_end
        assert cond.base[e.failure] == 1.0              # it fails while running...
        assert (base[e.failure:e.repair_end] == 0).all()  # ...and the belt stops for the repair
        by_asset.setdefault(e.asset, []).append((e.onset, e.repair_end))
    for spans in by_asset.values():
        spans.sort()
        assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # no overlap on one asset


def test_severity_shape(pair):
    _, scn, _ = pair
    ev, s = scn.events[0], sev(scn)
    assert (s[:ev.onset + 1] == 0).all() and (s[ev.repair_end:] == 0).all()
    assert s[ev.failure] == 1.0 and (s[ev.failure:ev.repair_end] == 1).all()
    rise = s[ev.onset:ev.failure]
    assert (np.diff(rise) > 0).all() and rise.max() < 1.0


def test_same_schedule_with_and_without_effects(pair):
    _, scn, twin = pair
    pd.testing.assert_frame_equal(scn.faults, twin.faults)
    assert (scn.belt_wide["run_state"] == twin.belt_wide["run_state"]).all()


def test_repair_stop_and_recovery(pair):
    kind, scn, twin = pair
    ev = scn.events[0]
    state = scn.belt_wide["run_state"].to_numpy()
    assert state[ev.failure - 1] == "running"
    assert (state[ev.failure:ev.repair_end] == "stopped").all()
    assert state[ev.repair_end] in ("ramp", "stopped")  # stopped if the repair ends inside the nightly stop
    assert "running" in state[ev.repair_end:ev.repair_end + 8 * 60]
    after = slice(ev.repair_end, None)
    for col in ZONE_SIGNALS:  # the asset is healthy again after the repair
        assert np.abs(zmat(scn, col)[after] - zmat(twin, col)[after]).max() < 0.1, col
    for col in ["motor_current_a", "tension_kn"]:
        assert np.abs(scn.belt_wide[col].to_numpy()[after] - twin.belt_wide[col].to_numpy()[after]).max() < 0.1
    # drive temperatures cool with a 25 / 40 min lag after the trip; healthy within a few hours
    later = slice(ev.repair_end + 180, None)
    for col in ["motor_temp_c", "gearbox_temp_c"]:
        assert np.abs(scn.belt_wide[col].to_numpy()[later] - twin.belt_wide[col].to_numpy()[later]).max() < 0.1


# ---- idler bearing ----------------------------------------------------------------------------------

def test_idler_effects(pair):
    kind, scn, twin = pair
    if kind != IDLER:
        pytest.skip("idler only")
    ev, s = scn.events[0], sev(scn)
    z = ev.zone - 1
    others = np.arange(Z) != z
    diffs = {c: zmat(scn, c) - zmat(twin, c) for c in ZONE_SIGNALS}
    for col, d in diffs.items():
        assert (d[:, others] == 0).all(), f"{col} changed outside the faulty zone"
    assert (diffs["belt_offset_mm"] == 0).all()

    # the drive cannot see an idler fault
    for col in ["motor_temp_c", "gearbox_temp_c"]:
        assert scn.belt_wide[col].equals(twin.belt_wide[col])
    cur = scn.belt_wide["motor_current_a"].to_numpy() - twin.belt_wide["motor_current_a"].to_numpy()
    assert 0.5 < cur.max() <= 1.6

    # early effects stay within about one noise sigma at s <= 0.1 (healthy sds: 0.3 C, 1.2 dB+, ~0.15, 4%)
    early = (s > 0) & (s <= 0.1)
    assert early.sum() > 300
    assert diffs["bearing_temp_c"][early, z].max() <= 0.3
    assert diffs["acoustic_db"][early, z].max() <= 1.2
    assert diffs["vibration_kurtosis"][early, z].max() <= 0.16
    assert (zmat(scn, "vibration_rms_mm_s")[early, z] / np.maximum(zmat(twin, "vibration_rms_mm_s")[early, z], 1e-9)
            ).max() <= 1.04

    # late effects are large
    last_hour = slice(ev.failure - 60, ev.failure)
    assert diffs["bearing_temp_c"][last_hour, z].mean() > 15
    assert diffs["acoustic_db"][last_hour, z].mean() > 7


# ---- drive overheating ------------------------------------------------------------------------------

def test_drive_effects(pair):
    kind, scn, twin = pair
    if kind != DRIVE_OVERHEAT:
        pytest.skip("drive only")
    s = sev(scn)
    for col in ZONE_SIGNALS:  # no zone sees a drive fault
        assert scn.zones[col].equals(twin.zones[col])
    for col in set(BELT_SIGNALS) - {"motor_temp_c", "gearbox_temp_c", "motor_current_a"}:
        assert scn.belt_wide[col].equals(twin.belt_wide[col]), col
    d = {c: scn.belt_wide[c].to_numpy().astype(float) - twin.belt_wide[c].to_numpy() for c in BELT_SIGNALS}
    assert d["motor_temp_c"].max() > 25 and d["gearbox_temp_c"].max() > 12
    # at first only the temperatures move: current stays inside its 4 A noise
    early = (s > 0) & (s < 0.3)
    assert d["motor_current_a"][early].max() < 0.6
    assert d["motor_temp_c"][(s > 0.25) & (s < 0.3)].max() > 3


# ---- belt coordinates ---------------------------------------------------------------

def test_coordinate_round_trip():
    D = np.cumsum(np.full(500, 240.0))
    c = 1234.5
    p = ground_position(c, D, 6000.0)
    assert np.allclose(belt_coordinate(p, D, 6000.0), c)
    assert np.isclose(p[24], (c + 6000.0) % 6000.0)  # 25 steps of 240 m = one lap, back where it started


# ---- determinism and ground truth -------------------------------------------------------------------

def test_fault_run_is_deterministic(pair):
    kind, scn, _ = pair
    if kind != DRIVE_OVERHEAT:
        pytest.skip("one fault type is enough; keeps the suite fast")
    again = one_fault(kind)
    assert scn.zones.equals(again.zones)
    assert scn.belt_wide.equals(again.belt_wide)
    assert scn.faults.equals(again.faults)


def test_fault_truth_stays_out_of_public(pair, tmp_path):
    from simulator.io import PUBLIC_COLUMNS, VISION_PUBLIC_COLUMNS, write_scenario
    kind, scn, _ = pair
    root = write_scenario(scn, config_from_dict({"scenario": f"s2_{kind}"}), tmp_path)
    assert (root / "truth" / "faults.parquet").exists()
    assert sorted(p.name for p in (root / "public").iterdir()) == sorted(
        [f"{n}.parquet" for n in [*PUBLIC_COLUMNS, *VISION_PUBLIC_COLUMNS]] + ["manifest.json"])
    public_text = (root / "public" / "manifest.json").read_text()
    assert scn.events[0].fault_id not in public_text and "fault" not in public_text
    assert set(scn.belt_wide["run_state"].unique()) <= {"stopped", "ramp", "running"}  # no "repair" state


# ---- every fault has its own character ----------------------------------------------------------------

def test_each_fault_draws_its_own_strength_shape_and_mix():
    cfg = Config(seed=3, days=30)
    events = schedule_faults(cfg.faults, 30, Z, make_streams(3)["faults"])
    assert len(events) == 6
    lo_s, hi_s = cfg.faults.strength
    lo_e, hi_e = cfg.faults.exponent
    lo_m, hi_m = cfg.faults.signal_mix
    for ev in events:
        assert lo_s <= ev.strength <= hi_s and lo_e <= ev.exponent <= hi_e
        assert len(ev.mix) == (5 if ev.type == IDLER else 3) and all(lo_m <= m <= hi_m for m in ev.mix)
    assert len({round(e.strength, 6) for e in events}) == len(events)      # no two alike
    assert len({round(e.exponent, 6) for e in events}) == len(events)
    again = schedule_faults(cfg.faults, 30, Z, make_streams(3)["faults"])
    assert [(e.strength, e.exponent, e.mix) for e in events] == [(e.strength, e.exponent, e.mix) for e in again]


def test_a_stronger_fault_has_a_larger_effect_and_the_growth_shape_follows_its_exponent():
    def temp_effect(strength, exponent):
        faults = {**FIXED, "strength": (strength, strength), "exponent": (exponent, exponent),
                  "drive_overheat": {"per_30d": 0}, "idler_bearing": {"per_30d": 5, **SHORT[IDLER]}}
        cfg = {"seed": 11, "days": 6, "faults": faults, "messiness": NO_MESS}
        scn = generate(config_from_dict(cfg))
        twin = generate(config_from_dict({**cfg, "faults": {**faults, "idler_bearing": {**faults["idler_bearing"],
                                                                                         **ZERO[IDLER]}}}))
        ev = scn.events[0]
        d = (zmat(scn, "acoustic_db") - zmat(twin, "acoustic_db"))[:, ev.zone - 1]
        return ev, d
    ev1, weak = temp_effect(0.5, 2.5)
    ev2, strong = temp_effect(1.0, 2.5)
    assert ev1.onset == ev2.onset and ev1.failure == ev2.failure
    assert strong.max() == pytest.approx(2 * weak.max(), rel=0.02)            # the effect scales with strength
    _, late = temp_effect(1.0, 3.5)
    _, linear = temp_effect(1.0, 1.0)
    span = slice(ev2.onset, ev2.failure)    # the belt is stopped at night, so compare the whole progression
    assert linear[span].sum() > strong[span].sum() > late[span].sum()        # a lower exponent grows earlier


def test_fault_ranges_are_validated_and_degenerate_ranges_give_the_fixed_fault():
    with pytest.raises(ValueError):
        config_from_dict({"faults": {"strength": [0, 1]}})
    with pytest.raises(ValueError):
        config_from_dict({"faults": {"exponent": [3.0, 2.0]}})
    cfg = Config(seed=3, days=30, faults=config_from_dict({"faults": FIXED}).faults)
    for ev in schedule_faults(cfg.faults, 30, Z, make_streams(3)["faults"]):
        assert (ev.strength, ev.exponent, ev.mix[0]) == (1.0, 2.5, 1.0)
