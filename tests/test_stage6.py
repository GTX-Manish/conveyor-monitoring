"""Stage 6: the scenario set builder and the demo seed finder, on a 2-day mini set."""
import json

import pandas as pd
import pytest
import yaml

from scripts import find_demo_seed as fds
from simulator import scenario_set
from simulator.config import Config
from simulator.generate import generate
from simulator.io import load_public
from simulator.scenario_set import STRESS_OVERRIDES, balance_check, default_specs, read_split, write_config

MINI = [dict(name="train_01", split="train", seed=101, days=2, overrides={}),
        dict(name="stress_01", split="stress", seed=401, days=2, overrides=STRESS_OVERRIDES)]
FIELDS = {"name", "split", "seed", "days", "config_hash", "path", "zone_rows", "public_mb", "seconds", "peak_mb",
          "fault_counts", "hard_negative_counts", "sensor_fault_counts", "fails_within_24h_rate"}


def build_set(*args, **kwargs):
    kwargs.setdefault("min_free_gb", 0.2)   # the test folder is on a small temporary disk
    return scenario_set.build_set(*args, **kwargs)


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    root = tmp_path_factory.mktemp("set")
    cfg_dir, out = root / "configs", root / "data"
    for spec in MINI:
        write_config(spec, cfg_dir)
    log = []
    first = build_set(cfg_dir, out, workers=2, report_path=root / "report.md", log=log.append)
    return cfg_dir, out, first, root


def test_default_set_has_unique_seeds_and_disjoint_splits():
    specs = default_specs()
    assert len(specs) == 27 and len({s["seed"] for s in specs}) == 27 and len({s["name"] for s in specs}) == 27
    assert [sum(s["split"] == k for s in specs) for k in ("train", "train_drive", "validation", "test", "stress",
                                                          "test_fresh")] == [8, 6, 6, 2, 1, 4]
    fresh = [s for s in specs if s["split"] == "test_fresh"]
    assert [s["seed"] for s in fresh] == [501, 502, 503, 504] and [s["name"] for s in fresh] == [
        "test_03", "test_04", "test_05", "test_06"]
    drive = [s for s in specs if s["split"] == "train_drive"]
    assert all(s["overrides"] == {"faults": {"drive_overheat": {"per_30d": 4.0}}} for s in drive)
    val = [s["overrides"]["signals"] for s in specs if s["split"] == "validation"]
    assert max(v["noise_random"][0] for v in val) >= 5 and max(v["drift_random"][0] for v in val) >= 3
    assert max(v["shared_random"][0] for v in val) >= 3 and min(v["noise_random"][0] for v in val) == 1
    assert all(s["days"] == 30 for s in specs)


def test_balance_check_fails_on_a_reused_seed():
    row = {"name": "a", "split": "train", "seed": 1, "fault_counts": {"idler_bearing": 1},
           "hard_negative_counts": {"heavy_load": 1, "heat_wave": 1, "dusty": 1}}
    errors, warnings = balance_check([row, {**row, "name": "b", "split": "test"}])
    assert errors and "seed 1" in errors[0]
    errors, warnings = balance_check([row, {**row, "name": "v", "seed": 2, "split": "validation"}])
    assert not errors and any("validation" in w for w in warnings)


def test_summary_has_every_field(mini):
    cfg_dir, out, first, root = mini
    saved = json.loads((out / "scenario_set.json").read_text())
    assert [r["name"] for r in saved["scenarios"]] == ["stress_01", "train_01"]
    for r in saved["scenarios"]:
        assert FIELDS <= set(r)
        assert r["zone_rows"] == 2 * 1440 * 120 and r["peak_mb"] < 3000
    assert set(saved["totals"]) == {"train", "stress"}
    assert "| `train_01` | train | 101 |" in (root / "report.md").read_text()
    assert read_split(cfg_dir / "stress_01.yaml") == "stress"


def test_second_run_skips_and_force_rebuilds(mini):
    cfg_dir, out, first, _ = mini
    assert len(first["built"]) == 2
    again = build_set(cfg_dir, out, log=lambda m: None)
    assert again["built"] == [] and sorted(again["skipped"]) == ["stress_01", "train_01"]
    forced = build_set(cfg_dir, out, only=["train_01"], force=True, log=lambda m: None)
    assert len(forced["built"]) == 1 and forced["skipped"] == []
    assert len(forced["scenarios"]) == 2  # the summary still covers the whole set


def test_changed_config_is_rebuilt(mini, tmp_path):
    cfg_dir, out, _, _ = mini
    other = tmp_path / "configs"
    write_config({**MINI[0], "overrides": {"signals": {"noise_scale": 1.5}}}, other)
    assert len(build_set(other, out, log=lambda m: None)["built"]) == 1
    assert len(build_set(cfg_dir, out, only=["train_01"], log=lambda m: None)["built"]) == 1  # restore for other tests


def test_stress_overrides_are_in_stress_only(mini):
    _, out, _, _ = mini
    used = {n: yaml.safe_load((out / n / "truth" / "config_used.yaml").read_text()) for n in ("train_01", "stress_01")}
    assert used["stress_01"]["signals"]["noise_scale"] == 2.0 and used["train_01"]["signals"]["noise_scale"] == 1.0
    assert used["stress_01"]["messiness"]["missing_rate"] == 0.03 and used["train_01"]["messiness"]["missing_rate"] == 0.01
    assert used["stress_01"]["signals"]["drift_random"] == [2.0, 3.0]
    assert used["train_01"]["signals"]["drift_random"] == [0.0, 3.0]
    assert used["stress_01"]["faults"]["idler_bearing"]["per_30d"] == 8.0
    assert used["stress_01"]["hard_negatives"]["dusty"]["per_30d"] == 3.0
    assert "seed" not in (out / "stress_01" / "public" / "manifest.json").read_text()


def test_two_builds_give_identical_public_tables(mini, tmp_path):
    cfg_dir, out, _, _ = mini
    build_set(cfg_dir, tmp_path / "data2", only=["train_01"], log=lambda m: None)
    a, b = load_public(out / "train_01"), load_public(tmp_path / "data2" / "train_01")
    for name in ("zones", "belt_wide", "layout"):
        assert a[name].equals(b[name]), name


def test_bad_arguments_are_refused(mini, tmp_path):
    cfg_dir, out, _, _ = mini
    with pytest.raises(ValueError, match="workers"):
        build_set(cfg_dir, out, workers=3)
    with pytest.raises(ValueError, match="no config"):
        build_set(cfg_dir, out, only=["nope"])
    with pytest.raises(RuntimeError, match="free"):
        build_set(cfg_dir, tmp_path / "x", min_free_gb=1e9)
    assert not any((tmp_path / "x").glob("*/public"))


def test_demo_seed_search_reports_a_seed_or_the_failed_rules():
    seed, ok, passed = fds.find(1000, 1020)
    assert seed is not None and set(ok) == {r[0] for r in fds.RULES}
    assert passed == all(ok.values())
    note = fds.rules_note(seed, ok, passed, 1000, 1020)
    assert ("passes every rule" in note) == passed and ("**no**" in note) == (not passed)


def test_schedule_only_scan_matches_the_full_generation():
    """The seed finder skips the physics; its schedules must be exactly what the simulator generates."""
    seed = 1003
    cfg, events, hard_negs, sensor, _ = fds.schedules(seed)
    scn = generate(Config(seed=seed, days=fds.DAYS))
    ts = scn.belt_wide["timestamp"]
    assert [(e.asset, ts[e.onset], ts[e.failure]) for e in events] == list(
        zip(scn.faults["asset"], scn.faults["onset_time"], scn.faults["failure_time"]))
    assert [(s.type, s.zone, s.start, s.end) for s in sensor] == [
        (s.type, s.zone, s.start, s.end) for s in scn.sensor_fault_events]
    real = scn.hard_negatives[scn.hard_negatives["type"] != "startup_transient"]
    assert [(h.type, ts[h.start]) for h in hard_negs] == list(zip(real["type"], pd.DatetimeIndex(real["start_time"])))
