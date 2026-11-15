"""Guards against ground truth or simulator internals reaching pipeline data."""
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from simulator.config import Config
from simulator.generate import generate
from simulator.io import PUBLIC_COLUMNS, LeakageError, check_public_columns, load_public, write_scenario

REPO = Path(__file__).resolve().parents[1]
# Live-path and model code. Folders that do not exist yet are skipped.
PIPELINE_DIRS = ["backend", "replayer", "frontend/src", "common", "model", "vision", "scripts"]
# The only files outside simulator/ that may read ground truth (BUILD_PLAN.md section 2).
TRUTH_READERS = {"scripts/find_demo_seed.py", "scripts/score.py", "scripts/make_model_data.py", "scripts/robustness.py",
                 "replayer/work_orders.py"}   # repair records, released only after each repair (see its docstring)
OTHER_TRUTH_FILES = ["layout_truth", "faults.parquet", "hard_negatives.parquet", "sensor_faults.parquet",
                     "conditions.parquet"]


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    cfg = Config(scenario="leak", seed=2, days=1)
    scn = generate(cfg)
    root = write_scenario(scn, cfg, tmp_path_factory.mktemp("data"))
    return scn, root


def test_public_folder_holds_only_allowlisted_files(written):
    _, root = written
    files = sorted(p.name for p in (root / "public").iterdir())
    assert files == sorted([f"{n}.parquet" for n in PUBLIC_COLUMNS] + ["manifest.json"])


def test_truth_files_are_not_public(written):
    _, root = written
    truth = {p.name for p in (root / "truth").iterdir()}
    assert {"layout_truth.parquet", "conditions.parquet", "config_used.yaml", "run_meta.json"} <= truth
    assert not truth & {p.name for p in (root / "public").iterdir()}


def test_public_columns_match_allowlist(written):
    _, root = written
    for name, cols in PUBLIC_COLUMNS.items():
        assert list(pd.read_parquet(root / "public" / f"{name}.parquet").columns) == cols


def test_hidden_parameters_absent_from_public(written):
    scn, _ = written
    hidden = (set(scn.layout_truth.columns) | set(scn.conditions.columns)) - {"global_zone", "timestamp"}
    for name in PUBLIC_COLUMNS:
        assert not hidden & set(getattr(scn, name).columns), name


def test_injected_label_column_is_rejected(written):
    scn, _ = written
    bad = scn.belt_wide.assign(fault_flag=0)
    with pytest.raises(LeakageError):
        check_public_columns("belt_wide", bad)
    renamed = scn.layout.rename(columns={"incline_deg": "temp_bias_c"})
    with pytest.raises(LeakageError):
        check_public_columns("layout", renamed)


def test_load_public_never_needs_truth(written, tmp_path):
    _, root = written
    copy = tmp_path / "copy"
    shutil.copytree(root / "public", copy / "public")  # no truth/ folder at all
    tables = load_public(copy)
    assert set(tables) == set(PUBLIC_COLUMNS) | {"manifest"}
    manifest_text = (copy / "public" / "manifest.json").read_text()
    assert "seed" not in manifest_text


def test_pipeline_code_never_touches_truth():
    """Static guard: live-path and model code may not reference truth files or simulator internals."""
    pattern = re.compile(r"""truth|layout_truth|labels\.parquet|faults\.parquet|hard_negatives\.parquet|"""
                         r"""sensor_faults\.parquet|conditions\.parquet|simulator\.generate""")
    offenders = []
    for d in PIPELINE_DIRS:
        if not (REPO / d).is_dir():
            continue
        for path in (REPO / d).rglob("*"):
            rel = str(path.relative_to(REPO))
            if path.suffix not in {".py", ".ts", ".tsx"} or rel in TRUTH_READERS:
                continue
            if pattern.search(path.read_text()):
                offenders.append(rel)
    assert not offenders, f"pipeline code references ground truth: {offenders}"


def test_model_data_script_reads_labels_only():
    """Training needs labels; it must not read any other truth table (conditions are never a model input)."""
    path = REPO / "scripts" / "make_model_data.py"
    if not path.exists():
        pytest.skip("scripts/make_model_data.py is not built yet")
    text = path.read_text()
    assert not [name for name in OTHER_TRUTH_FILES if name in text]


def test_work_orders_read_only_what_a_maintenance_system_knows(written):
    """The repair feed may know which asset failed and when, never when the fault started, and never early."""
    from replayer import work_orders as wo
    text = (REPO / "replayer" / "work_orders.py").read_text()
    assert wo.COLUMNS == ["asset", "type", "failure_time", "repair_end"]
    for forbidden in ("onset", "labels.parquet", "hard_negatives", "sensor_faults", "conditions", "layout_truth",
                      "severity", "stage"):
        assert forbidden not in text, forbidden
    orders = [{"asset": "Z005", "kind": "idler_bearing", "failed_ts": 1000, "repaired_ts": 8200},
              {"asset": "DRIVE", "kind": "drive_overheat", "failed_ts": 500, "repaired_ts": 7700}]
    feed = wo.WorkOrderFeed(orders)
    assert feed.due(7699) == []                                  # nothing before a repair has ended
    assert [o["asset"] for o in feed.due(7700)] == ["DRIVE"]     # exactly at the end of the repair
    assert feed.due(8199) == [] and [o["asset"] for o in feed.due(9000)] == ["Z005"] and feed.due(10**9) == []
    _, root = written
    for order in wo.load_work_orders(root):
        assert set(order) == {"asset", "kind", "failed_ts", "repaired_ts"}


def test_only_the_repair_feed_and_scoring_touch_work_orders():
    """Repair records must never reach the features or the model."""
    for name in ("model/features.py", "model/predict.py", "model/train.py", "model/alerting.py", "model/explain.py"):
        assert "work_order" not in (REPO / name).read_text(), name
