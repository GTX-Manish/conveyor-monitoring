"""Guards against ground truth or simulator internals reaching pipeline data."""
import re
import shutil
from pathlib import Path

import pandas as pd
import pytest

from simulator.config import Config
from simulator.generate import generate
from simulator.io import (PUBLIC_COLUMNS, LeakageError, check_public_columns, load_public,
                          write_scenario)

REPO = Path(__file__).resolve().parents[1]
PIPELINE_DIRS = ["backend", "model", "vision", "dashboard"]


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
    """Static guard for later stages: pipeline packages may not reference truth files or simulator internals."""
    pattern = re.compile(r"""truth|layout_truth|labels\.parquet|faults\.parquet|hard_negatives\.parquet|"""
                         r"""sensor_faults\.parquet|conditions\.parquet|simulator\.generate""")
    offenders = []
    for d in PIPELINE_DIRS:
        for path in (REPO / d).rglob("*.py") if (REPO / d).is_dir() else []:
            if pattern.search(path.read_text()):
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, f"pipeline code references ground truth: {offenders}"
