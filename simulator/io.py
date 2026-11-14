"""Writing and reading scenario folders, with the public / truth split enforced in code.

Layout on disk:

    data/<scenario>/public/   what the pipeline (backend, model, dashboard) may read
    data/<scenario>/truth/    ground truth and simulator internals; the pipeline must never read it

Every public table is checked against an explicit column allowlist before it is written, so a label,
fault flag or hidden simulator parameter cannot slip into pipeline data by accident. Pipeline code must
load data only through `load_public`, which never opens `truth/`.
"""
import json
from pathlib import Path

import pandas as pd

from .config import Config, dump_config
from .healthy import BELT_SIGNALS, ZONE_SIGNALS

PUBLIC_DIR = "public"
TRUTH_DIR = "truth"

PUBLIC_COLUMNS = {
    "zones": ["timestamp", "splice_section", "zone_in_section", "global_zone", "position_m", "run",
              "run_state", *ZONE_SIGNALS],
    "belt_wide": ["timestamp", "run_state", *BELT_SIGNALS],
    "layout": ["global_zone", "splice_section", "zone_in_section", "position_m", "run", "is_loading_zone",
               "is_unloading_zone", "is_curve", "curve_radius_m", "incline_deg"],
}

# Substrings that must never appear in a public column name.
FORBIDDEN_TOKENS = ["fault", "label", "stage", "severity", "failure", "truth", "hard_negative", "bias",
                    "vib_scale", "ambient", "dust", "load", "speed_frac", "repair", "onset"]
# Legitimate public names that happen to contain a forbidden token.
TOKEN_EXCEPTIONS = {"is_loading_zone", "is_unloading_zone"}


class LeakageError(RuntimeError):
    pass


def check_public_columns(name: str, df: pd.DataFrame) -> None:
    allowed = PUBLIC_COLUMNS[name]
    if list(df.columns) != allowed:
        raise LeakageError(f"public table {name!r} columns {list(df.columns)} != allowlist {allowed}")
    for col in df.columns:
        if col in TOKEN_EXCEPTIONS:
            continue
        hit = [tok for tok in FORBIDDEN_TOKENS if tok in col.lower()]
        if hit:
            raise LeakageError(f"public table {name!r} column {col!r} contains forbidden token(s) {hit}")


def write_scenario(scn, cfg: Config, out_root, manifest_extra=None) -> Path:
    root = Path(out_root) / cfg.scenario
    pub, tru = root / PUBLIC_DIR, root / TRUTH_DIR
    pub.mkdir(parents=True, exist_ok=True)
    tru.mkdir(parents=True, exist_ok=True)

    for name in PUBLIC_COLUMNS:
        df = getattr(scn, name)
        check_public_columns(name, df)
        df.to_parquet(pub / f"{name}.parquet", index=False)
    # Public manifest: only what a real plant would know about its own data feed.
    manifest = {"scenario": cfg.scenario, "start": cfg.start, "step_seconds": cfg.step_seconds,
                "n_timestamps": scn.meta["n_timestamps"], "n_zones": scn.meta["n_zones"],
                "files": sorted(f"{n}.parquet" for n in PUBLIC_COLUMNS), **(manifest_extra or {})}
    (pub / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Truth: hidden simulator parameters, conditions (analysis only), exact config and run metadata.
    # config_used/run_meta live here because from stage 2 on they carry fault rates and counts.
    scn.layout_truth.to_parquet(tru / "layout_truth.parquet", index=False)
    scn.conditions.to_parquet(tru / "conditions.parquet", index=False)
    scn.faults.to_parquet(tru / "faults.parquet", index=False)
    scn.hard_negatives.to_parquet(tru / "hard_negatives.parquet", index=False)
    scn.labels.to_parquet(tru / "labels.parquet", index=False)
    scn.sensor_faults.to_parquet(tru / "sensor_faults.parquet", index=False)
    dump_config(cfg, tru / "config_used.yaml")
    (tru / "run_meta.json").write_text(json.dumps(scn.meta, indent=2))
    return root


def load_public(scenario_dir) -> dict:
    """The only loader pipeline code may use. Reads `public/` and nothing else."""
    pub = Path(scenario_dir) / PUBLIC_DIR
    if not pub.is_dir():
        raise FileNotFoundError(f"no public folder in {scenario_dir}")
    tables = {name: pd.read_parquet(pub / f"{name}.parquet") for name in PUBLIC_COLUMNS}
    for name, df in tables.items():
        check_public_columns(name, df)
    tables["manifest"] = json.loads((pub / "manifest.json").read_text())
    return tables
