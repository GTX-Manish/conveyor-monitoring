"""Build the model tables: one row per (asset, 10-minute window) with features and labels.

    python -m scripts.make_model_data            # every scenario in data/scenario_set.json
    python -m scripts.make_model_data --only val_01

Features come from the public tables through model/features.py (the same code live scoring uses). Labels come
from each scenario's labels.parquet, the one ground-truth table this script may read, because training needs
labels. One scenario is in memory at a time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.ids import ASSETS, BELT_SIGNALS, DRIVE, N_ZONES, RUN_STATES, ZONE_SIGNALS, simulator_asset
from model.features import (
    HISTORY_WINDOWS,
    WINDOW_MIN,
    drive_feature_names,
    rolling_features,
    to_windows,
    window_aggregates,
    zone_feature_names,
)

REPO = Path(__file__).resolve().parents[1]
LABEL_COLUMNS = ["stage", "fails_within_24h", "hours_to_failure", "fault_id", "is_hard_negative_period"]


def public_blocks(scenario_dir: Path):
    """Public tables -> arrays shaped for `window_aggregates`, plus the layout facts and window end times."""
    pub = Path(scenario_dir) / "public"
    zones = pd.read_parquet(pub / "zones.parquet", columns=["global_zone", *ZONE_SIGNALS])
    belt = pd.read_parquet(pub / "belt_wide.parquet")
    layout = pd.read_parquet(pub / "layout.parquet").sort_values("global_zone")
    T = len(belt)
    assert len(zones) == T * N_ZONES and (zones["global_zone"].to_numpy()[:N_ZONES] == np.arange(1, N_ZONES + 1)).all()
    zone_vals = {s: to_windows(zones[s].to_numpy(dtype=np.float64).reshape(T, N_ZONES)) for s in ZONE_SIGNALS}
    belt_vals = {s: to_windows(belt[s].to_numpy(dtype=np.float64)) for s in BELT_SIGNALS}
    codes = belt["run_state"].astype(str).map({name: i for i, name in enumerate(RUN_STATES)}).to_numpy()
    facts = {"is_carry": (layout["run"] == "carry").to_numpy(), "is_loading": layout["is_loading_zone"].to_numpy(),
             "is_unloading": layout["is_unloading_zone"].to_numpy()}
    window_end = to_windows(belt["timestamp"].to_numpy())[:, -1]
    return zone_vals, belt_vals, to_windows(codes), facts, window_end


def window_labels(scenario_dir: Path, n_windows: int) -> dict:
    """Labels per (window, asset): worst stage in the window; the other labels as they stand at the window end."""
    lab = pd.read_parquet(Path(scenario_dir) / "truth" / "labels.parquet")
    A = N_ZONES + 1
    names = list(lab["asset"].cat.categories)
    assert [simulator_asset(a) for a in names] == ASSETS, "label assets do not map onto asset ids in order"
    grid = lambda col: to_windows(col.reshape(-1, A))[:n_windows]  # noqa: E731
    fault_ids = np.array(["", *lab["fault_id"].cat.categories], dtype=object)
    return {
        "stage": grid(lab["stage"].cat.codes.to_numpy()).max(axis=1).astype(np.int8),
        "fails_within_24h": grid(lab["fails_within_24h"].to_numpy())[:, -1],
        "hours_to_failure": grid(lab["hours_to_failure"].to_numpy())[:, -1].astype(np.float32),
        "fault_id": fault_ids[grid(lab["fault_id"].cat.codes.to_numpy())[:, -1] + 1],
        "is_hard_negative_period": grid(lab["is_hard_negative_period"].to_numpy()).any(axis=1),
    }


def build(scenario_dir: Path, out_dir: Path, name: str, groups=("zone", "drive")) -> dict:
    zone_vals, belt_vals, run_state, facts, window_end = public_blocks(scenario_dir)
    table = window_aggregates(zone_vals, belt_vals, run_state, facts)
    del zone_vals, belt_vals
    feats = rolling_features(table, facts)
    W = len(window_end)
    labels = window_labels(scenario_dir, W)
    keep = slice(HISTORY_WINDOWS - 1, W)   # rows with a full 24 h of history, as in live scoring
    n = W - (HISTORY_WINDOWS - 1)

    zone = pd.DataFrame(feats["zone"][keep].reshape(n * N_ZONES, -1), columns=zone_feature_names())
    zone.insert(0, "asset", pd.Categorical.from_codes(np.tile(np.arange(N_ZONES), n), categories=ASSETS))
    zone.insert(0, "window_end", np.repeat(window_end[keep], N_ZONES))
    drive = pd.DataFrame(feats["drive"][keep], columns=drive_feature_names())
    drive.insert(0, "asset", DRIVE)
    drive.insert(0, "window_end", window_end[keep])
    for col in LABEL_COLUMNS:
        zone[col] = labels[col][keep, :N_ZONES].reshape(-1)
        drive[col] = labels[col][keep, N_ZONES]
    for frame in (zone, drive):
        frame.insert(0, "scenario", name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if "zone" in groups:
        zone.to_parquet(out_dir / f"{name}_zone.parquet", index=False)
    if "drive" in groups:
        drive.to_parquet(out_dir / f"{name}_drive.parquet", index=False)
    return {"name": name, "windows": n, "zone_rows": len(zone),
            "advanced_rows": int((zone["stage"] == 2).sum()), "fail24_rows": int(zone["fails_within_24h"].sum())}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build model tables from scenarios.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--only", nargs="+")
    args = ap.parse_args(argv)
    data = Path(args.data)
    rows = json.loads((data / "scenario_set.json").read_text())["scenarios"]
    path = data / "model" / "index.json"
    index = json.loads(path.read_text()) if path.exists() else {}
    for r in rows:
        if args.only and r["name"] not in args.only:
            continue
        t0 = time.perf_counter()
        # scenarios made only to train the drive model get no zone table: nothing may train a zone model on them
        groups = ("drive",) if r["split"] == "train_drive" else ("zone", "drive")
        info = build(data / r["name"], data / "model", r["name"], groups)
        index[r["name"]] = {**info, "split": r["split"]}
        path.write_text(json.dumps(index, indent=2))   # after every scenario, so an interrupted run loses little
        print(f"{r['name']}: {info['zone_rows']:,} zone rows, {info['advanced_rows']:,} advanced, "
              f"{info['fail24_rows']:,} within 24 h, {time.perf_counter() - t0:.1f} s (window {WINDOW_MIN} min)",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
