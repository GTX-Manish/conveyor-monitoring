"""Train the failure models and tune their alert thresholds.

    python -m model.train                 # trains on the train split, tunes on validation, saves artifacts

Two groups: a zone model for the 120 idler zones and a drive model (different signals, very few events).
Each group has three LightGBM models:
  abnormal    advanced-stage windows against normal ones. The early stage is left out: by design it is below
              the noise (about the first 62% of a fault's life), so labelling it positive teaches noise.
  classifier  will this asset fail within 24 hours.
  regressor   hours to failure, trained on advanced-stage windows only.
Scenarios are split whole; no scenario gives rows to two splits.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from common.ids import ASSETS, DRIVE, N_ZONES

from .alerting import CALL_HORIZON_HOURS, HOLD_P_FAIL
from .evaluate import ADVANCED, FAILED, NORMAL, TARGET_PRECISION, tune_call_threshold, tune_threshold
from .features import WINDOW_MIN, drive_feature_names, zone_feature_names
from .predict import ARTIFACTS, KINDS

NORMAL_KEEP = 0.2          # (assumption) share of normal zone windows kept for training; all others are kept
PARAMS = {
    "zone": dict(num_leaves=31, min_child_samples=200, learning_rate=0.05, n_estimators=300,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0),
    # the drive has about 8 training events: a small model, or it memorises them
    "drive": dict(num_leaves=7, min_child_samples=30, learning_rate=0.05, n_estimators=150,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0),
}
MAX_POS_WEIGHT = 20.0      # cap on the class weight for the rare positive class


def load_group(data_dir: Path, names: list, group: str) -> pd.DataFrame:
    return pd.concat([pd.read_parquet(data_dir / "model" / f"{n}_{group}.parquet") for n in names],
                     ignore_index=True)


def training_rows(data_dir: Path, names: list, group: str, seed: int) -> pd.DataFrame:
    """Train split rows; for zones the normal windows are thinned one scenario at a time to fit in memory."""
    rng = np.random.default_rng(seed)
    parts = []
    for n in names:
        df = pd.read_parquet(data_dir / "model" / f"{n}_{group}.parquet")
        if group == "zone":
            plain = (df["stage"] == NORMAL).to_numpy()
            df = df[~plain | (rng.random(len(df)) < NORMAL_KEEP)]
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def fit_group(df: pd.DataFrame, group: str, seed: int) -> dict:
    feats = zone_feature_names() if group == "zone" else drive_feature_names()
    p = PARAMS[group]
    common = dict(num_leaves=p["num_leaves"], min_data_in_leaf=p["min_child_samples"],
                  learning_rate=p["learning_rate"], bagging_fraction=p["subsample"],
                  bagging_freq=p["subsample_freq"], feature_fraction=p["colsample_bytree"],
                  lambda_l2=p["reg_lambda"], seed=seed, num_threads=2, verbose=-1)
    stage = df["stage"].to_numpy()
    x = df[feats].to_numpy(dtype=np.float32)

    def fit(mask, y, **objective):
        data = lgb.Dataset(x[mask], label=np.asarray(y)[mask], feature_name=feats)
        return lgb.train({**common, **objective}, data, num_boost_round=p["n_estimators"])

    def classifier(mask, y):
        pos = int(y[mask].sum())
        weight = min((mask.sum() - pos) / max(pos, 1), MAX_POS_WEIGHT)
        return fit(mask, y.astype(np.float32), objective="binary", scale_pos_weight=weight)

    return {"abnormal": classifier(np.isin(stage, (NORMAL, ADVANCED)), stage == ADVANCED),
            "classifier": classifier(stage != FAILED, df["fails_within_24h"].to_numpy(dtype=bool)),
            "regressor": fit(stage == ADVANCED, df["hours_to_failure"].to_numpy(dtype=np.float32), objective="l1")}


def grids(df: pd.DataFrame, boosters: dict, group: str) -> dict:
    """One scenario's model table -> (W, A) grids of probabilities and labels for alert scoring."""
    feats = zone_feature_names() if group == "zone" else drive_feature_names()
    A = N_ZONES if group == "zone" else 1
    x = df[feats].to_numpy(dtype=np.float32)
    shape = lambda v: np.asarray(v).reshape(-1, A)  # noqa: E731
    run_col = "run_share"
    out = {
        "p_abnormal": shape(boosters["abnormal"].predict(x, num_threads=2)),
        "p_fail_24h": shape(boosters["classifier"].predict(x, num_threads=2)),
        "est_hours": shape(np.maximum(boosters["regressor"].predict(x, num_threads=2), 0.0)),
        "stage": shape(df["stage"].to_numpy()), "fault_id": shape(df["fault_id"].to_numpy(dtype=object)),
        "hours_to_failure": shape(df["hours_to_failure"].to_numpy()),
        "hard_negative": shape(df["is_hard_negative_period"].to_numpy()),
        "run_share": shape(df[run_col].to_numpy())[:, 0],
    }
    out["months"] = out["stage"].shape[0] * WINDOW_MIN / (60 * 24 * 30)
    return out


def candidates() -> np.ndarray:
    return np.round(np.concatenate([np.arange(0.05, 0.9, 0.05), np.arange(0.9, 0.995, 0.01),
                                    [0.995, 0.997, 0.999]]), 3)


def tune(val_grids: list) -> dict:
    """Thresholds from the validation scenarios, frozen afterwards.

    `abnormal` opens the alert. `predicted_failure` is the failure call inside an open alert; it is tuned on
    whether the part really failed within the claimed window (CALL_HORIZON_HOURS) after the call.
    """
    abn = tune_threshold([dict(prob=g["p_abnormal"], run_share=g["run_share"], stage=g["stage"],
                               fault_id=g["fault_id"], months=g["months"], hold=g["p_fail_24h"] >= HOLD_P_FAIL)
                          for g in val_grids], candidates())
    fail = tune_call_threshold(val_grids, abn["threshold"], candidates())
    drop = ("sweep", "hours_list", "eta_error_list")
    return {"abnormal": abn["threshold"], "predicted_failure": fail["threshold"],
            "abnormal_met_target": abn["met_target"], "predicted_failure_met_target": fail["met_target"],
            "abnormal_validation": {k: v for k, v in abn.items() if k not in drop},
            "predicted_failure_validation": {k: v for k, v in fail.items() if k not in drop},
            "abnormal_sweep": abn["sweep"],
            "predicted_failure_sweep": [{k: v for k, v in r.items() if k not in drop} for r in fail["sweep"]]}


def train(data_dir: Path, seed: int = 1, log=print) -> dict:
    index = json.loads((data_dir / "model" / "index.json").read_text())
    split = lambda s: sorted(n for n, r in index.items() if r["split"] == s)  # noqa: E731
    out = {"boosters": {}, "thresholds": {}, "val_grids": {}, "splits": {s: split(s) for s in
                                                                         ("train", "validation", "test", "stress")}}
    for group in ("zone", "drive"):
        t0 = time.perf_counter()
        df = training_rows(data_dir, split("train"), group, seed)
        boosters = fit_group(df, group, seed)
        rows = len(df)
        del df
        val = [grids(pd.read_parquet(data_dir / "model" / f"{n}_{group}.parquet"), boosters, group)
               for n in split("validation")]
        out["boosters"][group], out["thresholds"][group], out["val_grids"][group] = boosters, tune(val), val
        t = out["thresholds"][group]
        log(f"{group}: {rows:,} training rows, {time.perf_counter() - t0:.0f} s; abnormal threshold "
            f"{t['abnormal']} (precision {t['abnormal_validation']['precision']:.2f}, "
            f"{t['abnormal_validation']['detected']}/{t['abnormal_validation']['events']} events on validation, "
            f"target met: {t['abnormal_met_target']}); failure-call threshold {t['predicted_failure']} "
            f"({t['predicted_failure_validation']['came_true']} of {t['predicted_failure_validation']['calls']} calls "
            f"came true within {CALL_HORIZON_HOURS:.0f} h on validation, target met: {t['predicted_failure_met_target']})")
    return out


def save(result: dict, seed: int, directory=ARTIFACTS) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for group in ("zone", "drive"):
        for kind in KINDS:
            result["boosters"][group][kind].save_model(str(directory / f"{group}_{kind}.txt"))
    thresholds = {g: {k: v for k, v in t.items() if not k.endswith("_sweep")} for g, t in result["thresholds"].items()}
    meta = {"version": f"lgbm-w{WINDOW_MIN}-seed{seed}", "window_minutes": WINDOW_MIN, "seed": seed,
            "zone_features": zone_feature_names(), "drive_features": drive_feature_names(),
            "assets": len(ASSETS), "drive_asset": DRIVE, "target_precision": TARGET_PRECISION,
            "call_horizon_hours": CALL_HORIZON_HOURS,
            "normal_keep": NORMAL_KEEP, "params": PARAMS, "splits": result["splits"], "thresholds": thresholds}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train models and tune thresholds.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    result = train(Path(args.data), args.seed)
    save(result, args.seed)
    print(f"saved models and thresholds to {ARTIFACTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
