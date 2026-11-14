"""Load the saved models and score feature rows. Used by live scoring and by offline evaluation."""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .features import drive_feature_names, zone_feature_names

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
KINDS = ("abnormal", "classifier", "regressor")


class Models:
    def __init__(self, boosters: dict, meta: dict):
        self.boosters, self.meta = boosters, meta
        if meta["zone_features"] != zone_feature_names() or meta["drive_features"] != drive_feature_names():
            raise ValueError("saved models were trained on a different feature list; retrain (make train)")

    @classmethod
    def load(cls, directory=ARTIFACTS):
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text())
        boosters = {(group, kind): lgb.Booster(model_file=str(directory / f"{group}_{kind}.txt"))
                    for group in ("zone", "drive") for kind in KINDS}
        return cls(boosters, meta)

    @property
    def version(self) -> str:
        return self.meta["version"]

    def thresholds(self, group: str) -> dict:
        return self.meta["thresholds"][group]

    def predict(self, group: str, features) -> dict:
        """features: (n, F) -> p_abnormal, p_fail_24h and est_hours (hours to failure, never negative)."""
        x = np.asarray(features, dtype=np.float32)
        return {"p_abnormal": self.boosters[(group, "abnormal")].predict(x, num_threads=1),
                "p_fail_24h": self.boosters[(group, "classifier")].predict(x, num_threads=1),
                "est_hours": np.maximum(self.boosters[(group, "regressor")].predict(x, num_threads=1), 0.0)}

    def top_features(self, group: str, row, k: int = 3) -> list:
        """The k features pushing one row's abnormal score up the most (LightGBM contributions)."""
        names = self.meta[f"{group}_features"]
        contrib = self.boosters[(group, "abnormal")].predict(
            np.asarray(row, dtype=np.float32)[None, :], pred_contrib=True, num_threads=1)[0][:-1]
        order = np.argsort(-contrib)[:k]
        row = np.asarray(row, dtype=np.float64)
        return [{"name": names[i], "contribution": round(float(contrib[i]), 4),
                 "value": None if np.isnan(row[i]) else round(float(row[i]), 4)} for i in order if contrib[i] > 0]
