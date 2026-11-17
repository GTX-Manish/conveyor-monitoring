"""Vision on the live path: run the crack detector on each camera capture and keep one record per belt location.

Called by the engine (Engine.handle("captures", ...)), under its writer lock. Writes `captures`,
`vision_detections` and `vision_defects`; the API reads those tables. The de-duplication rules are in
vision/defects.py, shared with scripts/score_vision.py.

Vision is off, with the reason, when the model file, onnxruntime or the demo image folder is missing. Captures sent
then are refused, not stored, and the rest of the system runs as before.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
from pathlib import Path

from vision.defects import DefectTracker

log = logging.getLogger("vision")


class VisionService:
    def __init__(self, settings: dict, repo: Path):
        cfg = settings.get("vision") or {}
        self.cfg = cfg
        self.repo = Path(repo)
        self.model_path = self.repo / cfg.get("model", "vision/artifacts/best.onnx")
        self.images_dir = (self.repo / cfg.get("images_dir", "data/vision_demo")).resolve()
        self.eval_path = self.repo / cfg.get("eval", "vision/artifacts/vision_eval.json")
        self.belt_length = float(cfg.get("belt_length_m", 6000.0))
        self.threads = int(cfg.get("threads", 1))
        self.reason = None
        if not cfg:
            self.reason = "No vision settings in configs/app.yaml."
        elif not self.model_path.is_file():
            self.reason = f"The crack model ({self.model_path.relative_to(self.repo)}) is missing."
        elif importlib.util.find_spec("onnxruntime") is None:
            self.reason = "onnxruntime is not installed (pip install -r requirements.txt)."
        elif not self.images_dir.is_dir():
            self.reason = f"The camera images folder ({cfg.get('images_dir')}) is missing."
        self.enabled = self.reason is None
        self.tracker = DefectTracker(self.belt_length, float(cfg.get("splice_spacing_m", 500.0)),
                                     float(cfg.get("merge_m", 2.0)), float(cfg.get("splice_m", 1.5)))
        self.detector = None                     # loaded on the first capture
        self.cache: dict[str, tuple] = {}        # image -> (width, height, detections): the model is deterministic
        self.counts = dict.fromkeys(("captures", "rejected", "duplicates", "image_missing", "with_detections",
                                     "detections"), 0)

    # ------------------------------------------------------------------ helpers
    def threshold(self):
        if self.detector is not None:
            return self.detector.conf
        from vision.detector import frozen_threshold
        return frozen_threshold(self.eval_path)

    def _image(self, rel) -> Path | None:
        if not isinstance(rel, str) or not rel:
            return None
        path = (self.images_dir / rel).resolve()
        if self.images_dir not in path.parents or not path.is_file():   # nothing outside the image folder is served
            return None
        return path

    def image_path(self, rel) -> Path | None:
        return self._image(rel)

    def _detect(self, rel: str, path: Path):
        if rel not in self.cache:
            if self.detector is None:
                from vision.detector import Detector
                self.detector = Detector(self.model_path, threads=self.threads)
            from PIL import Image
            with Image.open(path) as im:
                im.load()
                dets = self.detector.detect(im)
                self.cache[rel] = (im.width, im.height, dets)
        return self.cache[rel]

    def _parse(self, c):
        if not isinstance(c, dict):
            return None
        cid, ts, station = c.get("capture_id"), c.get("ts"), c.get("station")
        try:
            ground, coord = float(c.get("ground_position_m")), float(c.get("belt_coordinate_m"))
        except (TypeError, ValueError):
            return None
        if (cid.__class__ is not int or cid < 1 or ts.__class__ is not int or ts < 0 or not isinstance(station, str)
                or not math.isfinite(ground) or not math.isfinite(coord) or not 0 <= coord < self.belt_length):
            return None
        return cid, ts, station, ground, coord, c.get("image")

    # ------------------------------------------------------------------ the write path
    def ingest(self, con, captures: list) -> dict:
        if not self.enabled:
            return {"enabled": False, "reason": self.reason, "accepted": 0}
        out = dict.fromkeys(("accepted", "rejected", "duplicates", "image_missing", "detections"), 0)
        for c in captures if isinstance(captures, list) else []:
            row = self._parse(c)
            if row is None:
                out["rejected"] += 1
                continue
            cid, ts, station, ground, coord, rel = row
            if con.execute("SELECT 1 FROM captures WHERE capture_id = ?", (cid,)).fetchone():
                out["duplicates"] += 1
                continue
            path = self._image(rel)
            if path is None:
                con.execute("INSERT INTO captures (capture_id, ts, station, ground_position_m, belt_coordinate_m, "
                            "image_path, status, n_detections) VALUES (?,?,?,?,?,?,'image_missing',0)",
                            (cid, ts, station, ground, coord, rel if isinstance(rel, str) else None))
                out["image_missing"] += 1
                out["accepted"] += 1
                continue
            w, h, dets = self._detect(rel, path)
            con.execute("INSERT INTO captures (capture_id, ts, station, ground_position_m, belt_coordinate_m, image_path,"
                        " image_w, image_h, status, n_detections) VALUES (?,?,?,?,?,?,?,?,'ok',?)",
                        (cid, ts, station, ground, coord, rel, w, h, len(dets)))
            loc, merged = self.tracker.add(cid, ts, station, coord, [d.score for d in dets])
            if dets:
                con.executemany("INSERT INTO vision_detections (capture_id, ts, label, score, belt_coordinate_m, "
                                "ground_position_m, bbox, defect_id) VALUES (?,?,?,?,?,?,?,?)",
                                [(cid, ts, "crack", d.score, coord, ground, json.dumps(d.as_dict()), loc.location_id)
                                 for d in dets])
                con.execute("UPDATE captures SET defect_id = ? WHERE capture_id = ?", (loc.location_id, cid))
                for gone in merged:
                    con.execute("UPDATE vision_detections SET defect_id = ? WHERE defect_id = ?", (loc.location_id, gone))
                    con.execute("UPDATE captures SET defect_id = ? WHERE defect_id = ?", (loc.location_id, gone))
                    con.execute("DELETE FROM vision_defects WHERE defect_id = ?", (gone,))
                self._save(con, loc)
                out["detections"] += len(dets)
            else:
                here = self.tracker._nearest(coord)
                if here is not None:                 # a look with no detection: laps_looked may change
                    self._save(con, here)
            out["accepted"] += 1
        for k in ("rejected", "duplicates", "image_missing", "detections"):
            self.counts[k] += out[k]
        self.counts["captures"] += out["accepted"]
        self.counts["with_detections"] = con.execute("SELECT COUNT(*) FROM captures WHERE n_detections > 0").fetchone()[0]
        return {"enabled": True, **out}

    def _save(self, con, loc) -> None:
        d = loc.as_dict()
        con.execute(
            "INSERT INTO vision_defects VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(defect_id) DO UPDATE SET "
            "belt_coordinate_m=excluded.belt_coordinate_m, laps=excluded.laps, laps_looked=excluded.laps_looked, "
            "stations=excluded.stations, n_captures=excluded.n_captures, n_detections=excluded.n_detections, "
            "first_ts=excluded.first_ts, last_ts=excluded.last_ts, confirmed=excluded.confirmed, "
            "confirmed_ts=excluded.confirmed_ts, best_capture_id=excluded.best_capture_id, best_score=excluded.best_score, "
            "at_splice=excluded.at_splice, splice_m=excluded.splice_m, splice_distance_m=excluded.splice_distance_m",
            (d["defect_id"], d["belt_coordinate_m"], d["laps"], d["laps_looked"], json.dumps(d["stations"]),
             d["n_captures"], d["n_detections"], d["first_ts"], d["last_ts"], int(d["confirmed"]), d["confirmed_ts"],
             d["best_capture_id"], d["best_score"], int(d["at_splice"]), d["splice_m"], d["splice_distance_m"]))

    # ------------------------------------------------------------------ read side helpers
    def evaluation(self) -> dict | None:
        """The detector's offline numbers (vision_eval.json) and the model credit, for the dashboard."""
        try:
            ev = json.loads(self.eval_path.read_text())
        except (OSError, ValueError):
            return None
        test, model = ev.get("test", {}), ev.get("model", {})
        frozen = test.get("at_frozen_threshold", {})
        return {"split": "BeltCrack9kd test", "images": test.get("images"), "map50": test.get("map50"),
                "precision": frozen.get("precision"), "recall": frozen.get("recall"),
                "threshold": ev.get("freeze", {}).get("threshold"), "credit": model.get("credit"),
                "license": "AGPL-3.0", "source_repo": model.get("source_repo"),
                "dataset": "BeltCrack (Huang, Ji, Ma and Ye, Pattern Recognition 2026), CC BY-SA 4.0"}
