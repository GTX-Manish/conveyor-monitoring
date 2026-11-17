"""Score the looping-belt layer on a scenario's simulated captures; writes vision/artifacts/dedup_eval.json.

    python -m scripts.score_vision                  # demo_01

Runs the shipped detector (vision/detector.py, frozen threshold) on every capture, feeds the detections to
vision/defects.py in time order (as the backend does live), and compares the resulting locations with the truth.
This script reads ground truth (truth/vision_truth.parquet, truth/vision_locations.json); it is one of the allowed
truth readers (tests/test_leakage.py).

Rules:
- A reported location is a confirmed defect or an entry in the "at splice, inspect" list. It matches a crack if any
  of its captures shows that crack.
- A crack is found if a reported location matches it. The crack placed near a splice counts as found and flagged
  when it is in the at-splice list; it is not missed and not false.
- Duplicates: reported locations matching a crack, beyond the first (target 0).
- False locations: reported locations matching no crack. A detection at a clean splice is a false hit at a splice.
- Laps to confirm: the number of laps the crack had been captured on when its location was confirmed.
- Single-frame false detections: detections in captures showing a clean image (a splice or a clean location).
  "Removed by confirmation": those whose location ended unconfirmed and away from a splice, so never reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from simulator.io import load_captures
from vision.defects import DefectTracker

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "vision" / "artifacts" / "dedup_eval.json"
DEMO_IMAGES = REPO / "data" / "vision_demo"


def run_tracker(captures: pd.DataFrame, scores_of) -> tuple[DefectTracker, dict]:
    """scores_of(image) -> detection scores. Returns the tracker and {capture_id: location_id or None}."""
    tracker = DefectTracker()
    joined: dict[int, int | None] = {}
    for c in captures.sort_values(["ts", "capture_id"]).itertuples():
        loc, merged = tracker.add(int(c.capture_id), float(c.ts), c.station, float(c.belt_coordinate_m),
                                  scores_of(c.image))
        joined[int(c.capture_id)] = loc.location_id if loc else None
        for gone in merged:
            for k, v in joined.items():
                if v == gone:
                    joined[k] = loc.location_id
    return tracker, joined


def score(captures: pd.DataFrame, truth: pd.DataFrame, locations: list, scores_of) -> dict:
    tracker, joined = run_tracker(captures, scores_of)
    t = truth.set_index("capture_id")
    crack_ids = [x["location_id"] for x in locations if x["kind"] == "crack"]
    near = {x["location_id"] for x in locations if x["kind"] == "crack" and x.get("near_splice")}
    reported = {x.location_id: x for x in tracker.defects() + tracker.at_splice()}
    members = {lid: {int(c[0]) for c in loc.captures} for lid, loc in tracker.locations.items()}

    cracks = []
    for cid in crack_ids:
        caps = t.index[t["location_id"] == cid]
        matches = [lid for lid in reported if members[lid] & set(caps)]
        laps_captured = sorted(t.loc[caps, "lap"].unique())
        entry = {"location_id": cid, "near_splice": cid in near,
                 "true_coordinate_m": round(next(x["belt_coordinate_m"] for x in locations if x["location_id"] == cid), 2),
                 "captures": int(len(caps)), "captures_with_detection": int(sum(joined[int(c)] is not None for c in caps)),
                 "laps_captured": len(laps_captured), "found": bool(matches), "reported_as": matches}
        if matches:
            loc = reported[matches[0]]
            entry.update({"reported_coordinate_m": round(loc.coordinate, 2), "laps_seen": loc.laps,
                          "laps_looked": loc.laps_looked,
                          "listed": "at_splice" if loc.at_splice else "defects", "confirmed": loc.confirmed})
            if loc.confirmed_ts is not None:
                caps_before = t.loc[[c for c in caps if captures.set_index("capture_id").loc[c, "ts"] <= loc.confirmed_ts]]
                entry["laps_to_confirm"] = int(caps_before["lap"].nunique())
        cracks.append(entry)

    crack_caps = set(t.index[t["kind"] == "crack"])
    false_locs = [lid for lid in reported if not members[lid] & crack_caps]
    false_at_splice = [lid for lid in false_locs if reported[lid].at_splice]
    clean = t[t["kind"] != "crack"]
    n_false = {int(c): len(scores_of(captures.set_index("capture_id").loc[c, "image"])) for c in clean.index}
    false_caps = [c for c, n in n_false.items() if n]
    final = {lid: loc for lid, loc in tracker.locations.items()}

    def removed(c):
        loc = final[joined[c]]
        return not loc.confirmed and not loc.at_splice
    by_kind = {k: int(sum(n_false[int(c)] for c in clean.index[clean["kind"] == k])) for k in ("splice", "clean")}
    return {
        "scenario_captures": int(len(captures)), "laps": int(truth["lap"].max()) if len(truth) else 0,
        "crack_locations": len(crack_ids),
        "crack_locations_found": sum(c["found"] for c in cracks),
        "near_splice_crack": next(({"found_and_flagged": c["found"] and c.get("listed") == "at_splice",
                                    "location_id": c["location_id"]} for c in cracks if c["near_splice"]), None),
        "duplicates": sum(max(0, len(c["reported_as"]) - 1) for c in cracks),
        "false_locations": len(false_locs),
        "false_locations_at_splice": len(false_at_splice),
        "false_confirmed_defects": len(false_locs) - len(false_at_splice),
        "laps_to_confirm": [c.get("laps_to_confirm") for c in cracks],
        "single_frame_false_detections": {
            "captures_with_a_false_detection": len(false_caps), "clean_captures": int(len(clean)),
            "detections": sum(n_false.values()), "at_splices": by_kind["splice"], "at_clean_locations": by_kind["clean"],
            "removed_by_confirmation": int(sum(n_false[c] for c in false_caps if removed(c))),
            "captures_removed_by_confirmation": int(sum(removed(c) for c in false_caps))},
        "reported": {"defects": [x.as_dict() for x in tracker.defects()],
                     "at_splice": [x.as_dict() for x in tracker.at_splice()],
                     "unconfirmed": len(tracker.unconfirmed())},
        "cracks": cracks,
        "notes": [
            "Each crack lap shows a different frame of a real crack video and each clean capture a randomly drawn "
            "clean crop, so frames are close to independent from lap to lap. A persistent stain or a joint that "
            "fools the model would repeat every lap and survive confirmation, so confirmation looks kinder here than "
            "it would on a real belt. Splices show one fixed image each, which is the persistent case.",
            "One lap is recorded per running day (an assumption of the simulator), seen by all three stations.",
        ],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default="demo_01")
    ap.add_argument("--data", default=str(REPO / "data"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    from vision.detector import Detector

    root = Path(args.data) / args.scenario
    captures = load_captures(root)
    truth = pd.read_parquet(root / "truth" / "vision_truth.parquet")
    locations = json.loads((root / "truth" / "vision_locations.json").read_text())
    det = Detector(threads=args.threads)
    cache: dict[str, list[float]] = {}

    def scores_of(image: str) -> list[float]:
        if image not in cache:
            cache[image] = [d.score for d in det.detect(DEMO_IMAGES / image)]
        return cache[image]

    result = {"scenario": args.scenario, "threshold": det.conf, **score(captures, truth, locations, scores_of)}
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("reported", "cracks", "notes")}, indent=1))


if __name__ == "__main__":
    main()
