"""Choose the lap-ratio confirmation rule (K, R) on a separate simulator run; writes vision/artifacts/dedup_tuning.json.

    python -m scripts.tune_dedup              # needs BeltCrack9kd_prep/ (vision/prepare_beltcrack9kd.py)

The rule (vision/defects.py): a location is confirmed while it has been seen on at least K laps and on at least a
fraction R of the laps it has been looked at.

The tuning run is the demo's generator and config with another seed (TUNE_SEED, used by no other scenario), so it
has its own crack, splice-image and clean placements, frames and noise. demo_01 is never generated or read here.
Crack frames are rendered from the dataset exactly as data/vision_demo/ is built (simulator.vision.crack_frame_jpeg);
clean images come from the same committed pool of 16 crops (there is no other).

Selection, fixed before the run:
1. A crack is detectable if the detector fires on its captures on at least 2 laps (no rule with K >= 2 can confirm
   anything less).
2. Keep the rules that find every detectable crack: a reported location (confirmed defect or "at a splice" entry)
   contains a capture of it.
3. Among those, keep the rules with the fewest false confirmed defects (reported away from a splice, showing no
   crack). The splice inspect list does not depend on the rule.
4. Simplest: the smallest K; then, among the R values tied for that K, the middle of the tied range (rounded down to
   the 0.05 grid), so R is not on the edge of what works.
Each rule is judged on the state at the end of the run (14 days, 14 recorded laps).
This script reads ground truth (the tuning run's own); it is one of the allowed truth readers (tests/test_leakage.py).
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

from scripts.score_vision import score
from simulator import vision as sv
from simulator.config import config_from_dict, config_to_dict, load_config
from simulator.generate import generate

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "vision" / "artifacts" / "dedup_tuning.json"
TUNE_SEED = 2027
K_GRID = (2, 3, 4)
R_GRID = tuple(round(x, 2) for x in np.arange(0.0, 0.951, 0.05))


def tuning_scenario():
    cfg = load_config(REPO / "configs" / "scenarios" / "demo_01.yaml")
    return config_from_dict({**config_to_dict(cfg), "seed": TUNE_SEED, "scenario": "dedup_tuning"})


def choose(table: list, detectable: set) -> dict:
    works = [r for r in table if detectable <= set(r["found_ids"])]
    if not works:
        return {"k": None, "r": None, "why": "no rule finds every detectable crack"}
    fewest = min(r["false_confirmed_defects"] for r in works)
    best = [r for r in works if r["false_confirmed_defects"] == fewest]
    k = min(r["k"] for r in best)
    tied = sorted(r["r"] for r in best if r["k"] == k)
    mid = (tied[0] + tied[-1]) / 2
    r = max(x for x in tied if x <= mid + 1e-9)
    return {"k": k, "r": r, "tied_r_range": [tied[0], tied[-1]], "false_confirmed_defects": fewest,
            "why": f"smallest K that finds all {len(detectable)} detectable cracks with the fewest false confirmed "
                   f"defects ({fewest}); R = middle of the tied range {tied[0]:.2f} to {tied[-1]:.2f}"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prep", default=str(REPO / "BeltCrack9kd_prep"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    from PIL import Image

    from vision.detector import Detector

    cfg = tuning_scenario()
    scn = generate(cfg)
    captures, truth, locations = scn.captures, scn.vision_truth, scn.vision_locations
    source = dict(zip(truth["capture_id"], truth["source"]))
    kind = dict(zip(truth["capture_id"], truth["kind"]))
    frames = sv.dataset_frame_paths(Path(args.prep))
    det = Detector(threads=args.threads)
    cache: dict[str, list[float]] = {}
    image_of = {}
    for cid, img in zip(captures["capture_id"], captures["image"]):
        if kind[cid] == "crack":
            _, seq, name = source[cid].split("/")
            image_of[img] = ("dataset", frames[(seq, int(name[:-4]))])
        else:
            image_of[img] = ("pool", sv.DEMO_DIR / img)

    def scores_of(image: str) -> list[float]:
        if image not in cache:
            where, path = image_of[image]
            data = sv.crack_frame_jpeg(path) if where == "dataset" else path.read_bytes()
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                cache[image] = [d.score for d in det.detect(im)]
        return cache[image]

    t = truth.set_index("capture_id")
    crack_ids = [x["location_id"] for x in locations if x["kind"] == "crack"]
    fired = {cid for cid, img in zip(captures["capture_id"], captures["image"]) if scores_of(img)}
    hit_laps = {cid: t.loc[[c for c in t.index[t["location_id"] == cid] if c in fired], "lap"].nunique()
                for cid in crack_ids}
    detectable = {cid for cid, n in hit_laps.items() if n >= 2}

    table = []
    for k in K_GRID:
        for r in R_GRID:
            s = score(captures, truth, locations, scores_of, confirm_laps=k, confirm_ratio=r)
            table.append({"k": k, "r": r, "found_ids": [c["location_id"] for c in s["cracks"] if c["found"]],
                          "crack_locations_found": s["crack_locations_found"],
                          "found_detectable": len(detectable & {c["location_id"] for c in s["cracks"] if c["found"]}),
                          "duplicates": s["duplicates"], "false_confirmed_defects": s["false_confirmed_defects"],
                          "false_locations_at_splice": s["false_locations_at_splice"],
                          "laps_to_confirm": s["laps_to_confirm"]})
    choice = choose(table, detectable)
    old = next(r for r in table if r["k"] == 2 and r["r"] == 0.0)
    final = score(captures, truth, locations, scores_of, confirm_laps=choice["k"], confirm_ratio=choice["r"])
    result = {
        "purpose": "choose K and R for the lap-ratio confirmation rule in vision/defects.py, away from the demo",
        "scenario": {"name": cfg.scenario, "seed": TUNE_SEED, "days": cfg.days,
                     "config": "configs/scenarios/demo_01.yaml with only the seed and name changed",
                     "captures": int(len(captures)), "laps": int(truth["lap"].max())},
        "selection_procedure": [line.strip() for line in __doc__.split("Selection, fixed before the run:")[1]
                                .split("Each rule")[0].strip().splitlines()],
        "cracks": [{"location_id": c["location_id"], "near_splice": c["near_splice"],
                    "sequence": next(x["sequence"] for x in locations if x["location_id"] == c["location_id"]),
                    "laps_with_a_detection": hit_laps[c["location_id"]],
                    "detectable": c["location_id"] in detectable,
                    "laps_seen_of_looked_at_old_rule": [c.get("laps_seen"), c.get("laps_looked")]}
                   for c in score(captures, truth, locations, scores_of, 2, 0.0)["cracks"]],
        "false_locations_old_rule": [
            {"belt_coordinate_m": x["belt_coordinate_m"], "laps": x["laps"], "laps_looked": x["laps_looked"]}
            for x in score(captures, truth, locations, scores_of, 2, 0.0)["reported"]["defects"]
            if not any(x["defect_id"] in c["reported_as"] for c in
                       score(captures, truth, locations, scores_of, 2, 0.0)["cracks"])],
        "choice": {"confirm_laps": choice["k"], "confirm_ratio": choice["r"], **{k: v for k, v in choice.items()
                                                                                   if k not in ("k", "r")}},
        "old_rule_on_this_run": {k: old[k] for k in ("crack_locations_found", "found_detectable", "duplicates",
                                                     "false_confirmed_defects", "false_locations_at_splice",
                                                     "laps_to_confirm")},
        "chosen_rule_on_this_run": {k: final[k] for k in ("crack_locations_found", "duplicates",
                                                          "false_confirmed_defects", "false_locations_at_splice",
                                                          "laps_to_confirm")},
        "grid": [{k: v for k, v in r.items() if k != "found_ids"} for r in table],
        "notes": [
            "The tuning run shares the 16 clean crops and the 9 test sequences with the demo (there are no others); "
            "its locations, frames, strides and noise are its own.",
            "The demo's results under the old rule had already been reported before this rule was written. The "
            "procedure above was fixed in this script before the tuning run, and demo_01 is not read here.",
        ],
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: result[k] for k in ("scenario", "cracks", "false_locations_old_rule", "choice",
                                             "old_rule_on_this_run", "chosen_rule_on_this_run")}, indent=1))


if __name__ == "__main__":
    main()
