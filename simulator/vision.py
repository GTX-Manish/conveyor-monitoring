"""Simulated belt-surface camera captures for the vision pipeline.

Three carry-side camera stations look at the belt. Things on the belt are fixed to belt coordinates
(simulator/belt_coords.py): the coordinate under a station at ground position p is (p - D) mod L, where D is the
belt's travel, which only grows while the belt runs.

Belt locations the cameras are simulated for:
- 3 or 4 crack locations at fixed belt coordinates, each pinned to one BeltCrack9kd test sequence. On every
  inspection lap the location shows that sequence's frame advanced by a seeded stride of 15 to 30 frames, so each
  lap is a different view of a real crack (wrapping round at the end of the sequence). One crack sits within 1.5 m
  of a splice on purpose.
- 12 splices, every 500 m. A splice shows one fixed clean image (a joint looks the same on every lap), so any
  detection there is a false hit at a splice.
- 4 clean locations, each capture showing a clean image drawn at random.
Clean images are box-free regions of test frames (the dataset has no crack-free frames), at the same scale as the
crack frames and centred on grey padding, so a clean capture looks like a narrower camera view.

(assumption) The cameras record one full lap per running day, starting at the first running minute after 12:00.
Recording every lap would need thousands of committed dataset frames; one lap a day still gives 14 laps in the
demo. On that lap every location passes CAM1, CAM2 and CAM3 in turn, so one lap gives three captures per location
within 750 s.

Outputs: public captures (capture_id, ts, station, ground position, belt coordinate, image path) and, separately,
the truth (which location each capture shows), written to truth/ by simulator.io. Image names are hashes, so the
path says nothing about what the image shows.

    python -m simulator.vision build-demo --prep BeltCrack9kd_prep    # (re)build data/vision_demo/ (needs the dataset)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO / "data" / "vision_demo"
CATALOG_NAME = "catalog.json"

STATIONS = (("CAM1", 0.0), ("CAM2", 1500.0), ("CAM3", 3000.0))      # carry side, ground positions in m
SPLICE_SPACING_M = 500.0
COORD_NOISE_M = 0.5
INSPECT_HOUR = 12                  # (assumption) one recorded lap a day, from the first running minute after 12:00
N_CRACKS = (3, 4)                  # inclusive range
N_CLEAN = 4
STRIDE = (15, 30)                  # frames advanced per lap, inclusive
NEAR_SPLICE_M = (0.5, 1.2)         # offset of the crack placed near a splice
MIN_GAP_M = 25.0                   # every other location keeps this far from splices and from each other
FRAME_SIZE = (640, 360)            # 1024 x 576 test frames reduced to fit the detector's 640 x 384 input
CLEAN_POOL = 16
JPEG_QUALITY = 85
PAD_GREY = 114                     # the detector's letterbox colour
# Chosen by eye for the demo: data34 and data38 are side views (rollers and steel fill the frame, the crack is a few
# pixels), so no crack location uses them; clean images come only from the four sequences that show belt surface edge
# to edge (the others include machinery, an object in front of the camera or pavement).
DEMO_EXCLUDED_SEQUENCES = ("data34", "data38")
CLEAN_SOURCE_SEQUENCES = ("data35", "data37", "data40", "data42")
CAPTURE_COLUMNS = ["capture_id", "ts", "station", "ground_position_m", "belt_coordinate_m", "image"]
TRUTH_COLUMNS = ["capture_id", "location_id", "kind", "lap", "source"]


def frame_file(seq: str, n: int) -> str:
    return f"frames/{hashlib.sha1(f'test/{seq}/{n}'.encode()).hexdigest()[:12]}.jpg"


def crop_file(seq: str, n: int, x: int, y: int) -> str:
    return f"frames/{hashlib.sha1(f'test/{seq}/{n}@{x},{y}'.encode()).hexdigest()[:12]}.jpg"


def load_catalog(demo_dir: Path = DEMO_DIR) -> dict | None:
    path = Path(demo_dir) / CATALOG_NAME
    return json.loads(path.read_text()) if path.exists() else None


def circ_dist(a, b, length):
    d = np.abs(np.asarray(a) - np.asarray(b)) % length
    return np.minimum(d, length - d)


def coordinate_under(station_m: float, travel_m, length: float):
    """Belt coordinate under a station: (station - travel) mod L."""
    return (station_m - np.asarray(travel_m)) % length


def place_locations(rng, catalog: dict, length: float) -> list[dict]:
    splices = np.arange(0.0, length, SPLICE_SPACING_M)
    seqs = sorted(s for s in catalog["sequences"] if s not in DEMO_EXCLUDED_SEQUENCES)
    n_cracks = int(rng.integers(N_CRACKS[0], N_CRACKS[1] + 1))
    chosen = [seqs[i] for i in rng.choice(len(seqs), n_cracks, replace=False)]
    pool = len(catalog["clean_pool"])
    splice_images = rng.choice(pool, len(splices), replace=False)
    locs = [{"location_id": i + 1, "kind": "splice", "belt_coordinate_m": float(c), "image": int(splice_images[i])}
            for i, c in enumerate(splices)]
    k = int(rng.integers(len(splices)))
    off = float(rng.uniform(*NEAR_SPLICE_M)) * (1 if rng.random() < 0.5 else -1)
    taken = list(splices)

    def free_coordinate():
        while True:
            c = float(rng.uniform(0.0, length))
            if circ_dist(c, np.array(taken), length).min() >= MIN_GAP_M:
                taken.append(c)
                return c

    for j, seq in enumerate(chosen):
        c = (splices[k] + off) % length if j == 0 else free_coordinate()
        locs.append({"location_id": len(locs) + 1, "kind": "crack", "belt_coordinate_m": float(c), "sequence": seq,
                     "near_splice": j == 0, "splice_m": float(splices[k]) if j == 0 else None})
    for _ in range(N_CLEAN):
        locs.append({"location_id": len(locs) + 1, "kind": "clean", "belt_coordinate_m": free_coordinate()})
    return locs


def inspection_starts(timestamp: pd.DatetimeIndex, running: np.ndarray) -> list[int]:
    """Step index of each recorded lap: the first running minute at or after INSPECT_HOUR on each day."""
    ts = np.asarray(timestamp.values, dtype="datetime64[m]")
    day0 = ts[0].astype("datetime64[D]")
    starts = []
    for d in range(int((ts[-1].astype("datetime64[D]") - day0).astype(int)) + 1):
        t0 = day0 + np.timedelta64(d, "D") + np.timedelta64(INSPECT_HOUR, "h")
        idx = np.flatnonzero((ts >= t0) & (ts < t0 + np.timedelta64(1, "D")) & running)
        if len(idx):
            starts.append(int(idx[0]))
    return starts


def simulate(timestamp: pd.DatetimeIndex, travel: np.ndarray, running: np.ndarray, length: float, rng,
             catalog: dict | None, step_s: int = 60):
    """(public captures, truth, locations). travel[t] is the belt travel at the end of step t."""
    if catalog is None:
        return pd.DataFrame(columns=CAPTURE_COLUMNS), pd.DataFrame(columns=TRUTH_COLUMNS), []
    travel = np.asarray(travel, dtype=float)
    locs = place_locations(rng, catalog, length)
    seq_frames = catalog["sequences"]
    pool = catalog["clean_pool"]
    position = {}                                            # crack location id -> current frame index
    epoch = (timestamp.values.astype("datetime64[s]").astype(np.int64))
    rows = []
    for lap, t0 in enumerate(inspection_starts(timestamp, running)):
        d0 = travel[t0 - 1] if t0 > 0 else 0.0
        for loc in locs:
            if loc["kind"] == "crack":
                frames = seq_frames[loc["sequence"]]
                if loc["location_id"] not in position:
                    position[loc["location_id"]] = int(rng.integers(len(frames)))
                else:
                    position[loc["location_id"]] = (position[loc["location_id"]]
                                                    + int(rng.integers(STRIDE[0], STRIDE[1] + 1))) % len(frames)
                n = frames[position[loc["location_id"]]]
                image, source = frame_file(loc["sequence"], n), f"test/{loc['sequence']}/{n}.jpg"
            # One traversal per lap: the point passes the first station, then the others downstream (750 s at 4 m/s).
            d_first = d0 + (STATIONS[0][1] - loc["belt_coordinate_m"] - d0) % length
            for station, p in STATIONS:
                d_pass = d_first + (p - STATIONS[0][1])
                t = int(np.searchsorted(travel, d_pass, side="left"))
                if t >= len(travel):
                    continue
                before = travel[t - 1] if t > 0 else 0.0
                frac = (d_pass - before) / (travel[t] - before) if travel[t] > before else 0.0
                ts = int(epoch[t] + round(frac * step_s))
                true_c = float(coordinate_under(p, d_pass, length))
                if loc["kind"] == "splice":
                    image, source = pool[loc["image"]]["file"], pool[loc["image"]]["source"]
                elif loc["kind"] == "clean":
                    item = pool[int(rng.integers(len(pool)))]
                    image, source = item["file"], item["source"]
                observed = float((true_c + rng.normal(0.0, COORD_NOISE_M)) % length)
                rows.append((ts, station, p, round(observed, 3), image, loc["location_id"], loc["kind"], lap + 1,
                             source))
    rows.sort(key=lambda r: (r[0], r[2], r[5]))
    df = pd.DataFrame(rows, columns=["ts", "station", "ground_position_m", "belt_coordinate_m", "image",
                                     "location_id", "kind", "lap", "source"])
    df.insert(0, "capture_id", np.arange(1, len(df) + 1, dtype=np.int64))
    df["ts"] = df["ts"].astype(np.int64)
    return df[CAPTURE_COLUMNS].copy(), df[TRUTH_COLUMNS].copy(), locs


# ------------------------------------------------------------------ building data/vision_demo/ (needs the dataset)
def _test_frames(prep: Path) -> tuple[dict, dict]:
    coco = json.loads((prep / "annotations" / "test_coco.json").read_text())
    boxes: dict[int, list] = {}
    for a in coco["annotations"]:
        boxes.setdefault(a["image_id"], []).append(a["bbox"])
    seqs: dict[str, list] = {}
    info = {}
    for im in coco["images"]:
        folder, name = im["file_name"].split("/")[-2:]
        seq, n = folder.rsplit("_", 1)[-1], int(name.rsplit(".", 1)[0])
        seqs.setdefault(seq, []).append(n)
        info[(seq, n)] = (prep / im["file_name"], boxes.get(im["id"], []), (im["width"], im["height"]))
    return {s: sorted(v) for s, v in sorted(seqs.items())}, info


def _clean_pool(info: dict, seqs: dict, rng, margin: int = 10) -> list[dict]:
    """CLEAN_POOL box-free FRAME_SIZE crops, spread over the sequences, at most one per frame."""
    from PIL import Image
    pool, used = [], set()
    order = [s for s in seqs if s in CLEAN_SOURCE_SEQUENCES]
    while len(pool) < CLEAN_POOL:
        seq = order[len(pool) % len(order)]
        for _ in range(500):
            n = seqs[seq][int(rng.integers(len(seqs[seq])))]
            if (seq, n) in used:
                continue
            path, boxes, (w, h) = info[(seq, n)]
            x, y = int(rng.integers(0, w - FRAME_SIZE[0] + 1)), int(rng.integers(0, h - FRAME_SIZE[1] + 1))
            if all(bx + bw + margin <= x or bx - margin >= x + FRAME_SIZE[0] or by + bh + margin <= y
                   or by - margin >= y + FRAME_SIZE[1] for bx, by, bw, bh in boxes):
                used.add((seq, n))
                # Reduced exactly as the crack frames are (1024 -> 640), so belt texture has the same scale, and centred
                # on the detector's own grey padding. Shown full size, a crop is zoomed 1.6x, and the model then fired on
                # 10 of 16 crops against 1 of 16 at the frames' own scale.
                s = FRAME_SIZE[0] / w
                with Image.open(path) as im:
                    region = im.convert("RGB").crop((x, y, x + FRAME_SIZE[0], y + FRAME_SIZE[1]))
                region = region.resize((round(FRAME_SIZE[0] * s), round(FRAME_SIZE[1] * s)), Image.Resampling.LANCZOS)
                crop = Image.new("RGB", FRAME_SIZE, (PAD_GREY,) * 3)
                crop.paste(region, ((FRAME_SIZE[0] - region.width) // 2, (FRAME_SIZE[1] - region.height) // 2))
                file = crop_file(seq, n, x, y)
                pool.append({"file": file, "source": f"test/{seq}/{n}.jpg", "crop_xywh": [x, y, *FRAME_SIZE],
                             "_image": crop})
                break
        else:
            raise SystemExit(f"no box-free crop found in {seq}")
    return pool


ATTRIBUTION = """# Demo images: attribution

The images in `frames/` are taken from the **BeltCrack** dataset by Huang, Ji, Ma and Ye (*Pattern Recognition*,
2026, doi:10.1016/j.patcog.2026.113598; data: doi:10.57760/sciencedb.31181), licensed
**CC BY-SA 4.0** (https://creativecommons.org/licenses/by-sa/4.0/). They come from the BeltCrack9kd test split.

Changes: each frame was reduced from 1024 x 576 (itself reduced from the original video resolution) to 640 x 360
and re-encoded as JPEG. Each clean image is a region of a test frame chosen to avoid every labelled crack, reduced
by the same factor and centred on a grey (114) 640 x 360 canvas.
File names are hashes. `catalog.json` lists the source frame (and crop) of every file. These images and this
folder are shared under the same licence, CC BY-SA 4.0.

Built by `python -m simulator.vision build-demo`. The demo scenario's captures (simulator/vision.py) refer to them.
"""


def crack_frame_jpeg(src: Path) -> bytes:
    """A test frame as the cameras show it: reduced to FRAME_SIZE and JPEG-encoded (used for data/vision_demo/ and
    for the tuning run in scripts/tune_dedup.py, so both see the same pixels)."""
    import io

    from PIL import Image
    buf = io.BytesIO()
    with Image.open(src) as im:
        im.convert("RGB").resize(FRAME_SIZE, Image.Resampling.LANCZOS).save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def dataset_frame_paths(prep: Path) -> dict:
    """(sequence, frame number) -> path of the 1024 px test frame in the output of vision/prepare_beltcrack9kd.py."""
    return {k: v[0] for k, v in _test_frames(Path(prep))[1].items()}


def build_demo(prep: Path, out: Path = DEMO_DIR, scenario_config: Path = REPO / "configs" / "scenarios" / "demo_01.yaml"):

    from .config import load_config
    from .generate import generate

    seqs, info = _test_frames(prep)
    pool = _clean_pool(info, seqs, np.random.default_rng(20261117))
    (out / "frames").mkdir(parents=True, exist_ok=True)
    catalog = {"dataset": "BeltCrack9kd test split (CC BY-SA 4.0), see ATTRIBUTION.md",
               "frame_size": list(FRAME_SIZE), "sequences": seqs,
               "clean_pool": [{k: v for k, v in p.items() if k != "_image"} for p in pool], "frames": []}
    for p in pool:
        p["_image"].save(out / p["file"], quality=JPEG_QUALITY)
    (out / CATALOG_NAME).write_text(json.dumps(catalog, indent=1))       # the simulator needs it to pick frames
    scn = generate(load_config(scenario_config))
    crack = scn.vision_truth[scn.vision_truth["kind"] == "crack"].drop_duplicates("source")
    for source in sorted(crack["source"]):
        _, seq, name = source.split("/")
        n = int(name[:-4])
        file = frame_file(seq, n)
        (out / file).write_bytes(crack_frame_jpeg(info[(seq, n)][0]))
        catalog["frames"].append({"file": file, "source": source})
    (out / CATALOG_NAME).write_text(json.dumps(catalog, indent=1))
    (out / "ATTRIBUTION.md").write_text(ATTRIBUTION)
    keep = {p["file"] for p in catalog["clean_pool"]} | {f["file"] for f in catalog["frames"]}
    for f in (out / "frames").glob("*.jpg"):
        if f"frames/{f.name}" not in keep:
            f.unlink()
    missing = sorted(set(scn.captures["image"]) - keep)
    if missing:
        raise SystemExit(f"demo captures refer to images that were not written: {missing[:5]}")
    size = sum((out / k).stat().st_size for k in keep) / 1e6
    print(f"{out}: {len(catalog['frames'])} crack frames, {len(pool)} clean crops, {size:.1f} MB; "
          f"{len(scn.captures)} demo captures")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Build data/vision_demo/ from the BeltCrack9kd test frames.")
    ap.add_argument("command", choices=["build-demo"])
    ap.add_argument("--prep", default="BeltCrack9kd_prep", help="output of vision/prepare_beltcrack9kd.py")
    args = ap.parse_args(argv)
    build_demo(Path(args.prep).resolve())


if __name__ == "__main__":
    main()
