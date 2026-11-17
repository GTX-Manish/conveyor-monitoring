"""Convert the extracted BeltCrack9kd dataset into a compact YOLO-format zip for Colab (plan step 3).

    python -m vision.prepare_dataset --root beltcrack_data --out beltcrack_yolo960

- Keeps the archive's own split (images/<split>/<sequence>/<n>.jpg with json/<split>/<sequence>/<n>.json, LabelMe).
- Each LabelMe polygon becomes its bounding box, class 0 "crack".
- Images are resized to LONG_SIDE (960) on the long side, JPEG q90. Labels are normalised, so they do not change.
- clean_val/: box-free crops of val frames, a stand-in for crack-free frames (the dataset has none), used only to
  measure false detections per frame. They are never used for training.
- Writes <out>/data.yaml, <out>/manifest.json, a label montage per split, and <out>.zip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

LONG_SIDE = 960
JPEG_QUALITY = 90
SPLITS = ("train", "val", "test")
TILE_EVERY = 10          # take tiles from every 10th val frame
TILE_MARGIN = 0.02       # a tile must keep this share of the frame size away from every box


def natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def read_labelme(json_path: Path) -> tuple[tuple[int, int], list[tuple[float, float, float, float]]]:
    """(width, height) and one (x1, y1, x2, y2) pixel box per polygon."""
    d = json.loads(json_path.read_text(encoding="utf-8"))
    boxes = []
    for shape in d.get("shapes", []):
        xs = [p[0] for p in shape["points"]]
        ys = [p[1] for p in shape["points"]]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return (int(d["imageWidth"]), int(d["imageHeight"])), boxes


def yolo_rows(size: tuple[int, int], boxes) -> list[str]:
    """Clip to the image and normalise. Degenerate boxes (under 1 px) are dropped and counted by the caller."""
    w, h = size
    rows = []
    for x1, y1, x2, y2 in boxes:
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        rows.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
    return rows


def resized(src: Path, long_side: int = LONG_SIDE) -> Image.Image:
    with Image.open(src) as im:
        im.draft("RGB", (long_side, long_side))       # JPEG DCT downscale first (never below the target), then resample
        im = im.convert("RGB")
        s = long_side / max(im.size)
        return im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)


def convert_one(job):
    src, json_path, dst_img, dst_lbl = job
    size, boxes = read_labelme(Path(json_path))
    rows = yolo_rows(size, boxes)
    img = resized(Path(src))
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    dst_lbl.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_img, quality=JPEG_QUALITY)
    dst_lbl.write_text("\n".join(rows) + ("\n" if rows else ""))
    return len(boxes), len(rows), hashlib.sha256(dst_img.read_bytes()).hexdigest()


def box_free_tiles(size, boxes, rng: random.Random, tries: int = 40):
    """Up to two half-width x half-height crops that keep TILE_MARGIN away from every box."""
    w, h = size
    tw, th = w // 2, h // 2
    mx, my = TILE_MARGIN * w, TILE_MARGIN * h
    found = []
    for _ in range(tries):
        x, y = rng.randint(0, w - tw), rng.randint(0, h - th)
        clear = all(x2 + mx <= x or x1 - mx >= x + tw or y2 + my <= y or y1 - my >= y + th for x1, y1, x2, y2 in boxes)
        overlaps_found = any(abs(x - fx) < tw // 2 and abs(y - fy) < th // 2 for fx, fy, _, _ in found)
        if clear and not overlaps_found:
            found.append((x, y, x + tw, y + th))
            if len(found) == 2:
                break
    return found


def montage(out: Path, split: str, n: int = 12, seed: int = 0):
    """Draw the boxes read back from the written YOLO files, so the montage checks the whole conversion."""
    imgs = sorted((out / "images" / split).glob("*.jpg"))
    pick = random.Random(seed).sample(imgs, min(n, len(imgs)))
    tiles = []
    for p in pick:
        im = Image.open(p).convert("RGB")
        im.thumbnail((480, 480))
        d = ImageDraw.Draw(im)
        for line in (out / "labels" / split / f"{p.stem}.txt").read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            d.rectangle(((cx - bw / 2) * im.width, (cy - bh / 2) * im.height,
                         (cx + bw / 2) * im.width, (cy + bh / 2) * im.height), outline=(0, 255, 0), width=2)
        d.text((4, 4), p.stem, fill=(255, 255, 0))
        tiles.append(im)
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 480, rows * 480), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 480, (i // cols) * 480))
    path = out / f"montage_{split}.jpg"
    sheet.save(path, quality=85)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="beltcrack_data")
    ap.add_argument("--out", default="beltcrack_yolo960")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args(argv)

    base = Path(args.root).resolve()
    roots = [p for p in base.rglob("json") if p.is_dir() and (p.parent / "images").is_dir()]
    if len(roots) != 1:
        raise SystemExit(f"expected one dataset folder holding images/ and json/ under {base}, found {roots}")
    root = roots[0].parent
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty; remove it first")

    jobs, meta = [], []
    for split in SPLITS:
        for jf in sorted((root / "json" / split).rglob("*.json"), key=lambda p: (natural(p.parent.name), natural(p.stem))):
            seq = jf.parent.name
            src = root / "images" / split / seq / f"{jf.stem}.jpg"
            if not src.is_file():
                raise SystemExit(f"no image for {jf}")
            stem = f"{seq}__{jf.stem}"
            jobs.append((str(src), str(jf), out / "images" / split / f"{stem}.jpg", out / "labels" / split / f"{stem}.txt"))
            meta.append((split, seq, stem))
    with ProcessPoolExecutor(args.workers) as ex:
        results = list(ex.map(convert_one, jobs, chunksize=32))

    counts = {s: Counter() for s in SPLITS}
    seqs = {s: Counter() for s in SPLITS}
    files = {}
    for (split, seq, stem), (n_src, n_out, sha) in zip(meta, results):
        counts[split]["images"] += 1
        counts[split]["source_boxes"] += n_src
        counts[split]["yolo_boxes"] += n_out
        seqs[split][seq] += 1
        files[f"images/{split}/{stem}.jpg"] = sha
    for split in SPLITS:
        if counts[split]["source_boxes"] != counts[split]["yolo_boxes"]:
            raise SystemExit(f"{split}: {counts[split]['source_boxes']} source boxes but "
                             f"{counts[split]['yolo_boxes']} written")

    # box-free val tiles, from the original-resolution frames
    rng = random.Random(0)
    tile_dir = out / "clean_val" / "images"
    tile_dir.mkdir(parents=True, exist_ok=True)
    n_tiles, tile_src = 0, Counter()
    by_seq: dict[str, list] = {}
    for (src, jf, _, _), (split, seq, stem) in zip(jobs, meta):
        if split == "val":
            by_seq.setdefault(seq, []).append((src, jf, stem))
    for seq, frames in sorted(by_seq.items()):
        for src, jf, stem in frames[::TILE_EVERY]:
            size, boxes = read_labelme(Path(jf))
            crops = box_free_tiles(size, boxes, rng)
            if not crops:
                continue
            with Image.open(src) as im:
                im = im.convert("RGB")
                for k, c in enumerate(crops):
                    t = im.crop(c)
                    s = LONG_SIDE / max(t.size)
                    t = t.resize((round(t.width * s), round(t.height * s)), Image.Resampling.LANCZOS)
                    t.save(tile_dir / f"{stem}__tile{k}.jpg", quality=JPEG_QUALITY)
                    n_tiles += 1
                    tile_src[seq] += 1

    (out / "data.yaml").write_text("train: images/train\nval: images/val\ntest: images/test\n"
                                   "names:\n  0: crack\n")
    manifest = {
        "dataset": "BeltCrack9kd (archive files named BeltCrack14ks.partNN.rar), ScienceDB doi:10.57760/sciencedb.31181",
        "split": "the archive's own folders: train 23 sequences, val 10, test 9",
        "long_side": LONG_SIDE, "jpeg_quality": JPEG_QUALITY, "label": "LabelMe polygon -> bounding box, class 0 crack",
        "counts": {s: dict(counts[s]) | {"sequences": len(seqs[s])} for s in SPLITS},
        "sequences": {s: dict(sorted(seqs[s].items(), key=lambda kv: natural(kv[0]))) for s in SPLITS},
        "clean_val": {"tiles": n_tiles, "from_val_frames_every": TILE_EVERY, "margin_share": TILE_MARGIN,
                      "per_sequence": dict(tile_src),
                      "note": "half-width x half-height box-free crops of val frames, at twice the full-frame scale; "
                              "may include background and unlabelled cracks; for false-hit measurement only"},
        "files_sha256": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    shown = [str(montage(out, s)) for s in SPLITS]
    print(json.dumps({"counts": manifest["counts"], "clean_val_tiles": n_tiles, "montages": shown}, indent=1))

    if not args.no_zip:
        zpath = out.with_suffix(".zip")
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:     # JPEGs do not compress further
            for p in sorted(out.rglob("*")):
                if p.is_file() and not p.name.startswith("montage_"):
                    z.write(p, p.relative_to(out.parent))
        print("zip:", zpath, f"{zpath.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
