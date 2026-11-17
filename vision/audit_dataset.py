"""Audit the extracted BeltCrack14ks dataset before any conversion or training (plan step 2).

Checks the layout against what the authors' code and paper say (official split by sequence, one class, txt lists for
train/val, COCO json for test), looks for leakage between splits (shared sequence folders, near-duplicate frames),
measures box sizes after downscaling, and lists the frames that no annotation file mentions.

    python -m vision.audit_dataset --root beltcrack_data [--workers 4]

Writes vision/artifacts/dataset_audit.json, plus montages of unlisted frames under vision/artifacts/audit/.
Nothing here is used for training; it only reports.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
OUT_DIR = Path(__file__).resolve().parent / "artifacts"
# 14ks split as read from the authors' code; 9kd totals from the paper (its split is not stated there).
PAPER = {"sequences": {"train": 18, "val": 5, "test": 6}, "images": {"train": 8773, "val": 2596, "test": 2718},
         "boxes_per_image": 2.65, "9kd_total": {"images": 9645, "sequences": 42},
         "14ks_total": {"images": 14087, "sequences": 29}}
NUM = r"-?\d+(?:\.\d+)?"
LINE = re.compile(rf"^(.+?\.(?:jpg|jpeg|png|bmp))((?:\s+{NUM},{NUM},{NUM},{NUM},{NUM})*)\s*$", re.I)
SMALL_PX = 8
# 256-bit dHash (16x16 gradients), compared only between frames of the same size. Neighbouring frames of one sequence
# already differ by about 30 to 65 bits (the belt moves), so a cross-split pair closer than NEAR_DUP_BITS is "as close
# as two neighbouring frames" and is listed for a by-eye look.
NEAR_DUP_BITS = 40


# ------------------------------------------------------------------------------------------------ path matching
class Resolver:
    """Map annotation paths written on the authors' machine to files here, by their last 1..6 path components."""

    def __init__(self, images: list[Path], max_k: int = 6):
        self.max_k = max_k
        self.index: list[dict[str, list[Path]]] = [defaultdict(list) for _ in range(max_k + 1)]
        for p in images:
            parts = p.parts
            for k in range(1, min(max_k, len(parts)) + 1):
                self.index[k]["/".join(parts[-k:])].append(p)

    def __call__(self, name: str) -> Path | None:
        parts = [x for x in name.replace("\\", "/").split("/") if x]
        for k in range(min(self.max_k, len(parts)), 0, -1):
            hits = self.index[k].get("/".join(parts[-k:]))
            if hits:
                return hits[0] if len(hits) == 1 else None
        return None


# ------------------------------------------------------------------------------------------------ readers
def read_txt(path: Path, resolve: Resolver):
    """`path x1,y1,x2,y2,cls ...` per line. Returns {image: [(x1, y1, x2, y2, cls)]}, unmatched count, classes."""
    out, unmatched, bad = {}, 0, 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        m = LINE.match(line.strip())
        if not m:
            bad += 1
            continue
        p = resolve(m.group(1))
        if p is None:
            unmatched += 1
            continue
        out[p] = [tuple(float(v) for v in b.split(",")) for b in m.group(2).split()]
    return out, unmatched, bad


def read_coco(path: Path, resolve: Resolver):
    data = json.loads(path.read_text(encoding="utf-8"))
    id_to_path, sizes, unmatched = {}, {}, 0
    for im in data["images"]:
        p = resolve(str(im.get("file_name", "")))
        if p is None:
            unmatched += 1
            continue
        id_to_path[im["id"]] = p
        sizes[p] = (im.get("width"), im.get("height"))
    boxes = {p: [] for p in id_to_path.values()}
    raw = []
    for ann in data["annotations"]:
        p = id_to_path.get(ann["image_id"])
        if p is None:
            continue
        raw.append((p, ann["bbox"], ann.get("category_id"), ann.get("iscrowd", 0)))
    info = {"categories": data.get("categories"), "n_images": len(data["images"]), "n_annotations": len(data["annotations"]),
            "image_id_min": min((im["id"] for im in data["images"]), default=None),
            "annotation_keys": sorted(data["annotations"][0].keys()) if data["annotations"] else []}
    return boxes, raw, sizes, unmatched, info


# ------------------------------------------------------------------------------------------------ per-image work
def image_facts(path: str):
    """Size from the header, plus a 256-bit dHash from a draft-mode (1/8 scale) decode, which keeps 4K JPEGs fast."""
    with Image.open(path) as im:
        size = im.size
        im.draft("L", (512, 512))
        g = np.asarray(im.convert("L").resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16)
    return size, np.packbits((g[:, 1:] > g[:, :-1]).flatten())


def natural(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def pct(values, qs=(0, 5, 25, 50, 75, 95, 100)):
    if len(values) == 0:
        return {}
    a = np.asarray(values, dtype=float)
    return {f"p{q}": round(float(np.percentile(a, q)), 2) for q in qs}


# ------------------------------------------------------------------------------------------------ main
def find_root(base: Path) -> Path:
    named = sorted((p for p in base.rglob("*") if p.is_dir() and "14ks" in p.name.lower()), key=lambda p: len(p.parts))
    return named[0] if named else base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="beltcrack_data")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--montage", type=int, default=24)
    args = ap.parse_args(argv)

    root = find_root(Path(args.root).resolve())
    files = [p for p in root.rglob("*") if p.is_file()]
    images = sorted(p for p in files if p.suffix.lower() in IMG_EXT)
    resolve = Resolver(images)
    report: dict = {"root": os.path.relpath(root), "n_image_files": len(images), "paper": PAPER}
    report["top_level"] = sorted(str(p.relative_to(root)) for p in root.iterdir())
    report["annotation_files"] = sorted(str(p.relative_to(root)) for p in files
                                        if p.suffix.lower() in {".txt", ".json", ".xml"} and "annotation" in str(p).lower())

    # ---- labels per official split. Two layouts are handled: the one the authors' code reads (txt lists for train/val,
    # COCO json for test) and the one this archive actually ships (one LabelMe json per image under json/<split>/<seq>/).
    split_boxes: dict[str, dict[Path, list]] = {}
    layout: dict = {}
    sizes_from_labels: dict[Path, tuple] = {}
    labelme = {s: root / "json" / s for s in ("train", "val", "test") if (root / "json" / s).is_dir()}
    test_raw: list = []
    if labelme:
        layout["format"] = "labelme json per image"
        for split, jdir in labelme.items():
            bx, no_image, shapes, labels, orig_names, out_of_bounds = {}, 0, Counter(), Counter(), Counter(), 0
            for jf in sorted(jdir.rglob("*.json")):
                d = json.loads(jf.read_text(encoding="utf-8"))
                img = root / "images" / split / jf.parent.name / f"{jf.stem}.jpg"
                if not img.is_file():
                    no_image += 1
                    continue
                W, H = d.get("imageWidth"), d.get("imageHeight")
                sizes_from_labels[img] = (W, H)
                orig_names[str(d.get("imagePath", "")).replace("\\", "/").split("/")[0]] += 1
                boxes = []
                for sh in d.get("shapes", []):
                    shapes[sh.get("shape_type")] += 1
                    labels[sh.get("label")] += 1
                    pts = np.asarray(sh["points"], dtype=float)
                    x1, y1 = pts.min(0)
                    x2, y2 = pts.max(0)
                    if W and (x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1):
                        out_of_bounds += 1
                    boxes.append((x1, y1, x2, y2, 0))
                bx[img] = boxes
            split_boxes[split] = bx
            layout[split] = {"json_dir": str(jdir.relative_to(root)), "json_files": len(bx) + no_image,
                             "json_without_image": no_image, "shape_types": dict(shapes), "labels": dict(labels),
                             "imagePath_first_component": dict(orig_names.most_common(5)),
                             "boxes_outside_image": out_of_bounds}
    else:
        for split in ("train", "val"):
            cands = [p for p in files if p.suffix.lower() == ".txt" and split in p.name.lower()]
            if not cands:
                layout[split] = "MISSING txt list"
                continue
            b, unmatched, bad = read_txt(cands[0], resolve)
            split_boxes[split] = b
            layout[split] = {"file": str(cands[0].relative_to(root)), "lines_matched": len(b), "unmatched": unmatched,
                             "unparsed_lines": bad}
        cands = [p for p in files if p.suffix.lower() == ".json" and "test" in str(p).lower()]
        if cands:
            b, test_raw, sizes_from_labels, unmatched, info = read_coco(cands[0], resolve)
            layout["test"] = {"file": str(cands[0].relative_to(root)), "unmatched": unmatched, **info}
            test_coco = b
    report["layout"] = layout

    # ---- image facts (size + hash) for every image file
    with ProcessPoolExecutor(args.workers) as ex:
        facts = dict(zip(images, ex.map(image_facts, map(str, images), chunksize=64)))
    sizes = {p: f[0] for p, f in facts.items()}
    hashes = {p: f[1] for p, f in facts.items()}
    report["image_sizes"] = {f"{w}x{h}": n for (w, h), n in Counter(sizes.values()).most_common()}
    report["label_size_mismatches"] = sum(1 for p, (w, h) in sizes_from_labels.items() if w and (w, h) != sizes[p])
    rel = [p.relative_to(root).parts for p in images]
    report["images_per_split_folder"] = dict(Counter(r[1] for r in rel if r[0] == "images"))

    # ---- LabelMe embeds a copy of the image: check on a sample that it is the same picture as the jpg on disk
    if labelme:
        import base64
        import io
        sample = random.Random(0).sample(sorted(sizes_from_labels), min(60, len(sizes_from_labels)))
        same, differ = 0, []
        for img in sample:
            jf = root / "json" / img.parent.parent.name / img.parent.name / f"{img.stem}.json"
            data = json.loads(jf.read_text(encoding="utf-8")).get("imageData")
            if not data:
                continue
            with Image.open(io.BytesIO(base64.b64decode(data))) as em:
                em.draft("L", (512, 512))
                g = np.asarray(em.convert("L").resize((17, 16), Image.Resampling.BILINEAR), dtype=np.int16)
            dist = int(np.unpackbits(np.packbits((g[:, 1:] > g[:, :-1]).flatten()) ^ hashes[img]).sum())
            if dist <= 10:
                same += 1
            else:
                differ.append([str(img.relative_to(root)), dist])
        report["embedded_image_check"] = {"sampled": len(sample), "same_as_jpg": same, "different": differ[:10]}

    # ---- COCO bbox convention: xywh vs xyxy, 0- vs 1-based (only for the COCO layout)
    if test_raw:
        over_xywh = sum(1 for p, (x, y, w, h), *_ in test_raw if x + w > sizes[p][0] + 1 or y + h > sizes[p][1] + 1)
        layout["test"]["bbox_checks"] = {"boxes": len(test_raw), "x+w beyond image": over_xywh}
        xyxy = over_xywh > 0.05 * len(test_raw)
        layout["test"]["bbox_format_inferred"] = "xyxy" if xyxy else "xywh"
        for p, (x, y, w, h), *_ in test_raw:
            test_coco[p].append((x, y, w, h, 0) if xyxy else (x, y, x + w, y + h, 0))
        split_boxes["test"] = test_coco

    # ---- counts and sequences
    seq_of = {p: str(p.parent.relative_to(root)) for p in images}
    counts, seqs_by_split = {}, {}
    for split, bx in split_boxes.items():
        seqs = defaultdict(list)
        for p in bx:
            seqs[seq_of[p]].append(p)
        seqs_by_split[split] = seqs
        nb = [len(v) for v in bx.values()]
        counts[split] = {"images": len(bx), "sequences": len(seqs), "boxes": int(sum(nb)),
                         "boxes_per_image": round(float(np.mean(nb)), 3) if nb else None,
                         "images_without_boxes": sum(1 for n in nb if n == 0),
                         "per_sequence": {s: len(v) for s, v in sorted(seqs.items())},
                         "frame_names_numeric": all(re.fullmatch(r"\d+", p.stem) for p in bx)}
    report["counts"] = counts

    # ---- leakage: shared sequence folders, shared images, near-duplicates
    leak: dict = {}
    splits = list(split_boxes)
    for i, a in enumerate(splits):
        for c in splits[i + 1:]:
            leak[f"sequences_in_{a}_and_{c}"] = sorted(set(seqs_by_split[a]) & set(seqs_by_split[c]))
            leak[f"images_in_{a}_and_{c}"] = len(set(split_boxes[a]) & set(split_boxes[c]))
    def bits(x, y):
        return np.unpackbits(x ^ y, axis=-1).sum(-1)

    # control: distance between neighbouring frames (n, n+1) within each sequence
    adjacent = []
    for bx in split_boxes.values():
        by_seq_frames = defaultdict(list)
        for p in bx:
            by_seq_frames[seq_of[p]].append(p)
        for ps in by_seq_frames.values():
            ps.sort(key=natural)
            adjacent += [int(bits(hashes[a], hashes[c])) for a, c in zip(ps, ps[1:])]
    leak["neighbouring_frames_within_a_sequence_bits"] = pct(adjacent)
    if "train" in split_boxes:
        tr_paths = list(split_boxes["train"])
        tr = np.stack([hashes[p] for p in tr_paths])
        tr_size = np.array([sizes[p] for p in tr_paths])
        for other in ("val", "test"):
            if other not in split_boxes:
                continue
            mins, per_seq, examples = [], Counter(), []
            for p in split_boxes[other]:
                d = bits(tr, hashes[p]).astype(float)
                d[(tr_size != np.array(sizes[p])).any(1)] = np.inf      # only frames of the same size
                j = int(d.argmin())
                mins.append(d[j])
                if d[j] <= NEAR_DUP_BITS:
                    per_seq[(seq_of[p], seq_of[tr_paths[j]])] += 1
                    if len(examples) < 10:
                        examples.append([str(p.relative_to(root)), str(tr_paths[j].relative_to(root)), int(d[j])])
            m = np.array(mins)
            leak[f"nearest_train_frame_{other}"] = {
                "bits_to_nearest_train_frame": pct(m[np.isfinite(m)]),
                f"frames_within_{NEAR_DUP_BITS}_bits": int(sum(per_seq.values())),
                "of": len(m), "by_sequence_pair": {f"{a} ~ {c}": n for (a, c), n in per_seq.most_common(20)},
                "examples": examples}
    report["leakage"] = leak

    # ---- box sizes after downscaling the long side
    sizes_report = {}
    for split, bx in split_boxes.items():
        entry = {}
        for target in (640, 960):
            short = []
            for p, boxes in bx.items():
                s = target / max(sizes[p])
                for x1, y1, x2, y2, _ in boxes:
                    w, h = abs(x2 - x1) * s, abs(y2 - y1) * s
                    short.append(min(w, h))
            small = round(float(np.mean(np.array(short) < SMALL_PX)), 4) if short else None
            entry[str(target)] = {"short_side_px": pct(short), f"share_short_side_under_{SMALL_PX}px": small}
        sizes_report[split] = entry
    report["box_sizes"] = sizes_report

    # ---- unlisted frames
    listed = set().union(*[set(b) for b in split_boxes.values()]) if split_boxes else set()
    all_boxes = {p: b for bx in split_boxes.values() for p, b in bx.items()}
    report["frames_with_empty_label"] = sum(1 for b in all_boxes.values() if not b)
    unlisted = [p for p in images if p not in listed or not all_boxes.get(p)]
    per_seq_split = {}
    for split, seqs in seqs_by_split.items():
        for s in seqs:
            per_seq_split[s] = split
    by_seq = defaultdict(list)
    for p in unlisted:
        by_seq[seq_of[p]].append(p)
    detail = {}
    for s, ps in sorted(by_seq.items()):
        listed_nums = sorted(int(p.stem) for p in images if seq_of[p] == s and p in listed and p.stem.isdigit())
        nums = [int(p.stem) for p in ps if p.stem.isdigit()]
        between = sum(1 for n in nums if listed_nums and listed_nums[0] < n < listed_nums[-1])
        detail[s] = {"split": per_seq_split.get(s, "none (sequence not in any list)"), "unlisted": len(ps),
                     "listed": len(listed_nums), "between_first_and_last_listed": between}
    report["unlisted"] = {"frames": len(unlisted), "by_split": dict(Counter(d["split"] for d in detail.values()
                                                                             for _ in range(d["unlisted"]))),
                          "sequences": detail}

    # ---- montage of unlisted frames (and full-resolution copies) for a by-eye crack check
    audit_dir = OUT_DIR / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    if unlisted:
        picked = random.Random(0).sample(unlisted, min(args.montage, len(unlisted)))
        sample = sorted(picked, key=lambda p: (seq_of[p], natural(p)))
        report["unlisted"]["montage_sample"] = [str(p.relative_to(root)) for p in sample]
        tiles = []
        for p in sample:
            with Image.open(p) as im:
                im = im.convert("RGB")
                im.thumbnail((960, 960))
                tiles.append(im)
        cols = 4
        tw, th = max(t.width for t in tiles), max(t.height for t in tiles)
        for k in range(0, len(tiles), 8):     # 8 per sheet keeps each sheet readable
            chunk = tiles[k:k + 8]
            rows = (len(chunk) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * tw, rows * th), "white")
            for i, t in enumerate(chunk):
                sheet.paste(t, ((i % cols) * tw, (i // cols) * th))
            sheet.save(audit_dir / f"unlisted_{k // 8 + 1}.jpg", quality=88)

    # ---- summary flags against the paper
    report["vs_paper"] = {
        split: {"images": [counts.get(split, {}).get("images"), PAPER["images"][split]],
                "sequences": [counts.get(split, {}).get("sequences"), PAPER["sequences"][split]]}
        for split in ("train", "val", "test")}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_audit.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: report[k] for k in ("vs_paper", "layout")}, indent=2, default=str))
    print("wrote", OUT_DIR / "dataset_audit.json")


if __name__ == "__main__":
    main()
