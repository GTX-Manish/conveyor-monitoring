# ---------------------------------------------------------------------------------------
# BeltCrack9kd preparation for the authors' BeltCrackDet code.
#
# Train frames are stored at exactly the size the authors' loader resizes them to
# (letterbox scale to 512, int() sizes, bicubic), so their loader's own resize becomes a
# no-op and training sees the same pixels as on full-resolution files (apart from JPEG
# re-encoding at quality 95). Box coordinates use the same int() truncation as the loader.
# Test frames are kept at 1024 on the long side, because the authors' test script
# stretches each frame to 512x512 without letterbox.
# ---------------------------------------------------------------------------------------
import json
import os
import re
import shutil
import subprocess
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

TRAIN_SIZE = 512
TEST_LONG_SIDE = 1024
JPEG_QUALITY = 95
HISTORY = 5                      # frames per sample in the authors' model (key frame + 4 before)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
BICUBIC = getattr(Image, "Resampling", Image).BICUBIC
NUM = r"-?\d+(?:\.\d+)?"
LINE_RE = re.compile(rf"^(\S+?\.(?:jpg|jpeg|png|bmp))((?:\s+{NUM},{NUM},{NUM},{NUM},{NUM})*)\s*$", re.I)


# ----------------------------------- archives -----------------------------------------
def is_first_volume(p):
    """The archive file to hand to the extractor; later parts of multi-part sets are skipped."""
    n = p.name.lower()
    m = re.search(r"\.part(\d+)\.rar$", n)
    if m:
        return int(m.group(1)) == 1
    m = re.search(r"\.(?:zip|7z)\.(\d+)$", n)
    if m:
        return int(m.group(1)) == 1
    return bool(re.search(r"\.(zip|7z|rar|tar|tgz|tar\.gz)$", n))


def find_archives(folders, skip_names=(), pattern="*"):
    """First volumes of archives under folders whose file name matches pattern, e.g. "*.rar"."""
    found = []
    for folder in folders:
        folder = Path(folder)
        if not folder.exists():
            continue
        for p in sorted(folder.rglob(pattern)):
            if p.is_file() and is_first_volume(p) and p.name not in skip_names and "runs" not in p.parts:
                found.append(p)
    return found


def extract_archive(archive, out, password):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"extracting {archive.name} ...", flush=True)
    low = archive.name.lower()
    unrar = shutil.which("unrar")
    seven = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")
    if low.endswith(".rar") and unrar:
        cmd, ok = [unrar, "x", "-o+", "-inul", f"-p{password}", str(archive), str(out) + os.sep], (0,)
    elif seven:
        cmd, ok = [seven, "x", "-y", "-bso0", "-bsp0", f"-p{password}", f"-o{out}", str(archive)], (0, 1)
    elif low.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out, pwd=password.encode())
        return
    elif low.endswith((".tar", ".tgz", ".tar.gz")):
        shutil.unpack_archive(str(archive), str(out))
        return
    else:
        raise RuntimeError(f"No extractor for {archive.name}: install unrar or 7-Zip "
                           "(Colab/Ubuntu: apt-get install -y unrar p7zip-full).")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in ok:
        raise RuntimeError(f"Extracting {archive.name} failed (exit {r.returncode}). Check the password and "
                           f"that every part of a multi-part archive is present.\n{r.stdout[-800:]}{r.stderr[-800:]}")


def extract_all(folders, out, password, skip_names=(), pattern="*"):
    archives = find_archives(folders, skip_names, pattern)
    for a in archives:
        extract_archive(a, out, password)
    for a in find_archives([out]):                 # archives packed inside the archive
        extract_archive(a, a.parent, password)
        a.unlink()
    return archives


# Labelled images per split in the authors' annotation files.
KNOWN_SPLITS = {"BeltCrack9kd": {"train": 7697, "test": 1948},
                "BeltCrack14ks": {"train": 8773, "test": 2718}}


def check_dataset(info, expect="BeltCrack9kd"):
    """Identify the data by its split sizes (archive names are not reliable). Stops on the other dataset."""
    # images without boxes are left out of the lists (as coco_to_txt.py does) but count towards the published sizes
    got = {sp: info[sp]["images"] + info[sp].get("without_boxes", 0) for sp in ("train", "test")}
    for name, ref in KNOWN_SPLITS.items():
        if got == ref:
            if name != expect:
                raise RuntimeError(f"These annotations are {name} ({got}), not {expect}.")
            return f"OK: matches {name} ({got['train']} train / {got['test']} test images)"
    return (f"WARNING: {got['train']} train / {got['test']} test images do not match the published "
            f"{expect} counts {KNOWN_SPLITS[expect]}. Check the annotation files listed above.")


def locate_root(base):
    return Path(base)          # the whole folder is searched for images and annotation files


# ----------------------------------- annotations --------------------------------------
class PathIndex:
    """Matches paths written on the authors' machine (/home/hjh/BeltCrack9kd/...) to local files."""

    def __init__(self, files, max_k=6):
        self.max_k = max_k
        self.idx = [defaultdict(list) for _ in range(max_k + 1)]
        for p in files:
            parts = p.parts
            for k in range(1, min(max_k, len(parts)) + 1):
                self.idx[k]["/".join(parts[-k:])].append(p)

    def resolve(self, name):
        parts = [x for x in str(name).replace("\\", "/").split("/") if x]
        for k in range(min(self.max_k, len(parts)), 0, -1):
            hits = self.idx[k].get("/".join(parts[-k:]))
            if hits:
                return hits[0] if len(hits) == 1 else None
        return None

    def exact(self, parts):
        """The one file whose last len(parts) path parts are exactly parts, else None (no shorter fallback)."""
        if not 1 <= len(parts) <= self.max_k:
            return None
        hits = self.idx[len(parts)].get("/".join(parts))
        return hits[0] if hits and len(hits) == 1 else None


def split_of(rel):
    for part in reversed(Path(rel).parts):
        low = part.lower()
        for key in ("train", "test", "val"):
            if key in low:
                return key
    return None


def frame_no(p):
    m = re.search(r"(\d+)$", Path(p).stem)
    return int(m.group(1)) if m else None


def read_txt(path, index):
    entries, unresolved = [], 0
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        p = index.resolve(m.group(1))
        if p is None:
            unresolved += 1
            continue
        boxes = [tuple(int(float(v)) for v in b.split(",")) for b in m.group(2).split()]
        entries.append((p, boxes))
    return entries, unresolved


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def is_coco(data):
    return isinstance(data, dict) and "images" in data and "annotations" in data


def is_labelme(data):
    return isinstance(data, dict) and isinstance(data.get("shapes"), list)


def labelme_image(path, data, index, image_set):
    """The image a LabelMe file belongs to: same folder and stem, then imagePath, then <split>/<seq>/<n>.<ext>."""
    path = Path(path)
    exts = sorted(IMG_EXT)
    for ext in exts:
        for p in (path.with_suffix(ext), path.with_suffix(ext.upper())):
            if p in image_set:
                return p
    ip = data.get("imagePath")
    if ip:
        p = Path(os.path.normpath(path.parent / str(ip).replace("\\", "/")))
        if p in image_set:
            return p
    if len(path.parts) >= 3:
        split, seq = path.parts[-3], path.parts[-2]
        for ext in exts:
            for e in (ext, ext.upper()):
                p = index.exact([split, seq, path.stem + e])
                if p is not None:
                    return p
    return None


def labelme_record(path, data, index, image_set, stats):
    """One LabelMe file as a COCO-style record (what the authors' LabelMe -> COCO step gives), or None if
    no image matches. Box per shape with >= 2 points: x, y = min; w, h = max - min, as floats."""
    p = labelme_image(path, data, index, image_set)
    if p is None:
        return None
    anns = []
    for s in data["shapes"]:
        stats["labels"][str(s.get("label"))] += 1
        stats["shape_types"][str(s.get("shape_type"))] += 1
        pts = s.get("points") or []
        if len(pts) < 2:
            stats["short_shapes"] += 1
            continue
        xs, ys = [float(q[0]) for q in pts], [float(q[1]) for q in pts]
        x, y = min(xs), min(ys)
        anns.append({"bbox": [x, y, max(xs) - x, max(ys) - y], "category_id": 1, "iscrowd": 0})
    return {"path": p, "anns": anns}


def read_coco(path, index, data=None):
    if data is None:
        data = load_json(path)
    if not is_coco(data):
        return None
    recs, unresolved = {}, 0
    for im in data["images"]:
        p = index.resolve(im.get("file_name", ""))
        if p is None:
            unresolved += 1
            continue
        recs[im["id"]] = {"id": im["id"], "path": p, "anns": []}
    for a in data["annotations"]:
        r = recs.get(a.get("image_id"))
        if r is not None:
            r["anns"].append({"bbox": [float(v) for v in a["bbox"]],
                              "category_id": int(a.get("category_id", 1)),
                              "iscrowd": int(a.get("iscrowd", 0))})
    cats = data.get("categories") or [{"id": 1, "name": "crack"}]
    return list(recs.values()), unresolved, cats


def coco_to_lines(recs):
    """Same rule as the authors' utils_coco/coco_to_txt.py: annotated images only, int() corners."""
    lines = []
    for r in recs:
        if not r["anns"]:
            continue
        boxes = []
        for a in r["anns"]:
            x, y, w, h = a["bbox"]
            x0, y0 = int(x), int(y)
            boxes.append((x0, y0, x0 + int(w), y0 + int(h), a["category_id"] - 1))
        lines.append((r["path"], boxes))
    return lines


def lines_to_coco(lines):
    recs = []
    for i, (p, boxes) in enumerate(lines, 1):
        recs.append({"id": i, "path": p, "anns": [
            {"bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "category_id": c + 1, "iscrowd": 0}
            for x1, y1, x2, y2, c in boxes]})
    return recs


# ----------------------------------- resizing -----------------------------------------
def target_size(iw, ih, mode):
    if mode == "train":                              # same arithmetic as utils/dataloader_for_CRACK.py
        scale = min(TRAIN_SIZE / iw, TRAIN_SIZE / ih)
        return int(iw * scale), int(ih * scale)
    s = min(1.0, TEST_LONG_SIDE / max(iw, ih))
    return max(1, int(round(iw * s))), max(1, int(round(ih * s)))


def resize_job(job):
    src, dst, mode = job
    with Image.open(src) as im:
        im.load()
        iw, ih = im.size
        nw, nh = target_size(iw, ih, mode)
        if im.mode != "RGB":
            im = im.convert("RGB")
        if (nw, nh) != (iw, ih):
            im = im.resize((nw, nh), BICUBIC)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        im.save(dst, quality=JPEG_QUALITY, subsampling=0)
    return src, (iw, ih, nw, nh)


def _seq_id(rel, used):
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", rel.as_posix()).strip("_") or "seq"
    sid, k = base, 2
    while sid in used:
        sid, k = f"{base}_{k}", k + 1
    used.add(sid)
    return sid


# ----------------------------------- main ----------------------------------------------
def prepare(root, out, workers=None, log=print):
    root, out = Path(root), Path(out)
    skip = {"runs", out.name}  # earlier training outputs and this script's own output
    images = [p for p in root.rglob("*")
              if p.is_file() and p.suffix.lower() in IMG_EXT and not skip & set(p.relative_to(root).parts)]
    if not images:
        raise RuntimeError(f"No images found under {root}")
    index = PathIndex(images)
    log(f"{len(images)} image files under {root}")

    txt, coco, unresolved, found = defaultdict(list), {}, defaultdict(int), []
    cats = [{"id": 1, "name": "crack"}]
    for f in sorted(root.rglob("*.txt")):
        entries, unres = read_txt(f, index)
        if not entries and not unres:
            continue
        sp = split_of(f.relative_to(root))
        found.append(f"{f.relative_to(root)}  [{sp}] {len(entries)} lines, {unres} unmatched")
        if sp:
            txt[sp] += entries
            unresolved[sp] += unres
    image_set = set(images)
    lm = defaultdict(list)                                   # split -> LabelMe records (one JSON per image)
    lm_stats = {"labels": defaultdict(int), "shape_types": defaultdict(int), "short_shapes": 0}
    lm_count = defaultdict(lambda: defaultdict(int))
    for f in sorted(root.rglob("*.json")):
        data = load_json(f)                                  # each file is parsed once
        sp = split_of(f.relative_to(root))
        if is_labelme(data):
            r = labelme_record(f, data, index, image_set, lm_stats)
            c = lm_count[sp]
            if r is None:
                c["unmatched"] += 1
            elif r["anns"]:
                c["with_boxes"] += 1
                c["boxes"] += len(r["anns"])
                lm[sp].append(r)
            else:
                c["without_boxes"] += 1
                lm[sp].append(r)
            continue
        res = read_coco(f, index, data)
        if res is None:
            continue
        recs, unres, c = res
        found.append(f"{f.relative_to(root)}  [{sp}] {len(recs)} images, {unres} unmatched")
        if sp and sp not in coco:
            coco[sp], cats = recs, c
            unresolved[sp] += unres
    for sp in ("train", "val", "test", None):
        if sp in lm_count:
            c = lm_count[sp]
            found.append(f"LabelMe json [{sp}] {c['with_boxes']} images with boxes, {c['boxes']} boxes, "
                         f"{c['without_boxes']} images without boxes, {c['unmatched']} unmatched"
                         + ("  (merged into train)" if sp == "val" else ""))
    for line in found:
        log("  annotation file: " + line)
    if lm_count:
        log(f"  LabelMe shape labels: {dict(lm_stats['labels'])}  shape_types: {dict(lm_stats['shape_types'])}")

    # The paper's 9kd train set is 33 sequences = the archive's 23 train + 10 val folders; test stays test.
    lm_train = lm["train"] + lm["val"]
    for i, r in enumerate(lm["test"], 1):
        r["id"] = i
    without = {"train": 0, "test": 0}
    if txt["train"]:
        train = txt["train"]
    elif coco.get("train"):
        train = coco_to_lines(coco["train"])
        without["train"] = sum(not r["anns"] for r in coco["train"])
    else:
        train = coco_to_lines(lm_train)
        without["train"] = sum(not r["anns"] for r in lm_train)
        unresolved["train"] += lm_count["train"]["unmatched"] + lm_count["val"]["unmatched"]
    if txt["test"]:
        test_lines = txt["test"]
        test_coco = coco.get("test") or lines_to_coco(test_lines)
    elif coco.get("test"):
        test_lines, test_coco = coco_to_lines(coco["test"]), coco["test"]
        without["test"] = sum(not r["anns"] for r in coco["test"])
    else:
        test_lines = coco_to_lines(lm["test"])
        test_coco = lm["test"] or lines_to_coco(test_lines)  # COCO test file keeps images without boxes
        without["test"] = sum(not r["anns"] for r in lm["test"])
        unresolved["test"] += lm_count["test"]["unmatched"]
    if not train or not test_lines:
        raise RuntimeError("Could not find both train and test annotations. Files found:\n  " + "\n  ".join(found))
    notes = []
    if lm_count["val"] and not (txt["train"] or coco.get("train")):
        notes.append(f"LabelMe val ({len(lm['val'])} images) merged into train: the paper's 9kd train set is "
                     "the 23 train + 10 val sequences.")
    if lm_stats["short_shapes"]:
        notes.append(f"{lm_stats['short_shapes']} LabelMe shapes with fewer than 2 points were skipped.")
    if txt["val"] or coco.get("val"):
        notes.append("A val annotation file exists but is not used: the authors' 9kd config validates on test.")
    for sp, n in unresolved.items():
        if n:
            notes.append(f"{n} {sp} annotation entries did not match an image file and were skipped.")
    n_listed = len(train) + len(test_lines)
    if sum(unresolved.values()) > 0.01 * max(1, n_listed):
        raise RuntimeError("More than 1% of annotation entries match no image file:\n  " + "\n  ".join(notes))

    # folders, frame numbers and the history frames each sample needs
    mode = {}
    for p, _ in train:
        mode.setdefault(p.parent, "train")
    shared = set()
    for p in [p for p, _ in test_lines] + [r["path"] for r in test_coco]:
        if mode.get(p.parent) == "train":
            shared.add(p.parent)
        mode[p.parent] = "test"
    if shared:
        notes.append(f"{len(shared)} sequence folders appear in both train and test.")

    frames = defaultdict(dict)
    for p in images:
        if p.parent in mode:
            n = frame_no(p)
            if n is not None and (n not in frames[p.parent] or p.suffix.lower() == ".jpg"):
                frames[p.parent][n] = p
    listed = [p for p, _ in train] + [p for p, _ in test_lines] + [r["path"] for r in test_coco]
    bad = [p for p in listed if frame_no(p) is None]
    if bad:
        raise RuntimeError(f"{len(bad)} labelled images have no frame number in their name, e.g. {bad[0]}")

    used, seq = set(), {}
    for folder in sorted(mode):
        seq[folder] = _seq_id(folder.relative_to(root), used)

    jobs, gaps = {}, 0
    for p in listed:
        folder, n = p.parent, frame_no(p)
        for k in range(HISTORY):
            m = max(n - k, 0)                       # same clamp as the authors' loaders
            dst = out / "frames" / seq[folder] / f"{m}.jpg"
            if dst in jobs and not (k == 0 and jobs[dst][0] != p):
                continue
            src = p if k == 0 else frames[folder].get(m)
            if src is None:                       # the authors' loader would crash here
                gaps += 1
                src = frames[folder][min(frames[folder], key=lambda x: (abs(x - m), x > m))]
            jobs[dst] = (src, dst, mode[folder])
    if gaps:
        notes.append(f"{gaps} history frames were missing on disk and were filled with the nearest frame.")

    log(f"resizing {len(jobs)} frames ...")
    sizes = {}
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        for i, (src, size) in enumerate(ex.map(resize_job, jobs.values(), chunksize=16), 1):
            sizes[src] = size
            if i % 1000 == 0:
                log(f"  {i}/{len(jobs)}")

    def rel(p):
        return f"frames/{seq[p.parent]}/{frame_no(p)}.jpg"

    ann = out / "annotations"
    ann.mkdir(parents=True, exist_ok=True)
    n_train_boxes = 0
    with open(ann / "train.txt", "w") as f:
        for p, boxes in train:
            iw, ih, nw, nh = sizes[p]
            out_boxes = [(int(x1 * nw / iw), int(y1 * nh / ih), int(x2 * nw / iw), int(y2 * nh / ih), c)
                         for x1, y1, x2, y2, c in boxes]
            n_train_boxes += len(out_boxes)
            f.write(rel(p) + "".join(" %d,%d,%d,%d,%d" % b for b in out_boxes) + "\n")
    n_test_boxes = 0
    with open(ann / "test.txt", "w") as f:
        for p, boxes in test_lines:
            iw, ih, nw, nh = sizes[p]
            sx, sy = nw / iw, nh / ih
            out_boxes = [(int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy)), c)
                         for x1, y1, x2, y2, c in boxes]
            n_test_boxes += len(out_boxes)
            f.write(rel(p) + "".join(" %d,%d,%d,%d,%d" % b for b in out_boxes) + "\n")
    cj = {"info": {"description": "BeltCrack9kd test split, rescaled for BeltCrackDet replication"},
          "licenses": [], "categories": cats, "images": [], "annotations": []}
    for r in test_coco:
        iw, ih, nw, nh = sizes[r["path"]]
        sx, sy = nw / iw, nh / ih
        cj["images"].append({"id": r["id"], "file_name": rel(r["path"]), "width": nw, "height": nh})
        for a in r["anns"]:
            x, y, w, h = a["bbox"]
            cj["annotations"].append({"id": len(cj["annotations"]) + 1, "image_id": r["id"],
                                      "category_id": a["category_id"], "iscrowd": a["iscrowd"],
                                      "bbox": [x * sx, y * sy, w * sx, h * sy], "area": w * sx * h * sy})
    (ann / "test_coco.json").write_text(json.dumps(cj))

    src_boxes = {"train": sum(len(b) for _, b in train), "test": sum(len(b) for _, b in test_lines)}
    assert n_train_boxes == src_boxes["train"] and n_test_boxes == src_boxes["test"]
    info = {
        "source_root": str(root),
        "train": {"images": len(train), "boxes": n_train_boxes, "sequences": len({p.parent for p, _ in train}),
                  "without_boxes": without["train"]},
        "test": {"images": len(test_lines), "boxes": n_test_boxes, "without_boxes": without["test"],
                 "sequences": len({p.parent for p, _ in test_lines}),
                 "coco_images": len(cj["images"]), "coco_boxes": len(cj["annotations"])},
        "frames_written": len(jobs),
        "source_sizes": sorted({f"{s[0]}x{s[1]}" for s in sizes.values()}),
        "train_frame_rule": f"letterbox-exact {TRAIN_SIZE}", "test_frame_rule": f"long side {TEST_LONG_SIDE}",
        "jpeg_quality": JPEG_QUALITY, "notes": notes,
    }
    (out / "prep_info.json").write_text(json.dumps(info, indent=1))
    for n in notes:
        log("NOTE: " + n)
    return info


def make_zip(folder, zip_path):
    folder, zip_path = Path(folder), Path(zip_path)
    tmp = zip_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, Path(folder.name) / p.relative_to(folder))
    os.replace(tmp, zip_path)
    return zip_path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Prepare BeltCrack9kd for the BeltCrackDet Colab notebook.")
    ap.add_argument("src", help="folder holding the downloaded 9kd archive(s), or the extracted dataset")
    ap.add_argument("--out", default="BeltCrack9kd_prep", help="output folder (default: BeltCrack9kd_prep)")
    ap.add_argument("--password", default="cv205")
    ap.add_argument("--pattern", default="*.rar",
                    help='archives to extract (default "*.rar", e.g. BeltCrack14ks.part01.rar ...)')
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    raw = out.parent / (out.name + "_raw")
    archives = find_archives([src], pattern=a.pattern)
    if raw.exists() and any(raw.iterdir()):
        print("using the earlier extraction in", raw)
        root = locate_root(raw)
    elif archives:
        print("extracting", [x.name for x in archives])
        extract_all([src], raw, a.password, pattern=a.pattern)
        root = locate_root(raw)
    else:
        root = locate_root(src)
    print("dataset root:", root)
    info = prepare(root, out, workers=a.workers)
    print(check_dataset(info))
    z = make_zip(out, out.parent / "BeltCrack9kd_prep.zip")
    print(f"\nDone. Upload {z} to your Google Drive dataset folder and run the notebook.")
    if raw.exists():
        print(f"You can delete the extracted copy: {raw}")
