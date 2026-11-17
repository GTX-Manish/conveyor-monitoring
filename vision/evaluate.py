"""Evaluate the pretrained crack detector: threshold on BeltCrack9kd val, then test scored once.

    python -m vision.evaluate --val beltcrack_yolo960 --test BeltCrack9kd_prep

- Val: the 1,961 val frames of beltcrack_yolo960 (YOLO labels). Chooses the operating threshold with the rule
  "lowest conf with P >= 0.9 if R there >= 0.6, else max F1" (P and R at IoU 0.5), then freezes it with the model's
  sha256 in vision_eval.json.
- Test: the 1,948 frames of BeltCrack9kd_prep/annotations/test_coco.json, scored once (the script refuses to score
  test again): COCO-style mAP50 and mAP50-95 on detections at conf 0.001 (101-point interpolation, up to 100
  detections per image), and P / R / F1 at conf 0.3 (the author's default) and at the frozen threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from vision.detector import AUTHOR_CONF, EVAL_PATH, MODEL_PATH, NMS_IOU, Detector

EVAL_CONF = 0.001
MAX_DET = 100
TARGET_P, MIN_R = 0.90, 0.60
IOU_THRESHOLDS = np.round(np.arange(0.5, 0.951, 0.05), 2)
CONF_GRID = np.round(np.arange(0.001, 1.0, 0.001), 3)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every box in a (n, 4) with every box in b (m, 4); boxes are x1, y1, x2, y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match(dets: np.ndarray, scores: np.ndarray, gts: np.ndarray, thr: float) -> np.ndarray:
    """COCO matching in one image: detections by falling score, each takes the unmatched labelled box with the
    highest IoU >= thr. Returns a true-positive flag per detection, in the order given."""
    order = np.argsort(-scores, kind="stable")
    ious = iou_matrix(dets, gts)
    taken = np.zeros(len(gts), bool)
    tp = np.zeros(len(dets), bool)
    for i in order:
        if len(gts) == 0:
            break
        cand = np.where(~taken, ious[i], -1.0)
        j = int(cand.argmax())
        if cand[j] >= thr:
            taken[j] = tp[i] = True
    return tp


def pooled(per_image: list[tuple[np.ndarray, np.ndarray, np.ndarray]], thr: float):
    """Scores and TP flags of all detections over all images, sorted by falling score, and the labelled-box count."""
    scores, tps, n_gt = [], [], 0
    for dets, sc, gts in per_image:
        scores.append(sc)
        tps.append(match(dets, sc, gts, thr))
        n_gt += len(gts)
    scores = np.concatenate(scores) if scores else np.zeros(0)
    tps = np.concatenate(tps) if tps else np.zeros(0, bool)
    order = np.argsort(-scores, kind="mergesort")
    return scores[order], tps[order], n_gt


def average_precision(tp_sorted: np.ndarray, n_gt: int) -> float:
    """COCO 101-point interpolated AP from TP flags sorted by falling score."""
    if n_gt == 0:
        return float("nan")
    tp = np.cumsum(tp_sorted)
    fp = np.cumsum(~tp_sorted)
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(recall, r, side="left")
        ap += precision[k] if k < len(precision) else 0.0
    return ap / 101


def pr_at(scores_sorted: np.ndarray, tp_sorted: np.ndarray, n_gt: int, conf: float) -> dict:
    """P, R, F1 for detections with score >= conf (matching is by score order, so a cut is exact)."""
    k = int(np.searchsorted(-scores_sorted, -conf, side="right"))
    tp = int(tp_sorted[:k].sum())
    p = tp / k if k else 0.0
    r = tp / n_gt if n_gt else 0.0
    return {"conf": round(float(conf), 4), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r) if p + r else 0.0, 4), "detections": k, "true_positives": tp}


def run_detector(det: Detector, items: list[tuple[Path, np.ndarray]], label: str):
    out = []
    for n, (path, gts) in enumerate(items, 1):
        ds = det.detect(path, conf=EVAL_CONF, max_det=MAX_DET)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in ds]).reshape(-1, 4)
        out.append((boxes, np.array([d.score for d in ds]), gts))
        if n % 250 == 0:
            print(f"  {label}: {n}/{len(items)}", flush=True)
    return out


def val_items(root: Path):
    from PIL import Image

    items = []
    for p in sorted((root / "images" / "val").glob("*.jpg")):
        w, h = Image.open(p).size
        rows = [[float(v) for v in line.split()[1:]]
                for line in (root / "labels" / "val" / f"{p.stem}.txt").read_text().splitlines() if line.strip()]
        g = np.array([[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]
                      for cx, cy, bw, bh in rows]).reshape(-1, 4)
        items.append((p, g))
    return items


def test_items(root: Path):
    cj = json.loads((root / "annotations" / "test_coco.json").read_text())
    boxes: dict[int, list] = {}
    for a in cj["annotations"]:
        x, y, w, h = a["bbox"]
        boxes.setdefault(a["image_id"], []).append([x, y, x + w, y + h])
    return [(root / im["file_name"], np.array(boxes.get(im["id"], [])).reshape(-1, 4)) for im in cj["images"]]


def choose_threshold(scores, tps, n_gt) -> dict:
    curve = [pr_at(scores, tps, n_gt, c) for c in CONF_GRID]
    p = np.array([c["precision"] for c in curve])
    f1 = np.array([c["f1"] for c in curve])
    ok = np.where(p >= TARGET_P)[0]
    cand_p = curve[int(ok[0])] if len(ok) else None
    cand_f = curve[int(f1.argmax())]
    use_p = cand_p is not None and cand_p["recall"] >= MIN_R
    chosen = cand_p if use_p else cand_f
    return {"rule": f"lowest conf with P >= {TARGET_P} if R there >= {MIN_R}, else max F1 (val frames, IoU 0.5)",
            "chosen": "precision>=0.9" if use_p else "max_f1", "threshold": chosen["conf"], "at_threshold": chosen,
            "candidate_precision_0.9": cand_p, "candidate_max_f1": cand_f}, curve


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default="beltcrack_yolo960", help="folder with images/val and labels/val (YOLO)")
    ap.add_argument("--test", default="BeltCrack9kd_prep", help="folder with annotations/test_coco.json")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)

    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    ev = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else {}
    if "test" in ev:
        raise SystemExit(f"test was already scored in {EVAL_PATH}; not scoring it again")
    det = Detector(threads=args.threads)
    settings = {"eval_conf": EVAL_CONF, "nms_iou": NMS_IOU, "max_det": MAX_DET, "match_iou_pr": 0.5,
                "map": "COCO-style, 101-point interpolation, one class, IoU 0.50:0.95"}

    if "freeze" not in ev:
        items = val_items(Path(args.val))
        res = run_detector(det, items, "val")
        s, t, n = pooled(res, 0.5)
        choice, curve = choose_threshold(s, t, n)
        aps = {thr: average_precision(pooled(res, thr)[1], n) for thr in IOU_THRESHOLDS}
        ev = {
            "model": json.loads((MODEL_PATH.parent / "model_source.json").read_text()),
            "settings": settings,
            "val": {"source": f"{args.val}/images/val (BeltCrack9kd val, 960 px long side)", "images": len(items),
                    "boxes": n, "map50": round(aps[0.5], 4), "map50_95": round(float(np.mean(list(aps.values()))), 4),
                    "at_author_conf_0.3": pr_at(s, t, n, AUTHOR_CONF)},
            "threshold_choice": choice,
            "freeze": {"threshold": choice["threshold"], "model_sha256": sha,
                       "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        }
        EVAL_PATH.write_text(json.dumps(ev, indent=1) + "\n")
        with open(EVAL_PATH.parent / "pr_curve_val.csv", "w") as f:
            f.write("conf,precision,recall,f1\n")
            for c in curve:
                f.write(f"{c['conf']},{c['precision']},{c['recall']},{c['f1']}\n")
        print("val:", json.dumps({k: ev["val"][k] for k in ("map50", "map50_95")}), "threshold", choice["threshold"],
              f"({choice['chosen']})")

    assert ev["freeze"]["model_sha256"] == sha, "best.onnx changed after the threshold was frozen"
    thr = ev["freeze"]["threshold"]
    items = test_items(Path(args.test))
    res = run_detector(det, items, "test")
    s, t, n = pooled(res, 0.5)
    aps = {thr_: average_precision(pooled(res, thr_)[1], n) for thr_ in IOU_THRESHOLDS}
    ev["test"] = {"source": f"{args.test}/annotations/test_coco.json (BeltCrack9kd test, 1024 px long side)",
                  "images": len(items), "boxes": n, "map50": round(aps[0.5], 4),
                  "map50_95": round(float(np.mean(list(aps.values()))), 4),
                  "at_author_conf_0.3": pr_at(s, t, n, AUTHOR_CONF), "at_frozen_threshold": pr_at(s, t, n, thr),
                  "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    EVAL_PATH.write_text(json.dumps(ev, indent=1) + "\n")
    print("test:", json.dumps(ev["test"], indent=1))


if __name__ == "__main__":
    main()
