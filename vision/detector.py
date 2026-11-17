"""Crack detector: the pretrained BeltCrack YOLOv8 model (ONNX, CPU).

Model: vision/artifacts/best.onnx, by Julián Neira Nova (github.com/JSNN15/conveyor_crack_detection_2026, AGPL-3.0);
see vision/artifacts/model_source.json and vision/artifacts/NOTICE.md. Pre- and post-processing follow the author's
det.js:

- input 640 x 384 (W x H): letterbox scale s = min(640 / W, 384 / H), resized size rounded half up, centred on a
  canvas filled with 114, RGB / 255, NCHW float32;
- output [1, 5, N]: rows cx, cy, w, h, score in input pixels; the letterbox is undone and boxes are clipped to the
  image; greedy NMS drops a box whose IoU with a kept, higher-scoring box is above 0.45.

The only deliberate difference from det.js: the canvas offset is a whole pixel ((640 - nw) // 2), where the browser
can draw at a half pixel. Both are undone with the same offset, so boxes stay in place.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ART = Path(__file__).resolve().parent / "artifacts"
MODEL_PATH = ART / "best.onnx"
EVAL_PATH = ART / "vision_eval.json"
IW, IH = 640, 384
PAD = 114
NMS_IOU = 0.45
AUTHOR_CONF = 0.3          # det.js default


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float

    def as_dict(self) -> dict:
        return {"x1": round(self.x1, 1), "y1": round(self.y1, 1), "x2": round(self.x2, 1), "y2": round(self.y2, 1),
                "score": round(self.score, 4)}


@dataclass(frozen=True)
class Letterbox:
    width: int             # original image size
    height: int
    scale: float
    dx: int                # offset of the resized image on the 640 x 384 canvas
    dy: int


def _round_half_up(x: float) -> int:  # JavaScript Math.round, as in det.js (Python's round() rounds half to even)
    return int(math.floor(x + 0.5))


def letterbox(img: Image.Image) -> tuple[np.ndarray, Letterbox]:
    """1 x 3 x 384 x 640 float32 tensor (RGB / 255) and the geometry needed to map boxes back."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(IW / w, IH / h)
    nw, nh = _round_half_up(w * s), _round_half_up(h * s)
    dx, dy = (IW - nw) // 2, (IH - nh) // 2
    canvas = Image.new("RGB", (IW, IH), (PAD, PAD, PAD))
    canvas.paste(img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.Resampling.BILINEAR), (dx, dy))
    x = np.asarray(canvas, dtype=np.float32) / 255.0
    return x.transpose(2, 0, 1)[None].copy(), Letterbox(w, h, s, dx, dy)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = NMS_IOU) -> list[int]:
    """Indices kept by greedy NMS, highest score first (boxes are x1, y1, x2, y2)."""
    keep: list[int] = []
    for i in np.argsort(-scores, kind="stable"):
        b = boxes[i]
        ok = True
        for k in keep:
            c = boxes[k]
            iw = max(0.0, min(b[2], c[2]) - max(b[0], c[0]))
            ih = max(0.0, min(b[3], c[3]) - max(b[1], c[1]))
            inter = iw * ih
            union = (b[2] - b[0]) * (b[3] - b[1]) + (c[2] - c[0]) * (c[3] - c[1]) - inter
            if inter / (union + 1e-9) > iou:
                ok = False
                break
        if ok:
            keep.append(int(i))
    return keep


def decode(output: np.ndarray, lb: Letterbox, conf: float, iou: float = NMS_IOU,
           max_det: int | None = None) -> list[Detection]:
    """Model output [1, 5, N] -> detections in original image pixels, best first."""
    rows = np.asarray(output, dtype=np.float32)[0]
    m = rows[4] >= conf
    cx, cy, bw, bh, sc = (r[m] for r in rows)
    x1 = np.clip((cx - bw / 2 - lb.dx) / lb.scale, 0, lb.width)
    y1 = np.clip((cy - bh / 2 - lb.dy) / lb.scale, 0, lb.height)
    x2 = np.clip((cx + bw / 2 - lb.dx) / lb.scale, 0, lb.width)
    y2 = np.clip((cy + bh / 2 - lb.dy) / lb.scale, 0, lb.height)
    boxes = np.stack([x1, y1, x2, y2], 1)
    keep = nms(boxes, sc, iou)[:max_det]
    return [Detection(*map(float, boxes[i]), float(sc[i])) for i in keep]


def frozen_threshold(path: Path = EVAL_PATH) -> float | None:
    """The operating threshold chosen on val (vision_eval.json), or None before it exists."""
    try:
        return float(json.loads(path.read_text())["freeze"]["threshold"])
    except (FileNotFoundError, KeyError, ValueError):
        return None


class Detector:
    def __init__(self, model_path: Path | str = MODEL_PATH, conf: float | None = None, threads: int = 1):
        import onnxruntime as ort  # imported here so the rest of the repo works without it

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        if list(shape) != [1, 3, IH, IW]:
            raise ValueError(f"expected model input [1, 3, {IH}, {IW}], got {shape}")
        self.conf = conf if conf is not None else (frozen_threshold() or AUTHOR_CONF)

    def raw(self, image: Image.Image | Path | str) -> tuple[np.ndarray, Letterbox]:
        img = image if isinstance(image, Image.Image) else Image.open(image)
        x, lb = letterbox(img)
        return self.session.run(None, {self.input_name: x})[0], lb

    def detect(self, image: Image.Image | Path | str, conf: float | None = None,
               max_det: int | None = None) -> list[Detection]:
        out, lb = self.raw(image)
        return decode(out, lb, self.conf if conf is None else conf, max_det=max_det)
