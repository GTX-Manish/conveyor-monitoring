"""vision/detector.py: the author's letterbox (640 x 384), decoding, NMS, and the model's ONNX outputs."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from vision import detector as det

FIXTURE = Path(__file__).parent / "fixtures" / "vision_onnx_outputs.json"
LOCAL_FRAMES = Path(__file__).resolve().parents[1] / "beltcrack_yolo960" / "images" / "val"


def synthetic_frame(w=960, h=540, seed=7) -> Image.Image:
    """A deterministic belt-like frame with a dark diagonal gash (no dataset image needed)."""
    rng = np.random.default_rng(seed)
    a = (rng.normal(95, 12, (h, w, 3))).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(a, "RGB")
    d = ImageDraw.Draw(img)
    d.line((300, 200, 620, 260), fill=(20, 20, 20), width=9)
    d.line((310, 205, 600, 250), fill=(200, 190, 170), width=2)
    return img


def test_letterbox_landscape_geometry_pad_and_scaling():
    x, lb = det.letterbox(Image.new("RGB", (1920, 1080), (255, 0, 0)))
    assert x.shape == (1, 3, 384, 640) and x.dtype == np.float32
    assert (lb.scale, lb.dx, lb.dy) == (1 / 3, 0, 12)          # 640 x 360, centred: 12 px above and below
    assert np.allclose(x[0, :, :12, :], 114 / 255) and np.allclose(x[0, :, 372:, :], 114 / 255)
    assert np.allclose(x[0, 0, 12:372], 1.0) and np.allclose(x[0, 1:, 12:372], 0.0)   # RGB order, / 255


def test_letterbox_portrait_is_centred_horizontally():
    x, lb = det.letterbox(Image.new("RGB", (540, 960), (0, 0, 255)))
    assert (lb.scale, lb.dx, lb.dy) == (0.4, 212, 0)           # 216 x 384
    assert np.allclose(x[0, :, :, :212], 114 / 255) and np.allclose(x[0, :, :, 428:], 114 / 255)
    assert np.allclose(x[0, 2, :, 212:428], 1.0)


def test_sizes_round_half_up_like_javascript():
    assert det._round_half_up(2.5) == 3 and det._round_half_up(3.5) == 4 and det._round_half_up(2.49) == 2
    _, lb = det.letterbox(Image.new("RGB", (1000, 750)))        # s = 0.512 -> 512 x 384
    assert (lb.dx, lb.dy) == (64, 0)


def _output(rows):
    out = np.zeros((1, 5, 8), np.float32)
    for i, r in enumerate(rows):
        out[0, :, i] = r
    return out


def test_decode_undoes_the_letterbox_and_filters_by_conf():
    _, lb = det.letterbox(Image.new("RGB", (1920, 1080)))
    out = _output([(320, 192, 60, 30, 0.9), (100, 100, 20, 20, 0.2)])
    (d,) = det.decode(out, lb, conf=0.3)
    assert (d.x1, d.y1, d.x2, d.y2) == pytest.approx((870, 495, 1050, 585))
    assert d.score == pytest.approx(0.9)


def test_decode_clips_boxes_to_the_image():
    _, lb = det.letterbox(Image.new("RGB", (1920, 1080)))
    (d,) = det.decode(_output([(5, 15, 40, 20, 0.8)]), lb, conf=0.3)
    assert d.x1 == 0 and d.y1 == 0


def test_nms_at_iou_045():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 12], [0, 0, 10, 24.5]], float)   # IoU 10/12, then 10/24.5
    keep = det.nms(boxes, np.array([0.9, 0.8, 0.7]))
    assert keep == [0, 2]                                       # 0.833 > 0.45 dropped; 0.408 <= 0.45 kept


def test_model_io_matches_the_author_model():
    pytest.importorskip("onnxruntime")
    d = det.Detector()
    assert d.session.get_inputs()[0].shape == [1, 3, 384, 640]
    out, _ = d.raw(Image.new("RGB", (640, 384), (114, 114, 114)))
    assert out.shape == (1, 5, 5040)


def test_onnx_output_matches_fixture_on_synthetic_frame():
    """Model file, onnxruntime and preprocessing together: raw outputs on a generated frame (runs in CI)."""
    pytest.importorskip("onnxruntime")
    fx = json.loads(FIXTURE.read_text())["synthetic"]
    out, _ = det.Detector().raw(synthetic_frame())
    got = out[0][:, :: fx["anchor_stride"]]
    assert np.allclose(got, np.array(fx["outputs"]), rtol=fx["rtol"], atol=fx["atol"])


def test_onnx_detections_match_fixture_on_val_frames():
    """Detections on a few BeltCrack9kd val frames. The frames are not in the repo until the dataset licence is
    confirmed, so this runs only where beltcrack_yolo960/ exists."""
    pytest.importorskip("onnxruntime")
    fx = json.loads(FIXTURE.read_text())["frames"]
    if not all((LOCAL_FRAMES / f["frame"]).exists() for f in fx["items"]):
        pytest.skip("BeltCrack9kd val frames are not available locally")
    d = det.Detector()
    for item in fx["items"]:
        got = d.detect(LOCAL_FRAMES / item["frame"], conf=fx["conf"])
        assert len(got) == len(item["detections"]), item["frame"]
        for g, e in zip(got, item["detections"]):
            a = np.array([[g.x1, g.y1, g.x2, g.y2]])
            b = np.array([[e["x1"], e["y1"], e["x2"], e["y2"]]])
            iw = max(0, min(a[0, 2], b[0, 2]) - max(a[0, 0], b[0, 0]))
            ih = max(0, min(a[0, 3], b[0, 3]) - max(a[0, 1], b[0, 1]))
            inter = iw * ih
            iou = inter / ((a[0, 2] - a[0, 0]) * (a[0, 3] - a[0, 1]) + (b[0, 2] - b[0, 0]) * (b[0, 3] - b[0, 1]) - inter)
            assert iou >= 0.99 and abs(g.score - e["score"]) <= 0.01, item["frame"]


def test_detector_uses_frozen_threshold(tmp_path):
    p = tmp_path / "vision_eval.json"
    assert det.frozen_threshold(p) is None
    p.write_text(json.dumps({"freeze": {"threshold": 0.42}}))
    assert det.frozen_threshold(p) == 0.42
