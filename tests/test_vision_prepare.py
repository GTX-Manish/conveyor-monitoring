"""The BeltCrack converter on a tiny synthetic LabelMe dataset: split kept, polygons become the right boxes, sizes."""

import json
from pathlib import Path

import pytest
from PIL import Image

from vision import prepare_dataset as prep


def _frame(root: Path, split: str, seq: str, n: int, size, polygons):
    img = root / "images" / split / seq / f"{n}.jpg"
    js = root / "json" / split / seq / f"{n}.json"
    img.parent.mkdir(parents=True, exist_ok=True)
    js.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 90, 90)).save(img)
    js.write_text(json.dumps({"imageWidth": size[0], "imageHeight": size[1], "imagePath": "..\\pictures\\x.jpg",
                              "shapes": [{"label": "crack", "shape_type": "polygon", "points": p} for p in polygons]}))


@pytest.fixture
def toy(tmp_path):
    root = tmp_path / "beltcrack_data" / "BeltCrack14ks"
    _frame(root, "train", "data1", 0, (1920, 1080), [[[100, 100], [500, 120], [480, 200]]])
    _frame(root, "val", "data24", 0, (3840, 2160), [[[0, 0], [3840, 0], [3840, 50]], [[10, 2000], [20, 2100], [15, 2150]]])
    _frame(root, "test", "data34", 0, (2160, 3840), [[[1000, 1000], [1100, 1000], [1100, 1200], [1000, 1200]]])
    out = tmp_path / "beltcrack_yolo960"
    prep.main(["--root", str(tmp_path / "beltcrack_data"), "--out", str(out), "--workers", "1"])
    return out


def test_split_and_names_kept(toy):
    assert [p.name for p in (toy / "images" / "train").glob("*.jpg")] == ["data1__0.jpg"]
    assert [p.name for p in (toy / "images" / "val").glob("*.jpg")] == ["data24__0.jpg"]
    assert [p.name for p in (toy / "images" / "test").glob("*.jpg")] == ["data34__0.jpg"]
    m = json.loads((toy / "manifest.json").read_text())
    assert m["counts"]["val"] == {"images": 1, "source_boxes": 2, "yolo_boxes": 2, "sequences": 1}


def test_polygon_becomes_normalised_box(toy):
    (row,) = (toy / "labels" / "test" / "data34__0.txt").read_text().split("\n")[:1]
    cls, cx, cy, w, h = row.split()
    assert cls == "0"
    assert float(cx) == pytest.approx(1050 / 2160, abs=1e-6)
    assert float(cy) == pytest.approx(1100 / 3840, abs=1e-6)
    assert float(w) == pytest.approx(100 / 2160, abs=1e-6)
    assert float(h) == pytest.approx(200 / 3840, abs=1e-6)


def test_long_side_is_960_and_orientation_kept(toy):
    assert Image.open(toy / "images" / "val" / "data24__0.jpg").size == (960, 540)
    assert Image.open(toy / "images" / "test" / "data34__0.jpg").size == (540, 960)
    assert Image.open(toy / "images" / "train" / "data1__0.jpg").size == (960, 540)

