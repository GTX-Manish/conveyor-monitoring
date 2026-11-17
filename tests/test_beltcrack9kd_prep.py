"""prepare_beltcrack9kd.py on a tiny synthetic LabelMe dataset laid out like the archive:
images/<split>/<seq>/<n>.jpg and json/<split>/<seq>/<n>.json, one polygon per crack."""

import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

_spec = importlib.util.spec_from_file_location(
    "prepare_beltcrack9kd", Path(__file__).resolve().parents[1] / "vision" / "prepare_beltcrack9kd.py")
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)

# a polygon whose rule-4 box differs from int(max): w = 301.5 - 100.7 = 200.8 -> x1 = 100 + 200 = 300 (not 301)
POLY = [[100.7, 50.2], [301.5, 80.6], [250.3, 200.95]]


def _frame(root, split, seq, n, size, polygons, json_dir=None, image_path="..\\..\\images\\x\\crack1_0.jpg"):
    img = root / "images" / split / seq / f"{n}.jpg"
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (90, 90, 90)).save(img)
    js = (json_dir or root / "json" / split / seq) / f"{n}.json"
    js.parent.mkdir(parents=True, exist_ok=True)
    js.write_text(json.dumps({
        "version": "5.5.0", "imagePath": image_path, "imageData": None, "imageWidth": size[0], "imageHeight": size[1],
        "shapes": [{"label": "crack", "shape_type": "polygon", "points": p} for p in polygons]}))


@pytest.fixture
def toy(tmp_path):
    root = tmp_path / "raw" / "BeltCrack14ks"
    for n in range(6):                                   # train data1: frame 2 has no shapes
        _frame(root, "train", "data1", n, (1000, 600), [] if n == 2 else [POLY])
    for n in range(4):                                   # train data3: portrait
        _frame(root, "train", "data3", n, (600, 1000), [POLY, [[5, 5], [40, 60]]])
    for n in range(3):                                   # val data24: merged into train
        _frame(root, "val", "data24", n, (800, 450), [POLY])
    for n in range(5):                                   # test data34: long side 2048 -> 1024
        _frame(root, "test", "data34", n, (2048, 1152), [POLY])
    for n in range(3):                                   # test data35: JSON beside the image (rule 3, first lookup)
        _frame(root, "test", "data35", n, (640, 360), [POLY], json_dir=root / "images" / "test" / "data35")
    out = tmp_path / "prep"
    info = prep.prepare(tmp_path / "raw", out, workers=1, log=lambda *a: None)
    return out, info


def _lines(path):
    rows = []
    for line in path.read_text().splitlines():
        name, *boxes = line.split()
        rows.append((name, [tuple(int(v) for v in b.split(",")) for b in boxes]))
    return rows


def test_counts_and_val_merged(toy):
    out, info = toy
    assert info["train"] == {"images": 6 - 1 + 4 + 3, "boxes": 5 + 8 + 3, "sequences": 3, "without_boxes": 1}
    assert info["test"]["images"] == 8 and info["test"]["without_boxes"] == 0
    names = [n for n, _ in _lines(out / "annotations" / "train.txt")]
    assert sum("val_data24" in n for n in names) == 3
    assert not any(n.endswith("data1/2.jpg") for n in names)             # left out, as coco_to_txt.py does
    assert any("val" in note and "merged into train" in note for note in info["notes"])


def test_check_dataset_adds_frames_without_boxes(toy, monkeypatch):
    _, info = toy
    monkeypatch.setitem(prep.KNOWN_SPLITS, "BeltCrack9kd", {"train": 13, "test": 8})
    assert prep.check_dataset(info) == "OK: matches BeltCrack9kd (13 train / 8 test images)"
    monkeypatch.setitem(prep.KNOWN_SPLITS, "BeltCrack9kd", {"train": 12, "test": 8})
    assert prep.check_dataset(info).startswith("WARNING")


def test_train_box_is_rule4_then_train_scaling(toy):
    out, _ = toy
    rows = dict(_lines(out / "annotations" / "train.txt"))
    x0, y0, x1, y1 = 100, 50, 100 + 200, 50 + 150                          # rule 4 on POLY
    nw, nh = 512, int(600 * 512 / 1000)                                    # letterbox-exact 512, int() sizes
    (key,) = [k for k in rows if k.endswith("data1/0.jpg")]
    assert rows[key] == [(int(x0 * nw / 1000), int(y0 * nh / 600), int(x1 * nw / 1000), int(y1 * nh / 600), 0)]
    assert rows[key] == [(51, 25, 153, 102, 0)]
    assert Image.open(out / key).size == (512, 307)
    (key,) = [k for k in rows if k.endswith("data3/1.jpg")]
    assert rows[key][1] == (int(5 * 307 / 600), int(5 * 512 / 1000), int(40 * 307 / 600), int(60 * 512 / 1000), 0)


def test_test_split_boxes_and_coco(toy):
    out, _ = toy
    rows = dict(_lines(out / "annotations" / "test.txt"))
    (key,) = [k for k in rows if k.endswith("data34/0.jpg")]
    assert rows[key] == [(round(100 * 0.5), round(50 * 0.5), round(300 * 0.5), round(200 * 0.5), 0)]
    assert Image.open(out / key).size == (1024, 576)
    cj = json.loads((out / "annotations" / "test_coco.json").read_text())
    assert len(cj["images"]) == 8 and len(cj["annotations"]) == 8
    im = next(i for i in cj["images"] if i["file_name"] == key)
    (a,) = [a for a in cj["annotations"] if a["image_id"] == im["id"]]
    assert a["bbox"] == pytest.approx([100.7 * 0.5, 50.2 * 0.5, 200.8 * 0.5, 150.75 * 0.5])
    assert sum(k.endswith(f"data35/{n}.jpg") for k in rows for n in range(3)) == 3


def test_every_sample_has_its_five_history_frames(toy):
    out, _ = toy
    cj = json.loads((out / "annotations" / "test_coco.json").read_text())
    names = [n for f in ("train.txt", "test.txt") for n, _ in _lines(out / "annotations" / f)]
    names += [i["file_name"] for i in cj["images"]]
    for name in names:
        folder, n = Path(name).parent, int(Path(name).stem)
        for k in range(prep.HISTORY):
            assert (out / folder / f"{max(n - k, 0)}.jpg").is_file(), (name, k)


def test_unmatched_json_still_stops_the_run(tmp_path):
    root = tmp_path / "raw" / "BeltCrack14ks"
    for n in range(3):
        _frame(root, "train", "data1", n, (1000, 600), [POLY])
        _frame(root, "test", "data34", n, (1000, 600), [POLY])
    orphan = root / "json" / "test" / "data34" / "9.json"
    orphan.write_text((root / "json" / "test" / "data34" / "0.json").read_text())
    with pytest.raises(RuntimeError, match="More than 1%"):
        prep.prepare(tmp_path / "raw", tmp_path / "prep", workers=1, log=lambda *a: None)
