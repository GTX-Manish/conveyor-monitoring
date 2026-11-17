# Vision report: crack detector

## Credits

The crack detector is a YOLOv8 model trained on BeltCrack by Julián Neira Nova
(github.com/JSNN15/conveyor_crack_detection_2026, AGPL-3.0). We integrated it, evaluated it on the official
BeltCrack9kd test split, and chose the operating threshold on validation. The looping-belt layer on top of it (belt
coordinates, cross-lap de-duplication, splice handling and the dashboard) is described at the end.

The model file is `vision/artifacts/best.onnx`, taken unchanged from the author's repo at commit `73cc79f`. Next to it:
`NOTICE.md`, the AGPL-3.0 text (`LICENSE-AGPL-3.0.txt`), and `model_source.json` (source files, git blob hashes,
sha256 `236f930a…529c`). The dataset is BeltCrack by Huang, Ji, Ma and Ye (*Pattern Recognition*, 2026;
doi:10.57760/sciencedb.31181), licensed CC BY-SA 4.0.

We evaluated training our own model, but the paper's BeltCrackDet needed about 74 GPU-hours on a T4, so we chose this
pretrained detector instead.

## Integration

`vision/detector.py` runs the model with onnxruntime on CPU (about 100 ms per frame on 4 threads). Pre- and
post-processing follow the author's `det.js`:

| Step | Setting |
|---|---|
| Input | 640 × 384 (W × H), NCHW float32, RGB / 255 |
| Letterbox | scale `min(640/W, 384/H)`, sizes rounded half up, centred, pad colour 114 |
| Output | `[1, 5, 5040]`: cx, cy, w, h, score in input pixels; one class (`grieta`, crack) |
| Post-processing | letterbox undone, boxes clipped to the image, greedy NMS at IoU 0.45 |
| Operating threshold | 0.363, chosen on val (below); the author's default is 0.3 |

One deliberate difference: the canvas offset is a whole pixel, while the browser can draw at a half pixel. Both sides
undo their own offset, so boxes stay in place. `tests/test_vision_detector.py` checks the letterbox geometry, decoding
and NMS. It also compares ONNX outputs against a fixture in two ways:
- **Generated frame:** raw outputs, which runs in CI.
- **Four BeltCrack9kd val frames:** IoU ≥ 0.99 and score within 0.01. This runs only where `beltcrack_yolo960/` is
  present, because dataset images are kept out of the repo.

## Evaluation protocol

`python -m vision.evaluate` (metric code unit-tested in `tests/test_vision_evaluate.py`):

1. **Val:** the 1,961 BeltCrack9kd val frames (960 px long side, 2,615 boxes).
   - Detections at conf 0.001, NMS 0.45, up to 100 per frame.
   - P, R and F1 at IoU 0.5 on a 0.001 conf grid.
   - The threshold rule, fixed before looking: the lowest conf with P ≥ 0.9 if R there is ≥ 0.6, otherwise max F1.
   - The threshold and the model's sha256 are written to `vision_eval.json`. The curve is in `pr_curve_val.csv`.
2. **Test, scored once:** the 1,948 BeltCrack9kd test frames, 2,482 boxes, from `test_coco.json` at 1024 px long
   side. The script refuses to score test a second time.
   - mAP50 and mAP50-95: COCO-style, 101-point interpolation, IoU 0.50:0.95.
   - P / R / F1 at conf 0.3 and at the frozen threshold.

### Reproducing the evaluation

The dataset is not in the repo. From the downloaded archive (8 RAR parts, password `cv205`):

```bash
python -m vision.prepare_beltcrack9kd dataset --out BeltCrack9kd_prep   # extracts to BeltCrack9kd_prep_raw/; test frames and test_coco.json
python -m vision.prepare_dataset --root BeltCrack9kd_prep_raw --out beltcrack_yolo960   # 960 px val frames and YOLO labels
python -m vision.evaluate --val beltcrack_yolo960 --test BeltCrack9kd_prep
```

`vision.evaluate` refuses to score test again while `vision_eval.json` holds a test result. Tests:
`tests/test_beltcrack9kd_prep.py` and `tests/test_vision_prepare.py` cover the two converters on synthetic data.

## Results

| Split | mAP50 | mAP50-95 | P / R / F1 at 0.3 | P / R / F1 at 0.363 (frozen) |
|---|---|---|---|---|
| Val (1,961 frames) | 73.3 | 34.7 | 76.5 / 71.8 / 74.1 | 82.0 / 68.9 / 74.9 |
| **Test (1,948 frames)** | **26.9** | **13.7** | 41.8 / 31.7 / 36.1 | 45.9 / 28.2 / 35.0 |

On val, precision ≥ 0.9 was reached only at conf 0.486, with recall 0.537, below the 0.6 floor. So the rule fell back
to max F1, at 0.363.

**Beside the paper's BeltCrack9kd rows** (arXiv 2506.17892, Table 3, test split):

| Model | mAP50 | P | R | F1 |
|---|---|---|---|---|
| YOLOX-s, single frame (paper) | 24.82 | 50.54 | 49.36 | 50.22 |
| BeltCrackDet, 5 frames (paper) | 26.98 | 51.06 | 54.76 | 52.83 |
| **This detector, single frame (ours, at 0.363)** | **26.9** | 45.9 | 28.2 | 35.0 |

Protocol differences:
- **Model:** YOLOv8 (the author's, export metadata "c2_e5"), not YOLOX-s.
- **Input:** 640 × 384 letterboxed, not 512 × 512. Our test frames were first reduced to 1024 px, then letterboxed.
- **Training data:** the author's, not the paper's 9kd train split (see the caveat below).
- **Operating point:** P and R are at our frozen threshold. The paper does not state its operating point, so the
  P / R columns compare less directly than mAP50.

### Caveat: which frames the author trained on

**We do not know which frames the author trained on, so test numbers may be optimistic if the author used 9kd test
frames.** The author's README reports results on "9 held-out videos" without naming them. The model metadata says
only "trained on full.yaml".

The two splits point in different directions:
- **Val is almost certainly not held out for this model.** Val scores 73.3 mAP50 against 26.9 on test, a gap no
  split difference in this dataset explains. The author most likely trained on 9kd val sequences, or on frames very
  close to them. So the threshold chosen on val is optimistic: val precision at 0.363 is 82%, test precision 46%.
- **Test looks held out.** Its mAP50 sits in the paper's single-frame range, and 9kd has exactly 9 test sequences.
  That is consistent with test having been held out, but it does not prove it.

For the looping-belt layer, the practical reading is to expect roughly the test numbers: about 46% of single-frame
detections are real cracks, and about 28% of crack boxes are found per frame. Confirmation across laps has to do the
filtering. The author's README also notes chevron-pattern belts were "almost absent from training", and 9kd has
several chevron sequences.

## The looping belt: one record per crack, not one per lap

The belt is a 6 km loop, so a crack passes every camera once per lap. The layer on top of the detector turns
detections into belt locations. Code: `simulator/vision.py` (captures), `vision/defects.py` (rules, shared by the
backend and the scorer), `backend/vision.py` (live), `scripts/score_vision.py` (scoring).

### Simulated captures

- **Cameras:** three carry-side stations at 0, 1500 and 3000 m (`CAM1` to `CAM3`). The belt coordinate under a
  station is (station − belt travel) mod 6000. Travel only grows while the belt runs. Each capture's coordinate has
  0.5 m Gaussian noise.
- **Locations:**
  - 3 or 4 cracks, each pinned to one BeltCrack9kd test sequence. Every lap shows that sequence's frame advanced by
    a seeded stride of 15 to 30 frames.
  - 12 splices, every 500 m, each showing one fixed clean image (a joint looks the same every lap).
  - 4 clean locations, each capture drawing a clean image at random.
  - One crack is placed within 1.5 m of a splice.
- **Clean images:** the dataset has no crack-free frames. Clean images are box-free regions of test frames, reduced
  by the same factor as the crack frames and centred on the detector's grey padding, so they look like a narrower
  view. Shown full size instead (zoomed 1.6×), the model fired on 10 of 16 of them, against 1 of 16 at the frames'
  own scale.
- **Assumption, one recorded lap per running day:** recording every lap would need thousands of committed dataset
  frames. On that lap each location passes CAM1, CAM2 and CAM3 in turn. The demo has 14 laps, 840 captures and 72
  images (2.6 MB in `data/vision_demo/`, CC BY-SA 4.0, see `ATTRIBUTION.md`).
- **Public and truth:** public captures hold only ts, station, ground position, belt coordinate and image path
  (hashed names). Which location a capture shows is in `truth/vision_truth.parquet`, under the leakage guard.
  Adding the `vision` random stream last left the demo's sensor files byte-identical (tested against their hashes).

Rebuild the demo images with `python -m simulator.vision build-demo` (needs `BeltCrack9kd_prep/`), and the scores
with `python -m scripts.score_vision` after `make demo-data`.

### Rules (`vision/defects.py`)

- **Joining:** every detection in a capture joins the location at that capture's belt coordinate. Captures within
  2 m are one location, and locations that drift within 2 m are merged.
- **Confirmation:** a location is confirmed once seen on 2 laps (captures less than 1125 s apart are one lap).
- **Splices:** a location within 1.5 m of a splice goes to a separate "at a splice, inspect" list, confirmed or not,
  and is never dropped.
- **Looks:** a capture with no detection that falls on a location counts as a look, so the dashboard can say
  "seen on 8 of 14 laps".

### Results on the demo (`vision/artifacts/dedup_eval.json`)

| | Result |
|---|---|
| Crack locations found | **3 of 4**, 0 duplicates |
| Crack placed 0.7 m from the splice at 500 m | found and flagged (in the inspect list, seen on 11 of 14 laps) |
| Laps to confirm (found cracks) | 2, 3 and 4 |
| Missed crack (sequence `data39`) | 0 detections in 42 captures |
| False confirmed defects (clean locations) | 4, seen on 3 to 5 of 7 to 14 laps |
| False entries in the inspect list | 1 (the splice at 3500 m, whose image fires on every lap) |
| Single-frame false detections | 60, in 60 of 672 clean captures; confirmation removed none |

The live backend gives the same locations as the scorer (checked by posting all 840 captures through the API).

**What this shows:**
- **De-duplication works:** each found crack is one location however many laps and cameras saw it, and the near-splice
  crack is listed for inspection rather than dropped.
- **The missed crack:** `data39` is a chevron-pattern belt with a large tear, and the model's author notes chevron
  belts were "almost absent from training".
- **"Seen on 2 laps" is a weak filter over a long run:** a location the model fires on now and then reaches 2 laps
  eventually. Over 14 laps every intermittent false hit did, so confirmation removed none here. The look count
  separates them: the real cracks were seen on 8 to 11 of 13 or 14 laps, the false locations on 3 to 5. A stricter rule
  (for example, seen on most laps looked) would help, but it is not built, because the brief's rule is 2 laps.
- **Few clean images:** the clean images come from a pool of 16, so a crop the model fires on comes back at more than
  one clean location. Three of the four false defects show the same crop, and the fourth and the splice at 3500 m show
  machinery at the frame's edge.
- **Flattering assumptions:** each crack lap shows a different frame of a real video and each clean capture an
  independently drawn crop. Persistent stains that fool the model would repeat every lap, so confirmation looks
  kinder here than it would on a real belt. The fixed splice images are the persistent case.
- **Limits:** splices are handled by coordinate only (the dataset has no splice images), and the images are a real
  dataset placed on a simulated belt. They are not linked to the sensors, and no sensor alert uses them.

### Backend and dashboard

- **`POST /api/captures`:** runs the detector on each capture (cached per image, since the model is deterministic)
  and updates `captures`, `vision_detections` and `vision_defects`.
- **Read endpoints:** `/api/vision/status`, `/api/vision/detections`, `/api/vision/defects` (two lists, `defects`
  and `at_splice`) and `/api/vision/image/{id}`. All return `enabled: false` with the reason if the model,
  onnxruntime or the image folder is missing.
- **Replayer:** posts captures as the minutes they were taken in are sent.
- **Vision tab:** confirmed locations (laps seen, cameras, best capture with boxes), the inspect list, the latest
  detections, and the detector's test numbers marked "public dataset, not linked to the sensors".
- **Overview:** shows both counts.
