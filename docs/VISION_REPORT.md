# Vision report: crack detector

## Credits

The crack detector is a YOLOv8 model trained on BeltCrack by Julián Neira Nova
(github.com/JSNN15/conveyor_crack_detection_2026, AGPL-3.0). We integrated it, evaluated it on the official
BeltCrack9kd test split, and chose the operating threshold on validation. The looping-belt layer on top of it (belt
coordinates, cross-lap de-duplication, splice handling and the dashboard) is the next step and is not built at this
commit.

The model file is `vision/artifacts/best.onnx`, taken unchanged from the author's repo at commit `73cc79f`. Next to it:
`NOTICE.md`, the AGPL-3.0 text (`LICENSE-AGPL-3.0.txt`), and `model_source.json` (source files, git blob hashes,
sha256 `236f930a…529c`). The dataset is BeltCrack by Huang, Ji, Ma and Ye (*Pattern Recognition*, 2026;
doi:10.57760/sciencedb.31181).

## Why a pretrained detector

Detection quality on BeltCrack is set by the model and the GPU-hours behind it. The looping-belt logic is what this
project adds. Our replication of the paper's BeltCrackDet needed about 74 GPU-hours on a T4, against a budget of 4,
and a published YOLOv8 trained on the same dataset was available. Using it costs no training time and still gets
a measurement on the official test split. So we spent the budget on integrating it, measuring it on the official split, and the looping-belt layer.

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
- **Four BeltCrack9kd val frames:** IoU ≥ 0.99 and score within 0.01. This runs only where the dataset is present,
  because dataset images stay out of the repo until the licence is confirmed.

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

## Not done yet

The looping-belt layer and the Vision tab are not built at this commit:
- simulated captures with belt coordinates;
- cross-lap de-duplication;
- the "at splice, inspect" list;
- backend ingest and the dashboard.

`/api/vision/*` still returns `enabled: false`.
