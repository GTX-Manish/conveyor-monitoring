"""vision/evaluate.py metric code on hand-checked cases (COCO-style matching and 101-point AP)."""

import numpy as np
import pytest

from vision import evaluate as ev


def B(*rows):
    return np.array(rows, float).reshape(-1, 4)


def test_perfect_detections_give_ap_1():
    res = [(B([0, 0, 10, 10]), np.array([0.9]), B([0, 0, 10, 10])),
           (B([5, 5, 20, 20]), np.array([0.8]), B([5, 5, 20, 20]))]
    _, tps, n = ev.pooled(res, 0.5)
    assert ev.average_precision(tps, n) == pytest.approx(1.0)


def test_false_positive_ranked_first_halves_ap():
    res = [(B([50, 50, 60, 60], [0, 0, 10, 10]), np.array([0.9, 0.5]), B([0, 0, 10, 10]))]
    _, tps, n = ev.pooled(res, 0.5)
    assert list(tps) == [False, True]
    assert ev.average_precision(tps, n) == pytest.approx(0.5)


def test_half_recall_counts_51_of_101_points():
    res = [(B([0, 0, 10, 10]), np.array([0.9]), B([0, 0, 10, 10], [30, 30, 40, 40]))]
    _, tps, n = ev.pooled(res, 0.5)
    assert ev.average_precision(tps, n) == pytest.approx(51 / 101)


def test_matching_is_by_score_and_each_label_is_used_once():
    dets = B([0, 0, 10, 9], [0, 0, 10, 10])        # the higher score takes the labelled box, the duplicate is a FP
    tp = ev.match(dets, np.array([0.9, 0.5]), B([0, 0, 10, 10]), 0.5)
    assert list(tp) == [True, False]
    tp = ev.match(dets, np.array([0.9, 0.5]), B([0, 0, 10, 10]), 0.95)
    assert list(tp) == [False, True]               # at IoU 0.95 only the exact box matches


def test_pr_at_cuts_by_score():
    res = [(B([0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]), np.array([0.9, 0.6, 0.2]),
            B([0, 0, 10, 10], [20, 20, 30, 30]))]
    s, t, n = ev.pooled(res, 0.5)
    assert ev.pr_at(s, t, n, 0.5) == {"conf": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5,
                                           "detections": 2, "true_positives": 1}
    assert ev.pr_at(s, t, n, 0.1)["recall"] == 1.0
    assert ev.pr_at(s, t, n, 0.9)["detections"] == 1      # score >= conf is kept


def test_threshold_rule_prefers_precision_09_when_recall_holds():
    # 10 labelled boxes; 7 TPs scored 0.9..0.6 and 3 FPs scored low: P >= 0.9 with R = 0.7 at conf 0.6
    dets = [[i * 20, 0, i * 20 + 10, 10] for i in range(10)]
    gts = B(*dets)
    boxes = B(*(dets[:7] + [[500, 500, 510, 510], [600, 600, 610, 610], [700, 700, 710, 710]]))
    scores = np.array([0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.3, 0.2, 0.1])
    s, t, n = ev.pooled([(boxes, scores, gts)], 0.5)
    choice, _ = ev.choose_threshold(s, t, n)
    assert choice["chosen"] == "precision>=0.9" and choice["at_threshold"]["recall"] == 0.7
    assert choice["threshold"] == 0.301                   # at 0.3 the FP scored 0.3 is kept: P = 7/8 < 0.9


def test_threshold_rule_falls_back_to_max_f1_when_recall_is_low():
    gts = B(*[[i * 20, 0, i * 20 + 10, 10] for i in range(10)])
    boxes = B([0, 0, 10, 10], [20, 0, 30, 10], [500, 500, 510, 510], [40, 0, 50, 10], [60, 0, 70, 10])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])           # P >= 0.9 only at R = 0.2
    s, t, n = ev.pooled([(boxes, scores, gts)], 0.5)
    choice, _ = ev.choose_threshold(s, t, n)
    assert choice["chosen"] == "max_f1" and choice["candidate_precision_0.9"]["recall"] == 0.2
    assert choice["at_threshold"]["recall"] == 0.4 and choice["threshold"] == 0.001
