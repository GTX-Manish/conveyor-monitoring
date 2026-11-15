"""Per-event scoring of alerts, and threshold tuning. Works on arrays; it opens no files.

An alert is FALSE when it opens while the asset's real stage is `normal`. An event is DETECTED when an alert
opens on its asset between the fault's onset and its failure. Scores are per event and per alert, never per row,
because rows are overwhelmingly normal and row accuracy would look good for a model that never alerts.
"""
import numpy as np

from .alerting import ACTIVE_RUN_SHARE, HOLD_P_FAIL, alert_spans
from .features import WINDOW_MIN

NORMAL, EARLY, ADVANCED, FAILED = 0, 1, 2, 3
TARGET_PRECISION = 0.9     # at most 1 to 2 false alerts in 20 (user's requirement)
LEAD_KEEP = 0.75           # (assumption) warning time we refuse to give up when choosing a cautious threshold
HOURS = WINDOW_MIN / 60.0


def events_from_labels(stage, fault_id) -> list:
    """[(asset, fault_id, onset_w, failure_w)] from (W, A) label grids."""
    out = []
    for a in np.flatnonzero((stage != NORMAL).any(axis=0)):
        ids = fault_id[:, a]
        for fid in [f for f in dict.fromkeys(ids.tolist()) if f]:
            rows = np.flatnonzero(ids == fid)
            developing = rows[np.isin(stage[rows, a], (EARLY, ADVANCED))]
            failed = rows[stage[rows, a] == FAILED]
            if len(developing) == 0:
                continue
            failure_w = int(failed[0]) if len(failed) else int(developing[-1]) + 1
            out.append((int(a), fid, int(developing[0]), failure_w))
    return out


def score(prob, threshold: float, run_share, stage, fault_id, hard_negative=None, months: float = 1.0,
          hold=None) -> dict:
    """Replay the alert rules on (W, A) probabilities and score them against the labels."""
    active = np.asarray(run_share) >= ACTIVE_RUN_SHARE
    spans = alert_spans(np.nan_to_num(prob, nan=0.0) >= threshold, active, hold=hold)
    events = events_from_labels(stage, fault_id)
    false_alerts = [(a, w) for a, w, _ in spans if stage[w, a] == NORMAL]
    true_alerts = [(a, w) for a, w, _ in spans if stage[w, a] != NORMAL]
    leads, detected_ids, open_at_failure = [], [], 0
    for a, fid, onset_w, failure_w in events:
        opens = [w for aa, w in true_alerts if aa == a and onset_w <= w < failure_w]
        if opens:
            detected_ids.append(fid)
            leads.append((failure_w - min(opens)) * HOURS)
        # was an alert still open in the last window before the part failed? An alert that opened and then
        # lapsed is no use to the operator at the moment that matters.
        last = failure_w - 1
        open_at_failure += any(aa == a and w <= last and (c is None or c > last) for aa, w, c in spans)
    n_false, n_det = len(false_alerts), len(detected_ids)
    n_assets = prob.shape[1]
    in_hn = sum(bool(hard_negative[w, a]) for a, w in false_alerts) if hard_negative is not None else None
    return {
        "threshold": float(threshold), "events": len(events), "detected": n_det, "open_at_failure": open_at_failure,
        "detection_rate": n_det / len(events) if events else float("nan"),
        "median_lead_hours": float(np.median(leads)) if leads else float("nan"),
        "min_lead_hours": float(np.min(leads)) if leads else float("nan"),
        "alerts": len(spans), "true_alerts": len(true_alerts), "false_alerts": n_false,
        "false_alerts_in_hard_negative_periods": in_hn,
        "false_alert_share": n_false / len(spans) if spans else 0.0,
        # stricter than the alert share: several alerts on one fault count once
        "precision": n_det / (n_det + n_false) if n_det + n_false else float("nan"),
        "false_alerts_per_asset_per_month": n_false / (n_assets * months),
    }


def pooled(parts: list) -> dict:
    """Combine `score` results from several scenarios (sums, then rates)."""
    tot = {k: sum(p[k] for p in parts) for k in ("events", "detected", "open_at_failure", "alerts", "true_alerts",
                                                  "false_alerts")}
    hn = [p["false_alerts_in_hard_negative_periods"] for p in parts]
    tot["false_alerts_in_hard_negative_periods"] = None if None in hn else sum(hn)
    tot["threshold"] = parts[0]["threshold"]
    tot["detection_rate"] = tot["detected"] / tot["events"] if tot["events"] else float("nan")
    leads = [p["median_lead_hours"] for p in parts if p["detected"]]
    tot["median_lead_hours"] = float(np.median(leads)) if leads else float("nan")
    tot["false_alert_share"] = tot["false_alerts"] / tot["alerts"] if tot["alerts"] else 0.0
    d, f = tot["detected"], tot["false_alerts"]
    tot["precision"] = d / (d + f) if d + f else float("nan")
    tot["false_alerts_per_asset_per_month"] = float(np.mean([p["false_alerts_per_asset_per_month"] for p in parts]))
    return tot


def tune_threshold(scenarios: list, candidates, target: float = TARGET_PRECISION, lead_keep: float = LEAD_KEEP) -> dict:
    """Pick the most cautious threshold that still does the job on the given scenarios.

    scenarios: list of dicts with prob, run_share, stage, fault_id, months. Among thresholds with precision >=
    target, keep those that detect the most events and give at least `lead_keep` of the best median lead time,
    then take the HIGHEST. Why the highest: when many thresholds are perfect on validation, validation cannot
    choose between them, and a low threshold is the first to break on noisier data. If no threshold reaches the
    target, the most precise one is returned and `met_target` is False, so the caller can say so.
    """
    rows = [pooled([score(s["prob"], t, s["run_share"], s["stage"], s["fault_id"], months=s["months"],
                          hold=s.get("hold")) for s in scenarios]) for t in candidates]
    ok = [r for r in rows if r["detected"] > 0 and r["precision"] >= target]
    if ok:   # most faults with an alert still open at failure, then most faults detected at all
        most = max((r["open_at_failure"], r["detected"]) for r in ok)
        ok = [r for r in ok if (r["open_at_failure"], r["detected"]) == most]
        best_lead = max(r["median_lead_hours"] for r in ok)
        best = max((r for r in ok if r["median_lead_hours"] >= lead_keep * best_lead), key=lambda r: r["threshold"])
        return {**best, "met_target": True, "sweep": rows}
    with_alerts = [r for r in rows if r["detected"] > 0] or rows
    best = max(with_alerts, key=lambda r: (np.nan_to_num(r["precision"]), r["detected"], r["threshold"]))
    return {**best, "met_target": False, "sweep": rows}


# ---------------------------------------------------------------- does a failure call come true?
HORIZON_HOURS = 24.0


def score_calls(p_abnormal, thr_abnormal: float, p_fail, thr_fail: float, run_share, stage, fault_id,
                hours_to_failure, est_hours=None) -> dict:
    """Score each "predicted failure" call on what really happened next.

    A call is made exactly as the live system makes it: an abnormal alert is open on the asset and the
    probability of failing within 24 h reaches its threshold. One call per alert, at the first such window.
    A call CAME TRUE if the part failed within 24 h of it, was EARLY if the part failed later than that, and was
    FALSE if the part was normal. The time estimate is compared with the real time at the moment of the call.
    """
    active = np.asarray(run_share) >= ACTIVE_RUN_SHARE
    spans = alert_spans(np.nan_to_num(p_abnormal, nan=0.0) >= thr_abnormal, active,
                        hold=np.nan_to_num(p_fail, nan=0.0) >= HOLD_P_FAIL)
    W = p_fail.shape[0]
    hours, eta_err, called_events, in_time_events = [], [], set(), set()
    out = {"calls": 0, "came_true": 0, "early": 0, "false": 0}
    for a, w_open, w_close in spans:
        end = W if w_close is None else w_close
        hit = np.flatnonzero((np.nan_to_num(p_fail[w_open:end, a], nan=0.0) >= thr_fail) & active[w_open:end])
        if len(hit) == 0:
            continue
        w = w_open + int(hit[0])
        out["calls"] += 1
        if stage[w, a] == NORMAL:
            out["false"] += 1
            continue
        left = float(hours_to_failure[w, a])        # real hours to failure at the call (0 once failed)
        hours.append(left)
        called_events.add(fault_id[w, a])
        if left <= HORIZON_HOURS:
            out["came_true"] += 1
            in_time_events.add(fault_id[w, a])
        else:
            out["early"] += 1
        if est_hours is not None:
            eta_err.append(abs(float(est_hours[w, a]) - left))
    events = events_from_labels(stage, fault_id)
    out.update(events=len(events), events_called=len(called_events), events_called_in_time=len(in_time_events),
               hours_list=hours, eta_error_list=eta_err, threshold=float(thr_fail))
    return _finish_calls(out)


def _finish_calls(out: dict) -> dict:
    out["share_came_true"] = out["came_true"] / out["calls"] if out["calls"] else float("nan")
    out["median_hours_call_to_failure"] = float(np.median(out["hours_list"])) if out["hours_list"] else float("nan")
    out["median_eta_error_hours"] = float(np.median(out["eta_error_list"])) if out["eta_error_list"] else float("nan")
    return out


def pooled_calls(parts: list) -> dict:
    keys = ("calls", "came_true", "early", "false", "events", "events_called", "events_called_in_time")
    out = {k: sum(p[k] for p in parts) for k in keys}
    out["hours_list"] = [h for p in parts for h in p["hours_list"]]
    out["eta_error_list"] = [h for p in parts for h in p["eta_error_list"]]
    out["threshold"] = parts[0]["threshold"]
    return _finish_calls(out)


def tune_call_threshold(scenarios: list, thr_abnormal: float, candidates, target: float = TARGET_PRECISION) -> dict:
    """Threshold for the failure call: at least `target` of calls must be followed by a failure within 24 h.

    Among thresholds that reach the target, take the one that calls the most events in time, then the lowest
    (the earliest warning that is still honest). If none reaches it, return the one with the best share and
    `met_target` False, so the caller can say so.
    """
    rows = [pooled_calls([score_calls(s["p_abnormal"], thr_abnormal, s["p_fail_24h"], t, s["run_share"], s["stage"],
                                      s["fault_id"], s["hours_to_failure"], s.get("est_hours")) for s in scenarios])
            for t in candidates]
    ok = [r for r in rows if r["calls"] > 0 and r["share_came_true"] >= target]
    if ok:
        best = max(ok, key=lambda r: (r["events_called_in_time"], -r["threshold"]))
        return {**best, "met_target": True, "sweep": rows}
    made = [r for r in rows if r["calls"] > 0] or rows
    best = max(made, key=lambda r: (np.nan_to_num(r["share_came_true"]), r["events_called_in_time"], r["threshold"]))
    return {**best, "met_target": False, "sweep": rows}


def row_metrics(p_fail, threshold: float, stage, hours_to_failure) -> dict:
    """Window-level numbers for the 24-hour classifier, for readers who want them.

    Plain accuracy is not reported: more than 99% of windows are negative, so a model that never calls a
    failure would score above 99%. Windows during a repair are left out (the part has already failed).
    """
    keep = stage != FAILED
    y = (np.isin(stage, (EARLY, ADVANCED)) & (np.nan_to_num(hours_to_failure, nan=np.inf) <= HORIZON_HOURS))[keep]
    p = np.nan_to_num(p_fail[keep], nan=0.0)
    pred = p >= threshold
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    order = np.argsort(-p, kind="stable")
    hits = y[order]
    cum = np.cumsum(hits)
    ap = float((cum[hits] / (np.flatnonzero(hits) + 1)).mean()) if hits.any() else float("nan")
    return {"windows": int(keep.sum()), "positive_windows": int(y.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else float("nan"),
            "recall": tp / (tp + fn) if tp + fn else float("nan"), "average_precision": ap,
            "positive_share": float(y.mean())}
