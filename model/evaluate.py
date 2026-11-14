"""Per-event scoring of alerts, and threshold tuning. Works on arrays; it opens no files.

An alert is FALSE when it opens while the asset's real stage is `normal`. An event is DETECTED when an alert
opens on its asset between the fault's onset and its failure. Scores are per event and per alert, never per row,
because rows are overwhelmingly normal and row accuracy would look good for a model that never alerts.
"""
import numpy as np

from .alerting import ACTIVE_RUN_SHARE, alert_spans
from .features import WINDOW_MIN

NORMAL, EARLY, ADVANCED, FAILED = 0, 1, 2, 3
TARGET_PRECISION = 0.9     # at most 1 to 2 false alerts in 20 (user's requirement)
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


def score(prob, threshold: float, run_share, stage, fault_id, hard_negative=None, months: float = 1.0) -> dict:
    """Replay the alert rules on (W, A) probabilities and score them against the labels."""
    active = np.asarray(run_share) >= ACTIVE_RUN_SHARE
    spans = alert_spans(np.nan_to_num(prob, nan=0.0) >= threshold, active)
    events = events_from_labels(stage, fault_id)
    false_alerts = [(a, w) for a, w, _ in spans if stage[w, a] == NORMAL]
    true_alerts = [(a, w) for a, w, _ in spans if stage[w, a] != NORMAL]
    leads, detected_ids = [], []
    for a, fid, onset_w, failure_w in events:
        opens = [w for aa, w in true_alerts if aa == a and onset_w <= w < failure_w]
        if opens:
            detected_ids.append(fid)
            leads.append((failure_w - min(opens)) * HOURS)
    n_false, n_det = len(false_alerts), len(detected_ids)
    n_assets = prob.shape[1]
    in_hn = sum(bool(hard_negative[w, a]) for a, w in false_alerts) if hard_negative is not None else None
    return {
        "threshold": float(threshold), "events": len(events), "detected": n_det,
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
    tot = {k: sum(p[k] for p in parts) for k in ("events", "detected", "alerts", "true_alerts", "false_alerts")}
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


def tune_threshold(scenarios: list, candidates, target: float = TARGET_PRECISION) -> dict:
    """Pick the threshold that detects the most events with precision >= target on the given scenarios.

    scenarios: list of dicts with prob, run_share, stage, fault_id, months. Ties go to the longer lead time,
    then to the higher (more cautious) threshold. If no threshold reaches the target, the most precise one is
    returned and `met_target` is False, so the caller can say so.
    """
    rows = []
    for t in candidates:
        r = pooled([score(s["prob"], t, s["run_share"], s["stage"], s["fault_id"], months=s["months"])
                    for s in scenarios])
        rows.append(r)
    ok = [r for r in rows if r["detected"] > 0 and r["precision"] >= target]
    if ok:
        best = max(ok, key=lambda r: (r["detected"], round(r["median_lead_hours"], 1), r["threshold"]))
        return {**best, "met_target": True, "sweep": rows}
    with_alerts = [r for r in rows if r["detected"] > 0] or rows
    best = max(with_alerts, key=lambda r: (np.nan_to_num(r["precision"]), r["detected"], r["threshold"]))
    return {**best, "met_target": False, "sweep": rows}
