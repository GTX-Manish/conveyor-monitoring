"""Score the models per event against two baselines, and write model/artifacts/eval_report.md.

    python -m scripts.score                    # one training seed (the saved models)
    python -m scripts.score --seeds 1 2 3      # retrain per seed in memory and report the spread

Thresholds are tuned on the validation scenarios and frozen; test and stress are scored once with them. This
script reads the model tables (features plus labels) and is one of the four allowed readers of ground truth.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common.ids import N_ZONES
from model.alerting import CALL_HORIZON_HOURS, HOLD_P_FAIL
from model.evaluate import ADVANCED, TARGET_PRECISION, pooled, pooled_calls, row_metrics, score, score_calls, tune_threshold
from model.predict import ARTIFACTS
from model.train import grids, save, train

SPLITS = ("validation", "test", "stress")
COLUMNS = ["events", "detected", "open_at_failure", "detection_rate", "median_lead_hours", "alerts", "false_alerts",
           "false_alert_share", "precision", "false_alerts_per_asset_per_month",
           "false_alerts_in_hard_negative_periods"]


def baseline_scores(df: pd.DataFrame, group: str) -> dict:
    """Two simple detectors on the same windows: a fixed temperature limit and a neighbour comparison."""
    A = N_ZONES if group == "zone" else 1
    shape = lambda v: np.asarray(v, dtype=np.float64).reshape(-1, A)  # noqa: E731
    if group == "zone":
        return {"Fixed bearing-temperature limit": shape(df["temp_reference"] + df["bearing_temp_c__dev10"]),
                "Deviation from neighbouring zones": shape(df["bearing_temp_c__nb6h"])}
    return {"Fixed motor-temperature limit": shape(df["temp_reference"] + df["motor_temp_c__dev10"])}


def evaluate_method(scenario_grids: dict, key, splits: dict) -> dict:
    """Tune on validation, freeze, score every split. key: grid name or a function grid -> (W, A) scores."""
    get = (lambda g: g[key]) if isinstance(key, str) else key
    # the model's alerts are held open while failure within 24 h is likely; the baselines have no such score
    hold = (lambda g: g["p_fail_24h"] >= HOLD_P_FAIL) if isinstance(key, str) else (lambda g: None)
    val = [dict(prob=get(g), run_share=g["run_share"], stage=g["stage"], fault_id=g["fault_id"], months=g["months"],
                hold=hold(g)) for n, g in scenario_grids.items() if n in splits["validation"]]
    values = np.concatenate([v["prob"].ravel() for v in val])
    values = values[np.isfinite(values)]
    if isinstance(key, str):
        from model.train import candidates
        cand = candidates()
    else:  # baselines are in physical units: sweep the upper tail of what validation shows
        cand = np.unique(np.quantile(values, [0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999,
                                              0.99995, 0.99999]))
    tuned = tune_threshold(val, cand)
    out = {"threshold": tuned["threshold"], "met_target": tuned["met_target"]}
    for split in SPLITS:
        parts = [score(get(g), tuned["threshold"], g["run_share"], g["stage"], g["fault_id"], g["hard_negative"],
                       g["months"], hold=hold(g)) for n, g in scenario_grids.items() if n in splits[split]]
        out[split] = pooled(parts) if parts else None
    return out


def regressor_error(scenario_grids: dict, names: list):
    err = []
    for n in names:
        g = scenario_grids[n]
        adv = g["stage"] == ADVANCED
        err.append(np.abs(g["est_hours"][adv] - g["hours_to_failure"][adv]))
    err = np.concatenate(err) if err else np.array([])
    return float(np.median(err)) if len(err) else float("nan")


def run_seed(data: Path, seed: int, save_artifacts: bool) -> dict:
    result = train(data, seed)
    if save_artifacts:
        save(result, seed)
    out = {}
    for group in ("zone", "drive"):
        names = [n for s in SPLITS for n in result["splits"][s]]
        frames = {n: pd.read_parquet(data / "model" / f"{n}_{group}.parquet") for n in names}
        g = {n: grids(df, result["boosters"][group], group) for n, df in frames.items()}
        methods = {"LightGBM `abnormal` model": evaluate_method(g, "p_abnormal", result["splits"])}
        base = {n: baseline_scores(df, group) for n, df in frames.items()}
        for name in next(iter(base.values())):
            for n in g:
                g[n][name] = base[n][name]
            methods[name] = evaluate_method(g, (lambda grid, name=name: grid[name]), result["splits"])
        thr = result["thresholds"][group]
        calls, rows = {}, {}
        for split in SPLITS:
            part = [g[n] for n in result["splits"][split]]
            calls[split] = pooled_calls([score_calls(
                x["p_abnormal"], thr["abnormal"], x["p_fail_24h"], thr["predicted_failure"], x["run_share"],
                x["stage"], x["fault_id"], x["hours_to_failure"], x["est_hours"]) for x in part]) if part else None
            rows[split] = row_metrics(np.concatenate([x["p_fail_24h"].ravel() for x in part]), thr["predicted_failure"],
                                      np.concatenate([x["stage"].ravel() for x in part]),
                                      np.concatenate([x["hours_to_failure"].ravel() for x in part])) if part else None
        out[group] = {"methods": methods, "failure_calls": calls, "window_metrics": rows,
                      "alert_sweep_validation": thr["abnormal_sweep"],
                      "call_sweep_validation": thr["predicted_failure_sweep"],
                      "alert_threshold": thr["abnormal"],
                      "failure_call_threshold": thr["predicted_failure"],
                      "failure_call_target_met": thr["predicted_failure_met_target"],
                      "eta_median_abs_error_hours": {s: regressor_error(g, result["splits"][s]) for s in SPLITS}}
    return out


def fmt(v, digits=2):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return "n/a" if np.isnan(v) else f"{v:.{digits}f}"
    return str(v)


def report(runs: dict, path: Path) -> None:
    seeds = list(runs)
    first = runs[seeds[0]]
    H = CALL_HORIZON_HOURS
    lines = ["# Evaluation report", "",
             "Generated by `python -m scripts.score`. Do not edit by hand.", "",
             "- Scores are **per event and per alert**, not per row. An event is detected when an alert opens on its "
             "asset between the fault's onset and its failure. An alert is false when it opens while the asset's real "
             "stage is `normal`.",
             f"- Thresholds are tuned on the validation scenarios for a precision of at least {TARGET_PRECISION:.0%} "
             "(at most 1 to 2 false alerts in 20). Among thresholds that meet it and detect the most events, the most "
             "cautious (highest) one that keeps at least 75% of the best lead time is taken, then frozen. "
             "Test and stress are scored once.",
             "- Precision here is detected events / (detected events + false alerts). Several alerts on one fault "
             "count once, so it is stricter than the false share of all alerts.",
             "- The same alert rules apply to every method: open after 2 windows above the threshold, close after 6 "
             "below, hold while the belt is stopped. The model's alerts also stay open while its probability of "
             "failing within 24 h is above 50%; this can keep an alert open but can never open one.",
             "- **Alert still open at failure** counts the faults where an alert was open in the last window before "
             "the part failed. An alert that opened and then lapsed is counted as detected but not here.", ""]
    lines += ["## Precision and recall at a glance", "",
              "**Precision** is the share of the model's alerts (or failure calls) that were right. **Recall** is the "
              "share of real faults the model caught. Alerts are counted per fault: recall is faults with an alert "
              "before failure, precision is those faults against those faults plus false alerts. For failure calls, "
              f"precision is calls followed by a failure within {H:.0f} h, recall is faults that got such a call.", "",
              "| Asset | Split | Faults | Alert precision | Alert recall | Failure-call precision | Failure-call recall |",
              "|---|---|---|---|---|---|---|"]
    for group, name in (("zone", "Idler bearings"), ("drive", "Drive")):
        for split in SPLITS:
            a = first[group]["methods"]["LightGBM `abnormal` model"][split]
            c = first[group]["failure_calls"][split]
            if a is None:
                continue
            lines.append(f"| {name} | {split} | {a['events']} | {fmt(a['precision'])} ({a['detected']} right, "
                         f"{a['false_alerts']} false) | {fmt(a['detection_rate'])} ({a['detected']} of {a['events']}) | "
                         f"{fmt(c['precision'])} ({c['came_true']} of {c['calls']}) | "
                         f"{fmt(c['recall'])} ({c['events_called_in_time']} of {c['events']}) |")
    lines += ["", "These are small samples on simulated faults. They do not show the model works on a real belt.", ""]
    for group, title in (("zone", "Idler bearing faults (zone model, 120 zones)"),
                         ("drive", "Drive overheating (drive model, 1 asset)")):
        lines += [f"## {title}", ""]
        for split in SPLITS:
            lines += [f"### {split.capitalize()} scenarios" + (" (thresholds tuned here)" if split == "validation" else ""),
                      "",
                      "| Method | Threshold | Target met on validation | Events | Detected | "
                      "Alert still open at failure | Detection rate | "
                      "Median lead time (h) | Alerts | False alerts | False share of alerts | Precision | "
                      "False alerts per asset per month | False alerts in hard-negative periods |",
                      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
            for method, m in first[group]["methods"].items():
                r = m[split]
                if r is None:
                    continue
                lines.append(f"| {method} | {fmt(m['threshold'], 3)} | {'yes' if m['met_target'] else '**no**'} | "
                             + " | ".join(fmt(r[c]) for c in COLUMNS) + " |")
            lines.append("")
        e = first[group]["eta_median_abs_error_hours"]
        lines += ["Median absolute error of the hours-to-failure estimate on advanced-stage windows: "
                  + ", ".join(f"{s} {fmt(e[s], 1)} h" for s in SPLITS) + ".", ""]
        met = first[group]["failure_call_target_met"]
        lines += ["### Does a failure call come true?", "",
                  "A failure call is made when an alert is open and the classifier's failure risk score reaches "
                  f"its threshold ({fmt(first[group]['failure_call_threshold'], 3)}). The call claims the part will "
                  f"fail within **{H:.0f} hours**. It **came true** if the part failed within {H:.0f} h of the call, "
                  "was **early** if the part failed later than that, and was "
                  "**false** if the part was healthy. The threshold was tuned on validation so that at least "
                  f"{TARGET_PRECISION:.0%} of calls come true; that target was "
                  + ("met there." if met else "**not met** there, and the best available threshold is used."), "",
                  f"| Split | Faults | Calls made | Came true within {H:.0f} h | Early (failed later) | False | "
                  "Precision (calls that came true) | Recall (faults called in time) | "
                  "Median hours from call to failure | "
                  "Median error of the time estimate at the call (h) |", "|---|---|---|---|---|---|---|---|---|---|"]
        for split in SPLITS:
            c = first[group]["failure_calls"][split]
            if c is None:
                continue
            lines.append(f"| {split} | {c['events']} | {c['calls']} | {c['came_true']} | {c['early']} | {c['false']} | "
                         f"{fmt(c['precision'])} | {fmt(c['recall'])} ({c['events_called_in_time']} of {c['events']}) | "
                         f"{fmt(c['median_hours_call_to_failure'], 1)} | {fmt(c['median_eta_error_hours'], 1)} |")
        lines += ["", "#### Trade-off on validation: failure-call threshold", "",
                  "A lower threshold calls failures earlier and more often; a higher one waits. Use this table to "
                  "balance precision against recall. It is computed on the validation scenarios only, so that the "
                  "test scenarios stay untouched by the choice.", "",
                  "| Threshold | Calls | Came true | Early | False | Precision | Recall | "
                  "Median hours from call to failure |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in first[group]["call_sweep_validation"]:
            mark = " **(chosen)**" if abs(r["threshold"] - first[group]["failure_call_threshold"]) < 1e-9 else ""
            lines.append(f"| {fmt(r['threshold'], 3)}{mark} | {r['calls']} | {r['came_true']} | {r['early']} | "
                         f"{r['false']} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                         f"{fmt(r['median_hours_call_to_failure'], 1)} |")
        lines += ["", "#### Trade-off on validation: alert threshold", "",
                  "| Threshold | Alerts | Faults detected | Alert open at failure | False alerts | Precision | Recall | "
                  "Median lead time (h) |", "|---|---|---|---|---|---|---|---|"]
        for r in first[group]["alert_sweep_validation"]:
            mark = " **(chosen)**" if abs(r["threshold"] - first[group]["alert_threshold"]) < 1e-9 else ""
            lines.append(f"| {fmt(r['threshold'], 3)}{mark} | {r['alerts']} | {r['detected']} of {r['events']} | "
                         f"{r['open_at_failure']} | {r['false_alerts']} | {fmt(r['precision'])} | "
                         f"{fmt(r['detection_rate'])} | {fmt(r['median_lead_hours'], 1)} |")
        lines += ["", "Window-level numbers for the 24-hour classifier at the same threshold. Plain accuracy is not "
                  "reported: over 99% of windows are negative, so a model that never calls a failure would score "
                  "above 99%.", "",
                  "| Split | Windows | Truly within 24 h of failure | Precision | Recall | Average precision |",
                  "|---|---|---|---|---|---|"]
        for split in SPLITS:
            r = first[group]["window_metrics"][split]
            if r is None:
                continue
            lines.append(f"| {split} | {r['windows']:,} | {r['positive_windows']:,} ({100 * r['positive_share']:.2f}%) | "
                         f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['average_precision'])} |")
        lines.append("")
        if len(seeds) > 1:
            lines += [f"### Spread over training seeds {', '.join(map(str, seeds))} (`abnormal` model)", "",
                      "| Split | Detection rate (mean, min to max) | Median lead time h (mean, min to max) | "
                      "False alerts (mean, min to max) | Precision (mean, min to max) |", "|---|---|---|---|---|"]
            for split in SPLITS:
                rs = [runs[s][group]["methods"]["LightGBM `abnormal` model"][split] for s in seeds]
                if rs[0] is None:
                    continue
                cell = lambda k: (f"{np.nanmean([r[k] for r in rs]):.2f} "  # noqa: E731
                                  f"({np.nanmin([r[k] for r in rs]):.2f} to {np.nanmax([r[k] for r in rs]):.2f})")
                lines.append(f"| {split} | {cell('detection_rate')} | {cell('median_lead_hours')} | "
                             f"{cell('false_alerts')} | {cell('precision')} |")
            lines.append("")
    path.write_text("\n".join(lines) + "\n")


def summary(run: dict) -> dict:
    """The held-out test numbers in a few figures, for the dashboard to show beside its live track record."""
    out = {"split": "test", "call_horizon_hours": CALL_HORIZON_HOURS, "faults": 0, "caught": 0, "open_at_failure": 0,
           "false_alerts": 0, "calls": 0, "calls_came_true": 0, "faults_called": 0}
    for group in ("zone", "drive"):
        a, c = run[group]["methods"]["LightGBM `abnormal` model"]["test"], run[group]["failure_calls"]["test"]
        out["faults"] += a["events"]
        out["caught"] += a["detected"]
        out["open_at_failure"] += a["open_at_failure"]
        out["false_alerts"] += a["false_alerts"]
        out["calls"] += c["calls"]
        out["calls_came_true"] += c["came_true"]
        out["faults_called"] += c["events_called_in_time"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score models and baselines per event.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    args = ap.parse_args(argv)
    runs = {}
    for i, seed in enumerate(args.seeds):
        runs[seed] = run_seed(Path(args.data), seed, save_artifacts=(i == 0))
    report(runs, ARTIFACTS / "eval_report.md")
    (ARTIFACTS / "eval.json").write_text(json.dumps(runs, indent=2, default=float))
    (ARTIFACTS / "summary.json").write_text(json.dumps(summary(runs[args.seeds[0]]), indent=2))
    print(f"wrote {ARTIFACTS / 'eval_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
