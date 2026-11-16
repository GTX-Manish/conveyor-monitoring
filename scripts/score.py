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

import lightgbm as lgb
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


def evaluate_method(scenario_grids: dict, key, splits: dict, scored=SPLITS) -> dict:
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
    for split in scored:
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
    return evaluate_result(data, result)[0]


def load_frozen(data: Path) -> dict:
    """The saved models and thresholds as a `result` for evaluate_result. Nothing is trained and nothing is written.

    The threshold sweeps (only used for the report's trade-off tables) are recomputed from the saved boosters on the
    validation scenarios, and the recomputed thresholds must equal the saved ones, or the artifacts are not what the
    report would claim."""
    from model.train import tune
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    index = json.loads((data / "model" / "index.json").read_text())
    splits = {s: sorted(n for n, r in index.items() if r["split"] == s)
              for s in ("train", "train_drive", "validation", "test", "stress", "test_fresh")}
    result = {"boosters": {}, "thresholds": {}, "splits": splits, "trained_on": meta.get("trained_on", {})}
    for group in ("zone", "drive"):
        result["boosters"][group] = {k: lgb.Booster(model_file=str(ARTIFACTS / f"{group}_{k}.txt"))
                                     for k in ("abnormal", "classifier", "regressor")}
        val = [grids(pd.read_parquet(data / "model" / f"{n}_{group}.parquet"), result["boosters"][group], group)
               for n in splits["validation"]]
        tuned = tune(val)
        saved = meta["thresholds"][group]
        if tuned["abnormal"] != saved["abnormal"] or tuned["predicted_failure"] != saved["predicted_failure"]:
            raise RuntimeError(f"{group}: thresholds recomputed from the saved models "
                               f"({tuned['abnormal']}, {tuned['predicted_failure']}) differ from meta.json "
                               f"({saved['abnormal']}, {saved['predicted_failure']}); the artifacts are inconsistent")
        result["thresholds"][group] = tuned
    return result


def evaluate_result(data: Path, result: dict, scored=SPLITS):
    """Score a `result` (trained or loaded). Returns (numbers, per-scenario grids), the grids for per-scenario detail."""
    out, all_grids = {}, {}
    for group in ("zone", "drive"):
        names = [n for s in ("validation", *scored) for n in result["splits"][s]]
        names = list(dict.fromkeys(names))
        frames = {n: pd.read_parquet(data / "model" / f"{n}_{group}.parquet") for n in names}
        g = {n: grids(df, result["boosters"][group], group) for n, df in frames.items()}
        methods = {"LightGBM `abnormal` model": evaluate_method(g, "p_abnormal", result["splits"], scored)}
        if methods["LightGBM `abnormal` model"]["threshold"] != result["thresholds"][group]["abnormal"]:
            raise RuntimeError(f"{group}: the alert threshold used for scoring differs from the saved one")
        base = {n: baseline_scores(df, group) for n, df in frames.items()}
        for name in next(iter(base.values())):
            for n in g:
                g[n][name] = base[n][name]
            methods[name] = evaluate_method(g, (lambda grid, name=name: grid[name]), result["splits"], scored)
        thr = result["thresholds"][group]
        calls, rows = {}, {}
        for split in scored:
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
                      "eta_median_abs_error_hours": {s: regressor_error(g, result["splits"][s]) for s in scored}}
        all_grids[group] = g
    return out, all_grids


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


FRESH = "test_fresh"
FRESH_REPORT = ARTIFACTS / "fresh_test_report.md"
RESULTS_MARK = "## Results (scored once)"


def write_fresh_report(data: Path, result: dict, out: dict, grids_by_group: dict, path: Path = FRESH_REPORT) -> None:
    """Results of the fresh test scenarios, appended below the criteria that were committed before any scoring.

    Only reads the frozen models' outputs; scoring logic is `model.evaluate.score` and `score_calls`, unchanged."""
    if not path.exists() or "Criteria set before scoring" not in path.read_text():
        raise RuntimeError(f"{path} must exist with its 'Criteria set before scoring' section before scoring")
    head = path.read_text().split(RESULTS_MARK)[0].rstrip() + "\n"
    names = result["splits"][FRESH]
    if not names:
        raise RuntimeError("no test_fresh scenarios in data/model/index.json")
    H = CALL_HORIZON_HOURS
    lines = [RESULTS_MARK, "", "Frozen models, one scoring run. Generated by `python -m scripts.score --frozen --fresh`.",
             "",
             "### Scenarios", "", "| Scenario | Seed | Noise x | Drift x | Shared x | Bearing faults | Drive faults |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        meta = json.loads((data / n / "truth" / "run_meta.json").read_text())
        lines.append(f"| {n} | {meta['seed']} | {meta['noise_scale_effective']:.2f} | {meta['drift_scale_effective']:.2f} | "
                     f"{meta['shared_scale_effective']:.2f} | {meta['fault_counts']['idler_bearing']} | "
                     f"{meta['fault_counts']['drive_overheat']} |")
    lines += ["", "The drive fault rate in these scenarios is boosted to 3 per 30 days (default is 1) so there are enough "
              "drive events to judge; all other settings are at default.", "",
              f"### Precision and recall at a glance (split `{FRESH}`)", "",
              f"| Asset | Faults | Alerts caught | False alerts | Alert precision | Alert recall | Calls | "
              f"Calls true within {H:.0f} h | Call precision | Call recall |", "|---|---|---|---|---|---|---|---|---|---|"]
    verdict = {}
    for group, name in (("zone", "Bearings"), ("drive", "Drive")):
        a = out[group]["methods"]["LightGBM `abnormal` model"][FRESH]
        c = out[group]["failure_calls"][FRESH]
        verdict[group] = (a["precision"], a["detection_rate"])
        lines.append(f"| {name} | {a['events']} | {a['detected']} | {a['false_alerts']} | {fmt(a['precision'])} | "
                     f"{fmt(a['detection_rate'])} | {c['calls']} | {c['came_true']} | {fmt(c['precision'])} | "
                     f"{fmt(c['recall'])} ({c['events_called_in_time']} of {c['events']}) |")
    lines += ["", "### Hours from failure call to failure", ""]
    for group, name in (("zone", "Bearings"), ("drive", "Drive")):
        c = out[group]["failure_calls"][FRESH]
        lines.append(f"- {name}: median {fmt(c['median_hours_call_to_failure'], 1)} h; each call: "
                     + (", ".join(f"{h:.1f}" for h in sorted(c["hours_list"])) or "none"))
    lines += ["", "### Per scenario", "",
              "| Scenario | Bearings caught | Bearing false alerts | Drive caught | Drive false alerts |",
              "|---|---|---|---|---|"]
    for n in names:
        cells = []
        for group in ("zone", "drive"):
            x = grids_by_group[group][n]
            thr = result["thresholds"][group]
            s = score(x["p_abnormal"], thr["abnormal"], x["run_share"], x["stage"], x["fault_id"], x["hard_negative"], 1.0,
                      hold=x["p_fail_24h"] >= HOLD_P_FAIL)
            cells += [f"{s['detected']} of {s['events']}", str(s["false_alerts"])]
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    zp, zr = verdict["zone"]
    _, dr = verdict["drive"]
    ok = lambda b: "met" if b else "**not met**"  # noqa: E731
    lines += ["", "### Against the criteria set in advance", "", "| Criterion | Result | |", "|---|---|---|",
              f"| Bearings alert recall at least 0.75 | {fmt(zr)} | {ok(zr >= 0.75)} |",
              f"| Bearings alert precision at least 0.90 | {fmt(zp)} | {ok(zp >= 0.90)} |",
              f"| Drive alert recall at least 0.50 | {fmt(dr)} | {ok(dr >= 0.50)} |"]
    old = out["zone"]["methods"]["LightGBM `abnormal` model"]["test"]
    olddrive = out["drive"]["methods"]["LightGBM `abnormal` model"]["test"]
    lines += ["", "For comparison, the old test scenarios (test_01 and test_02, used for diagnosis in V7 and V8, so no "
              f"longer clean): bearings {old['detected']} of {old['events']} caught with {old['false_alerts']} false "
              f"alerts; drive {olddrive['detected']} of {olddrive['events']} caught with {olddrive['false_alerts']} "
              "false alerts.", ""]
    path.write_text(head + "\n" + "\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score models and baselines per event.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--frozen", action="store_true",
                    help="score the saved models and thresholds: nothing is trained, nothing in model/artifacts is "
                         "overwritten except the standard report files")
    ap.add_argument("--fresh", action="store_true",
                    help="with --frozen: score only the test_fresh split into fresh_test_report.md and write nothing else")
    args = ap.parse_args(argv)
    if args.fresh and not args.frozen:
        ap.error("--fresh needs --frozen")
    if args.frozen:
        result = load_frozen(Path(args.data))
        if args.fresh:
            out, grids_by_group = evaluate_result(Path(args.data), result, scored=("test", FRESH))
            write_fresh_report(Path(args.data), result, out, grids_by_group)
            print(f"wrote {FRESH_REPORT}")
            return 0
        runs = {1: evaluate_result(Path(args.data), result)[0]}
        args.seeds = [1]
        report(runs, ARTIFACTS / "eval_report.md")
        (ARTIFACTS / "eval.json").write_text(json.dumps(runs, indent=2, default=float))
        (ARTIFACTS / "summary.json").write_text(json.dumps(summary(runs[1]), indent=2))
        print(f"wrote {ARTIFACTS / 'eval_report.md'} from the saved models (nothing was trained)")
        return 0
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
