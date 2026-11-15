"""How does the frozen model cope with data that is NOT like its training data?

    python -m scripts.robustness            # writes model/artifacts/robustness_report.md

The evaluation in scripts/score.py uses scenarios drawn from the same simulator settings as training. A model can
score perfectly there and still be brittle, so this script builds fresh 30-day scenarios that are deliberately
different (weaker faults, other growth shapes, much more sensor noise) and scores the saved model on them with its
thresholds frozen. Nothing is retuned here. It reads each scenario's labels, so it is one of the allowed readers of
ground truth.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from model.alerting import HOLD_P_FAIL
from model.evaluate import score, score_calls
from model.predict import ARTIFACTS
from model.train import grids
from scripts.make_model_data import build
from simulator.config import config_from_dict
from simulator.generate import generate
from simulator.io import write_scenario

REPO = Path(__file__).resolve().parents[1]
FIXED = {"strength": [1.0, 1.0], "signal_mix": [1.0, 1.0], "exponent": [2.5, 2.5]}   # the old identical fault


def _faults(strength=(0.3, 1.5), exponent=(1.5, 3.5), mix=(0.5, 1.5), idler=8.0, drive=2.0):
    return {"strength": list(strength), "exponent": list(exponent), "signal_mix": list(mix),
            "idler_bearing": {"per_30d": idler}, "drive_overheat": {"per_30d": drive}}


def _only(noise=1.0, drift=0.0, shared=0.0):
    """Exactly these messiness levels, nothing random: base noise x `noise`, drift x `drift`, shared x `shared`."""
    return {"noise_scale": noise, "noise_random": [1.0, 1.0], "drift_random": [drift, drift],
            "shared_random": [shared, shared]}


# name -> (seed, faults, signals). Seeds are far from the 101 to 401 used by the scenario set. An empty signals
# dict means the simulator's own random noise, drift and shared-noise levels, as in training. The "fixed" cases
# set exactly one kind of messiness and switch the others off, to isolate it. Training saw noise 1 to 5x,
# drift 0 to 3x and shared noise 0 to 3x.
CASES = {
    "like training (everything random)": (701, _faults(), {}),
    "faults 0.5 to 0.75 of normal strength": (702, _faults(strength=(0.5, 0.75)), {}),
    "faults 0.2 to 0.4 of normal strength": (703, _faults(strength=(0.2, 0.4)), {}),
    "slow, nearly linear growth": (704, _faults(exponent=(1.0, 1.4)), {}),
    "very late, sharp growth": (705, _faults(exponent=(3.5, 4.5)), {}),
    "clean: no noise change, no drift, no shared noise": (709, _faults(), _only()),
    "noise fixed at 5x (top of training range)": (707, _faults(), _only(noise=5.0)),
    "noise fixed at 8x (beyond training range)": (708, _faults(), _only(noise=8.0)),
    "noise fixed at 12x (far beyond)": (710, _faults(), _only(noise=12.0)),
    "drift fixed at 3x (top of training range)": (711, _faults(), _only(drift=3.0)),
    "drift fixed at 6x (beyond training range)": (712, _faults(), _only(drift=6.0)),
    "drift fixed at 10x (far beyond)": (713, _faults(), _only(drift=10.0)),
    "shared noise fixed at 3x (top of training range)": (714, _faults(), _only(shared=3.0)),
    "shared noise fixed at 6x (beyond training range)": (715, _faults(), _only(shared=6.0)),
    "all three at the top of the training range": (716, _faults(), _only(noise=5.0, drift=3.0, shared=3.0)),
}


def run_case(name, seed, faults, signals, boosters, meta, out_dir: Path) -> dict:
    cfg = config_from_dict({"scenario": f"robust_{seed}", "seed": seed, "days": 30, "faults": faults, "signals": signals})
    root = write_scenario(generate(cfg), cfg, out_dir)
    build(root, out_dir / "model", cfg.scenario)
    out = {}
    for group in ("zone", "drive"):
        g = grids(pd.read_parquet(out_dir / "model" / f"{cfg.scenario}_{group}.parquet"), boosters[group], group)
        t = meta["thresholds"][group]
        a = score(g["p_abnormal"], t["abnormal"], g["run_share"], g["stage"], g["fault_id"], g["hard_negative"], 1.0,
                  hold=g["p_fail_24h"] >= HOLD_P_FAIL)
        c = score_calls(g["p_abnormal"], t["abnormal"], g["p_fail_24h"], t["predicted_failure"], g["run_share"],
                        g["stage"], g["fault_id"], g["hours_to_failure"], g["est_hours"])
        out[group] = {"alerts": a, "calls": c}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score the frozen model on deliberately different data.")
    ap.add_argument("--out", default=str(REPO / "data" / "robustness"))
    ap.add_argument("--only", nargs="+", help="case names to run (default all)")
    args = ap.parse_args(argv)
    meta = json.loads((ARTIFACTS / "meta.json").read_text())
    kinds = ("abnormal", "classifier", "regressor")
    boosters = {g: {k: lgb.Booster(model_file=str(ARTIFACTS / f"{g}_{k}.txt")) for k in kinds} for g in ("zone", "drive")}
    out_dir = Path(args.out)
    results = {}
    for name, (seed, faults, signals) in CASES.items():
        if args.only and name not in args.only:
            continue
        results[name] = run_case(name, seed, faults, signals, boosters, meta, out_dir)
        z = results[name]["zone"]["alerts"]
        print(f"{name}: bearing faults caught {z['detected']}/{z['events']}, false alerts {z['false_alerts']}", flush=True)

    H = meta["call_horizon_hours"]
    lines = ["# Robustness report", "",
             "Generated by `python -m scripts.robustness`. Do not edit by hand.", "",
             "The saved model, with its thresholds frozen, scored on fresh 30-day scenarios that are deliberately "
             "unlike its training data. Nothing is retuned. Each case has about 8 bearing faults and 2 drive faults, "
             "so every number is a small sample.", "",
             f"**Alert**: an alert opened on the faulty part before it failed (recall = faults caught; precision = "
             f"faults caught against faults caught plus false alerts). **Call**: a failure call that came true within "
             f"{H:.0f} h of the call.", ""]
    for group, title in (("zone", "Idler bearings (zone model)"), ("drive", "Drive")):
        lines += [f"## {title}", "",
                  "| Case | Faults | Caught | Alert still open at failure | False alerts | Alert precision | "
                  f"Calls | Came true within {H:.0f} h | Early | False calls |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in results.items():
            a, c = r[group]["alerts"], r[group]["calls"]
            prec = "–" if a["precision"] != a["precision"] else f"{a['precision']:.2f}"
            lines.append(f"| {name} | {a['events']} | {a['detected']} | {a['open_at_failure']} | {a['false_alerts']} | "
                         f"{prec} | {c['calls']} | {c['came_true']} | {c['early']} | {c['false']} |")
        lines.append("")
    (ARTIFACTS / "robustness_report.md").write_text("\n".join(lines) + "\n")
    (ARTIFACTS / "robustness.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {ARTIFACTS / 'robustness_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
