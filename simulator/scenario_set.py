"""Build the whole scenario set from committed configs (BUILD_PLAN.md section 4).

    python -m simulator.scenario_set --write-configs      # (re)write configs/scenarios/*.yaml from the table below
    python -m simulator.scenario_set                      # build every scenario that is missing or out of date
    python -m simulator.scenario_set --only demo_01 --force

Each scenario is generated in its own process, so memory is returned between scenarios and the peak is measured
per scenario. This script reads its own truth output to fill the summary; it is one of the four allowed readers.
"""
import argparse
import hashlib
import json
import resource
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import config_from_dict, config_to_dict, load_config
from .generate import generate
from .io import PUBLIC_COLUMNS, write_scenario

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "configs" / "scenarios"
MIN_FREE_GB = 5.0
MAX_PEAK_MB = 3000.0
MAX_WORKERS = 2

# (assumption) harder data for the stress scenario: more noise, more missing data, more events of every kind
STRESS_OVERRIDES = {
    # noise doubled on top of its random level, and drift and shared noise held in the top third of their range
    "signals": {"noise_scale": 2.0, "drift_random": [2.0, 3.0], "shared_random": [2.0, 3.0]},
    "messiness": {"missing_rate": 0.03, "outages_per_30d": 12.0, "stuck_per_30d": 8.0},
    "faults": {"idler_bearing": {"per_30d": 8.0}, "drive_overheat": {"per_30d": 2.0}},
    "hard_negatives": {"heavy_load": {"per_30d": 6.0}, "dusty": {"per_30d": 3.0}},
}


def _levels(noise, drift, shared) -> dict:
    """Exactly these messiness levels, nothing random."""
    return {"noise_random": [noise, noise], "drift_random": [drift, drift], "shared_random": [shared, shared]}


# (noise x, drift x, shared noise x) for each validation scenario: quiet, mixed, noisy, drifting, shared, all high.
VALIDATION_LEVELS = [(1.0, 0.0, 0.0), (3.0, 1.5, 1.5), (5.0, 0.0, 0.0), (2.0, 3.0, 0.0), (2.0, 0.0, 3.0),
                     (4.0, 3.0, 3.0)]


def default_specs() -> list:
    """name, split, seed, days, overrides. The demo scenario is added by scripts/find_demo_seed.py."""
    specs = [(f"train_{i:02d}", "train", 100 + i, 30, {}) for i in range(1, 9)]
    # Validation is stratified, not random: the thresholds are tuned here, so it must cover the whole range of
    # sensor messiness the other splits draw from (noise 1 to 5x, drift 0 to 3x, shared noise 0 to 3x).
    specs += [(f"val_{i:02d}", "validation", 200 + i, 30, {"signals": _levels(*levels)})
              for i, levels in enumerate(VALIDATION_LEVELS, start=1)]
    # Extra drive-fault scenarios for the drive model only (handover, final day). The fault stream is shared, so raising
    # the drive rate inside train_01 to train_08 would also change their bearing faults; separate scenarios leave the
    # zone model, every existing scenario and the demo untouched.
    specs += [(f"train_drive_{i:02d}", "train_drive", 600 + i, 30, {"faults": {"drive_overheat": {"per_30d": 4.0}}})
              for i in range(1, 7)]
    specs += [(f"test_{i:02d}", "test", 300 + i, 30, {}) for i in (1, 2)]
    specs += [("stress_01", "stress", 401, 30, STRESS_OVERRIDES)]
    # Fresh test (final day): new seeds, scored once with the frozen models. The drive rate is boosted to 3 per 30 days
    # (default 1) so there are enough drive events to judge; everything else is default and random.
    specs += [(f"test_{i:02d}", "test_fresh", 500 + i - 2, 30, {"faults": {"drive_overheat": {"per_30d": 3.0}}})
              for i in range(3, 7)]
    return [dict(name=n, split=s, seed=seed, days=d, overrides=o) for n, s, seed, d, o in specs]


def write_config(spec: dict, config_dir=CONFIG_DIR) -> Path:
    """One YAML per scenario: the full default config plus overrides, with the split recorded beside it."""
    import yaml
    cfg = config_from_dict({"scenario": spec["name"], "seed": spec["seed"], "days": spec["days"], **spec["overrides"]})
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{spec['name']}.yaml"
    with open(path, "w") as fh:
        fh.write(f"# split: {spec['split']}\n")
        yaml.safe_dump(config_to_dict(cfg), fh, sort_keys=False)
    return path


def read_split(path: Path) -> str:
    first = Path(path).read_text().splitlines()[0]
    if not first.startswith("# split:"):
        raise ValueError(f"{path} has no '# split:' first line")
    return first.split(":", 1)[1].strip()


def config_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(config_to_dict(cfg), sort_keys=True).encode()).hexdigest()[:16]


def is_current(root: Path, digest: str) -> bool:
    manifest = root / "public" / "manifest.json"
    if not manifest.exists() or not (root / "truth" / "summary.json").exists():
        return False
    if json.loads(manifest.read_text()).get("config_hash") != digest:
        return False
    return all((root / "public" / f"{n}.parquet").exists() for n in PUBLIC_COLUMNS)


def _build_one(config_path: str, split: str, out_root: str) -> dict:
    """Runs in a child process: generate, write, summarise from the truth tables."""
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    digest = config_hash(cfg)
    scn = generate(cfg)
    root = write_scenario(scn, cfg, out_root, manifest_extra={"config_hash": digest})
    lab = scn.labels
    summary = {
        "name": cfg.scenario, "split": split, "seed": cfg.seed, "days": cfg.days, "config_hash": digest,
        "path": str(root), "zone_rows": len(scn.zones),
        "public_mb": round(sum(p.stat().st_size for p in (root / "public").iterdir()) / 1e6, 1),
        "noise_scale_effective": round(scn.meta["noise_scale_effective"], 2),
        "drift_scale_effective": round(scn.meta["drift_scale_effective"], 2),
        "shared_scale_effective": round(scn.meta["shared_scale_effective"], 2),
        "fault_counts": scn.meta["fault_counts"],
        "hard_negative_counts": scn.meta["hard_negative_counts"],
        "sensor_fault_counts": scn.meta["sensor_fault_counts"],
        "fails_within_24h_rate": float(lab["fails_within_24h"].mean()),
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 0),
    }
    (root / "truth" / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def balance_check(rows: list):
    """Return (errors, warnings). Errors fail the build."""
    errors, warnings = [], []
    seeds, names = {}, {}
    for r in rows:
        if r["seed"] in seeds:
            errors.append(f"seed {r['seed']} is used by both {seeds[r['seed']]} and {r['name']}")
        seeds[r["seed"]] = r["name"]
        if r["name"] in names and names[r["name"]] != r["split"]:
            errors.append(f"{r['name']} appears in two splits")
        names[r["name"]] = r["split"]
        if r["split"] == "demo":
            continue   # the demo is curated by its own rules (scripts/find_demo_seed.py)
        for kind in ("heavy_load", "heat_wave", "dusty"):
            if r["hard_negative_counts"].get(kind, 0) < 1:
                warnings.append(f"{r['name']}: no {kind} hard negative (try another seed)")
        if r["fault_counts"].get("idler_bearing", 0) < 1:
            warnings.append(f"{r['name']}: no idler fault (try another seed)")
    for split in ("validation", "test"):
        part = [r for r in rows if r["split"] == split]
        if not part:
            continue
        idler = sum(r["fault_counts"].get("idler_bearing", 0) for r in part)
        drive = sum(r["fault_counts"].get("drive_overheat", 0) for r in part)
        if idler < 2 or drive < 1:
            warnings.append(f"{split}: only {idler} idler and {drive} drive events (need 2 and 1; try another seed)")
    return errors, warnings


def split_totals(rows: list) -> dict:
    out = {}
    for r in rows:
        t = out.setdefault(r["split"], {"scenarios": 0, "idler_bearing": 0, "drive_overheat": 0, "zone_rows": 0,
                                        "public_mb": 0.0, "_pos": 0.0})
        t["scenarios"] += 1
        t["idler_bearing"] += r["fault_counts"].get("idler_bearing", 0)
        t["drive_overheat"] += r["fault_counts"].get("drive_overheat", 0)
        t["zone_rows"] += r["zone_rows"]
        t["public_mb"] = round(t["public_mb"] + r["public_mb"], 1)
        t["_pos"] += r["fails_within_24h_rate"]
    for t in out.values():
        t["fails_within_24h_rate"] = t.pop("_pos") / t["scenarios"]
    return out


def write_report(rows, errors, warnings, path, demo_note=None) -> None:
    lines = ["# Scenario set report", "",
             "Generated by `python -m simulator.scenario_set`. Do not edit by hand.", "",
             "| Scenario | Split | Seed | Days | Noise x | Drift x | Shared x | Idler faults | Drive faults | "
             "Heavy load | Heat wave | Dusty | "
             "Outages | Stuck | `fails_within_24h` rate | Public MB | Seconds | Peak MB |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        f, h, s = r["fault_counts"], r["hard_negative_counts"], r["sensor_fault_counts"]
        lines.append(f"| `{r['name']}` | {r['split']} | {r['seed']} | {r['days']} | {r.get('noise_scale_effective', '')} | "
                     f"{r.get('drift_scale_effective', '')} | {r.get('shared_scale_effective', '')} | "
                     f"{f.get('idler_bearing', 0)} | "
                     f"{f.get('drive_overheat', 0)} | {h.get('heavy_load', 0)} | {h.get('heat_wave', 0)} | "
                     f"{h.get('dusty', 0)} | {s.get('outage', 0)} | {s.get('stuck', 0)} | "
                     f"{100 * r['fails_within_24h_rate']:.3f}% | {r['public_mb']} | {r['seconds']} | {r['peak_mb']:.0f} |")
    lines += ["", "## Totals per split", "",
              "| Split | Scenarios | Idler faults | Drive faults | Zone rows | Public MB | Mean `fails_within_24h` rate |",
              "|---|---|---|---|---|---|---|"]
    for split, t in split_totals(rows).items():
        lines.append(f"| {split} | {t['scenarios']} | {t['idler_bearing']} | {t['drive_overheat']} | "
                     f"{t['zone_rows']:,} | {t['public_mb']} | {100 * t['fails_within_24h_rate']:.3f}% |")
    lines += ["", "## Balance check", ""]
    lines += [f"- **ERROR:** {e}" for e in errors] + [f"- Warning: {w}" for w in warnings]
    if not errors and not warnings:
        lines.append("- No seed is used twice, no scenario is in two splits, and every scenario and split has the "
                     "events it needs.")
    if demo_note:
        lines += ["", "## Demo scenario", "", demo_note]
    Path(path).write_text("\n".join(lines) + "\n")


def build_set(config_dir=CONFIG_DIR, out="data", only=None, workers=1, force=False, report_path=None,
              min_free_gb=MIN_FREE_GB, log=print) -> dict:
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"--workers must be 1 or {MAX_WORKERS} (this machine has 2 usable cores and limited RAM)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < min_free_gb:
        raise RuntimeError(f"only {free_gb:.1f} GB free under {out}; need {min_free_gb} GB. Nothing was written.")

    paths = sorted(Path(config_dir).glob("*.yaml"))
    if only:
        missing = set(only) - {p.stem for p in paths}
        if missing:
            raise ValueError(f"no config for scenario(s): {sorted(missing)}")
        paths = [p for p in paths if p.stem in only]
    if not paths:
        raise ValueError(f"no scenario configs in {config_dir}; run with --write-configs first")

    jobs, rows, skipped = [], {}, []
    seen_seeds = {}
    for p in paths:
        cfg = load_config(p)
        if cfg.scenario != p.stem:
            raise ValueError(f"{p}: scenario name {cfg.scenario!r} does not match the file name")
        if cfg.seed in seen_seeds:
            raise ValueError(f"seed {cfg.seed} is used by both {seen_seeds[cfg.seed]} and {cfg.scenario}")
        seen_seeds[cfg.seed] = cfg.scenario
        root = out / cfg.scenario
        if not force and is_current(root, config_hash(cfg)):
            rows[cfg.scenario] = json.loads((root / "truth" / "summary.json").read_text())
            skipped.append(cfg.scenario)
        else:
            jobs.append((str(p), read_split(p), str(out)))

    if jobs:
        # one task per child process, so each scenario's peak memory is its own and is freed afterwards
        with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
            for summary in pool.map(_build_one, *zip(*jobs)):
                rows[summary["name"]] = summary
                log(f"built {summary['name']}: {summary['seconds']} s, peak {summary['peak_mb']:.0f} MB")
                if summary["peak_mb"] > MAX_PEAK_MB:
                    raise RuntimeError(f"{summary['name']} peaked at {summary['peak_mb']:.0f} MB (limit {MAX_PEAK_MB})")
    if skipped:
        log(f"up to date, skipped: {', '.join(skipped)}")

    # the set summary always covers every scenario on disk that has a config, not only this call's selection
    all_rows = []
    for p in sorted(Path(config_dir).glob("*.yaml")):
        s = out / p.stem / "truth" / "summary.json"
        if p.stem in rows:
            all_rows.append(rows[p.stem])
        elif s.exists():
            all_rows.append(json.loads(s.read_text()))
    errors, warnings = balance_check(all_rows)
    result = {"scenarios": all_rows, "totals": split_totals(all_rows), "errors": errors, "warnings": warnings,
              "built": [j[0] for j in jobs], "skipped": skipped}
    (out / "scenario_set.json").write_text(json.dumps(result, indent=2))
    if report_path:
        demo = next((r for r in all_rows if r["split"] == "demo"), None)
        note_file = Path(config_dir) / "demo_01.rules.md"
        note = note_file.read_text() if demo and note_file.exists() else None
        write_report(all_rows, errors, warnings, report_path, note)
    for w in warnings:
        log(f"warning: {w}")
    if errors:
        raise RuntimeError("balance check failed: " + "; ".join(errors))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the scenario set.")
    ap.add_argument("--only", nargs="+", help="scenario names to build")
    ap.add_argument("--workers", type=int, default=1,
                    help="1 (default) or 2. On the development machine 2 was about 20x slower per scenario.")
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    ap.add_argument("--out", default="data")
    ap.add_argument("--write-configs", action="store_true", help="write configs/scenarios/*.yaml and stop")
    args = ap.parse_args(argv)
    if args.write_configs:
        for spec in default_specs():
            print("wrote", write_config(spec).relative_to(REPO))
        return 0
    t0 = time.perf_counter()
    try:
        result = build_set(only=args.only, workers=args.workers, force=args.force, out=args.out,
                           report_path=REPO / "docs" / "scenario_set_report.md")
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(result['built'])} built, {len(result['skipped'])} skipped in {time.perf_counter() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
