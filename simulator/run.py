"""CLI: python -m simulator.run --days 14 --seed 1"""
import argparse
import time

from .config import Config, config_from_dict, config_to_dict, load_config
from .generate import generate
from .io import write_scenario


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate a conveyor-belt scenario.")
    ap.add_argument("--config", help="YAML config (defaults used if omitted)")
    ap.add_argument("--days", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scenario", help="scenario name (folder under --out)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else Config()
    overrides = {k: v for k, v in {"days": args.days, "seed": args.seed, "scenario": args.scenario}.items()
                 if v is not None}
    if overrides:
        cfg = config_from_dict({**config_to_dict(cfg), **overrides})
    if args.scenario is None and cfg.scenario == "scenario":
        cfg = config_from_dict({**config_to_dict(cfg), "scenario": f"dev_seed{cfg.seed}_{cfg.days}d"})

    t0 = time.perf_counter()
    scn = generate(cfg)
    root = write_scenario(scn, cfg, args.out)
    print(f"{cfg.scenario}: {scn.meta['zone_rows']:,} zone rows, generated in {scn.meta['generate_seconds']} s, "
          f"total {time.perf_counter() - t0:.1f} s -> {root}")


if __name__ == "__main__":
    main()
