"""Find a seed whose 14-day scenario tells the demo story (BUILD_PLAN.md section 4).

    python -m scripts.find_demo_seed            # scan, then write configs/scenarios/demo_01.yaml and its rules note

Only the schedules are drawn (faults, hard negatives, sensor faults), not the signal physics, so a seed takes
about a millisecond. Each schedule has its own random stream, so the schedule found here is exactly the one the
full generation produces (checked by a test). This script handles ground truth by design: it is one of the four
allowed readers and is never imported by the live path.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from simulator.conditions import duty_cycle_base
from simulator.config import Config
from simulator.faults import DRIVE_OVERHEAT, IDLER, apply_repair_stops, schedule_faults
from simulator.generate import make_streams
from simulator.hard_negatives import HEAT_WAVE, HEAVY_LOAD, schedule_hard_negatives
from simulator.healthy import ZONE_SIGNALS
from simulator.messiness import OUTAGE, STUCK, schedule_sensor_faults
from simulator.scenario_set import CONFIG_DIR, write_config

DAY = 1440
DAYS = 14
N_ZONES = 120
RULES = [
    ("idler", "An idler fault with onset on day 2 or later and failure between day 9 and day 12",
     "A full progression is visible"),
    ("drive", "Exactly one drive overheat, failing between day 5 and day 13, at least 24 h from the idler failure",
     "A second signature"),
    ("heat_wave", "A heat wave overlapping days 3 to 8 that ends before the idler fault's advanced stage",
     "The model must stay calm while it is hot"),
    ("heavy_load", "A heavy-load spell covering at least an hour of a night shift (21:00 to 23:00) while running",
     "A visible, harmless spike"),
    ("stuck", "A stuck sensor on a healthy zone, starting between day 3 and day 10, with 2 h or more of running time",
     "Shows 'sensor suspect'"),
    ("outage", "An outage of at least 2 hours after day 1, on a healthy zone",
     "A silent suite must not look healthy"),
]


def schedules(seed: int, days: int = DAYS):
    """The scenario's schedules without its signals. Mirrors the order of draws in simulator.generate."""
    cfg = Config(seed=seed, days=days)
    rng = make_streams(seed)
    n_steps = days * DAY
    hard_negs = schedule_hard_negatives(cfg.hard_negatives, days, rng["hard_negatives"])
    events = schedule_faults(cfg.faults, days, N_ZONES, rng["faults"])
    base = apply_repair_stops(events, duty_cycle_base(cfg.conditions, cfg.start, n_steps),
                              cfg.conditions.ramp_minutes)
    sensor = schedule_sensor_faults(cfg.messiness, days, N_ZONES, ZONE_SIGNALS, rng["messiness"])
    return cfg, events, hard_negs, sensor, base


def check(seed: int) -> dict:
    """Return {rule: bool} for one seed."""
    cfg, events, hard_negs, sensor, base = schedules(seed)
    running = base >= 1.0
    minute_of_day = np.arange(len(base)) % DAY
    night = running & (minute_of_day >= 21 * 60) & (minute_of_day < 23 * 60)
    idlers = [e for e in events if e.type == IDLER]
    drives = [e for e in events if e.type == DRIVE_OVERHEAT]
    faulty_zones = {e.zone for e in idlers}
    exponent = cfg.faults.progression_exponent
    early_share = cfg.labels.early_below ** (1.0 / exponent)   # share of a fault's life spent in the early stage

    ok = dict.fromkeys((r[0] for r in RULES), False)
    stars = [e for e in idlers if e.onset >= 1 * DAY and 8 * DAY <= e.failure < 12 * DAY]
    ok["heavy_load"] = any(night[h.start:h.end].sum() >= 60 for h in hard_negs if h.type == HEAVY_LOAD)
    ok["stuck"] = any(s.type == STUCK and s.zone not in faulty_zones and 2 * DAY <= s.start < 10 * DAY
                      and running[s.start:s.end].sum() >= 120 for s in sensor)
    ok["outage"] = any(s.type == OUTAGE and s.zone not in faulty_zones and s.start >= DAY
                       and s.end - s.start >= 120 for s in sensor)
    best = (False, False)
    for star in stars:   # the drive and heat-wave rules are judged against the star idler fault
        advanced_start = star.onset + early_share * (star.failure - star.onset)
        drive = (len(drives) == 1 and 4 * DAY <= drives[0].failure < 13 * DAY
                 and abs(drives[0].failure - star.failure) >= DAY)
        heat = any(h.type == HEAT_WAVE and h.start < 8 * DAY and h.end > 2 * DAY and h.end <= advanced_start
                   for h in hard_negs)
        if drive + heat > sum(best):
            best = (drive, heat)
    ok["idler"] = bool(stars)
    ok["drive"], ok["heat_wave"] = best
    return ok


def find(first: int = 1000, last: int = 3000):
    """First seed passing every rule, else the best match. Returns (seed, rule results, passed_all)."""
    best = (None, None, -1)
    for seed in range(first, last + 1):
        ok = check(seed)
        n = sum(ok.values())
        if n == len(RULES):
            return seed, ok, True
        if n > best[2]:
            best = (seed, ok, n)
    return best[0], best[1], False


def rules_note(seed, ok, passed, first, last) -> str:
    lines = [f"The demo scenario is **curated**: seed **{seed}** is the first seed in {first} to {last} whose 14-day "
             "schedule passes every rule below." if passed else
             f"No seed in {first} to {last} passes every rule. Seed **{seed}** is the best match.", "",
             "| Rule | Why | Passed |", "|---|---|---|"]
    lines += [f"| {text} | {why} | {'yes' if ok[key] else '**no**'} |" for key, text, why in RULES]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find the demo seed.")
    ap.add_argument("--first", type=int, default=1000)
    ap.add_argument("--last", type=int, default=3000)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    seed, ok, passed = find(args.first, args.last)
    if seed is None:
        print("no seeds scanned", file=sys.stderr)
        return 1
    failed = [k for k, v in ok.items() if not v]
    print(f"seed {seed}: " + ("passes every rule" if passed else f"best match; failed rule(s): {failed}"))
    if not args.no_write:
        path = write_config(dict(name="demo_01", split="demo", seed=seed, days=DAYS, overrides={}))
        Path(CONFIG_DIR / "demo_01.rules.md").write_text(rules_note(seed, ok, passed, args.first, args.last))
        print(f"wrote {path}")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
