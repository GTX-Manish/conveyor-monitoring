"""Replays a scenario's public tables into the backend over HTTP, one batch per simulated minute.

    python -m replayer.run --scenario demo_01 --inject-faults

It first posts the first 24 simulated hours as fast as the backend takes them (the model needs a day of
history), then replays at the speed set through /api/replay. With --inject-faults it also adds the kinds of bad
input the simulator does not produce: duplicate rows, out-of-range values, rows for unknown zones and batches
out of order. What it added is written to data/<scenario>/replayer_injected.json, so the backend's ingest
counts can be checked against it.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import yaml

from common.ids import BELT_SIGNALS, N_ZONES, RUN_STATES, ZONE_SIGNALS

REPO = Path(__file__).resolve().parents[1]
OUT_OF_RANGE = {"vibration_rms_mm_s": 999.0, "vibration_kurtosis": -5.0, "bearing_temp_c": 1500.0,
                "acoustic_db": -20.0, "belt_offset_mm": 9999.0}
UNKNOWN_ZONE = 999


def load_public(scenario_dir: Path):
    pub = Path(scenario_dir) / "public"
    zones = pd.read_parquet(pub / "zones.parquet", columns=["timestamp", "global_zone", "run_state", *ZONE_SIGNALS])
    belt = pd.read_parquet(pub / "belt_wide.parquet")
    T = len(belt)
    ts = (belt["timestamp"].astype("datetime64[s]").astype("int64")).to_numpy()
    codes = belt["run_state"].astype(str).map({n: i for i, n in enumerate(RUN_STATES)}).to_numpy()
    z = {s: zones[s].to_numpy(dtype=np.float64).reshape(T, N_ZONES) for s in ZONE_SIGNALS}
    b = {s: belt[s].to_numpy(dtype=np.float64) for s in BELT_SIGNALS}
    return ts, codes, z, b


class Injector:
    """Seeded input faults. Counts are of what was actually sent, so they must equal the backend's counts."""

    def __init__(self, cfg: dict, enabled: bool):
        self.cfg, self.enabled = cfg, enabled
        self.rng = np.random.default_rng(cfg["seed"])
        self.counts = {"duplicate_rows": 0, "out_of_range_values": 0, "unknown_zone_rows": 0, "swapped_batches": 0}
        self.day_done = set()

    def apply(self, rows: list, ts: int, minute_index: int) -> list:
        if not self.enabled:
            return rows
        cfg, rng = self.cfg, self.rng
        # out-of-range first, so a duplicated row carries the same bad value
        for i in np.flatnonzero(rng.random(len(rows)) < cfg["out_of_range_value_rate"] * len(ZONE_SIGNALS)):
            sig = ZONE_SIGNALS[int(rng.integers(len(ZONE_SIGNALS)))]
            if rows[i][sig] is not None:
                rows[i][sig] = OUT_OF_RANGE[sig]
                self.counts["out_of_range_values"] += 1
        dup = np.flatnonzero(rng.random(len(rows)) < cfg["duplicate_row_rate"])
        rows = rows + [dict(rows[i]) for i in dup]
        self.counts["duplicate_rows"] += len(dup)
        day = minute_index // 1440
        if day not in self.day_done and minute_index % 1440 >= 720 and cfg["unknown_zone_rows_per_day"] >= 1:
            self.day_done.add(day)
            for _ in range(int(cfg["unknown_zone_rows_per_day"])):
                rows.append({**rows[0], "zone_id": UNKNOWN_ZONE})
                self.counts["unknown_zone_rows"] += 1
        return rows


def as_json_lists(a) -> list:
    """Float array -> nested lists rounded to 4 decimals with None for NaN (what goes on the wire)."""
    obj = np.round(a, 4).astype(object)
    obj[np.isnan(a)] = None
    return obj.tolist()


def batches(ts, codes, z, b, injector: Injector, chunk: int = 720):
    """Yield (minute_index, sim_ts, body), with every Nth batch swapped with the next when injecting."""
    swap = int(injector.cfg["swap_every_batches"]) if injector.enabled else 0
    zone_ids = list(range(1, N_ZONES + 1))
    s0, s1, s2, s3, s4 = ZONE_SIGNALS
    held = None
    for start in range(0, len(ts), chunk):       # convert half a day at a time: cheap and small in memory
        stop = min(start + chunk, len(ts))
        zl = [as_json_lists(z[s][start:stop]) for s in ZONE_SIGNALS]
        bl = {s: as_json_lists(b[s][start:stop]) for s in BELT_SIGNALS}
        for k in range(stop - start):
            i, t, rs = start + k, int(ts[start + k]), int(codes[start + k])
            rows = [{"ts": t, "zone_id": zid, "run_state": rs, s0: v0, s1: v1, s2: v2, s3: v3, s4: v4}
                    for zid, v0, v1, v2, v3, v4 in zip(zone_ids, zl[0][k], zl[1][k], zl[2][k], zl[3][k], zl[4][k])]
            body = {"batch": injector.apply(rows, t, i),
                    "belt": {"ts": t, "run_state": rs, **{s: bl[s][k] for s in BELT_SIGNALS}}}
            if swap and (i + 1) % swap == 0 and i + 1 < len(ts):
                held = (i, t, body)              # send the next minute first, then this one
                continue
            yield i, t, body
            if held is not None:
                injector.counts["swapped_batches"] += 1
                yield held
                held = None


def run(scenario: str, base_url: str, data_dir: Path, settings: dict, inject: bool, max_minutes=None,
        speed=None, client=None, log=print) -> dict:
    scenario_dir = Path(data_dir) / scenario
    ts, codes, z, b = load_public(scenario_dir)
    if max_minutes:
        ts, codes = ts[:max_minutes], codes[:max_minutes]
    injector = Injector(settings["inject"], inject)
    preload = int(settings["replay"]["preload_hours"] * 60)
    own = client is None
    client = client or httpx.Client(base_url=base_url, timeout=30.0)
    sent = 0
    try:
        if speed is not None:
            client.post("/api/replay", json={"speed": speed})
        state = client.post("/api/replay", json={"action": "play"}).json()
        t_wall, t_sim, started = None, None, time.time()
        for i, sim_ts, body in batches(ts, codes, z, b, injector):
            if i >= preload:
                if t_wall is None:
                    log(f"history preload done: {preload} minutes in {time.time() - started:.1f} s; replay starts")
                    t_wall, t_sim = time.time(), sim_ts
                while state["state"] == "paused":
                    time.sleep(0.3)
                    state = client.get("/api/replay").json()
                    t_wall, t_sim = time.time(), sim_ts          # do not race to catch up after a pause
                rate = state["speed"] * 3600.0                   # simulated seconds per real second
                due = t_wall + (sim_ts - t_sim) / rate
                delay = due - time.time()
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    t_wall, t_sim = time.time(), sim_ts          # running behind: re-anchor, do not burst
            reply = client.post("/api/ingest", json=body).json()["replay"]
            if reply["speed"] != state["speed"]:
                t_wall, t_sim = time.time(), sim_ts
            state = {**state, **reply}
            sent += 1
        client.post("/api/replay", json={"action": "finished"})
    finally:
        if own:
            client.close()
    summary = {"scenario": scenario, "batches_sent": sent, "inject_faults": inject, **injector.counts}
    (scenario_dir / "replayer_injected.json").write_text(json.dumps(summary, indent=2))
    log(f"replay finished: {sent} batches in {time.time() - started:.0f} s; {injector.counts}")
    return summary


def main(argv=None):
    settings = yaml.safe_load((REPO / "configs" / "app.yaml").read_text())
    ap = argparse.ArgumentParser(description="Replay a scenario into the backend.")
    ap.add_argument("--scenario", default=settings["server"]["scenario"])
    ap.add_argument("--url", default=f"http://{settings['server']['host']}:{settings['server']['port']}")
    ap.add_argument("--data", default=str(REPO / settings["server"]["data_dir"]))
    ap.add_argument("--inject-faults", action="store_true")
    ap.add_argument("--speed", type=float, help="simulated hours per real second")
    ap.add_argument("--max-minutes", type=int)
    args = ap.parse_args(argv)
    run(args.scenario, args.url, Path(args.data), settings, args.inject_faults, args.max_minutes, args.speed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
