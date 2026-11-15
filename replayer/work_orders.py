"""Repair records for the replay: what a plant's maintenance system would report AFTER a repair.

A real plant learns that a part failed from its work orders. The simulator has no maintenance system, so this
module takes four facts per repair from the simulator's fault table: which asset, what kind of fault, when it
failed and when the repair ended. It is the one place in the live path that reads the answer key, and it is
fenced in three ways (all tested):
  - it reads only the four columns named below, never when a fault STARTED and never any label;
  - a record is released only once the replayed time has reached the end of that repair;
  - the backend stores records to score past predictions and never feeds them to the model.
"""
from pathlib import Path

import pandas as pd

from common.ids import simulator_asset

COLUMNS = ["asset", "type", "failure_time", "repair_end"]


def load_work_orders(scenario_dir) -> list:
    """Repair records sorted by the time they become known (the end of the repair)."""
    table = pd.read_parquet(Path(scenario_dir) / "truth" / "faults.parquet", columns=COLUMNS)
    epoch = lambda col: col.astype("datetime64[s]").astype("int64").tolist()  # noqa: E731
    orders = [{"asset": simulator_asset(a), "kind": k, "failed_ts": f, "repaired_ts": r}
              for a, k, f, r in zip(table["asset"], table["type"], epoch(table["failure_time"]), epoch(table["repair_end"]))]
    return sorted(orders, key=lambda o: o["repaired_ts"])


class WorkOrderFeed:
    """Hands out each record once, and only when the replay has reached the end of its repair."""

    def __init__(self, orders: list):
        self.pending = sorted(orders, key=lambda o: o["repaired_ts"])

    def due(self, sim_ts: int) -> list:
        ready = [o for o in self.pending if o["repaired_ts"] <= sim_ts]
        self.pending = [o for o in self.pending if o["repaired_ts"] > sim_ts]
        return ready
