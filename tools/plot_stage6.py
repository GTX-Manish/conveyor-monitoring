"""Stage 6 plot: one timeline per scenario with faults, hard negatives and sensor faults.

Usage: python -m tools.plot_stage6   (writes docs/stage6_scenario_set.png; reads data/*/truth, analysis only)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "docs" / "stage6_scenario_set.png"
ROWS = [("idler fault (onset to failure)", "tab:red"), ("drive overheat (onset to failure)", "tab:purple"),
        ("repair stop", "black"), ("heavy load", "tab:orange"), ("heat wave", "gold"), ("dusty spell", "tab:brown"),
        ("sensor outage", "tab:blue"), ("stuck sensor", "tab:cyan")]
COLOUR = dict(ROWS)


def bars(ax, y, spans, colour, height=0.8):
    ax.broken_barh([(s, e - s) for s, e in spans], (y - height / 2, height), color=colour, alpha=0.85, lw=0)


def main():
    rows = json.loads((REPO / "data" / "scenario_set.json").read_text())["scenarios"]
    fig, axes = plt.subplots(len(rows), 1, figsize=(15, 1.5 * len(rows) + 1.2), sharex=False)
    for ax, r in zip(axes, rows):
        truth = Path(r["path"]) / "truth"
        t0 = pd.Timestamp(json.loads((Path(r["path"]) / "public" / "manifest.json").read_text())["start"])
        day = lambda col: ((col - t0) / pd.Timedelta(days=1)).to_numpy()  # noqa: E731
        f = pd.read_parquet(truth / "faults.parquet")
        h = pd.read_parquet(truth / "hard_negatives.parquet")
        s = pd.read_parquet(truth / "sensor_faults.parquet")
        if len(f):
            on, fail, rep = day(f["onset_time"]), day(f["failure_time"]), day(f["repair_end"])
            for kind, y, name in (("idler_bearing", 4, "idler fault (onset to failure)"),
                                  ("drive_overheat", 3, "drive overheat (onset to failure)")):
                m = (f["type"] == kind).to_numpy()
                bars(ax, y, zip(on[m], fail[m]), COLOUR[name])
            bars(ax, 3.5, zip(fail, rep + 0.05), "black", height=1.8)
        for kind, name in (("heavy_load", "heavy load"), ("heat_wave", "heat wave"), ("dusty", "dusty spell")):
            part = h[h["type"] == kind]
            bars(ax, 2, zip(day(part["start_time"]), day(part["end_time"])), COLOUR[name])
        for kind, name, y in (("outage", "sensor outage", 1), ("stuck", "stuck sensor", 0.3)):
            part = s[s["type"] == kind]
            bars(ax, y, zip(day(part["start_time"]), day(part["end_time"]) + 0.03), COLOUR[name], height=0.6)
        ax.set_xlim(0, r["days"])
        ax.set_ylim(-0.3, 4.7)
        ax.set_yticks([0.65, 2, 3.5], ["sensor", "hard neg", "faults"], fontsize=7)
        ax.set_title(f"{r['name']} ({r['split']}, seed {r['seed']}): {r['fault_counts']}", fontsize=8, loc="left")
        ax.grid(axis="x", alpha=0.3)
    axes[-1].set_xlabel("day")
    fig.legend(handles=[Patch(color=c, label=n) for n, c in ROWS], loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (1.5 * len(rows) + 1.2)))
    fig.savefig(OUT, dpi=90)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
