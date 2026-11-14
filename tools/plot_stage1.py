"""Stage 1 sanity plot: two days of healthy signals, compared by eye with spec section 5.5.

Usage: python -m tools.plot_stage1   (writes docs/stage1_healthy.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simulator.config import Config  # noqa: E402
from simulator.generate import generate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "stage1_healthy.png"


def main():
    scn = generate(Config(seed=1, days=3))
    z, bw, cond = scn.zones, scn.belt_wide, scn.conditions
    t0 = cond["timestamp"].iloc[0] + __import__("pandas").Timedelta(days=1)
    t1 = t0 + __import__("pandas").Timedelta(days=2)
    sel = lambda df: df[(df["timestamp"] >= t0) & (df["timestamp"] < t1)]  # noqa: E731
    z, bw, cond = sel(z), sel(bw), sel(cond)
    picks = {1: "zone 1 (loading)", 30: "zone 30 (carry)", 90: "zone 90 (return)"}

    fig, ax = plt.subplots(5, 1, figsize=(13, 15), sharex=True)
    for g, label in picks.items():
        zz = z[z["global_zone"] == g]
        ax[0].plot(zz["timestamp"], zz["bearing_temp_c"], lw=0.8, label=label)
        ax[1].plot(zz["timestamp"], zz["vibration_rms_mm_s"], lw=0.5, label=label)
        ax[2].plot(zz["timestamp"], zz["acoustic_db"], lw=0.5, label=label)
    ax[0].plot(cond["timestamp"], cond["ambient_c"], "k--", lw=1, label="ambient (truth)")
    ax[0].set_ylabel("bearing temp C")
    ax[1].set_ylabel("vib RMS mm/s")
    ax[2].set_ylabel("acoustic dB")
    ax[3].plot(bw["timestamp"], bw["motor_temp_c"], label="motor temp C")
    ax[3].plot(bw["timestamp"], bw["gearbox_temp_c"], label="gearbox temp C")
    ax[3].set_ylabel("drive temp C")
    ax[4].plot(bw["timestamp"], bw["motor_current_a"], lw=0.6, label="motor current A")
    ax4b = ax[4].twinx()
    ax4b.plot(cond["timestamp"], cond["load"], "g", lw=0.8, label="load (truth)")
    ax4b.set_ylabel("load")
    ax[4].set_ylabel("current A")
    for a in ax:
        a.legend(loc="upper left", fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("Stage 1 healthy baseline, seed 1, two running days")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
