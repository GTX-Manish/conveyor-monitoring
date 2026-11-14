"""Stage 3 sanity plot: hard negatives over a default 30-day run, and the labels of one idler fault.

Usage: python -m tools.plot_stage3   (writes docs/stage3_hard_negatives_labels.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simulator.config import Config  # noqa: E402
from simulator.faults import IDLER  # noqa: E402
from simulator.generate import generate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "stage3_hard_negatives_labels.png"
Z, A = 120, 121
COLOURS = {"heavy_load": "tab:orange", "heat_wave": "tab:red", "dusty": "tab:brown"}


def main():
    scn = generate(Config(seed=1, days=30))
    ts = scn.belt_wide["timestamp"].to_numpy()
    cond, bw, hn = scn.conditions, scn.belt_wide, scn.hard_negatives
    mat = lambda c: scn.zones[c].to_numpy().reshape(-1, Z)  # noqa: E731
    print(hn[hn["type"] != "startup_transient"].to_string())
    print(scn.meta["hard_negative_counts"], scn.meta["fault_counts"])

    fig, ax = plt.subplots(7, 1, figsize=(14, 22))
    ax[0].plot(ts, cond["load"], lw=0.4)
    ax[0].set_ylabel("load (truth)")
    ax[1].plot(ts, cond["ambient_c"], lw=0.6, color="k", label="ambient (truth)")
    ax[1].plot(ts, mat("bearing_temp_c")[:, 29], lw=0.4, label="bearing temp, zone 30")
    ax[1].set_ylabel("C")
    ax[2].plot(ts, cond["dust"], lw=0.5, color="tab:brown", label="dust (truth)")
    ax[2].set_ylabel("dust")
    ax2 = ax[2].twinx()
    ax2.plot(ts, mat("acoustic_db")[:, 29], lw=0.2, alpha=0.6, label="acoustic dB, zone 30")
    ax2.set_ylim(30, 75)
    ax[3].plot(ts, bw["motor_temp_c"], lw=0.5, label="motor temp C")
    ax[3].plot(ts, bw["motor_current_a"] / 10, lw=0.3, label="motor current / 10 (A)")
    for a in ax[:4]:
        for _, e in hn.iterrows():
            if e["type"] in COLOURS:
                a.axvspan(e["start_time"], e["end_time"], color=COLOURS[e["type"]], alpha=0.18)
    ax[0].set_title("Hard negatives over 30 days (orange heavy load, red heat wave, brown dusty spell); "
                    "all labelled as not faults")

    # start-up transient: one morning start
    start = hn[hn["type"] == "startup_transient"].iloc[3]
    i = int((ts >= start["start_time"].to_datetime64()).argmax())
    w = slice(i - 15, i + 45)
    ax[4].plot(ts[w], mat("vibration_rms_mm_s")[w, 29], marker=".", label="vib RMS mm/s, zone 30")
    ax[4].axvspan(start["start_time"], start["end_time"], color="tab:green", alpha=0.2, label="start-up transient")
    ax[4].set_title("Start-up transient: vibration x1.3 for the first 10 minutes")

    # labels for one idler fault
    ev = next(e for e in scn.events if e.type == IDLER)
    lab = scn.labels
    grid = lambda c: lab[c].to_numpy().reshape(-1, A)[:, ev.zone - 1]  # noqa: E731
    w = slice(max(ev.onset - 1440, 0), ev.repair_end + 720)
    ax[5].plot(ts[w], mat("bearing_temp_c")[w, ev.zone - 1], lw=0.5, color="C3", label=f"bearing temp, zone {ev.zone}")
    ax[5].plot(ts[w], mat("bearing_temp_c")[w, (ev.zone - 2) % Z], lw=0.5, color="0.6", label="neighbour")
    ax[5].set_title(f"Labels for idler fault {ev.fault_id}, zone {ev.zone}")
    codes = lab["stage"].cat.codes.to_numpy().reshape(-1, A)[:, ev.zone - 1]
    ax[6].plot(ts[w], codes[w], drawstyle="steps-post", label="stage (0 normal, 1 early, 2 advanced, 3 failed)")
    ax[6].plot(ts[w], grid("fails_within_24h")[w] * 3.5, drawstyle="steps-post", label="fails_within_24h (x3.5)")
    ax6 = ax[6].twinx()
    ax6.plot(ts[w], grid("hours_to_failure")[w], color="tab:green", lw=1, label="hours_to_failure")
    ax6.set_ylabel("hours to failure")
    ax6.legend(loc="upper right", fontsize=8)

    for a in ax:
        a.legend(loc="upper left", fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=95)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
