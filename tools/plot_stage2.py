"""Stage 2 sanity plot: one fault of each type from a default 30-day run.

Usage: python -m tools.plot_stage2   (writes docs/stage2_faults.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from simulator.config import Config  # noqa: E402
from simulator.faults import BELT_DAMAGE, DRIVE_OVERHEAT, IDLER  # noqa: E402
from simulator.generate import generate  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "stage2_faults.png"
Z = 120


def main():
    scn = generate(Config(seed=1, days=30))
    ts = scn.belt_wide["timestamp"].to_numpy()
    bw = scn.belt_wide
    mat = lambda c: scn.zones[c].to_numpy().reshape(-1, Z)  # noqa: E731
    first = lambda kind: next(e for e in scn.events if e.type == kind)  # noqa: E731
    marks = lambda a, e: [a.axvline(ts[i], color=c, ls=":", lw=1)  # noqa: E731
                          for i, c in ((e.onset, "g"), (e.failure, "r"), (e.repair_end, "k"))]
    print(scn.faults.to_string())

    fig, ax = plt.subplots(7, 1, figsize=(13, 22))

    e = first(IDLER)
    z, nb = e.zone - 1, (e.zone - 2) % Z
    w = slice(max(e.onset - 1440, 0), e.repair_end + 720)
    for a, col, lab in ((ax[0], "bearing_temp_c", "bearing temp C"), (ax[1], "acoustic_db", "acoustic dB"),
                        (ax[2], "vibration_rms_mm_s", "vib RMS mm/s")):
        a.plot(ts[w], mat(col)[w, nb], lw=0.5, color="0.6", label=f"neighbour zone {nb + 1}")
        a.plot(ts[w], mat(col)[w, z], lw=0.5, color="C3", label=f"faulty zone {z + 1}")
        a.set_ylabel(lab)
        marks(a, e)
    ax[0].set_title(f"Idler bearing {e.fault_id}, zone {e.zone} (green onset, red failure, black repair end)")

    e = first(DRIVE_OVERHEAT)
    w = slice(max(e.onset - 1440, 0), e.repair_end + 720)
    ax[3].plot(ts[w], bw["motor_temp_c"].to_numpy()[w], label="motor temp C")
    ax[3].plot(ts[w], bw["gearbox_temp_c"].to_numpy()[w], label="gearbox temp C")
    ax[3].plot(ts[w], bw["motor_current_a"].to_numpy()[w] / 10, lw=0.5, label="motor current / 10 (A)")
    ax[3].set_title(f"Drive overheat {e.fault_id}")
    marks(ax[3], e)

    e = first(BELT_DAMAGE)
    zc = 29
    w = slice(e.failure - 300, e.failure + 30)
    ax[4].plot(ts[w], mat("vibration_kurtosis")[w, zc], lw=0.7, label=f"kurtosis, zone {zc + 1} (carry)")
    ax[4].plot(ts[w], mat("vibration_kurtosis")[w, 89], lw=0.7, label="kurtosis, zone 90 (return)")
    ax[4].set_title(f"Belt damage {e.fault_id}: last 5 h before failure, one spike per 25 min lap")
    w = slice(e.failure - 120, e.failure)
    im = ax[5].imshow(mat("vibration_kurtosis")[w].T, aspect="auto", origin="lower", vmin=2.5, vmax=8,
                      extent=[0, 120, 1, Z], cmap="magma")
    ax[5].set_ylabel("zone")
    ax[5].set_xlabel("minutes (last 2 h before failure)")
    ax[5].set_title("Kurtosis by zone and minute: the defect sweeps round the loop")
    fig.colorbar(im, ax=ax[5], pad=0.01)
    w = slice(max(e.onset - 720, 0), e.repair_end + 720)
    ax[6].plot(ts[w], bw["tension_kn"].to_numpy()[w], lw=0.5, label="tension kN")
    ax[6].set_title("Belt tension over the belt-damage event")
    marks(ax[6], e)

    for a in ax:
        if a is not ax[5]:
            a.legend(loc="upper left", fontsize=8)
            a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=100)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
