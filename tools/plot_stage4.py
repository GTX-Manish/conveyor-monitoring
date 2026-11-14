"""Stage 4 sanity plot: an outage and a stuck sensor against the clean signal, plus missing readings.

Usage: python -m tools.plot_stage4   (writes docs/stage4_messiness.png)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from simulator.config import config_from_dict  # noqa: E402
from simulator.generate import generate  # noqa: E402
from simulator.messiness import OUTAGE, STUCK  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "stage4_messiness.png"
Z = 120


def main():
    messy = generate(config_from_dict({"seed": 1, "days": 30}))
    clean = generate(config_from_dict({"seed": 1, "days": 30, "messiness": {
        "missing_rate": 0, "outages_per_30d": 0, "stuck_per_30d": 0}}))
    ts = messy.belt_wide["timestamp"].to_numpy()
    mat = lambda s, c: s.zones[c].to_numpy().reshape(-1, Z)  # noqa: E731
    print(messy.sensor_faults.to_string())
    print({k: messy.meta[k] for k in ("missing_rate_measured", "zone_nan_share", "sensor_fault_counts")})

    fig, ax = plt.subplots(3, 1, figsize=(13, 11))
    ev = next(e for e in messy.sensor_fault_events if e.type == OUTAGE)
    w = slice(max(ev.start - 360, 0), ev.end + 360)
    ax[0].plot(ts[w], mat(clean, "bearing_temp_c")[w, ev.zone - 1], color="0.7", lw=2, label="true value (never public)")
    ax[0].plot(ts[w], mat(messy, "bearing_temp_c")[w, ev.zone - 1], color="C0", lw=0.8, label="reported")
    ax[0].axvspan(ts[ev.start], ts[ev.end], color="tab:red", alpha=0.12, label="outage (truth)")
    ax[0].set_title(f"Outage {ev.sensor_fault_id}: zone {ev.zone} suite silent, bearing temperature shown")

    ev = max((e for e in messy.sensor_fault_events if e.type == STUCK), key=lambda e: e.end - e.start)
    w = slice(max(ev.start - 360, 0), min(ev.end + 360, len(ts)))
    ax[1].plot(ts[w], mat(clean, ev.signal)[w, ev.zone - 1], color="0.7", lw=1.5, label="true value (never public)")
    ax[1].plot(ts[w], mat(messy, ev.signal)[w, ev.zone - 1], color="C3", lw=0.8, label="reported")
    ax[1].axvspan(ts[ev.start], ts[ev.end], color="tab:purple", alpha=0.12, label="stuck (truth)")
    ax[1].set_title(f"Stuck sensor {ev.sensor_fault_id}: zone {ev.zone}, {ev.signal} frozen at {ev.stuck_value:.2f}")

    w = slice(6 * 60 + 1440, 9 * 60 + 1440)
    y = mat(messy, "acoustic_db")[w, 29]
    ax[2].plot(ts[w], y, marker=".", lw=0.6, label="acoustic dB, zone 30 (reported)")
    gaps = np.isnan(y)
    ax[2].plot(ts[w][gaps], np.full(gaps.sum(), np.nanmin(y) - 0.5), "rx", label="missing reading")
    ax[2].set_title("Random missing readings (about 1%): gaps, never filled")

    for a in ax:
        a.legend(loc="upper left", fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT, dpi=100)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
