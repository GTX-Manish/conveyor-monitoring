"""Equipment faults: scheduling, repair stops and signal effects (SIMULATOR_SPEC.md section 7).

Everything here is ground truth. Events are scheduled before the physics runs (repair stops shape the
run state) and their effects are added after the healthy signals exist. Effects are deterministic given
the schedule: all randomness is in `schedule_faults`, drawn from the fault stream only.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conditions import lag
from .config import Config, FaultsCfg

IDLER, DRIVE_OVERHEAT = "idler_bearing", "drive_overheat"
SNAP_SLACK_MIN = 12 * 60  # room for moving a failure out of the daily stopped window


@dataclass
class FaultEvent:
    fault_id: str
    type: str
    asset: str                 # zone number as text, or 'DRIVE'
    zone: int                  # 1-based zone for idler faults, else 0
    onset: int                 # step indices
    failure: int
    repair_end: int


def _count(rate_per_30d, days, rng):
    x = rate_per_30d * days / 30.0
    n = int(np.floor(x))
    return n + int(rng.random() < x - n)  # always draws, so the stream stays aligned


def _place(n, dur_lo_min, dur_hi_min, repair_min, n_steps, cfg, rng, gap_min=None):
    """Draw up to n (onset, failure, repair_end) triples; non-overlapping if gap_min is given."""
    earliest = int(cfg.min_start_hours * 60)
    margin = int(cfg.min_end_margin_hours * 60) + SNAP_SLACK_MIN
    out = []
    for _ in range(n):
        for _try in range(50):
            dur = int(rng.uniform(dur_lo_min, dur_hi_min))
            latest = n_steps - margin - repair_min - dur
            if latest < earliest:
                continue  # this duration does not fit the scenario; try a shorter one
            onset = int(rng.integers(earliest, latest + 1))
            failure = onset + dur
            end = failure + repair_min
            if gap_min is not None and any(onset < e + gap_min + SNAP_SLACK_MIN and o < end + gap_min + SNAP_SLACK_MIN
                                           for o, _, e in out):
                continue
            out.append((onset, failure, end))
            break
    return out


def schedule_faults(cfg: FaultsCfg, days: int, n_zones: int, rng) -> list:
    """Draw fault events. Failure times are provisional until `apply_repair_stops` snaps them."""
    n_steps = days * 1440
    gap = int(cfg.same_asset_gap_hours * 60)
    raw = []

    c = cfg.idler_bearing
    n = min(_count(c.per_30d, days, rng), n_zones)
    slots = _place(n, c.min_days * 1440, c.max_days * 1440, int(c.repair_hours * 60), n_steps, cfg, rng)
    zones = rng.choice(np.arange(1, n_zones + 1), size=len(slots), replace=False)  # distinct zones: no overlap
    for (o, f, e), z in zip(slots, zones):
        raw.append((IDLER, str(int(z)), int(z), o, f, e))

    c = cfg.drive_overheat
    n = _count(c.per_30d, days, rng)
    for o, f, e in _place(n, c.min_hours * 60, c.max_hours * 60, int(c.repair_hours * 60), n_steps, cfg, rng, gap):
        raw.append((DRIVE_OVERHEAT, "DRIVE", 0, o, f, e))

    raw.sort(key=lambda r: (r[3], r[0], r[1]))
    return [FaultEvent(f"F{i + 1:03d}", *r) for i, r in enumerate(raw)]


def apply_repair_stops(events, base, ramp_minutes: float):
    """Snap each failure to a minute when the belt is running, then stop the belt for the repair window.

    An asset fails while it is working, so a failure drawn inside a stop (the nightly window or another
    repair) moves to the next running minute. Returns the new duty-cycle factor; events are updated in place.
    """
    base = np.array(base, dtype=float)
    ramp = max(int(round(ramp_minutes)), 1)
    for ev in sorted(events, key=lambda e: e.failure):
        repair = ev.repair_end - ev.failure
        running = np.flatnonzero(base[ev.failure:] >= 1.0)
        if len(running) == 0:
            raise RuntimeError(f"no running minute left for the failure of {ev.fault_id}")
        ev.failure += int(running[0])
        ev.repair_end = ev.failure + repair
        base[ev.failure:ev.repair_end] = 0.0
        # restart ramp after the repair (never above what the duty cycle allows)
        k = np.arange(ev.repair_end, min(ev.repair_end + ramp, len(base)))
        base[k] = np.minimum(base[k], (k - ev.repair_end + 1) / ramp)
    return base


def severity(ev: FaultEvent, n_steps: int, exponent: float):
    """s(t): 0 before onset, rising to 1 at failure, 1 during the repair window, 0 after repair_end."""
    s = np.zeros(n_steps)
    t = np.arange(ev.onset, ev.failure)
    s[t] = ((t - ev.onset) / (ev.failure - ev.onset)) ** exponent
    s[ev.failure:ev.repair_end] = 1.0
    return s


def apply_fault_effects(zone_sig, belt_sig, events, cond, cfg: Config) -> None:
    """Add fault effects to the healthy signals in place. Per-event loop; vectorised over time."""
    fc, sc = cfg.faults, cfg.signals
    n_steps = zone_sig["bearing_temp_c"].shape[0]
    speed = cond.speed

    for ev in events:
        s = severity(ev, n_steps, fc.progression_exponent)
        # s is 0 after repair_end; the extra 12 h only lets the thermal lag decay smoothly
        live = slice(ev.onset, min(n_steps, ev.repair_end + 720))
        sl, sp = s[live], speed[live]

        if ev.type == IDLER:
            c, z = fc.idler_bearing, ev.zone - 1
            # heat needs a turning roller, and the bearing housing is a thermal mass
            zone_sig["bearing_temp_c"][live, z] += lag(c.temp_c * sl ** c.temp_exp * sp, sc.temp_tau_min)
            zone_sig["acoustic_db"][live, z] += c.acoustic_db * sl ** c.acoustic_exp * sp
            zone_sig["vibration_rms_mm_s"][live, z] *= 1.0 + c.vib_gain * sl ** c.vib_exp * np.minimum(sp, 1.0)
            zone_sig["vibration_kurtosis"][live, z] += c.kurtosis_add * sl * sp
            belt_sig["motor_current_a"][live] += c.current_a * sl ** c.current_exp * sp
            # the drive cannot see idler faults: motor and gearbox temperature are deliberately untouched

        elif ev.type == DRIVE_OVERHEAT:
            c = fc.drive_overheat
            rise = sl ** c.exp * sp
            belt_sig["motor_temp_c"][live] += lag(c.motor_temp_c * rise, sc.motor_temp_tau_min)
            belt_sig["gearbox_temp_c"][live] += lag(c.gearbox_temp_c * rise, sc.gearbox_temp_tau_min)
            belt_sig["motor_current_a"][live] += c.current_a * sl ** c.current_exp * sp


def faults_frame(events, timestamps) -> pd.DataFrame:
    ts = pd.DatetimeIndex(timestamps)
    return pd.DataFrame({
        "fault_id": [e.fault_id for e in events],
        "type": [e.type for e in events],
        "asset": [e.asset for e in events],
        "onset_time": [ts[e.onset] for e in events],
        "failure_time": [ts[e.failure] for e in events],
        "repair_end": [ts[e.repair_end] for e in events],
        "duration_hours": [(e.failure - e.onset) / 60.0 for e in events],
        "repair_hours": [(e.repair_end - e.failure) / 60.0 for e in events],
    }).astype({"onset_time": "datetime64[us]", "failure_time": "datetime64[us]", "repair_end": "datetime64[us]"}
              if events else {})
