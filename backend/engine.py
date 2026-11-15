"""The single writer: validates ingested batches, stores them, tracks sensor health and runs live scoring.

Everything that changes state goes through `Engine.submit`, one item at a time under one lock (SQLite has one
writer), in this order:
  1. validate rows (reject, de-duplicate, range-check) and store them, counting what was wrong;
  2. advance the watermark, so the steps below see minutes in time order even if batches arrive out of order;
  3. per minute: update sensor health (stuck, no data);
  4. per 10-minute window: aggregate, extend the 24 h window table, compute features, score, update alerts.
Steps 3 and 4 use model/features.py and model/alerting.py, the same code the offline training and scoring use.
"""
import json
import logging
import math
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common.ids import ASSETS, BELT_SIGNALS, N_ZONES, RUN_STATES, RUNNING, ZONE_SIGNALS, zone_asset
from model.alerting import ACTIVE_RUN_SHARE, Debouncer, severity, severity_label
from model.features import HISTORY_WINDOWS, WINDOW_MIN, append_windows, rolling_features, window_aggregates

log = logging.getLogger("engine")
SCHEMA = Path(__file__).with_name("schema.sql")
STEP = 60
WINDOW_S = WINDOW_MIN * STEP
MODEL_ALERT = {True: "bearing_degradation", False: "drive_overheating"}   # keyed by "is a zone"
SIGNAL_LABEL = {"vibration_rms_mm_s": "vibration", "vibration_kurtosis": "vibration kurtosis",
                "bearing_temp_c": "bearing temperature", "acoustic_db": "acoustic level",
                "belt_offset_mm": "belt offset"}


def connect(path, read_only: bool = False) -> sqlite3.Connection:
    if read_only:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False, timeout=5)
    else:
        con = sqlite3.connect(str(path), check_same_thread=False, timeout=5, isolation_level=None)
        # synchronous=OFF: this database is deleted and rebuilt on every start, so surviving a power cut is
        # worth nothing, while waiting for the disk on every checkpoint cut ingest from about 150 to 24
        # batches per second on a busy disk (measured). A system of record would use NORMAL.
        for pragma in ("journal_mode=WAL", "synchronous=OFF", "wal_autocheckpoint=4000", "temp_store=MEMORY",
                       "cache_size=-65536"):
            con.execute(f"PRAGMA {pragma}")
    con.row_factory = sqlite3.Row
    return con


def parse_ts(value):
    """Epoch seconds from an integer or an ISO-8601 string; None if it is neither or not on a whole minute."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value) and float(value).is_integer():
        ts = int(value)
    elif isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        ts = int((dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp())
    else:
        return None
    return ts if ts >= 0 and ts % STEP == 0 else None


def parse_run_state(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 0 <= value < len(RUN_STATES):
        return value
    if isinstance(value, str) and value in RUN_STATES:
        return RUN_STATES.index(value)
    return None


class Engine:
    def __init__(self, db_path, layout_rows: list, settings: dict, models=None, scenario: str = ""):
        """layout_rows: one dict per zone (the public layout). models: model.predict.Models or None."""
        self.settings, self.models, self.scenario = settings, models, scenario
        self.db_path = Path(db_path)
        self.ranges = {k: tuple(v) for k, v in settings["ingest"]["ranges"].items()}
        self.zone_ranges = [(sig, *self.ranges[sig]) for sig in ZONE_SIGNALS]
        self.grace_s = int(settings["ingest"]["grace_minutes"] * STEP)
        self.commit_every = int(settings["ingest"]["commit_every_batches"])
        self.stuck_after = int(settings["data_quality"]["stuck_after_identical"])
        self.no_data_after = int(settings["data_quality"]["no_data_after_minutes"])
        self.dq_severity = float(settings["data_quality"]["alert_severity"])
        self.layout_rows = sorted(layout_rows, key=lambda r: r["zone_id"])
        assert [r["zone_id"] for r in self.layout_rows] == list(range(1, N_ZONES + 1))
        self.layout = {"is_carry": np.array([r["run"] == "carry" for r in self.layout_rows]),
                       "is_loading": np.array([bool(r["is_loading"]) for r in self.layout_rows]),
                       "is_unloading": np.array([bool(r["is_unloading"]) for r in self.layout_rows])}

        self.lock = threading.RLock()            # guards the in-memory state read by the API
        # ingest state
        self.seen = {}                           # ts -> bool (Z,) zones already stored for that minute
        self.belt_seen = set()
        self.buffers = {}                        # ts -> minute record not yet processed
        self.max_ts = None
        self.next_ts = None                      # next minute to process, in order
        self.watermark = None                    # last processed minute
        self.start_ts = None                     # first minute of the run
        self.totals = dict.fromkeys(("batches", "rows", "null_values", "rejected_rows", "out_of_range",
                                     "duplicates", "late_rows"), 0)
        self.pending_commits = 0
        self.timing = {"busy_s": 0.0}             # seconds spent processing
        # data quality state
        S = len(ZONE_SIGNALS)
        self.last_val = np.full((N_ZONES, S), np.nan)
        self.same_run = np.zeros((N_ZONES, S), dtype=np.int32)
        self.stuck = np.zeros((N_ZONES, S), dtype=bool)
        self.stuck_since = np.zeros((N_ZONES, S), dtype=np.int64)
        self.miss_run = np.zeros(N_ZONES, dtype=np.int32)
        self.no_data = np.zeros(N_ZONES, dtype=bool)
        self.no_data_since = np.zeros(N_ZONES, dtype=np.int64)
        self.open_events = {}                    # (zone, signal or None) -> quality_events.event_id
        # live scoring state
        self.window_minutes = []                 # minute records of the window being filled
        self.table = {}                          # window table, last 24 h
        self.n_windows = 0
        A = len(ASSETS)
        self.debouncer = Debouncer(A)
        self.p_abnormal = np.zeros(A)
        self.p_fail = np.zeros(A)
        self.est_hours = np.full(A, np.nan)
        self.scored_ts = None
        if models is not None:
            z, d = models.thresholds("zone"), models.thresholds("drive")
            self.thr_abnormal = np.array([z["abnormal"]] * N_ZONES + [d["abnormal"]])
            self.thr_fail = np.array([z["predicted_failure"]] * N_ZONES + [d["predicted_failure"]])
        self.replay = {"state": "idle", "speed": float(settings["replay"]["default_speed"])}

        self._init_db()

    # ------------------------------------------------------------------ database
    def _init_db(self) -> None:
        for suffix in ("", "-wal", "-shm"):      # every start is a fresh run
            Path(str(self.db_path) + suffix).unlink(missing_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = connect(self.db_path)
        self.con.executescript(SCHEMA.read_text())
        self.con.execute("BEGIN")
        self.con.executemany(
            "INSERT INTO zones VALUES (?,?,?,?,?,?,?,?,?)",
            [(r["zone_id"], r["section"], r["zone_in_section"], r["position_m"], r["run"], int(r["is_loading"]),
              int(r["is_unloading"]), r.get("incline_deg"), r.get("curve_radius_m")) for r in self.layout_rows])
        self.con.executemany("INSERT INTO sensor_health VALUES (?,?,'ok',0,NULL,NULL)",
                             [(z, s) for z in range(1, N_ZONES + 1) for s in ZONE_SIGNALS])
        meta = {"scenario": self.scenario, "replay_state": self.replay["state"], "replay_position_ts": "",
                "replay_speed": str(self.replay["speed"]), "schema_version": "1",
                "model_version": self.models.version if self.models else "",
                "thresholds": json.dumps(self.models.meta["thresholds"]) if self.models else "{}"}
        self.con.executemany("INSERT INTO meta VALUES (?,?)", list(meta.items()))
        self.con.execute("COMMIT")
        self.con.execute("BEGIN")

    def _commit(self) -> None:
        self.con.execute("COMMIT")
        self.con.execute("BEGIN")
        self.pending_commits = 0

    def _set_meta(self, **values) -> None:
        self.con.executemany("INSERT INTO meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                             [(k, str(v)) for k, v in values.items()])

    # ------------------------------------------------------------------ the single writer
    def start(self) -> None:
        pass

    def stop(self) -> None:
        with self.lock:
            self._commit()
            self.con.close()

    def submit(self, kind: str, payload) -> None:
        """Process one item now, holding the writer lock. Every change to the database goes through here.

        Why not a queue and a worker thread: measured on the 2-core development machine, a worker thread
        competing with the web server for Python's interpreter lock ran about 7 times slower than the same
        work done directly in the request (24 against 180 batches per second). The caller waiting for the
        work is also the back-pressure: the replayer cannot send faster than the engine can take.
        """
        t0 = time.perf_counter()
        with self.lock:
            try:
                self.handle(kind, payload)
                if self.pending_commits >= self.commit_every or kind != "ingest":
                    self._commit()
            except Exception:  # one bad batch must not stop the system
                log.exception("failed to process a %s item", kind)
            finally:
                self.timing["busy_s"] += time.perf_counter() - t0

    def drain(self) -> None:
        """Commit what has been processed (tests, shutdown)."""
        with self.lock:
            self._commit()

    def handle(self, kind: str, payload) -> None:
        """Process one item now (the engine thread calls this; tests may call it directly)."""
        if kind == "ingest":
            self.ingest(payload)
        elif kind == "replay":
            self.set_replay(**payload)
        elif kind == "flush":
            self.advance(flush=True)

    # ------------------------------------------------------------------ replay state
    def set_replay(self, state=None, speed=None) -> None:
        if state is not None:
            self.replay["state"] = state
            if state == "finished":
                self.advance(flush=True)
        if speed is not None:
            self.replay["speed"] = float(speed)
        self._set_meta(replay_state=self.replay["state"], replay_speed=self.replay["speed"])

    # ------------------------------------------------------------------ step 1: validate and store
    def _clean(self, signal: str, value, counts: dict):
        if value is None:
            counts["null_values"] += 1
            return None
        lo, hi = self.ranges[signal]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not lo <= value <= hi:
            counts["out_of_range"] += 1      # also catches NaN and infinity: comparisons with them are false
            return None
        return float(value)

    def _minute(self, ts: int) -> dict:
        rec = self.buffers.get(ts)
        if rec is None:
            rec = self.buffers[ts] = {"z": np.full((len(ZONE_SIGNALS), N_ZONES), np.nan),
                                      "belt": np.full(len(BELT_SIGNALS), np.nan), "run_state": None}
        return rec

    def ingest(self, body: dict) -> dict:
        counts = dict.fromkeys(("rows", "null_values", "rejected_rows", "out_of_range", "duplicates", "late_rows"), 0)
        rows = body.get("batch") or []
        counts["rows"] = len(rows)
        store, sim_ts = [], None
        parsed = {}    # a batch is one minute, so nearly every row carries the same timestamp
        live = {}      # ts -> (zone indexes, value rows, run state) for minutes not yet processed
        ranges, seen_by_ts, watermark = self.zone_ranges, self.seen, self.watermark
        nulls = bad = 0
        for r in rows:
            if r.__class__ is not dict:
                counts["rejected_rows"] += 1
                continue
            raw = r.get("ts")
            ts = parsed.get(raw, 0) if raw.__class__ in (int, str) else None
            if ts == 0:
                ts = parsed[raw] = parse_ts(raw)
            rs = r.get("run_state")
            if rs.__class__ is not int or not 0 <= rs <= 2:
                rs = parse_run_state(rs)
            zone = r.get("zone_id")
            if ts is None or rs is None or zone.__class__ is not int or not 1 <= zone <= N_ZONES:
                counts["rejected_rows"] += 1
                continue
            seen = seen_by_ts.get(ts)
            if seen is None:
                seen = seen_by_ts[ts] = np.zeros(N_ZONES, dtype=bool)
            if seen[zone - 1]:
                counts["duplicates"] += 1    # the first row for (ts, zone) wins
                continue
            seen[zone - 1] = True
            vals = []
            for sig, lo, hi in ranges:
                v = r.get(sig)
                if v is None:
                    nulls += 1
                elif (v.__class__ is float or v.__class__ is int) and lo <= v <= hi:
                    pass
                else:                        # not a number, NaN, infinity or outside the physical range
                    bad += 1
                    v = None
                vals.append(v)
            store.append((ts, zone, rs, *vals))
            if sim_ts is None or ts > sim_ts:
                sim_ts = ts
            if watermark is not None and ts <= watermark:
                counts["late_rows"] += 1     # stored, but too late for health tracking and scoring
            else:
                group = live.get(ts)
                if group is None:
                    group = live[ts] = ([], [], rs)
                group[0].append(zone - 1)
                group[1].append(vals)
        counts["null_values"] += nulls
        counts["out_of_range"] += bad
        for ts, (idx, values, rs) in live.items():
            rec = self._minute(ts)
            rec["z"][:, idx] = np.array(values, dtype=np.float64).T      # None becomes NaN
            rec["run_state"] = rs
        if store:
            self.con.executemany("INSERT INTO readings VALUES (?,?,?,?,?,?,?,?)", store)

        belt = body.get("belt")
        if isinstance(belt, dict):
            ts, rs = parse_ts(belt.get("ts")), parse_run_state(belt.get("run_state"))
            if ts is None or rs is None:
                counts["rejected_rows"] += 1
            elif ts in self.belt_seen:
                counts["duplicates"] += 1
            else:
                self.belt_seen.add(ts)
                vals = [self._clean(s, belt.get(s), counts) for s in BELT_SIGNALS]
                self.con.execute("INSERT INTO belt_readings VALUES (?,?,?,?,?,?,?,?,?,?)", (ts, rs, *vals))
                sim_ts = ts if sim_ts is None else max(sim_ts, ts)
                if self.watermark is None or ts > self.watermark:
                    rec = self._minute(ts)
                    rec["belt"][:] = [np.nan if v is None else v for v in vals]
                    if rec["run_state"] is None:
                        rec["run_state"] = rs
            counts["rows"] += 1

        self.con.execute("INSERT INTO ingest_log (received_wall_ts, sim_ts, n_rows, n_null_values, n_rejected_rows,"
                         " n_out_of_range, n_duplicates) VALUES (?,?,?,?,?,?,?)",
                         (int(time.time()), sim_ts or 0, counts["rows"], counts["null_values"],
                          counts["rejected_rows"], counts["out_of_range"], counts["duplicates"]))
        self.totals["batches"] += 1
        for k, v in counts.items():
            self.totals[k] += v
        self.pending_commits += 1
        if sim_ts is not None:
            self.max_ts = sim_ts if self.max_ts is None else max(self.max_ts, sim_ts)
            self.advance()
        return counts

    # ------------------------------------------------------------------ step 2: watermark
    def advance(self, flush: bool = False) -> None:
        """Process minutes in order. A minute is ready when all zones reported or the grace period has passed."""
        if self.max_ts is None:
            return
        if self.next_ts is None:
            self.next_ts = self.start_ts = min(self.buffers)
        edge = self.max_ts if flush else self.max_ts - self.grace_s
        while self.next_ts <= self.max_ts:
            seen = self.seen.get(self.next_ts)
            if not (self.next_ts <= edge or (seen is not None and seen.all())):
                break
            self.process_minute(self.next_ts)
            self.watermark = self.next_ts
            self.next_ts += STEP
        if self.watermark is not None:
            self._set_meta(replay_position_ts=self.watermark)

    def process_minute(self, ts: int) -> None:
        rec = self.buffers.pop(ts, None)
        if rec is None:   # nothing arrived for this minute: every suite was silent
            rec = {"z": np.full((len(ZONE_SIGNALS), N_ZONES), np.nan), "belt": np.full(len(BELT_SIGNALS), np.nan),
                   "run_state": None}
        rec["ts"] = ts
        if rec["run_state"] is None:
            rec["run_state"] = 0
        self.update_quality(ts, rec["z"].T, rec["run_state"] == RUNNING)
        if ts % WINDOW_S == 0:
            self.window_minutes = []
        self.window_minutes.append(rec)
        if ts % WINDOW_S == WINDOW_S - STEP:
            if len(self.window_minutes) == WINDOW_MIN:
                self.close_window(ts)
            self.window_minutes = []

    # ------------------------------------------------------------------ step 3: sensor health
    def update_quality(self, ts: int, v, running: bool) -> None:
        """v: (Z, S) readings for one minute, NaN where missing."""
        valid = ~np.isnan(v)
        all_missing = ~valid.any(axis=1)
        self.miss_run = np.where(all_missing, self.miss_run + 1, 0)
        new_silent = (self.miss_run >= self.no_data_after) & ~self.no_data
        back = self.no_data & ~all_missing

        same = valid & (v == self.last_val)
        changed = valid & ~same
        if running:   # repeats only count while the belt runs; a different value always resets
            self.same_run = np.where(same, self.same_run + 1, self.same_run)
        self.same_run = np.where(changed, 1, self.same_run)
        self.last_val = np.where(valid, v, self.last_val)
        new_stuck = (self.same_run >= self.stuck_after) & ~self.stuck
        freed = self.stuck & changed

        if not (new_silent.any() or back.any() or new_stuck.any() or freed.any()):
            return
        for z in np.flatnonzero(new_silent):
            self.no_data[z], self.no_data_since[z] = True, ts - (self.no_data_after - 1) * STEP
            self._open_event(z, None, "outage", int(self.no_data_since[z]))
        for z in np.flatnonzero(back):
            self.no_data[z] = False
            self._close_event(z, None, ts)
        for z, s in zip(*np.nonzero(new_stuck)):
            self.stuck[z, s] = True
            self.stuck_since[z, s] = ts - (self.stuck_after - 1) * STEP
            self._open_event(z, s, "stuck", int(self.stuck_since[z, s]))
        for z, s in zip(*np.nonzero(freed)):
            self.stuck[z, s] = False
            self._close_event(z, s, ts)
        for z in set(np.flatnonzero(new_silent | back)) | set(np.nonzero(new_stuck | freed)[0]):
            self._sync_zone_health(int(z), ts)

    def _open_event(self, z: int, s, kind: str, start_ts: int) -> None:
        cur = self.con.execute("INSERT INTO quality_events (zone_id, signal, type, start_ts) VALUES (?,?,?,?)",
                               (int(z) + 1, None if s is None else ZONE_SIGNALS[s], kind, start_ts))
        self.open_events[(int(z), None if s is None else int(s))] = cur.lastrowid

    def _close_event(self, z: int, s, ts: int) -> None:
        event_id = self.open_events.pop((int(z), None if s is None else int(s)), None)
        if event_id is not None:
            self.con.execute("UPDATE quality_events SET end_ts=? WHERE event_id=?", (ts, event_id))

    def _sync_zone_health(self, z: int, ts: int) -> None:
        """Write one zone's sensor states and open or close its data-quality alerts to match."""
        for s, sig in enumerate(ZONE_SIGNALS):
            state = "no_data" if self.no_data[z] else "stuck" if self.stuck[z, s] else "ok"
            since = (self.no_data_since[z] if state == "no_data" else self.stuck_since[z, s] if state == "stuck" else ts)
            last = None if np.isnan(self.last_val[z, s]) else float(self.last_val[z, s])
            self.con.execute("UPDATE sensor_health SET state=?, since_ts=?, last_seen_ts=?, last_value=? "
                             "WHERE zone_id=? AND signal=? AND state<>?",
                             (state, int(since), ts, last, z + 1, sig, state))
        row = self.layout_rows[z]
        where = f"zone {z + 1} (S{row['section']}, {row['position_m']:.0f} m)"
        self._data_quality_alert("sensor_outage", z, bool(self.no_data[z]), ts,
                                 f"No data from {where}: all five sensors silent since "
                                 f"{_clock(self.no_data_since[z])}. Its status is unknown, not normal.")
        names = [SIGNAL_LABEL[sig] for s, sig in enumerate(ZONE_SIGNALS) if self.stuck[z, s]]
        self._data_quality_alert("sensor_stuck", z, bool(names), ts,
                                 f"Sensor suspect at {where}: {', '.join(names)} "
                                 f"repeating one value for {self.stuck_after} or more readings.")

    def _data_quality_alert(self, kind: str, z: int, active: bool, ts: int, message: str) -> None:
        asset = zone_asset(z + 1)
        open_id = self.con.execute("SELECT alert_id FROM alerts WHERE type=? AND asset=? AND state='open'",
                                   (kind, asset)).fetchone()
        if active and open_id is None:
            self.con.execute(
                "INSERT INTO alerts (type, source, asset, zone_id, position_m, severity, severity_label, message,"
                " state, created_ts, updated_ts) VALUES (?,?,?,?,?,?,?,?,'open',?,?)",
                (kind, "data_quality", asset, z + 1, self.layout_rows[z]["position_m"], self.dq_severity,
                 "medium", message, ts, ts))   # data-quality alerts are always 'medium' (BUILD_PLAN section 6)
        elif active:
            self.con.execute("UPDATE alerts SET message=?, updated_ts=? WHERE alert_id=?", (message, ts, open_id[0]))
        elif open_id is not None:
            self.con.execute("UPDATE alerts SET state='closed', closed_ts=?, updated_ts=? WHERE alert_id=?",
                             (ts, ts, open_id[0]))

    # ------------------------------------------------------------------ step 4: live scoring
    def close_window(self, ts: int) -> None:
        mins = self.window_minutes
        zone_vals = {s: np.stack([m["z"][i] for m in mins])[None] for i, s in enumerate(ZONE_SIGNALS)}
        belt_vals = {s: np.array([m["belt"][i] for m in mins])[None] for i, s in enumerate(BELT_SIGNALS)}
        run_state = np.array([m["run_state"] for m in mins])[None]
        self.table = append_windows(self.table, window_aggregates(zone_vals, belt_vals, run_state, self.layout))
        self.n_windows += 1
        if self.models is None or len(self.table["run_share"]) < HISTORY_WINDOWS:
            return   # the model needs 24 h of history behind every score
        feats = rolling_features(self.table, self.layout)
        zone_x, drive_x = feats["zone"][-1], feats["drive"][-1][None, :]
        zp, dp = self.models.predict("zone", zone_x), self.models.predict("drive", drive_x)
        active = bool(self.table["run_share"][-1] >= ACTIVE_RUN_SHARE)
        if active:   # while the belt is stopped the last running values stand
            self.p_abnormal = np.concatenate([zp["p_abnormal"], dp["p_abnormal"]])
            self.p_fail = np.concatenate([zp["p_fail_24h"], dp["p_fail_24h"]])
            self.est_hours = np.concatenate([zp["est_hours"], dp["est_hours"]])
        opened, closed = self.debouncer.step(self.p_abnormal >= self.thr_abnormal, active)
        self.scored_ts = ts
        status = self.model_status()
        self.con.executemany(
            "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?)",
            [(ts, a, float(self.p_abnormal[i]), float(self.p_fail[i]),
              float(self.est_hours[i]) if status[i] == "predicted_failure" else None, status[i], self.models.version)
             for i, a in enumerate(ASSETS)])
        if not active:
            return
        for i in np.flatnonzero(self.debouncer.is_open | closed):
            x = zone_x[i] if i < N_ZONES else drive_x[0]
            self._model_alert(int(i), ts, x, bool(opened[i]), bool(closed[i]))

    def model_status(self) -> list:
        out = np.where(self.debouncer.is_open,
                       np.where(self.p_fail >= self.thr_fail, "predicted_failure", "abnormal"), "normal")
        return out.tolist()

    def _model_alert(self, i: int, ts: int, x, opened: bool, closed: bool) -> None:
        is_zone = i < N_ZONES
        kind, asset = MODEL_ALERT[is_zone], ASSETS[i]
        if closed:
            self.con.execute("UPDATE alerts SET state='closed', closed_ts=?, updated_ts=? "
                             "WHERE type=? AND asset=? AND state='open'", (ts, ts, kind, asset))
            return
        p_fail = float(self.p_fail[i])
        eta = float(self.est_hours[i]) if p_fail >= self.thr_fail[i] else None
        sev = severity(p_fail, eta)
        if is_zone:
            row = self.layout_rows[i]
            where, pos, zone_id = (f"Zone {i + 1} (S{row['section']}, {row['position_m']:.0f} m, {row['run']} run)",
                                   row["position_m"], i + 1)
            what = "idler bearing degrading"
        else:
            where, pos, zone_id, what = "Drive", None, None, "motor and gearbox overheating"
        when = (f"Failure expected in about {eta:.0f} h." if eta is not None
                else f"Chance of failing within a day: {p_fail:.0%}.")
        message = f"{where}: {what}. {when}"
        evidence = json.dumps({"top_features": self.models.top_features("zone" if is_zone else "drive", x),
                               "p_abnormal": round(float(self.p_abnormal[i]), 4), "p_fail_24h": round(p_fail, 4)})
        fail_ts = int(ts + eta * 3600) if eta is not None else None
        if opened:
            self.con.execute(
                "INSERT INTO alerts (type, source, asset, zone_id, position_m, severity, severity_label,"
                " predicted_failure_ts, eta_hours, message, evidence, state, created_ts, updated_ts)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,'open',?,?)",
                (kind, "model", asset, zone_id, pos, sev, severity_label(sev), fail_ts, eta, message, evidence, ts, ts))
        else:
            self.con.execute(
                "UPDATE alerts SET severity=?, severity_label=?, predicted_failure_ts=?, eta_hours=?, message=?,"
                " evidence=?, updated_ts=? WHERE type=? AND asset=? AND state='open'",
                (sev, severity_label(sev), fail_ts, eta, message, evidence, ts, kind, asset))

    # ------------------------------------------------------------------ read side (called by the API)
    def status(self) -> dict:
        """Current status of every asset. Predicted failure beats sensor suspect beats the model's verdict."""
        with self.lock:
            model = self.model_status() if self.models is not None else ["normal"] * len(ASSETS)
            scoring = self.scored_ts is not None
            assets = []
            for i, asset in enumerate(ASSETS):
                is_zone = i < N_ZONES
                sensors = ({sig: ("no_data" if self.no_data[i] else "stuck" if self.stuck[i, s] else "ok")
                            for s, sig in enumerate(ZONE_SIGNALS)} if is_zone else {})
                suspect = any(v != "ok" for v in sensors.values())
                quality = ("no_data" if is_zone and self.no_data[i] else "sensor_suspect" if suspect else "ok")
                status = model[i] if model[i] == "predicted_failure" else "no_data" if suspect else model[i]
                shown = scoring and model[i] == "predicted_failure"   # a time is only given with a failure call
                assets.append({
                    "asset": asset, "zone_id": i + 1 if is_zone else None, "status": status,
                    "model_status": model[i], "data_quality": quality, "sensors": sensors,
                    "p_abnormal": round(float(self.p_abnormal[i]), 4) if scoring else None,
                    "p_fail_24h": round(float(self.p_fail[i]), 4) if scoring else None,
                    "est_hours_to_failure": round(float(self.est_hours[i]), 1)
                    if shown and np.isfinite(self.est_hours[i]) else None})
            return {"sim_ts": self.watermark, "start_ts": self.start_ts, "scored_ts": self.scored_ts,
                    "scoring": scoring,
                    "model_version": self.models.version if self.models else None,
                    "windows_of_history": min(self.n_windows, HISTORY_WINDOWS),
                    "windows_needed": HISTORY_WINDOWS, "replay": dict(self.replay), "assets": assets}

    def ingest_totals(self) -> dict:
        with self.lock:
            return {**self.totals, **{k: round(v, 1) for k, v in self.timing.items()}}


def _clock(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%d %b %H:%M")
