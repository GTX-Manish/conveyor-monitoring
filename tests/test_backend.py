"""Backend: ingest validation, watermark, sensor health, alerts, the status rule, the API and the replayer."""
import json

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from backend.app import REPO, create_app, load_layout
from backend.engine import Engine, connect, parse_ts
from common.ids import ASSETS, BELT_SIGNALS, DRIVE, N_ZONES, ZONE_SIGNALS, asset_zone, simulator_asset, zone_asset
from model.alerting import Debouncer, alert_spans, severity, severity_label
from model.features import HISTORY_WINDOWS, drive_feature_names, zone_feature_names
from simulator.config import Config
from simulator.generate import generate
from simulator.io import write_scenario

T0 = 1767225600   # 2026-01-01 00:00:00 UTC
SETTINGS = yaml.safe_load((REPO / "configs" / "app.yaml").read_text())
LAYOUT = [{"zone_id": z, "section": (z - 1) // 10 + 1, "zone_in_section": (z - 1) % 10 + 1,
           "position_m": 50.0 * z - 25, "run": "carry" if z <= 60 else "return", "is_loading": z <= 2,
           "is_unloading": z in (59, 60), "incline_deg": 0.0, "curve_radius_m": None} for z in range(1, N_ZONES + 1)]


class StubModels:
    """Stands in for the trained models: fixed thresholds, scores set by the test."""
    version = "stub"
    meta = {"thresholds": {}}

    def __init__(self):
        self.p_abnormal = np.zeros(len(ASSETS))
        self.p_fail = np.zeros(len(ASSETS))
        self.seen = []

    def thresholds(self, group):
        return {"abnormal": 0.5, "predicted_failure": 0.8}

    def predict(self, group, x):
        self.seen.append((group, np.array(x)))
        part = slice(0, N_ZONES) if group == "zone" else slice(N_ZONES, None)
        return {"p_abnormal": self.p_abnormal[part], "p_fail_24h": self.p_fail[part],
                "est_hours": np.full(len(self.p_fail[part]), 6.0)}

    def top_features(self, group, row, k=3):
        return [{"name": "bearing_temp_c__dev6h", "contribution": 1.0, "value": 9.0}]


@pytest.fixture
def engine(tmp_path):
    e = Engine(tmp_path / "app.sqlite", LAYOUT, SETTINGS, models=None, scenario="unit")
    yield e
    e.con.close()


def minute(i, value=None, run_state=2, zones=range(1, N_ZONES + 1), **override):
    """One batch for minute i. Values differ per minute and zone unless given."""
    ts = T0 + 60 * i
    rows = []
    for z in zones:
        base = 1.0 + 0.001 * i + 0.01 * z if value is None else value
        row = {"ts": ts, "zone_id": z, "run_state": run_state,
               **{s: base + k for k, s in enumerate(ZONE_SIGNALS)}}
        row.update(override.get(f"z{z}", {}))
        rows.append(row)
    return {"batch": rows, "belt": {"ts": ts, "run_state": run_state, **{s: 1.0 for s in BELT_SIGNALS}}}


def rows(engine, sql, *params):
    engine._commit()
    return [dict(r) for r in engine.con.execute(sql, params).fetchall()]


# ------------------------------------------------------------------ ids
def test_asset_ids_round_trip():
    assert len(ASSETS) == 121 and ASSETS[0] == "Z001" and ASSETS[119] == "Z120" and ASSETS[-1] == DRIVE
    assert [asset_zone(a) for a in ASSETS] == list(range(1, 121)) + [None]
    assert [simulator_asset(a) for a in [str(z) for z in range(1, 121)] + ["DRIVE"]] == ASSETS   # no gaps
    for bad in ("Z000", "Z121", "z001", "47", ""):
        with pytest.raises(ValueError):
            asset_zone(bad)
    with pytest.raises(ValueError):
        zone_asset(0)


def test_signal_names_match_the_simulator():
    from common import ids
    from simulator import healthy
    from simulator.generate import RUN_STATES
    assert ids.ZONE_SIGNALS == healthy.ZONE_SIGNALS and ids.BELT_SIGNALS == healthy.BELT_SIGNALS
    assert ids.RUN_STATES == RUN_STATES


# ------------------------------------------------------------------ ingest validation
def test_bad_rows_are_rejected_and_counted(engine):
    body = minute(0)
    body["batch"] += [{**body["batch"][0], "zone_id": 999}, {**body["batch"][0], "zone_id": "7"},
                      {**body["batch"][1], "ts": "not a time"}, {**body["batch"][2], "ts": T0 + 30},
                      {**body["batch"][3], "run_state": "sideways"}, "not a row"]
    counts = engine.ingest(body)
    assert counts["rejected_rows"] == 6 and counts["duplicates"] == 0
    assert rows(engine, "SELECT COUNT(*) n FROM readings")[0]["n"] == N_ZONES
    log = rows(engine, "SELECT * FROM ingest_log")[0]
    assert log["n_rejected_rows"] == 6 and log["n_rows"] == N_ZONES + 6 + 1 and log["sim_ts"] == T0


def test_duplicates_first_row_wins(engine):
    body = minute(0)
    body["batch"].append({**body["batch"][4], "bearing_temp_c": 77.0})
    assert engine.ingest(body)["duplicates"] == 1
    assert engine.ingest(minute(0))["duplicates"] == N_ZONES + 1      # the whole batch again, belt row included
    stored = rows(engine, "SELECT bearing_temp_c FROM readings WHERE zone_id = 5")
    assert len(stored) == 1 and stored[0]["bearing_temp_c"] != 77.0


def test_out_of_range_and_null_become_null_never_filled(engine):
    body = minute(0, z1={"bearing_temp_c": 1500.0, "acoustic_db": None}, z2={"vibration_rms_mm_s": -1.0},
                  z3={"belt_offset_mm": "wide"}, z4={"vibration_kurtosis": float("nan")})
    body["belt"]["motor_temp_c"] = 9999.0
    counts = engine.ingest(body)
    assert counts["out_of_range"] == 5 and counts["null_values"] == 1
    r = rows(engine, "SELECT * FROM readings WHERE zone_id <= 4 ORDER BY zone_id")
    assert r[0]["bearing_temp_c"] is None and r[0]["acoustic_db"] is None and r[0]["vibration_rms_mm_s"] is not None
    assert r[1]["vibration_rms_mm_s"] is None and r[2]["belt_offset_mm"] is None and r[3]["vibration_kurtosis"] is None
    assert rows(engine, "SELECT motor_temp_c, tension_kn FROM belt_readings")[0] == {"motor_temp_c": None,
                                                                                    "tension_kn": 1.0}


def test_timestamps_accept_epoch_and_iso():
    assert parse_ts(T0) == T0 == parse_ts("2026-01-01T00:00:00Z") == parse_ts("2026-01-01 00:00:00")
    assert parse_ts(T0 + 1) is None and parse_ts(True) is None and parse_ts(None) is None and parse_ts(-60) is None


def test_out_of_order_batches_are_processed_in_time_order(engine):
    seen = []
    original = engine.process_minute
    engine.process_minute = lambda ts: (seen.append((ts - T0) // 60), original(ts))
    for i in (0, 1, 3, 2, 4, 5):          # minute 3 arrives before minute 2
        engine.ingest(minute(i))
    assert seen == [0, 1, 2, 3, 4, 5] and engine.watermark == T0 + 300
    assert engine.ingest(minute(1))["late_rows"] == 0   # a duplicate, not a late row
    assert engine.totals["late_rows"] == 0


def test_a_missing_batch_is_given_up_after_the_grace_period(engine):
    for i in (0, 1, 3):                   # minute 2 never arrives
        engine.ingest(minute(i))
    assert engine.watermark == T0 + 60    # still waiting for minute 2
    engine.ingest(minute(4))
    assert engine.watermark == T0 + 240   # 2 minutes of grace passed: minute 2 was processed as silent
    assert engine.ingest(minute(2))["late_rows"] == N_ZONES
    assert rows(engine, "SELECT COUNT(*) n FROM readings WHERE ts = ?", T0 + 120)[0]["n"] == N_ZONES   # still stored


# ------------------------------------------------------------------ sensor health
def test_stuck_after_15_identical_running_values_and_clears(engine):
    frozen = {"z7": {"acoustic_db": 55.5}}
    for i in range(14):
        engine.ingest(minute(i, **frozen))
    assert not engine.stuck.any()
    engine.ingest(minute(14, **frozen))
    assert engine.stuck[6, ZONE_SIGNALS.index("acoustic_db")] and engine.stuck.sum() == 1
    health = rows(engine, "SELECT * FROM sensor_health WHERE state <> 'ok'")
    assert [(h["zone_id"], h["signal"], h["state"], h["since_ts"]) for h in health] == [(7, "acoustic_db", "stuck", T0)]
    alert = rows(engine, "SELECT * FROM alerts")
    assert len(alert) == 1 and alert[0]["type"] == "sensor_stuck" and alert[0]["asset"] == "Z007"
    assert alert[0]["severity"] == 0.5 and alert[0]["severity_label"] == "medium" and alert[0]["source"] == "data_quality"
    assert engine.status()["assets"][6]["status"] == "no_data" and engine.status()["assets"][5]["status"] == "normal"

    engine.ingest(minute(15))             # the value moves again
    assert not engine.stuck.any()
    assert rows(engine, "SELECT state, closed_ts FROM alerts") == [{"state": "closed", "closed_ts": T0 + 900}]
    assert rows(engine, "SELECT type, end_ts FROM quality_events") == [{"type": "stuck", "end_ts": T0 + 900}]


def test_repeats_while_stopped_do_not_count_as_stuck(engine):
    for i in range(30):
        engine.ingest(minute(i, run_state=0, z7={"acoustic_db": 55.5}))
    assert not engine.stuck.any()


def test_missing_readings_do_not_break_or_fake_a_stuck_run(engine):
    for i in range(25):                   # every third reading is missing; the other 16 repeat
        engine.ingest(minute(i, z7={"acoustic_db": None if i % 3 == 0 else 55.5}))
        assert engine.stuck.any() == (i >= 22)                      # the 15th repeat arrives at minute 22
    assert rows(engine, "SELECT COUNT(*) n FROM alerts WHERE type='sensor_outage'")[0]["n"] == 0


def test_no_data_after_5_silent_minutes_and_clears(engine):
    silent = {"z9": dict.fromkeys(ZONE_SIGNALS)}
    engine.ingest(minute(0))
    for i in range(1, 5):
        engine.ingest(minute(i, **silent))
    assert not engine.no_data.any()       # 4 silent minutes: nothing yet
    engine.ingest(minute(5, **silent))
    assert engine.no_data[8] and engine.no_data.sum() == 1
    a = engine.status()["assets"][8]
    assert a["status"] == "no_data" and a["data_quality"] == "no_data" and set(a["sensors"].values()) == {"no_data"}
    alert = rows(engine, "SELECT type, asset, state FROM alerts")
    assert alert == [{"type": "sensor_outage", "asset": "Z009", "state": "open"}]
    assert rows(engine, "SELECT start_ts FROM quality_events")[0]["start_ts"] == T0 + 60

    engine.ingest(minute(6))
    assert not engine.no_data.any() and engine.status()["assets"][8]["status"] == "normal"
    assert rows(engine, "SELECT state FROM alerts")[0]["state"] == "closed"


def test_one_missing_reading_raises_nothing(engine):
    for i in range(10):
        engine.ingest(minute(i, z9={"bearing_temp_c": None} if i == 4 else {}))
    assert rows(engine, "SELECT COUNT(*) n FROM alerts")[0]["n"] == 0
    assert rows(engine, "SELECT COUNT(*) n FROM quality_events")[0]["n"] == 0


def test_rows_that_never_arrive_count_as_silence(engine):
    engine.ingest(minute(0))
    for i in range(1, 9):
        engine.ingest(minute(i, zones=[z for z in range(1, N_ZONES + 1) if z != 30]))
    assert engine.no_data[29] and engine.no_data.sum() == 1


# ------------------------------------------------------------------ alerts and the status rule
def test_debounce_opens_after_2_and_closes_after_6():
    above = np.array([0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1], dtype=bool)[:, None]
    assert alert_spans(above, np.ones(13, bool)) == [(0, 4, 11)]
    active = np.ones(13, bool)
    active[6:9] = False                   # belt stopped: the counters hold
    assert alert_spans(above, active) == [(0, 4, None)]
    d = Debouncer(1)
    assert [bool(d.step([True])[0][0]) for _ in range(3)] == [False, True, False]   # opens once


def test_severity_bands():
    assert [severity_label(v) for v in (0.0, 0.24, 0.25, 0.49, 0.5, 0.74, 0.75, 1.0)] == [
        "low", "low", "medium", "medium", "high", "high", "critical", "critical"]
    assert severity(0.3) == 0.3 and severity(1.0, 0.0) == 1.0
    assert severity(0.9, 2.0) > severity(0.9, 20.0)     # hours from failure looks worse than 20 h away
    assert severity(0.9, 48.0) == pytest.approx(0.54)


def test_failure_calls_are_scored_on_what_happened_next():
    from model.evaluate import row_metrics, score_calls
    W = 700                                             # three assets, 10-minute windows
    stage = np.zeros((W, 3), dtype=np.int8)
    htf = np.full((W, 3), np.nan)
    fault = np.full((W, 3), "", dtype=object)
    for a, (onset, failure, fid) in enumerate([(50, 300, "F1"), (50, 650, "F2")]):
        stage[onset:failure, a], fault[onset:failure, a] = 2, fid
        htf[onset:failure, a] = (failure - np.arange(onset, failure)) / 6.0
        stage[failure:failure + 12, a], htf[failure:failure + 12, a], fault[failure:failure + 12, a] = 3, 0.0, fid
    p_abn, p_fail = np.zeros((W, 3)), np.zeros((W, 3))
    p_abn[60:300, 0], p_fail[240:300, 0] = 1.0, 0.9     # call 10 h before failure: came true
    p_abn[60:650, 1], p_fail[100:650, 1] = 1.0, 0.9     # call 92 h before failure: early (the claim is 72 h)
    p_abn[60:120, 2], p_fail[70:120, 2] = 1.0, 0.9      # healthy part: false
    est = np.full((W, 3), 12.0)
    r = score_calls(p_abn, 0.5, p_fail, 0.8, np.ones(W), stage, fault, htf, est)
    assert (r["calls"], r["came_true"], r["early"], r["false"]) == (3, 1, 1, 1)
    assert r["events"] == 2 and r["events_called"] == 2 and r["events_called_in_time"] == 1
    assert r["share_came_true"] == pytest.approx(1 / 3)
    assert sorted(r["hours_list"]) == pytest.approx([10.0, 550 / 6])
    assert r["median_eta_error_hours"] == pytest.approx(np.median([2.0, 550 / 6 - 12]))
    assert r["precision"] == pytest.approx(1 / 3) and r["recall"] == 0.5
    assert score_calls(p_abn, 0.5, p_fail, 0.95, np.ones(W), stage, fault, htf)["calls"] == 0
    m = row_metrics(p_fail, 0.8, stage, htf)
    assert m["positive_windows"] == 2 * 144 and m["tp"] == 60 + 144 and m["fp"] == 406 + 50
    assert m["recall"] == pytest.approx(204 / 288) and 0 < m["average_precision"] <= 1


def test_reasons_are_plain_sentences():
    from model.explain import explain, reasons
    assert explain("bearing_temp_c__dev6h", 9.83) == (
        "Bearing temperature is 9.8 °C higher than the other zones on its run (6 h average)")
    assert explain("vibration_rms_mm_s__dev24h", 0.1823) == (
        "Vibration is 20% higher than the other zones on its run (24 h average)")
    assert explain("acoustic_db__rise1h", 1.24).startswith("Acoustic level has risen 1.2 dB in the last hour")
    assert explain("motor_temp_c__dev1h", 38.4) == (
        "Motor temperature is 38.4 °C above the idler bearings' temperature (1 h average)")
    assert explain("motor_minus_gearbox_1h", 10.7) == "The motor is 10.7 °C hotter than the gearbox (1 h average)"
    assert explain("bearing_temp_c__nb6h", -2.0).startswith("Bearing temperature is 2.0 °C lower than the two neighbouring")
    for context in ("run_share", "is_carry", "temp_reference", "throughput", "motor_current_a__mean1h"):
        assert explain(context, 1.0) is None            # context, not a symptom
    assert explain("bearing_temp_c__dev6h", None) is None and explain("bearing_temp_c__dev6h", float("nan")) is None
    feats = [{"name": "run_share", "value": 1.0}, {"name": "acoustic_db__dev24h", "value": 1.2},
             {"name": "vibration_kurtosis__dev24h", "value": 0.37}, {"name": "acoustic_db__dev24h", "value": 1.2}]
    assert len(reasons(feats)) == 2 and reasons(feats)[0].startswith("Acoustic level is 1.2 dB higher")


def scoring_engine(tmp_path):
    models = StubModels()
    e = Engine(tmp_path / "s.sqlite", LAYOUT, SETTINGS, models=models, scenario="unit")
    e.table = {}
    return e, models


def run_windows(e, first_minute, n_windows, **kw):
    for i in range(first_minute, first_minute + 10 * n_windows):
        e.ingest(minute(i, **kw))
    return first_minute + 10 * n_windows


def test_model_alert_lifecycle_and_status_rule(tmp_path):
    e, models = scoring_engine(tmp_path)
    try:
        t = run_windows(e, 0, HISTORY_WINDOWS)          # 24 h of history: scoring starts on the last window
        assert e.status()["scoring"] and rows(e, "SELECT COUNT(*) n FROM predictions")[0]["n"] == len(ASSETS)
        assert rows(e, "SELECT COUNT(*) n FROM alerts")[0]["n"] == 0

        models.p_abnormal[[10, 20, 120]] = 0.9          # zones 11 and 21, and the drive
        models.p_fail[20] = 0.95
        t = run_windows(e, t, 1)
        assert rows(e, "SELECT COUNT(*) n FROM alerts")[0]["n"] == 0        # one window is not enough
        t = run_windows(e, t, 1)
        alerts = {a["asset"]: a for a in rows(e, "SELECT * FROM alerts")}
        assert set(alerts) == {"Z011", "Z021", "DRIVE"}
        assert alerts["Z011"]["type"] == "bearing_degradation" and alerts["DRIVE"]["type"] == "drive_overheating"
        assert alerts["Z011"]["eta_hours"] is None and alerts["Z021"]["eta_hours"] == 6.0
        assert alerts["Z021"]["predicted_failure_ts"] == alerts["Z021"]["created_ts"] + 6 * 3600
        assert alerts["Z021"]["severity"] > alerts["Z011"]["severity"]
        assert json.loads(alerts["Z021"]["evidence"])["top_features"][0]["name"] == "bearing_temp_c__dev6h"
        assert json.loads(alerts["Z021"]["evidence"])["reasons"] == [
            "Bearing temperature is 9.0 °C higher than the other zones on its run (6 h average)"]
        assert "What the model saw: Bearing temperature is 9.0 °C higher" in alerts["Z021"]["message"]
        st = {a["asset"]: a for a in e.status()["assets"]}
        assert (st["Z011"]["status"], st["Z021"]["status"], st["Z001"]["status"]) == (
            "abnormal", "predicted_failure", "normal")

        # the status rule: predicted failure beats sensor suspect beats the model's abnormal or normal
        stuck = {f"z{z}": {"acoustic_db": 50.0} for z in (11, 21, 31)}
        t = run_windows(e, t, 2, **stuck)
        st = {a["asset"]: a for a in e.status()["assets"]}
        assert st["Z021"]["status"] == "predicted_failure" and st["Z021"]["data_quality"] == "sensor_suspect"
        assert st["Z011"]["status"] == "no_data" and st["Z011"]["model_status"] == "abnormal"
        assert st["Z031"]["status"] == "no_data" and st["Z031"]["model_status"] == "normal"

        t = run_windows(e, t, 3)                        # more windows above: still one open alert per asset
        assert rows(e, "SELECT COUNT(*) n FROM alerts WHERE source='model'")[0]["n"] == 3

        models.p_abnormal[:] = 0.0
        t = run_windows(e, t, 6)                        # zone 21 still has a 95% failure score: its alert is held
        assert [a["asset"] for a in rows(e, "SELECT asset FROM alerts WHERE source='model' AND state='open'")] == ["Z021"]
        models.p_abnormal[[10, 120]] = 0.9
        t = run_windows(e, t, 2)                        # the other two reopen; the hold never opened anything
        assert rows(e, "SELECT COUNT(*) n FROM alerts WHERE source='model'")[0]["n"] == 5
        models.p_abnormal[:] = 0.0
        models.p_fail[:] = 0.0
        t = run_windows(e, t, 5)
        assert rows(e, "SELECT COUNT(*) n FROM alerts WHERE source='model' AND state='open'")[0]["n"] == 3
        t = run_windows(e, t, 1)                        # the sixth window below closes them
        assert rows(e, "SELECT COUNT(*) n FROM alerts WHERE source='model' AND state='open'")[0]["n"] == 0
        with pytest.raises(Exception):                  # the database itself refuses a second open alert
            e.con.execute("INSERT INTO alerts (type, source, asset, severity, severity_label, message, state,"
                          " created_ts, updated_ts) VALUES ('sensor_stuck','x','Z031',0,'low','m','open',0,0)")
            e.con.execute("INSERT INTO alerts (type, source, asset, severity, severity_label, message, state,"
                          " created_ts, updated_ts) VALUES ('sensor_stuck','x','Z031',0,'low','m','open',0,0)")
    finally:
        e.con.close()


def test_outcomes_compare_predictions_with_repair_records(tmp_path):
    e, models = scoring_engine(tmp_path)
    try:
        t = run_windows(e, 0, HISTORY_WINDOWS)
        models.p_abnormal[[10, 20, 30]] = 0.9           # zones 11, 21 and 31 look abnormal
        t = run_windows(e, t, 2)                        # alerts open
        open_ts = rows(e, "SELECT created_ts FROM alerts WHERE asset='Z011'")[0]["created_ts"]
        models.p_fail[10] = 0.95                        # failure call on zone 11 (stub estimate: 6 h)
        t = run_windows(e, t, 6)
        call_ts = rows(e, "SELECT MIN(ts) t FROM predictions WHERE asset='Z011' AND status='predicted_failure'")[0]["t"]

        # a record dated after the data received so far is refused; so is an unknown asset
        future = T0 + 60 * t + 3600
        assert not e.work_order({"asset": "Z011", "kind": "idler_bearing", "failed_ts": future, "repaired_ts": future})
        assert not e.work_order({"asset": "Q9", "kind": "x", "failed_ts": T0, "repaired_ts": T0})
        assert rows(e, "SELECT COUNT(*) n FROM outcomes")[0]["n"] == 0

        failed = T0 + 60 * t - 600
        before = rows(e, "SELECT * FROM predictions ORDER BY asset, ts")
        assert e.work_order({"asset": "Z011", "kind": "idler_bearing", "failed_ts": failed, "repaired_ts": failed + 300})
        assert not e.work_order({"asset": "Z011", "kind": "idler_bearing", "failed_ts": failed, "repaired_ts": failed + 300})
        assert e.work_order({"asset": "Z050", "kind": "idler_bearing", "failed_ts": failed, "repaired_ts": failed + 300})
        assert rows(e, "SELECT * FROM predictions ORDER BY asset, ts") == before       # scoring only, no feedback
        out = {o["asset"]: o for o in rows(e, "SELECT * FROM outcomes")}
        assert out["Z050"]["kind"] == "missed" and out["Z050"]["alert_id"] is None
        o = out["Z011"]
        assert o["kind"] == "confirmed" and o["alert_created_ts"] == open_ts and o["call_ts"] == call_ts
        assert o["lead_hours"] == pytest.approx((failed - open_ts) / 3600)
        assert o["call_lead_hours"] == pytest.approx((failed - call_ts) / 3600) and o["within_horizon"] == 1
        assert o["eta_at_call"] == 6.0 and o["eta_error_hours"] == pytest.approx(abs(6.0 - (failed - call_ts) / 3600))

        models.p_abnormal[:] = 0.0                      # all alerts close; zones 21 and 31 never failed
        models.p_fail[:] = 0.0
        t = run_windows(e, t, 6)
        kinds = {o["asset"]: o["kind"] for o in rows(e, "SELECT * FROM outcomes")}
        assert kinds == {"Z011": "confirmed", "Z050": "missed", "Z021": "false_alarm", "Z031": "false_alarm"}

        # zone 21 fails a day later: its lapsed alert was an early warning, not a false alarm
        t = run_windows(e, t, 6)
        later = T0 + 60 * t - 600
        assert e.work_order({"asset": "Z021", "kind": "idler_bearing", "failed_ts": later, "repaired_ts": later + 60})
        z21 = rows(e, "SELECT kind, failed_ts, lead_hours FROM outcomes WHERE asset='Z021'")
        assert len(z21) == 1 and z21[0]["kind"] == "early_warning" and z21[0]["failed_ts"] == later   # one row per failure
        assert z21[0]["lead_hours"] > 0

        with TestClient(create_app(SETTINGS, engine=e)) as client:
            reply = client.get("/api/outcomes").json()
            assert reply["summary"] == {"failures": 3, "warned": 1, "missed": 1, "false_alarms": 1, "early_warnings": 1,
                                        "failure_calls": 1, "failure_calls_came_true": 1, "call_horizon_hours": 72.0,
                                        "recall": pytest.approx(1 / 3), "precision": pytest.approx(2 / 3)}
            assert len(reply["outcomes"]) == 4 and reply["outcomes"][0]["asset"] == "Z021"
            assert client.post("/api/work_orders", json={"orders": [{"asset": "Z031", "kind": "idler_bearing",
                                                                     "failed_ts": later, "repaired_ts": later + 60}]}
                               ).json() == {"received": 1}
            after = client.get("/api/outcomes").json()["summary"]
            assert after["early_warnings"] == 2 and after["false_alarms"] == 0 and after["failures"] == 4
    finally:
        pass


def test_alerts_hold_while_the_belt_is_stopped(tmp_path):
    e, models = scoring_engine(tmp_path)
    try:
        t = run_windows(e, 0, HISTORY_WINDOWS)
        models.p_abnormal[10] = 0.9
        t = run_windows(e, t, 2)
        models.p_abnormal[:] = 0.0
        t = run_windows(e, t, 12, run_state=0)          # two hours stopped with low scores
        assert rows(e, "SELECT state FROM alerts")[0]["state"] == "open"
        assert e.status()["assets"][10]["status"] == "abnormal"
    finally:
        e.con.close()


# ------------------------------------------------------------------ features: live equals training
@pytest.fixture(scope="module")
def scenario(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    cfg = Config(scenario="feat", seed=5, days=2)
    write_scenario(generate(cfg), cfg, out)
    return out


def test_live_features_equal_training_features(scenario, tmp_path):
    from model.features import rolling_features, window_aggregates
    from replayer.run import Injector, batches, load_public
    from scripts.make_model_data import public_blocks

    zone_vals, belt_vals, run_state, facts, _ = public_blocks(scenario / "feat")
    trained = rolling_features(window_aggregates(zone_vals, belt_vals, run_state, facts), facts)

    models = StubModels()
    e = Engine(tmp_path / "f.sqlite", load_layout(scenario / "feat"), SETTINGS, models=models, scenario="feat")
    try:
        for _, _, body in batches(*load_public(scenario / "feat"), Injector(SETTINGS["inject"], False)):
            e.ingest(body)
        e.advance(flush=True)
        live_zone = np.stack([x for g, x in models.seen if g == "zone"])
        live_drive = np.stack([x[0] for g, x in models.seen if g == "drive"])
        n = 2 * 144 - (HISTORY_WINDOWS - 1)
        assert live_zone.shape == (n, N_ZONES, len(zone_feature_names()))
        assert live_drive.shape == (n, len(drive_feature_names()))
        # the replayer rounds to 4 decimals on the wire, which is far below any signal's noise
        np.testing.assert_allclose(live_zone, trained["zone"][HISTORY_WINDOWS - 1:], rtol=0, atol=2e-3, equal_nan=True)
        np.testing.assert_allclose(live_drive, trained["drive"][HISTORY_WINDOWS - 1:], rtol=2e-4, atol=2e-3,
                                   equal_nan=True)
    finally:
        e.con.close()


# ------------------------------------------------------------------ API and replayer, end to end
def test_api_and_replayer_end_to_end(scenario, tmp_path):
    from replayer.run import run

    settings = json.loads(json.dumps(SETTINGS))
    settings["replay"]["preload_hours"] = 48            # everything unthrottled: the test does not wait on a clock
    settings["inject"].update(duplicate_row_rate=0.002, out_of_range_value_rate=0.001, swap_every_batches=50)
    e = Engine(tmp_path / "api.sqlite", load_layout(scenario / "feat"), settings, models=StubModels(), scenario="feat")
    with TestClient(create_app(settings, engine=e)) as client:
        assert client.get("/api/health").json()["ok"] is True
        assert client.post("/api/replay", json={"speed": 99}).status_code == 422
        assert client.post("/api/replay", json={"action": "dance"}).status_code == 422
        sent = run("feat", "", scenario, settings, inject=True, max_minutes=600, client=client, log=lambda m: None,
                   work_orders=True)
        e.drain()

        # what the replayer says it injected is exactly what the backend counted
        stats = client.get("/api/ingest/stats").json()
        log = rows(e, "SELECT SUM(n_duplicates) d, SUM(n_out_of_range) o, SUM(n_rejected_rows) r, COUNT(*) b "
                      "FROM ingest_log")[0]
        assert log["b"] == sent["batches_sent"] == 600 == stats["totals"]["batches"]
        assert sent["duplicate_rows"] > 0 and sent["out_of_range_values"] > 0 and sent["swapped_batches"] == 11
        assert (log["d"], log["o"], log["r"]) == (sent["duplicate_rows"], sent["out_of_range_values"],
                                                  sent["unknown_zone_rows"])
        assert stats["totals"]["late_rows"] == 0        # swapped batches were put back in order
        assert rows(e, "SELECT COUNT(*) n FROM readings")[0]["n"] == 600 * N_ZONES
        assert json.loads((scenario / "feat" / "replayer_injected.json").read_text()) == sent

        assert client.get("/api/replay").json()["state"] == "finished"
        status = client.get("/api/status").json()
        assert len(status["assets"]) == 121 and status["sim_ts"] == T0 + 599 * 60
        layout = client.get("/api/layout").json()
        assert len(layout["zones"]) == 120 and len(layout["sections"]) == 12
        series = client.get("/api/zones/5/series", params={"signal": "bearing_temp_c"}).json()
        assert 0 < len(series["points"]) <= 500 and series["points"][0]["value"] is not None
        assert client.get("/api/zones/5/series", params={"signal": "drop table"}).status_code == 422
        assert client.get("/api/zones/500/series", params={"signal": "acoustic_db"}).status_code == 404
        assert len(client.get("/api/belt/series", params={"signal": "motor_temp_c"}).json()["points"]) > 0
        assert "alerts" in client.get("/api/alerts").json()
        assert client.get("/api/alerts", params={"state": "nope"}).status_code == 422
        assert client.get("/api/predictions", params={"asset": "Z005"}).json()["asset"] == "Z005"
        assert client.get("/api/predictions", params={"asset": "Q1"}).status_code == 404
        assert "sensors" in client.get("/api/sensors/health").json()
        empty = client.get("/api/outcomes").json()["summary"]          # nothing settled yet: no percentages
        assert empty["failures"] == 0 and empty["precision"] is None and empty["recall"] is None
        import backend.app as app_module
        app_module.MODEL_SUMMARY = tmp_path / "summary.json"
        assert client.get("/api/model/evaluation").status_code == 404
        (tmp_path / "summary.json").write_text('{"faults": 12, "caught": 12}')
        assert client.get("/api/model/evaluation").json() == {"faults": 12, "caught": 12}
        for path in ("status", "detections", "defects"):   # vision placeholders: present, empty, saying so
            assert client.get(f"/api/vision/{path}").json() == {
                "enabled": False, "reason": "The camera pipeline is not part of this run.", "items": []}


def test_read_only_connections_cannot_write(engine):
    engine._commit()
    con = connect(engine.db_path, read_only=True)
    with pytest.raises(Exception):
        con.execute("DELETE FROM zones")
    con.close()


# ---- final-day handover: the zone model must never move when the drive model is retrained ----------------------

def test_the_zone_model_never_trains_on_extra_splits():
    from model.train import training_splits
    assert training_splits("zone") == ("train",)
    assert training_splits("zone", ("train_drive", "test_fresh")) == ("train",)       # asked, refused
    assert training_splits("drive") == ("train",)
    assert training_splits("drive", ("train_drive",)) == ("train", "train_drive")


def test_saving_one_group_leaves_the_other_groups_files_untouched(tmp_path):
    import hashlib

    import lightgbm as lgb

    from model.train import save
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(200, 3)), rng.integers(0, 2, 200)

    def boosters():
        return {k: lgb.train({"verbose": -1, "num_threads": 1, "objective": "binary", "min_data_in_leaf": 5},
                             lgb.Dataset(x, label=y), num_boost_round=3) for k in ("abnormal", "classifier", "regressor")}

    def thresholds(a):
        return {"abnormal": a, "predicted_failure": a, "abnormal_met_target": True, "predicted_failure_met_target": True,
                "abnormal_sweep": [], "predicted_failure_sweep": []}

    both = {"boosters": {"zone": boosters(), "drive": boosters()}, "thresholds": {"zone": thresholds(0.9),
            "drive": thresholds(0.5)}, "splits": {"train": ["a"]}, "trained_on": {"zone": ["a"], "drive": ["a"]}}
    save(both, 1, tmp_path)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    zone_before = {p.name: digest(p) for p in tmp_path.glob("zone_*.txt")}
    meta_before = json.loads((tmp_path / "meta.json").read_text())
    assert len(zone_before) == 3

    only_drive = {"boosters": {"drive": boosters()}, "thresholds": {"drive": thresholds(0.7)}, "splits": {"train": ["a"]},
                  "trained_on": {"drive": ["a", "d1"]}}
    save(only_drive, 1, tmp_path, version_suffix="-drive2")
    assert {p.name: digest(p) for p in tmp_path.glob("zone_*.txt")} == zone_before          # byte-identical
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["thresholds"]["zone"] == meta_before["thresholds"]["zone"]                  # zone thresholds untouched
    assert meta["thresholds"]["drive"]["abnormal"] == 0.7 and meta["trained_on"]["drive"] == ["a", "d1"]
    assert meta["trained_on"]["zone"] == ["a"] and meta["version"] == "lgbm-w10-seed1-drive2"
    assert meta["zone_features"] == meta_before["zone_features"]


def test_fresh_test_scenarios_are_in_no_training_or_tuning_path():
    from model.train import training_splits
    for group in ("zone", "drive"):
        assert not {"test", "test_fresh", "stress", "validation"} & set(training_splits(group, ("train_drive",)))
    source = (REPO / "model" / "train.py").read_text()
    assert "test_fresh" not in source and "test_0" not in source
