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
    from simulator import healthy
    from simulator.generate import RUN_STATES
    from common import ids
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
        sent = run("feat", "", scenario, settings, inject=True, max_minutes=600, client=client, log=lambda m: None)
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
        for path in ("status", "detections", "defects"):   # vision placeholders: present, empty, saying so
            assert client.get(f"/api/vision/{path}").json() == {
                "enabled": False, "reason": "The camera pipeline is not part of this run.", "items": []}


def test_read_only_connections_cannot_write(engine):
    engine._commit()
    con = connect(engine.db_path, read_only=True)
    with pytest.raises(Exception):
        con.execute("DELETE FROM zones")
    con.close()
