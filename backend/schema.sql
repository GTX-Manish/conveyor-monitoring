-- Timestamps are INTEGER UTC epoch seconds. run_state: 0 stopped, 1 ramp, 2 running. Missing values are NULL
-- and are never filled in. Labels and other simulator internals are never loaded into this database.
CREATE TABLE zones (zone_id INTEGER PRIMARY KEY, section INTEGER NOT NULL, zone_in_section INTEGER NOT NULL,
  position_m REAL NOT NULL, run TEXT NOT NULL, is_loading INTEGER NOT NULL, is_unloading INTEGER NOT NULL,
  incline_deg REAL, curve_radius_m REAL);
CREATE TABLE readings (ts INTEGER NOT NULL, zone_id INTEGER NOT NULL, run_state INTEGER NOT NULL,
  vibration_rms_mm_s REAL, vibration_kurtosis REAL, bearing_temp_c REAL, acoustic_db REAL, belt_offset_mm REAL,
  PRIMARY KEY (ts, zone_id)) WITHOUT ROWID;
CREATE INDEX readings_zone_ts ON readings(zone_id, ts);
CREATE TABLE belt_readings (ts INTEGER PRIMARY KEY, run_state INTEGER NOT NULL, throughput_tph REAL, motor_current_a REAL,
  motor_temp_c REAL, gearbox_temp_c REAL, drive_speed_ms REAL, belt_speed_ms REAL, slip_pct REAL, tension_kn REAL);
CREATE TABLE sensor_health (zone_id INTEGER NOT NULL, signal TEXT NOT NULL, state TEXT NOT NULL,  -- 'ok'|'no_data'|'stuck'
  since_ts INTEGER NOT NULL, last_seen_ts INTEGER, last_value REAL, PRIMARY KEY (zone_id, signal)) WITHOUT ROWID;
CREATE TABLE quality_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, zone_id INTEGER NOT NULL, signal TEXT,
  type TEXT NOT NULL,  -- 'outage'|'stuck'
  start_ts INTEGER NOT NULL, end_ts INTEGER);
CREATE TABLE predictions (ts INTEGER NOT NULL, asset TEXT NOT NULL,  -- 'Z001'..'Z120' or 'DRIVE'
  p_abnormal REAL NOT NULL, p_fail_24h REAL NOT NULL, est_hours_to_failure REAL,
  status TEXT NOT NULL,  -- 'normal'|'abnormal'|'predicted_failure'
  model_version TEXT NOT NULL, PRIMARY KEY (asset, ts)) WITHOUT ROWID;
CREATE TABLE alerts (alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
  type TEXT NOT NULL,    -- 'bearing_degradation'|'drive_overheating'|'sensor_stuck'|'sensor_outage'
  source TEXT NOT NULL,  -- 'model'|'data_quality'
  asset TEXT NOT NULL, zone_id INTEGER, position_m REAL,
  severity REAL NOT NULL, severity_label TEXT NOT NULL, predicted_failure_ts INTEGER, eta_hours REAL,
  message TEXT NOT NULL, evidence TEXT,  -- JSON: {"top_features":[{"name":..,"contribution":..} x3],"p_fail_24h":..}
  state TEXT NOT NULL,   -- 'open'|'closed'
  created_ts INTEGER NOT NULL, updated_ts INTEGER NOT NULL, closed_ts INTEGER);
CREATE UNIQUE INDEX alerts_one_open ON alerts(type, asset) WHERE state = 'open';
CREATE TABLE ingest_log (batch_id INTEGER PRIMARY KEY AUTOINCREMENT, received_wall_ts INTEGER NOT NULL, sim_ts INTEGER NOT NULL,
  n_rows INTEGER NOT NULL, n_null_values INTEGER NOT NULL, n_rejected_rows INTEGER NOT NULL,
  n_out_of_range INTEGER NOT NULL, n_duplicates INTEGER NOT NULL);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- keys: scenario, replay_state, replay_position_ts, replay_speed, model_version, thresholds (JSON), schema_version

-- Vision placeholders: the camera pipeline is not built yet (the image dataset is still downloading). These
-- tables stay empty so adding vision later needs no schema change.
CREATE TABLE captures (capture_id INTEGER PRIMARY KEY, ts INTEGER NOT NULL, station TEXT NOT NULL,
  ground_position_m REAL NOT NULL, belt_coordinate_m REAL, image_path TEXT);
CREATE TABLE vision_detections (detection_id INTEGER PRIMARY KEY AUTOINCREMENT, capture_id INTEGER NOT NULL,
  ts INTEGER NOT NULL, label TEXT NOT NULL, score REAL NOT NULL, belt_coordinate_m REAL, ground_position_m REAL,
  bbox TEXT, defect_id INTEGER);

-- Repair records arrive after a repair (in a plant: closed work orders). They are used only to score past
-- predictions; the model never reads them.
CREATE TABLE work_orders (order_id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT NOT NULL, zone_id INTEGER,
  kind TEXT NOT NULL, failed_ts INTEGER NOT NULL, repaired_ts INTEGER NOT NULL, received_ts INTEGER NOT NULL,
  UNIQUE (asset, failed_ts));
-- One row per resolved case. kind: 'confirmed' (an alert was open when the part failed), 'missed' (it failed
-- with no alert open), 'false_alarm' (an alert closed and no failure followed), 'early_warning' (an alert
-- lapsed, and the part did fail later).
CREATE TABLE outcomes (outcome_id INTEGER PRIMARY KEY AUTOINCREMENT, asset TEXT NOT NULL, kind TEXT NOT NULL,
  alert_id INTEGER, order_id INTEGER, alert_created_ts INTEGER, call_ts INTEGER, failed_ts INTEGER,
  lead_hours REAL, call_lead_hours REAL, within_horizon INTEGER, eta_at_call REAL, eta_error_hours REAL,
  resolved_ts INTEGER NOT NULL);
