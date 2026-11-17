// Types and helpers for the backend API. All times are UTC epoch seconds of SIMULATED time.
export type Status = "normal" | "abnormal" | "predicted_failure" | "no_data";
export type SensorState = "ok" | "no_data" | "stuck";

export interface Asset {
  asset: string; zone_id: number | null; status: Status; model_status: string;
  data_quality: "ok" | "no_data" | "sensor_suspect"; sensors: Record<string, SensorState>;
  p_abnormal: number | null; p_fail_24h: number | null; est_hours_to_failure: number | null;
}
export interface StatusReply {
  sim_ts: number | null; start_ts: number | null; scored_ts: number | null; scoring: boolean; model_version: string | null;
  windows_of_history: number; windows_needed: number; replay: { state: string; speed: number }; assets: Asset[];
}
export interface Zone {
  zone_id: number; section: number; zone_in_section: number; position_m: number; run: "carry" | "return";
  is_loading: boolean; is_unloading: boolean; incline_deg: number | null; curve_radius_m: number | null;
}
export interface Layout {
  zones: Zone[]; sections: { section: number; name: string; run: string; zones: number[] }[];
  zone_signals: string[]; belt_signals: string[];
}
export interface Feature { name: string; contribution: number; value: number | null }
export interface Alert {
  alert_id: number; type: string; source: "model" | "data_quality"; asset: string; zone_id: number | null;
  position_m: number | null; severity: number; severity_label: "low" | "medium" | "high" | "critical";
  predicted_failure_ts: number | null; eta_hours: number | null; message: string;
  evidence: { top_features: Feature[]; reasons?: string[]; p_fail_24h: number; p_abnormal: number } | null;
  state: "open" | "closed"; created_ts: number; updated_ts: number; closed_ts: number | null;
}
export interface Series {
  signal: string; from: number; to: number; bucket_seconds: number;
  points: { ts: number; value: number | null; n: number; run_state: number }[];
}
export interface Predictions {
  asset: string;
  points: { ts: number; p_abnormal: number; p_fail_24h: number; est_hours_to_failure: number | null; status: string }[];
}
export interface IngestStats {
  totals: Record<string, number>; watermark_ts: number | null;
  recent: { batch_id: number; sim_ts: number; n_rows: number; n_null_values: number; n_rejected_rows: number;
            n_out_of_range: number; n_duplicates: number }[];
}
export interface SensorHealth {
  sensors: { zone_id: number; section: number; position_m: number; signal: string; state: SensorState; since_ts: number;
             last_value: number | null }[];
  events: { event_id: number; zone_id: number; signal: string | null; type: string; start_ts: number; end_ts: number | null }[];
}

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}
export async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

// ---- words and formats -------------------------------------------------------------------------------
export const STATUS_LABEL: Record<Status, string> = {
  normal: "Normal", abnormal: "Abnormal", predicted_failure: "Predicted failure", no_data: "No data / sensor suspect",
};
export const STATUS_VAR: Record<Status, string> = {
  normal: "var(--st-normal)", abnormal: "var(--st-abnormal)", predicted_failure: "var(--st-failure)", no_data: "var(--st-nodata)",
};
export const STATUS_GLYPH: Record<Status, string> = { normal: "", abnormal: "•", predicted_failure: "!", no_data: "?" };
export const SEVERITY_VAR: Record<string, string> = {
  low: "var(--st-nodata)", medium: "var(--st-abnormal)", high: "var(--st-serious)", critical: "var(--st-failure)",
};

export const SIGNAL_LABEL: Record<string, string> = {
  vibration_rms_mm_s: "Vibration RMS (mm/s)", vibration_kurtosis: "Vibration kurtosis", bearing_temp_c: "Bearing temperature (°C)",
  acoustic_db: "Acoustic level (dB)", belt_offset_mm: "Belt offset (mm)", throughput_tph: "Throughput (t/h)",
  motor_current_a: "Motor current (A)", motor_temp_c: "Motor temperature (°C)", gearbox_temp_c: "Gearbox temperature (°C)",
  drive_speed_ms: "Drive speed (m/s)", belt_speed_ms: "Belt speed (m/s)", slip_pct: "Slip (%)", tension_kn: "Tension (kN)",
};
const SHORT: Record<string, string> = {
  vibration_rms_mm_s: "Vibration", vibration_kurtosis: "Vibration kurtosis", bearing_temp_c: "Bearing temperature",
  acoustic_db: "Acoustic level", belt_offset_mm: "Belt offset", motor_temp_c: "Motor temperature",
  gearbox_temp_c: "Gearbox temperature", motor_current_a: "Motor current", tension_kn: "Tension", slip_pct: "Slip",
  throughput_tph: "Throughput", belt_speed_ms: "Belt speed",
};
const KIND: Record<string, string> = {
  dev10: "above its peers, last 10 min", dev1h: "above its peers, 1 h average", dev6h: "above its peers, 6 h average",
  dev24h: "above its peers, 24 h average", rise1h: "rising: last hour against the last day",
  rise6h: "rising: last 6 h against the last day", nb6h: "above the neighbouring zones, 6 h average",
  peak1h: "peak in the last hour, against its peers", spread1h: "more variable than its peers, last hour",
  mean10: "level, last 10 min", mean1h: "level, 1 h average", mean24h: "level, 24 h average",
};
/** "bearing_temp_c__dev6h" -> "Bearing temperature above its peers, 6 h average". */
export function featureLabel(name: string): string {
  const [sig, kind] = name.split("__");
  if (kind && SHORT[sig]) return `${SHORT[sig]} ${KIND[kind] ?? kind}`;
  const plain: Record<string, string> = {
    motor_minus_gearbox_1h: "Motor hotter than gearbox, 1 h average", current_per_load_1h: "Motor current per tonne, 1 h average",
    current_per_load_24h: "Motor current per tonne, 24 h average", temp_reference: "Temperature of the other zones",
    missing_share: "Share of readings missing", any_flat: "A sensor repeating one value", run_share: "Share of the window running",
  };
  return plain[name] ?? name.replace(/_/g, " ");
}

/** Simulated time as "Day 3, 14:20" (day 1 is the first day of the scenario). */
export function simTime(ts: number | null | undefined, t0?: number | null): string {
  if (ts == null) return "–";
  const d = new Date(ts * 1000);
  const hhmm = `${String(d.getUTCHours()).padStart(2, "0")}:${String(d.getUTCMinutes()).padStart(2, "0")}`;
  if (t0 == null) return `${d.getUTCDate()} ${d.toLocaleString("en", { month: "short", timeZone: "UTC" })} ${hhmm}`;
  return `Day ${Math.floor((ts - t0) / 86400) + 1}, ${hhmm}`;
}
export const pct = (v: number | null | undefined) => (v == null ? "–" : `${Math.round(v * 100)}%`);
export const hours = (v: number | null | undefined) => (v == null ? "–" : v < 1 ? "under 1 h" : `${Math.round(v)} h`);
export const count = (v: number | undefined) => (v == null ? "–" : v.toLocaleString("en"));
export const assetName = (a: string) => (a === "DRIVE" ? "Drive" : `Zone ${parseInt(a.slice(1), 10)}`);

/** An alert's message split into its summary and the list of things the model saw. */
export function alertText(a: Alert): { summary: string; reasons: string[] } {
  const summary = a.message.split(" What the model saw:")[0];
  const reasons = a.evidence?.reasons?.length ? a.evidence.reasons : (a.evidence?.top_features ?? []).map((f) => featureLabel(f.name));
  return { summary, reasons };
}

export interface Outcome {
  outcome_id: number; asset: string; kind: "confirmed" | "missed" | "false_alarm" | "early_warning";
  alert_created_ts: number | null; call_ts: number | null; failed_ts: number | null; lead_hours: number | null;
  call_lead_hours: number | null; within_horizon: number | null; eta_at_call: number | null; eta_error_hours: number | null;
  resolved_ts: number; fault_kind: string | null; position_m: number | null; section: number | null;
}
export interface Outcomes {
  outcomes: Outcome[];
  summary: { failures: number; warned: number; missed: number; false_alarms: number; early_warnings: number;
             failure_calls: number; failure_calls_came_true: number; call_horizon_hours: number;
             precision: number | null; recall: number | null };
}

/** The fresh test: scenarios never used for training, tuning or diagnosis, scored once (fresh_test_summary.json). */
export interface FreshAsset {
  faults: number; alerts_caught: number; false_alerts: number; alert_precision: number; alert_recall: number;
  calls: number; calls_true: number; call_precision: number; call_recall: number; faults_called: number;
  median_hours_call_to_failure: number;
}
export interface FreshTest {
  split: string; scenarios: { name: string; seed: number }[]; drive_rate_per_30d: number; call_horizon_hours: number;
  assets: { bearings: FreshAsset; drive: FreshAsset };
}

/** One sentence for the Overview, built only from the fresh-test file's numbers. */
export function freshTestSentence(f: FreshTest): string {
  const b = f.assets.bearings, d = f.assets.drive;
  const faults = b.faults + d.faults, wrong = b.false_alerts + d.false_alerts;
  const calls = b.calls + d.calls, right = b.calls_true + d.calls_true, called = b.faults_called + d.faults_called;
  const pctOf = (n: number, of: number) => `${Math.round((100 * n) / Math.max(of, 1))}%`;
  return `Fresh test, ${faults} simulated faults in scenarios never used for training, tuning or diagnosis: `
    + `bearings ${b.alerts_caught} of ${b.faults} caught (${pctOf(b.alerts_caught, b.faults)}), `
    + `drive ${d.alerts_caught} of ${d.faults} caught (${pctOf(d.alerts_caught, d.faults)}), `
    + `${wrong} false alert${wrong === 1 ? "" : "s"}. `
    + `${calls} failure predictions were made and ${right === calls ? "all came true" : `${right} came true`} `
    + `within ${f.call_horizon_hours} h; ${2 * called === faults ? "half the faults got one" : `${called} of ${faults} faults got one`}. `
    + "Simulated data only.";
}

/** Held-out test figures from the offline evaluation (scripts/score.py). Not shown in the dashboard: the old test
 *  scenarios were used for diagnosis, so the fresh test is what the Overview quotes. */
export interface OfflineEvaluation {
  split: string; call_horizon_hours: number; faults: number; caught: number; open_at_failure: number; false_alerts: number;
  calls: number; calls_came_true: number; faults_called: number;
}

// ---- vision -------------------------------------------------------------------------------------------
export interface Box { x1: number; y1: number; x2: number; y2: number; score: number }
export interface CaptureView {
  capture_id: number; ts: number; station: string; ground_position_m: number; belt_coordinate_m: number;
  image_w: number | null; image_h: number | null; boxes: Box[]; image_url: string;
}
export interface VisionDefect {
  defect_id: number; belt_coordinate_m: number; laps: number; laps_looked: number; n_captures: number; n_detections: number;
  first_ts: number; last_ts: number; confirmed_ts: number | null; best_score: number; stations: string[]; confirmed: boolean;
  at_splice: boolean; splice_m: number | null; splice_distance_m: number | null; best_capture: CaptureView | null;
}
export interface VisionDefects { enabled: boolean; reason?: string; defects: VisionDefect[]; at_splice: VisionDefect[]; unconfirmed: number }
export interface VisionDetection extends CaptureView { defect_id: number | null; at_splice: boolean; confirmed: boolean; max_score: number | null }
export interface VisionStatus {
  enabled: boolean; reason: string | null; threshold: number | null; confirm_laps: number; confirm_ratio: number;
  counts: { captures: number; with_detections: number; detections: number; image_missing: number; defects: number; at_splice: number; unconfirmed: number };
  stations: { station: string; ground_position_m: number; captures: number }[];
  evaluation: { split: string; images: number; map50: number; precision: number; recall: number; threshold: number;
                credit: string; license: string; source_repo: string; dataset: string } | null;
  settings: { merge_m: number; splice_m: number; splice_spacing_m: number; belt_length_m: number };
}
