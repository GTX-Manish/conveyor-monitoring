import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  Alert, IngestStats, Layout, SEVERITY_VAR, SIGNAL_LABEL, SensorHealth, StatusReply, assetName, count, featureLabel, get, hours,
  pct, simTime,
} from "./api";
import { StatusChip } from "./Detail";

const TYPE_LABEL: Record<string, string> = {
  bearing_degradation: "Idler bearing degrading", drive_overheating: "Drive overheating",
  sensor_stuck: "Sensor stuck", sensor_outage: "Sensor suite silent",
};

function Severity({ label }: { label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap capitalize">
      <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: SEVERITY_VAR[label] }} />{label}
    </span>
  );
}

export function AnomaliesTab({ alerts, origin, onSelect }: { alerts: Alert[]; origin: number | null; onSelect: (asset: string) => void }) {
  const [showClosed, setShowClosed] = useState(false);
  const shown = alerts.filter((a) => showClosed || a.state === "open");
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs text-ink2">
        <span>Sorted by severity. One alert per fault: a fault that stays active updates its alert.</span>
        <label className="flex items-center gap-1.5"><input type="checkbox" checked={showClosed} onChange={(e) => setShowClosed(e.target.checked)} />Show closed</label>
      </div>
      {shown.length === 0 && <p className="py-6 text-center text-sm text-ink3">No {showClosed ? "" : "open "}alerts.</p>}
      <ul className="space-y-2">
        {shown.map((a) => (
          <li key={a.alert_id} onClick={() => onSelect(a.asset)}
            className={`cursor-pointer rounded border border-line p-2.5 text-sm hover:bg-raised ${a.state === "closed" ? "opacity-60" : ""}`}>
            <div className="flex flex-wrap items-baseline justify-between gap-x-3">
              <div className="font-medium">{TYPE_LABEL[a.type] ?? a.type} · {assetName(a.asset)}{a.position_m != null ? ` (${a.position_m.toFixed(0)} m)` : ""}</div>
              <div className="flex items-center gap-3 text-xs">
                <Severity label={a.severity_label} />
                {a.eta_hours != null && <span className="tabular font-medium">fails in about {hours(a.eta_hours)}</span>}
                <span className="text-ink3">{a.state === "open" ? `since ${simTime(a.created_ts, origin)}` : `closed ${simTime(a.closed_ts, origin)}`}</span>
              </div>
            </div>
            <p className="mt-1 text-ink2">{a.message}</p>
            {a.evidence && a.evidence.top_features.length > 0 && (
              <div className="mt-1.5 text-xs text-ink2">
                <span className="text-ink3">Why: </span>
                {a.evidence.top_features.map((f, i) => (
                  <span key={f.name}>{i > 0 && "; "}{featureLabel(f.name)}{f.value != null && <span className="tabular text-ink"> ({f.value > 0 ? "+" : ""}{f.value.toFixed(2)})</span>}</span>
                ))}
              </div>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function PredictionsTab({ status, layout, origin, onSelect }: {
  status: StatusReply; layout: Layout; origin: number | null; onSelect: (asset: string) => void;
}) {
  if (!status.scoring)
    return <p className="py-6 text-center text-sm text-ink3">{status.model_version
      ? `The model scores once it has 24 h of history (${status.windows_of_history} of ${status.windows_needed} ten-minute windows so far).`
      : "No model is loaded, so there are no predictions. Sensor health still works."}</p>;
  const ranked = [...status.assets].sort((a, b) => (b.p_fail_24h ?? 0) - (a.p_fail_24h ?? 0) || (b.p_abnormal ?? 0) - (a.p_abnormal ?? 0));
  const flagged = ranked.filter((a) => a.model_status !== "normal");
  const rows = flagged.length >= 8 ? flagged : ranked.slice(0, 8);
  return (
    <div>
      <p className="mb-2 text-xs text-ink2">
        What, where and when, sorted by risk. {flagged.length} of {status.assets.length} assets are flagged by the model; scored at {simTime(status.scored_ts, origin)}.
      </p>
      <table className="w-full text-sm">
        <thead className="text-left text-xs text-ink3">
          <tr><th className="py-1 font-normal">Where</th><th className="font-normal">What</th><th className="font-normal">Fails within 24 h</th>
            <th className="font-normal">Abnormal</th><th className="text-right font-normal">When</th></tr>
        </thead>
        <tbody>
          {rows.map((a) => {
            const z = a.zone_id ? layout.zones[a.zone_id - 1] : null;
            const when = a.est_hours_to_failure != null && status.scored_ts != null && a.model_status === "predicted_failure"
              ? `in about ${hours(a.est_hours_to_failure)} (${simTime(status.scored_ts + a.est_hours_to_failure * 3600, origin)})` : "–";
            return (
              <tr key={a.asset} onClick={() => onSelect(a.asset)} className="cursor-pointer border-t border-line hover:bg-raised">
                <td className="py-1.5"><span className="font-medium">{assetName(a.asset)}</span>{z && <span className="text-ink3"> · S{z.section}, {z.position_m.toFixed(0)} m</span>}</td>
                <td><StatusChip asset={{ ...a, status: a.model_status as typeof a.status }} /></td>
                <td><Meter value={a.p_fail_24h} /></td>
                <td><Meter value={a.p_abnormal} /></td>
                <td className="tabular text-right">{when}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Meter({ value }: { value: number | null }) {
  return (
    <span className="flex items-center gap-2">
      <span className="h-1.5 w-16 overflow-hidden rounded-full bg-raised"><span className="block h-full rounded-full" style={{ width: `${(value ?? 0) * 100}%`, background: "var(--series-1)" }} /></span>
      <span className="tabular w-9 text-right">{pct(value)}</span>
    </span>
  );
}

export function DataHealthTab({ stats, health, origin, onSelect }: {
  stats?: IngestStats; health?: SensorHealth; origin: number | null; onSelect: (asset: string) => void;
}) {
  const t = stats?.totals ?? {};
  const tiles: [string, number | undefined, string][] = [
    ["Rows received", t.rows, "zone and drive readings"],
    ["Missing values", t.null_values, "stored as missing, never filled in"],
    ["Out of range", t.out_of_range, "impossible values, stored as missing"],
    ["Duplicates", t.duplicates, "ignored; the first row wins"],
    ["Rejected rows", t.rejected_rows, "unknown zone or bad timestamp"],
    ["Late rows", t.late_rows, "stored, too late to score"],
  ];
  return (
    <div className="space-y-4">
      <div>
        <dl className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {tiles.map(([label, value, note]) => (
            <div key={label} className="rounded border border-line p-2">
              <dt className="text-xs text-ink3">{label}</dt>
              <dd className="tabular text-lg font-semibold leading-tight">{count(value)}</dd>
              <dd className="text-xs text-ink3">{note}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-1.5 text-xs text-ink3">
          {count(t.batches)} batches, processed up to {simTime(stats?.watermark_ts, origin)}. The simulator only produces missing values;
          the replayer adds duplicates, out-of-range values, unknown zones and out-of-order batches on purpose.
        </p>
      </div>
      <div>
        <h3 className="mb-1 text-sm font-semibold">Sensors not to be trusted right now</h3>
        {(health?.sensors.length ?? 0) === 0 ? <p className="text-sm text-ink3">None. Every sensor is reporting changing values.</p> : (
          <table className="w-full text-sm">
            <thead className="text-left text-xs text-ink3"><tr><th className="py-1 font-normal">Zone</th><th className="font-normal">Sensor</th><th className="font-normal">State</th><th className="font-normal">Since</th></tr></thead>
            <tbody>
              {health!.sensors.map((s) => (
                <tr key={`${s.zone_id}-${s.signal}`} onClick={() => onSelect(`Z${String(s.zone_id).padStart(3, "0")}`)} className="cursor-pointer border-t border-line hover:bg-raised">
                  <td className="py-1.5 font-medium">Zone {s.zone_id} <span className="font-normal text-ink3">· S{s.section}</span></td>
                  <td>{SIGNAL_LABEL[s.signal]?.replace(/ \(.*\)/, "") ?? s.signal}</td>
                  <td>{s.state === "stuck" ? `Stuck at ${s.last_value?.toFixed(2) ?? "?"}` : "No data"}</td>
                  <td className="tabular text-ink2">{simTime(s.since_ts, origin)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
      <div>
        <h3 className="mb-1 text-sm font-semibold">Sensor events so far</h3>
        {(health?.events.length ?? 0) === 0 ? <p className="text-sm text-ink3">None yet.</p> : (
          <ul className="max-h-48 space-y-0.5 overflow-y-auto text-sm">
            {health!.events.map((e) => (
              <li key={e.event_id} className="tabular flex justify-between gap-3 border-t border-line py-1">
                <span>Zone {e.zone_id}: {e.type === "stuck" ? `${SIGNAL_LABEL[e.signal ?? ""]?.replace(/ \(.*\)/, "") ?? "sensor"} stuck` : "all sensors silent"}</span>
                <span className="text-ink2">{simTime(e.start_ts, origin)} to {e.end_ts ? simTime(e.end_ts, origin) : "now"}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

export function VisionTab() {
  const q = useQuery({ queryKey: ["vision"], queryFn: () => get<{ enabled: boolean; reason: string }>("/api/vision/status"), refetchInterval: 30000 });
  return (
    <div className="py-6 text-center text-sm text-ink2">
      <p className="font-medium text-ink">Vision is not enabled in this run.</p>
      <p className="mx-auto mt-1 max-w-md">
        {q.data?.reason ?? "The camera pipeline is not part of this run."} The belt-surface defect detector will show
        its detections here, and mark defects on the map, once it is added. The sensor pipeline above does not depend on it.
      </p>
    </div>
  );
}
