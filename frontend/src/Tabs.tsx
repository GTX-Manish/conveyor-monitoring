import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import {
  Alert, IngestStats, Layout, Outcome, Outcomes, SEVERITY_VAR, SIGNAL_LABEL, SensorHealth, StatusReply, alertText, assetName, count, get, hours,
  pct, simTime,
} from "./api";
import { StatusChip } from "./Detail";

export const ALERT_TITLE: Record<string, string> = {
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

/** What the model saw, as a short list. Shown in full: an alert should say why it exists. */
export function Reasons({ alert }: { alert: Alert }) {
  const { reasons } = alertText(alert);
  if (reasons.length === 0) return null;
  return (
    <div className="mt-2 text-sm">
      <div className="font-medium">What the model saw</div>
      <ul className="mt-0.5 list-disc space-y-0.5 pl-5 text-ink2">{reasons.map((r) => <li key={r}>{r}</li>)}</ul>
    </div>
  );
}

export function AnomaliesTab({ alerts, origin, onSelect }: { alerts: Alert[]; origin: number | null; onSelect: (asset: string) => void }) {
  const [showClosed, setShowClosed] = useState(false);
  const shown = alerts.filter((a) => showClosed || a.state === "open");
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs text-ink2">
        <span>Most serious first. Click a card to open its zone. Critical is shown only when a failure is predicted within 72 h.</span>
        <label className="flex items-center gap-1.5"><input type="checkbox" checked={showClosed} onChange={(e) => setShowClosed(e.target.checked)} />Also show past alerts</label>
      </div>
      {shown.length === 0 && <p className="rounded-lg border border-line py-8 text-center text-sm text-ink3">No {showClosed ? "" : "open "}alerts. The belt looks healthy.</p>}
      <ul className="space-y-2">
        {shown.map((a) => (
          <li key={a.alert_id} onClick={() => onSelect(a.asset)}
            className={`cursor-pointer rounded border border-line p-2.5 text-sm hover:bg-raised ${a.state === "closed" ? "opacity-60" : ""}`}>
            <div className="flex flex-wrap items-baseline justify-between gap-x-3">
              <div className="font-medium">{ALERT_TITLE[a.type] ?? a.type} · {assetName(a.asset)}{a.position_m != null ? ` (${a.position_m.toFixed(0)} m)` : ""}</div>
              <div className="flex items-center gap-3 text-xs">
                <Severity label={a.severity_label} />
                {a.eta_hours != null && <span className="tabular font-medium">fails in about {hours(a.eta_hours)}</span>}
                <span className="text-ink3">{a.state === "open" ? `since ${simTime(a.created_ts, origin)}` : `closed ${simTime(a.closed_ts, origin)}`}</span>
              </div>
            </div>
            <p className="mt-1 text-ink2">{alertText(a).summary}</p>
            <Reasons alert={a} />
          </li>
        ))}
      </ul>
    </div>
  );
}

export function PredictionsTab({ status, layout, origin, onSelect }: {
  status: StatusReply; layout: Layout; origin: number | null; onSelect: (asset: string) => void;
}) {
  const [showAll, setShowAll] = useState(false);
  if (!status.scoring)
    return <p className="py-6 text-center text-sm text-ink3">{status.model_version
      ? `The model scores once it has 24 h of history (${status.windows_of_history} of ${status.windows_needed} ten-minute windows so far).`
      : "No model is loaded, so there are no predictions. Sensor health still works."}</p>;
  const ranked = [...status.assets].sort((a, b) => (b.p_fail_24h ?? 0) - (a.p_fail_24h ?? 0) || (b.p_abnormal ?? 0) - (a.p_abnormal ?? 0));
  const flagged = ranked.filter((a) => a.model_status !== "normal");
  const rows = showAll ? ranked.slice(0, 15) : flagged;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-xs text-ink2">
        <span>{flagged.length === 0 ? "The model flags nothing right now." : `The model flags ${flagged.length} of ${status.assets.length} parts.`} Last checked {simTime(status.scored_ts, origin)}.</span>
        <label className="flex items-center gap-1.5"><input type="checkbox" checked={showAll} onChange={(e) => setShowAll(e.target.checked)} />Also show the next most at risk</label>
      </div>
      {rows.length === 0 && <p className="rounded-lg border border-line py-8 text-center text-sm text-ink3">No part is expected to fail. Tick the box above to see the parts the model ranks highest anyway.</p>}
      <table className={`w-full text-sm ${rows.length === 0 ? "hidden" : ""}`}>
        <thead className="text-left text-xs text-ink3">
          <tr><th className="py-1 font-normal">Where</th><th className="font-normal">What the model says</th><th className="font-normal">Failure risk score</th>
            <th className="font-normal">How unusual it looks</th><th className="text-right font-normal">Expected failure</th></tr>
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


const OUTCOME_LABEL: Record<Outcome["kind"], string> = {
  confirmed: "Alert open at failure", missed: "Missed", false_alarm: "False alarm", early_warning: "Warned, but the alert had closed",
};
const OUTCOME_VAR: Record<Outcome["kind"], string> = {
  confirmed: "var(--st-normal)", missed: "var(--st-failure)", false_alarm: "var(--st-abnormal)", early_warning: "var(--st-nodata)",
};

/** What the model said, set against what the repair records later showed. */
export function TrackRecordTab({ origin, onSelect }: { origin: number | null; onSelect: (asset: string) => void }) {
  const q = useQuery({ queryKey: ["outcomes"], queryFn: () => get<Outcomes>("/api/outcomes"), refetchInterval: 4000 });
  const s = q.data?.summary;
  const rows = q.data?.outcomes ?? [];
  const tiles: [string, string, string][] = s ? [
    ["Failures with an alert open", `${s.warned} of ${s.failures}`, "the alert was open when the part failed"],
    ["Warned, but the alert had closed", String(s.early_warnings), "it opened earlier and lapsed before the failure"],
    ["Failures missed", String(s.missed), "no alert at any point"],
    ["False alarms", String(s.false_alarms), "an alert closed and no failure followed"],
    ["Failure calls that came true", `${s.failure_calls_came_true} of ${s.failure_calls}`, `the part failed within ${s.call_horizon_hours} h of the call`],
  ] : [];
  return (
    <div className="space-y-4">
      <dl className="grid grid-cols-2 gap-2 lg:grid-cols-5">
        {tiles.map(([label, value, note]) => (
          <div key={label} className="rounded-lg border border-line p-3">
            <dt className="text-xs text-ink3">{label}</dt>
            <dd className="tabular text-2xl font-semibold leading-tight">{value}</dd>
            <dd className="text-xs text-ink3">{note}</dd>
          </div>
        ))}
      </dl>
      <p className="text-xs text-ink3">
        A prediction can only be checked after the fact. When a repair ends, a repair record arrives (in a plant, a closed
        work order) saying which part failed and when. Each record is compared here with what the model had said. The model
        never sees these records. The demo is one curated scenario; the full evaluation is in the project's evaluation report.
      </p>
      {rows.length === 0 ? (
        <p className="rounded-lg border border-line py-8 text-center text-sm text-ink3">
          Nothing to check yet. Cases appear here once a part has failed and been repaired, or an alert has closed.
        </p>
      ) : (
        <ul className="space-y-2">
          {rows.map((o) => (
            <li key={o.outcome_id} onClick={() => onSelect(o.asset)} className="cursor-pointer rounded-lg border border-line p-3 text-sm hover:bg-raised">
              <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                <span className="font-medium">{assetName(o.asset)}{o.position_m != null ? ` (${o.position_m.toFixed(0)} m)` : ""}</span>
                <span className="flex items-center gap-1.5 text-xs">
                  <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: OUTCOME_VAR[o.kind] }} />{OUTCOME_LABEL[o.kind]}
                </span>
              </div>
              <p className="mt-1 text-ink2">{outcomeSentence(o, origin)}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function outcomeSentence(o: Outcome, origin: number | null): string {
  const failed = o.failed_ts != null ? `Failed on ${simTime(o.failed_ts, origin)}.` : "";
  if (o.kind === "missed") return `${failed} No alert was open: the model did not see this coming.`;
  if (o.kind === "false_alarm")
    return `An alert opened on ${simTime(o.alert_created_ts, origin)} and closed on ${simTime(o.resolved_ts, origin)}. No failure has followed${o.call_ts != null ? ", although a failure was predicted" : ""}.`;
  const warned = o.kind === "early_warning"
    ? `${failed} An alert first opened ${hours(o.lead_hours)} earlier but had closed again by the time the part failed.`
    : `${failed} The first alert came ${hours(o.lead_hours)} earlier and an alert was open when it failed.`;
  if (o.call_ts == null) return `${warned} No failure call was made before it failed.`;
  const verdict = o.within_horizon ? "The prediction came true within 72 h." : "The prediction was early: the part lasted longer than 72 h.";
  return `${warned} A failure was predicted ${hours(o.call_lead_hours)} before it happened, with ${hours(o.eta_at_call)} estimated at the time (off by ${hours(o.eta_error_hours)}). ${verdict}`;
}
