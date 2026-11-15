import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import {
  Alert, Asset, Layout, Predictions, SIGNAL_LABEL, STATUS_LABEL, STATUS_VAR, Series, StatusReply, get, hours, pct, simTime,
} from "./api";
import type { Selection } from "./App";

const RANGES = [{ label: "6 h", h: 6 }, { label: "24 h", h: 24 }, { label: "3 days", h: 72 }, { label: "7 days", h: 168 }];
const DRIVE_SIGNALS = ["motor_temp_c", "gearbox_temp_c", "motor_current_a", "tension_kn", "throughput_tph", "belt_speed_ms"];

export function StatusChip({ asset }: { asset: Asset }) {
  return (
    <span className="inline-flex items-center gap-1.5 whitespace-nowrap text-sm">
      <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: asset.status === "normal" ? "var(--st-normal-quiet)" : STATUS_VAR[asset.status] }} />
      {STATUS_LABEL[asset.status]}
    </span>
  );
}

export function Detail({ layout, status, alerts, selection, onSelect, origin }: {
  layout: Layout; status: StatusReply; alerts: Alert[]; selection: Selection; onSelect: (s: Selection) => void; origin: number | null;
}) {
  const [range, setRange] = useState(24);
  const [plot, setPlot] = useState("model");
  const picker = (
    <label className="mb-4 flex flex-wrap items-center gap-2 text-sm text-ink2">
      Show
      <select className="rounded border border-line bg-surface px-2 py-1.5 text-sm text-ink"
        value={!selection ? "" : selection.kind === "drive" ? "drive" : selection.kind === "zone" ? `z${selection.zone}` : `s${selection.section}`}
        onChange={(e) => {
          const v = e.target.value;
          onSelect(v === "" ? null : v === "drive" ? { kind: "drive" } : v[0] === "z" ? { kind: "zone", zone: Number(v.slice(1)) } : { kind: "section", section: Number(v.slice(1)) });
        }}>
        <option value="">Choose a zone…</option>
        <option value="drive">Drive (motor and gearbox)</option>
        {layout.sections.map((sec) => (
          <optgroup key={sec.section} label={`Section ${sec.name} (${sec.run} run)`}>
            <option value={`s${sec.section}`}>All of section {sec.name}</option>
            {[...sec.zones].sort((a, b) => a - b).map((z) => <option key={z} value={`z${z}`}>Zone {z}</option>)}
          </optgroup>
        ))}
      </select>
      <span className="text-ink3">or click a block on the map in Overview.</span>
    </label>
  );
  if (!selection)
    return (
      <div>
        {picker}
        <p className="rounded-lg border border-line py-8 text-center text-sm text-ink3">
          Choose a zone, a section or the drive to see its status, its sensors and a chart of its readings.
        </p>
      </div>
    );

  if (selection.kind === "section") {
    const sec = layout.sections.find((s) => s.section === selection.section)!;
    return (
      <div>
        {picker}
        <h2 className="mb-2 text-sm font-semibold">Section {sec.name}: zones {Math.min(...sec.zones)} to {Math.max(...sec.zones)} ({sec.run} run)</h2>
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-ink3">
            <tr><th className="py-1 font-normal">Zone</th><th className="font-normal">Position</th><th className="font-normal">Status</th>
              <th className="text-right font-normal">Chance of failing within 24 h</th></tr>
          </thead>
          <tbody>
            {[...sec.zones].sort((a, b) => a - b).map((id) => {
              const a = status.assets[id - 1], z = layout.zones[id - 1];
              return (
                <tr key={id} onClick={() => onSelect({ kind: "zone", zone: id })} className="cursor-pointer border-t border-line hover:bg-raised">
                  <td className="py-1.5 font-medium">Zone {id}</td>
                  <td className="tabular text-ink2">{z.position_m.toFixed(0)} m</td>
                  <td><StatusChip asset={a} /></td>
                  <td className="tabular text-right">{pct(a.p_fail_24h)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }

  const isDrive = selection.kind === "drive";
  const asset = isDrive ? status.assets[status.assets.length - 1] : status.assets[selection.zone - 1];
  const zone = isDrive ? null : layout.zones[selection.zone - 1];
  const mine = alerts.filter((a) => a.asset === asset.asset);
  const to = status.sim_ts ?? 0, from = to - range * 3600;
  const signals = isDrive ? DRIVE_SIGNALS : layout.zone_signals;
  const shown = plot === "model" || signals.includes(plot) ? plot : "model";
  const sentence: Record<string, string> = {
    normal: "Nothing unusual here.",
    abnormal: "The model sees unusual behaviour here. No failure is predicted within a day yet.",
    predicted_failure: "The model predicts a failure here within about a day.",
    no_data: "The sensors here are silent or stuck, so the real condition is unknown.",
  };
  const stuck = Object.entries(asset.sensors).filter(([, st]) => st !== "ok");
  return (
    <div>
      {picker}
      <div className="mb-3 rounded-lg border border-line p-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold">{zone ? `Zone ${zone.zone_id}` : "Drive (motor and gearbox)"}</h2>
            {zone && (
              <button className="text-sm text-ink2 underline-offset-2 hover:underline" onClick={() => onSelect({ kind: "section", section: zone.section })}>
                Section S{zone.section} · {zone.position_m.toFixed(0)} m along the belt · {zone.run} run
              </button>
            )}
          </div>
          <StatusChip asset={asset} />
        </div>
        <p className="mt-2 text-sm text-ink2">{sentence[asset.status]}</p>
        <dl className="tabular mt-3 grid grid-cols-2 gap-2 text-sm sm:max-w-md">
          <Stat label="Chance of failing within 24 h" value={pct(asset.p_fail_24h)} />
          <Stat label="Estimated time to failure" value={asset.est_hours_to_failure != null ? hours(asset.est_hours_to_failure) : "Not predicted"} />
        </dl>
        {zone && (
          <p className="mt-3 text-sm text-ink2">
            <span className="text-ink3">Sensors: </span>
            {stuck.length === 0 ? "all five are reporting normally."
              : stuck.map(([sig, st]) => `${SIGNAL_LABEL[sig].replace(/ \(.*\)/, "")} ${st === "stuck" ? "is stuck on one value" : "is silent"}`).join("; ") + "."}
          </p>
        )}
        {mine.map((a) => (
          <p key={a.alert_id} className="mt-3 rounded border border-line bg-raised p-2 text-sm">
            <span className="font-medium">Open alert ({a.severity_label}):</span> {a.message}
          </p>
        ))}
      </div>

      <div className="rounded-lg border border-line p-4">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
          <div className="flex flex-wrap gap-1" role="group" aria-label="What to plot">
            {["model", ...signals].map((sig) => (
              <button key={sig} onClick={() => setPlot(sig)} aria-pressed={shown === sig}
                className={`whitespace-nowrap rounded border px-2.5 py-1 text-sm ${shown === sig ? "border-ink bg-raised font-semibold" : "border-line text-ink2 hover:bg-raised"}`}>
                {sig === "model" ? "Model output" : SIGNAL_LABEL[sig].replace(/ \(.*\)/, "")}
              </button>
            ))}
          </div>
          <div className="flex gap-1" role="group" aria-label="Time range">
            {RANGES.map((r) => (
              <button key={r.h} onClick={() => setRange(r.h)} aria-pressed={range === r.h}
                className={`whitespace-nowrap rounded border px-2 py-1 text-xs ${range === r.h ? "border-ink bg-raised font-semibold" : "border-line text-ink2 hover:bg-raised"}`}>{r.label}</button>
            ))}
          </div>
        </div>
        {shown === "model"
          ? <PredictionChart asset={asset.asset} from={from} to={to} origin={origin} />
          : <SignalChart url={isDrive ? `/api/belt/series?signal=${shown}` : `/api/zones/${zone!.zone_id}/series?signal=${shown}`}
              signal={shown} from={from} to={to} origin={origin} />}
        <p className="mt-1 text-xs text-ink3">Readings are averaged for display. A gap in a line is missing data; it is never filled in.</p>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-line p-2">
      <dt className="text-xs text-ink3">{label}</dt>
      <dd className="text-lg font-semibold leading-tight">{value}</dd>
    </div>
  );
}

const axis = { tick: { fontSize: 10, fill: "var(--text-muted)" }, stroke: "var(--border)", tickLine: false } as const;
const tooltipStyle = {
  contentStyle: { background: "var(--surface-1)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, padding: "6px 8px" },
  labelStyle: { color: "var(--text-secondary)", marginBottom: 2 }, itemStyle: { color: "var(--text-primary)", padding: 0 },
} as const;

// Coarse 5-minute steps on the query window, so a poll every few seconds reuses the same request.
const snap = (ts: number) => ts - (ts % 300);

function SignalChart({ url, signal, from, to, origin }: { url: string; signal: string; from: number; to: number; origin: number | null }) {
  const f = snap(from), t = snap(to);
  const q = useQuery({
    queryKey: ["series", url, f, t], queryFn: () => get<Series>(`${url}&from=${f}&to=${t}`),
    refetchInterval: 5000, placeholderData: (prev) => prev,
  });
  const points = q.data?.points ?? [];
  return (
    <figure className="mb-1">
      <figcaption className="text-xs font-medium text-ink2">{SIGNAL_LABEL[signal]}</figcaption>
      <div className="h-[260px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid stroke="var(--grid)" vertical={false} />
            <XAxis dataKey="ts" type="number" domain={[f, t]} tickFormatter={(v) => simTime(v, origin)} {...axis} tickCount={5} minTickGap={50} />
            <YAxis width={44} domain={["auto", "auto"]} {...axis} tickCount={5} tickFormatter={(v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1))} />
            <Tooltip {...tooltipStyle} isAnimationActive={false} labelFormatter={(v) => simTime(v as number, origin)}
              formatter={(v: number) => [v == null ? "missing" : v.toFixed(2), SIGNAL_LABEL[signal]]} />
            <Line dataKey="value" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

function PredictionChart({ asset, from, to, origin }: { asset: string; from: number; to: number; origin: number | null }) {
  const f = snap(from), t = snap(to);
  const q = useQuery({
    queryKey: ["pred", asset, f, t], queryFn: () => get<Predictions>(`/api/predictions?asset=${asset}&from=${f}&to=${t}`),
    refetchInterval: 5000, placeholderData: (prev) => prev,
  });
  const points = q.data?.points ?? [];
  return (
    <figure className="mb-2">
      <figcaption className="flex flex-wrap items-center gap-x-4 text-xs font-medium text-ink2">
        Model output
        <span className="flex items-center gap-1 font-normal"><i className="inline-block h-0.5 w-4" style={{ background: "var(--series-1)" }} />How unusual it looks</span>
        <span className="flex items-center gap-1 font-normal"><i className="inline-block h-0.5 w-4" style={{ background: "var(--series-2)" }} />Chance of failing within 24 h</span>
      </figcaption>
      <div className="h-[260px]">
        {points.length === 0 ? (
          <div className="flex h-full items-center justify-center text-xs text-ink3">No predictions in this range yet. The model needs a day of history first.</div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={points} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="var(--grid)" vertical={false} />
              <XAxis dataKey="ts" type="number" domain={[f, t]} tickFormatter={(v) => simTime(v, origin)} {...axis} tickCount={5} minTickGap={50} />
              <YAxis width={44} domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={(v: number) => `${v * 100}%`} {...axis} />
              <Tooltip {...tooltipStyle} isAnimationActive={false} labelFormatter={(v) => simTime(v as number, origin)}
                formatter={(v: number, name) => [pct(v), name === "p_abnormal" ? "How unusual it looks" : "Chance of failing within 24 h"]} />
              <Line dataKey="p_abnormal" stroke="var(--series-1)" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line dataKey="p_fail_24h" stroke="var(--series-2)" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </figure>
  );
}
