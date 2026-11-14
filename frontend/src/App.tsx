import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  Alert, IngestStats, Layout, SensorHealth, StatusReply, assetName, count, get, post, simTime,
} from "./api";
import { Detail } from "./Detail";
import { RacetrackMap } from "./RacetrackMap";
import { AnomaliesTab, DataHealthTab, PredictionsTab, VisionTab } from "./Tabs";

export type Selection = { kind: "zone"; zone: number } | { kind: "section"; section: number } | { kind: "drive" } | null;

const TABS = ["Active anomalies", "Model predictions", "Data health", "Vision"] as const;

export default function App() {
  const qc = useQueryClient();
  // Deep links: ?zone=107, ?section=7 or ?drive=1 open the drill-down; ?tab=health picks a tab.
  const params = new URLSearchParams(window.location.search);
  const [selection, setSelection] = useState<Selection>(() =>
    params.get("zone") ? { kind: "zone", zone: Math.min(120, Math.max(1, Number(params.get("zone")) || 1)) }
      : params.get("section") ? { kind: "section", section: Math.min(12, Math.max(1, Number(params.get("section")) || 1)) }
        : params.get("drive") ? { kind: "drive" } : null);
  const [tab, setTab] = useState<(typeof TABS)[number]>(
    () => TABS.find((t) => t.toLowerCase().includes((params.get("tab") ?? "active").toLowerCase())) ?? "Active anomalies");
  const layout = useQuery({ queryKey: ["layout"], queryFn: () => get<Layout>("/api/layout"), refetchInterval: false, staleTime: Infinity });
  const status = useQuery({ queryKey: ["status"], queryFn: () => get<StatusReply>("/api/status") });
  const alerts = useQuery({ queryKey: ["alerts"], queryFn: () => get<{ alerts: Alert[] }>("/api/alerts?state=all") });
  const stats = useQuery({ queryKey: ["ingest"], queryFn: () => get<IngestStats>("/api/ingest/stats") });
  const health = useQuery({ queryKey: ["sensors"], queryFn: () => get<SensorHealth>("/api/sensors/health") });

  // "Day 1" is the day the run starts on.
  const start = status.data?.start_ts ?? null;
  const origin = start == null ? null : start - (start % 86400);

  const selectAsset = (asset: string) =>
    setSelection(asset === "DRIVE" ? { kind: "drive" } : { kind: "zone", zone: parseInt(asset.slice(1), 10) });
  const control = async (body: { action?: string; speed?: number }) => {
    await post("/api/replay", body);
    qc.invalidateQueries({ queryKey: ["status"] });
  };

  if (status.isError && !status.data)
    return <Centered>The backend is not answering at /api. Start it with <code>make demo</code>.</Centered>;
  if (!layout.data || !status.data) return <Centered>Loading…</Centered>;

  const s = status.data;
  const open = (alerts.data?.alerts ?? []).filter((a) => a.state === "open");
  return (
    <div className="mx-auto max-w-[1400px] px-4 pb-10">
      <TopBar status={s} stats={stats.data} origin={origin} onControl={control} />
      <RacetrackMap layout={layout.data} status={s} selection={selection} onSelect={setSelection} />
      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <section className="rounded-lg border border-line p-4">
          <Detail layout={layout.data} status={s} alerts={open} selection={selection} onSelect={setSelection} origin={origin} />
        </section>
        <section className="rounded-lg border border-line p-4">
          <div role="tablist" className="mb-3 flex flex-wrap gap-1 border-b border-line">
            {TABS.map((t) => (
              <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
                className={`-mb-px border-b-2 px-3 py-2 text-sm ${tab === t ? "border-ink font-semibold text-ink" : "border-transparent text-ink2 hover:text-ink"}`}>
                {t}
                {t === "Active anomalies" && open.length > 0 && (
                  <span className="tabular ml-1.5 rounded-full bg-raised px-1.5 py-0.5 text-xs">{open.length}</span>
                )}
              </button>
            ))}
          </div>
          {tab === "Active anomalies" && <AnomaliesTab alerts={alerts.data?.alerts ?? []} origin={origin} onSelect={selectAsset} />}
          {tab === "Model predictions" && <PredictionsTab status={s} layout={layout.data} origin={origin} onSelect={selectAsset} />}
          {tab === "Data health" && <DataHealthTab stats={stats.data} health={health.data} origin={origin} onSelect={selectAsset} />}
          {tab === "Vision" && <VisionTab />}
        </section>
      </div>
      <p className="mt-6 text-xs text-ink3">
        All data comes from a simulator. Times are simulated time. {s.model_version ? `Model ${s.model_version}.` : "No model is loaded."}
        {selection?.kind === "zone" ? ` Showing ${assetName(`Z${String(selection.zone).padStart(3, "0")}`)}.` : ""}
      </p>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex h-screen items-center justify-center p-8 text-center text-ink2">{children}</div>;
}

function TopBar({ status, stats, origin, onControl }: {
  status: StatusReply; stats?: IngestStats; origin: number | null; onControl: (b: { action?: string; speed?: number }) => void;
}) {
  const { state, speed } = status.replay;
  const playing = state === "playing";
  const t = stats?.totals;
  const problems = t ? t.rejected_rows + t.out_of_range + t.duplicates : 0;
  const label: Record<string, string> = { idle: "Waiting for data", playing: "Playing", paused: "Paused", finished: "Replay finished" };
  return (
    <header className="sticky top-0 z-10 mb-3 flex flex-wrap items-center gap-x-6 gap-y-2 border-b border-line bg-surface py-3">
      <h1 className="text-base font-semibold">Conveyor monitoring</h1>
      <div>
        <div className="text-xs text-ink3">Simulated time</div>
        <div className="tabular text-lg font-semibold leading-tight">{simTime(status.sim_ts, origin)}</div>
      </div>
      <div className="flex items-center gap-1.5">
        <button onClick={() => onControl({ action: playing ? "pause" : "play" })} disabled={state === "finished" || state === "idle"}
          className="rounded border border-line px-3 py-1.5 text-sm font-medium hover:bg-raised disabled:opacity-40">
          {playing ? "❚❚ Pause" : "▶ Play"}
        </button>
        {[1, 3].map((v) => (
          <button key={v} onClick={() => onControl({ speed: v })} aria-pressed={speed === v} title={`${v} simulated hour${v > 1 ? "s" : ""} per second`}
            className={`tabular rounded border px-2.5 py-1.5 text-sm ${speed === v ? "border-ink bg-raised font-semibold" : "border-line hover:bg-raised"}`}>
            {v}×
          </button>
        ))}
        <span className="ml-1 text-sm text-ink2">{label[state] ?? state}</span>
      </div>
      {!status.scoring && status.model_version && (
        <div className="text-sm text-ink2">
          Building 24 h of history before the model scores:{" "}
          <span className="tabular font-medium text-ink">{status.windows_of_history}/{status.windows_needed}</span> windows
        </div>
      )}
      <div className="ml-auto text-right">
        <div className="text-xs text-ink3">Ingest</div>
        <div className="tabular text-sm">
          {count(t?.rows)} rows · <span className={problems ? "font-medium" : ""}>{count(problems)} bad inputs handled</span>
        </div>
      </div>
    </header>
  );
}
