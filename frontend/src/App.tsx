import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useCallback, useState } from "react";
import {
  Alert, FreshTest, IngestStats, Layout, Outcomes, SEVERITY_VAR, SensorHealth, StatusReply, assetName, freshTestSentence, get, hours, post,
  simTime,
} from "./api";
import { Detail } from "./Detail";
import { RacetrackMap } from "./RacetrackMap";
import { ALERT_TITLE, AnomaliesTab, DataHealthTab, PredictionsTab, TrackRecordTab, VisionTab } from "./Tabs";
import { STEPS, Tour } from "./Tour";

export type Selection = { kind: "zone"; zone: number } | { kind: "section"; section: number } | { kind: "drive" } | null;

// One tab per question a user has. Only one is on screen at a time.
export const TABS = ["Overview", "Alerts", "Predictions", "Track record", "Zone details", "Data health", "Vision"] as const;
export type TabName = (typeof TABS)[number];
const TAB_HINT: Record<TabName, string> = {
  Overview: "Is anything wrong?", Alerts: "What exactly is wrong?", Predictions: "What will fail, and when?",
  "Track record": "Were the predictions right?", "Zone details": "Show me one zone", "Data health": "Can I trust the sensors?", Vision: "Belt surface cameras",
};
const TOUR_SEEN = "conveyor-tour-seen";

function seenTour(): boolean {
  try { return window.localStorage.getItem(TOUR_SEEN) === "1"; } catch { return false; }
}

export default function App() {
  const qc = useQueryClient();
  // Deep links: ?tab=alerts picks a tab; ?zone=107, ?section=7 or ?drive=1 open Zone details; ?tour=3 opens
  // the tour at step 3 and ?tour=0 keeps it closed.
  const params = new URLSearchParams(window.location.search);
  const linked = params.get("zone") || params.get("section") || params.get("drive");
  const [selection, setSelection] = useState<Selection>(() =>
    params.get("zone") ? { kind: "zone", zone: Math.min(120, Math.max(1, Number(params.get("zone")) || 1)) }
      : params.get("section") ? { kind: "section", section: Math.min(12, Math.max(1, Number(params.get("section")) || 1)) }
        : params.get("drive") ? { kind: "drive" } : null);
  const [tab, setTab] = useState<TabName>(() =>
    TABS.find((t) => params.get("tab") && t.toLowerCase().startsWith(params.get("tab")!.toLowerCase())) ?? (linked ? "Zone details" : "Overview"));
  const [tourStep, setTourStep] = useState<number | null>(() => {
    const t = params.get("tour");
    if (t != null) return Number(t) >= 1 ? Math.min(STEPS.length, Number(t)) - 1 : null;
    return seenTour() || linked || params.get("tab") ? null : 0;
  });

  const layout = useQuery({ queryKey: ["layout"], queryFn: () => get<Layout>("/api/layout"), refetchInterval: false, staleTime: Infinity });
  const status = useQuery({ queryKey: ["status"], queryFn: () => get<StatusReply>("/api/status") });
  const alerts = useQuery({ queryKey: ["alerts"], queryFn: () => get<{ alerts: Alert[] }>("/api/alerts?state=all") });
  const stats = useQuery({ queryKey: ["ingest"], queryFn: () => get<IngestStats>("/api/ingest/stats"), enabled: tab === "Data health" });
  const health = useQuery({ queryKey: ["sensors"], queryFn: () => get<SensorHealth>("/api/sensors/health"), enabled: tab === "Data health" });

  // "Day 1" is the day the run starts on.
  const start = status.data?.start_ts ?? null;
  const origin = start == null ? null : start - (start % 86400);

  const select = (s: Selection) => { setSelection(s); if (s) setTab("Zone details"); };
  const selectAsset = (asset: string) =>
    select(asset === "DRIVE" ? { kind: "drive" } : { kind: "zone", zone: parseInt(asset.slice(1), 10) });
  const control = async (body: { action?: string; speed?: number }) => {
    await post("/api/replay", body);
    qc.invalidateQueries({ queryKey: ["status"] });
  };
  const closeTour = useCallback(() => {
    setTourStep(null);
    try { window.localStorage.setItem(TOUR_SEEN, "1"); } catch { /* private window: the tour just shows again next time */ }
  }, []);

  if (status.isError && !status.data)
    return <Centered>The backend is not answering at /api. Start it with <code>make demo</code>.</Centered>;
  if (!layout.data || !status.data) return <Centered>Loading…</Centered>;

  const s = status.data;
  const open = (alerts.data?.alerts ?? []).filter((a) => a.state === "open");
  return (
    <div className="mx-auto max-w-[1280px] px-4 pb-10">
      <TopBar status={s} origin={origin} onControl={control} onTour={() => setTourStep(0)} />
      <nav role="tablist" aria-label="Views" data-tour="tabs" className="mb-4 flex flex-wrap gap-1 border-b border-line">
        {TABS.map((t) => (
          <button key={t} role="tab" aria-selected={tab === t} onClick={() => setTab(t)} title={TAB_HINT[t]}
            className={`-mb-px border-b-2 px-4 py-2.5 text-sm ${tab === t ? "border-ink font-semibold text-ink" : "border-transparent text-ink2 hover:text-ink"}`}>
            {t}
            {t === "Alerts" && open.length > 0 && <span className="tabular ml-1.5 rounded-full bg-raised px-1.5 py-0.5 text-xs">{open.length}</span>}
          </button>
        ))}
      </nav>

      <main data-tour={tab === "Overview" ? undefined : "panel"}>
        <p className="mb-3 text-sm text-ink2">{TAB_HINT[tab]}</p>
        {tab === "Overview" && <Overview layout={layout.data} status={s} open={open} origin={origin} selection={selection}
          onSelect={select} onSelectAsset={selectAsset} onSeeAll={() => setTab("Alerts")} onTrackRecord={() => setTab("Track record")} />}
        {tab === "Alerts" && <AnomaliesTab alerts={alerts.data?.alerts ?? []} origin={origin} onSelect={selectAsset} />}
        {tab === "Predictions" && <PredictionsTab status={s} layout={layout.data} origin={origin} onSelect={selectAsset} />}
        {tab === "Track record" && <TrackRecordTab origin={origin} onSelect={selectAsset} />}
        {tab === "Zone details" && <Detail layout={layout.data} status={s} alerts={open} selection={selection} onSelect={setSelection} origin={origin} />}
        {tab === "Data health" && <DataHealthTab stats={stats.data} health={health.data} origin={origin} onSelect={selectAsset} />}
        {tab === "Vision" && <VisionTab />}
      </main>

      <p className="mt-8 text-xs text-ink3">
        All data comes from a simulator. Times are simulated time. {s.model_version ? `Model ${s.model_version}.` : "No model is loaded."}
      </p>
      {tourStep != null && <Tour step={tourStep} onStep={setTourStep} onClose={closeTour} setTab={setTab} />}
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return <div className="flex h-screen items-center justify-center p-8 text-center text-ink2">{children}</div>;
}

function TopBar({ status, origin, onControl, onTour }: {
  status: StatusReply; origin: number | null; onControl: (b: { action?: string; speed?: number }) => void; onTour: () => void;
}) {
  const { state, speed } = status.replay;
  const playing = state === "playing";
  const label: Record<string, string> = { idle: "Waiting for data", playing: "Playing", paused: "Paused", finished: "Replay finished" };
  return (
    <header className="sticky top-0 z-10 flex flex-wrap items-center gap-x-6 gap-y-2 bg-surface py-3">
      <h1 className="text-base font-semibold">Conveyor monitoring</h1>
      <div data-tour="clock" className="flex flex-wrap items-center gap-x-4 gap-y-2">
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
      </div>
      {!status.scoring && status.model_version && (
        <div className="text-sm text-ink2">
          Collecting a day of history before predictions start:{" "}
          <span className="tabular font-medium text-ink">{Math.round((100 * status.windows_of_history) / status.windows_needed)}%</span>
        </div>
      )}
      <button onClick={onTour} className="ml-auto rounded border border-line px-3 py-1.5 text-sm hover:bg-raised">? Tour</button>
    </header>
  );
}

function Overview({ layout, status, open, origin, selection, onSelect, onSelectAsset, onSeeAll, onTrackRecord }: {
  layout: Layout; status: StatusReply; open: Alert[]; origin: number | null; selection: Selection;
  onSelect: (s: Selection) => void; onSelectAsset: (asset: string) => void; onSeeAll: () => void; onTrackRecord: () => void;
}) {
  const top = open.slice(0, 5);
  const headline = open.length === 0 ? "Everything looks normal."
    : `${open.length} ${open.length === 1 ? "thing needs" : "things need"} attention.`;
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-semibold">{headline}</h2>
      <RacetrackMap layout={layout} status={status} selection={selection} onSelect={onSelect} />
      <Accuracy onOpen={onTrackRecord} />
      <section data-tour="attention" className="rounded-lg border border-line p-4">
        <div className="mb-2 flex items-baseline justify-between">
          <h3 className="text-sm font-semibold">Needs attention</h3>
          {open.length > 0 && <button onClick={onSeeAll} className="text-sm text-ink2 underline underline-offset-2 hover:text-ink">See all alerts</button>}
        </div>
        {top.length === 0 ? (
          <p className="text-sm text-ink3">Nothing right now. Problems appear here as soon as the model or a sensor check finds one.</p>
        ) : (
          <ul>
            {top.map((a) => (
              <li key={a.alert_id} onClick={() => onSelectAsset(a.asset)}
                className="flex cursor-pointer flex-wrap items-center gap-x-3 gap-y-0.5 border-t border-line py-2 text-sm first:border-t-0 hover:bg-raised">
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: SEVERITY_VAR[a.severity_label] }} />
                <span className="font-medium">{ALERT_TITLE[a.type] ?? a.type}</span>
                <span className="text-ink2">{assetName(a.asset)}{a.position_m != null ? `, ${a.position_m.toFixed(0)} m along the belt` : ""}</span>
                <span className="ml-auto text-ink3">
                  {a.eta_hours != null ? <span className="font-medium text-ink">fails in about {hours(a.eta_hours)} · </span> : null}
                  <span className="capitalize">{a.severity_label}</span> · since {simTime(a.created_ts, origin)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

const share = (n: number, of: number) => (of > 0 ? `${Math.round((100 * n) / of)}%` : "–");

/** The model's record in this run, set against what really happened, plus the offline test figures. */
function Accuracy({ onOpen }: { onOpen: () => void }) {
  const live = useQuery({ queryKey: ["outcomes"], queryFn: () => get<Outcomes>("/api/outcomes"), refetchInterval: 4000 });
  const fresh = useQuery({
    queryKey: ["fresh-test"], queryFn: () => get<FreshTest>("/api/model/fresh-test"), refetchInterval: false, retry: false, staleTime: Infinity,
  });
  const s = live.data?.summary;
  const real = s ? s.warned + s.early_warnings : 0;
  const settled = s ? real + s.false_alarms : 0;
  const tiles: [string, string, string][] = s ? [
    ["Failures caught (recall)", share(s.warned, s.failures),
      s.failures ? `${s.warned} of ${s.failures} failed parts had an alert open` : "no part has failed yet"],
    ["Alerts that were real (precision)", share(real, settled),
      settled ? `${real} of ${settled} settled alerts were followed by a failure` : "no alert has been settled yet"],
    [`Failure predictions that came true`, share(s.failure_calls_came_true, s.failure_calls),
      s.failure_calls ? `${s.failure_calls_came_true} of ${s.failure_calls} failed within ${s.call_horizon_hours} h` : "no prediction has been settled yet"],
  ] : [];
  const f = fresh.data;
  return (
    <section data-tour="accuracy" className="rounded-lg border border-line p-4">
      <div className="mb-2 flex items-baseline justify-between">
        <h3 className="text-sm font-semibold">Model accuracy so far in this run</h3>
        <button onClick={onOpen} className="text-sm text-ink2 underline underline-offset-2 hover:text-ink">See each case</button>
      </div>
      <dl className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        {tiles.map(([label, value, note]) => (
          <div key={label} className="rounded border border-line p-3">
            <dt className="text-xs text-ink3">{label}</dt>
            <dd className="tabular text-2xl font-semibold leading-tight">{value}</dd>
            <dd className="text-xs text-ink3">{note}</dd>
          </div>
        ))}
      </dl>
      <p className="mt-2 text-xs text-ink3">
        Checked against repair records, which arrive after each repair. {s && s.failures > 0 ? `This rests on ${s.failures} failure${s.failures === 1 ? "" : "s"} in one demo scenario. ` : ""}
        {f && freshTestSentence(f)}
      </p>
    </section>
  );
}
