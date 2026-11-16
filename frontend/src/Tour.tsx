import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import type { TabName } from "./App";

// A short guided walkthrough. Each step points at one part of the page (found by its data-tour attribute),
// switches to the tab it lives on, and explains it in a sentence or two.
interface Step { tab: TabName; target?: string; title: string; text: string }

export const STEPS: Step[] = [
  { tab: "Overview", title: "Welcome",
    text: "This dashboard watches a 6 km conveyor belt and warns before parts fail. The data comes from a simulator and is replayed fast, so days pass in minutes. This tour takes about a minute." },
  { tab: "Overview", target: "clock", title: "Time and playback",
    text: "This is the simulated time. Use Pause and Play to stop and resume the replay, and 1× or 3× to change its speed." },
  { tab: "Overview", target: "tabs", title: "One tab per question",
    text: "Overview: is anything wrong? Alerts: what exactly? Predictions: what will fail, and when? Track record: were the predictions right? Zone details: show me one zone. Data health: can I trust the sensors?" },
  { tab: "Overview", target: "map", title: "The belt map",
    text: "Each block is a 50 m stretch of belt with its own sensors (a zone). The top row carries material from loading to unloading; the bottom row is the belt returning. Click any block to open that zone." },
  { tab: "Overview", target: "legend", title: "Four statuses",
    text: "Green is normal. Amber with a dot means the model sees unusual behaviour. Red with ! means a failure is predicted within 72 hours. Grey stripes with ? mean the sensors are silent or stuck, so the status is unknown, not normal." },
  { tab: "Overview", target: "accuracy", title: "Is the model any good?",
    text: "The model's record in this run so far: of the parts that failed, how many it had warned about (recall), and of its alerts, how many were real (precision). It fills in as failures happen and are repaired. The line below it gives the result of a fresh test, on scenarios never used for training, tuning or diagnosis." },
  { tab: "Overview", target: "attention", title: "What needs attention",
    text: "The few things worth looking at right now, most serious first. Click one to open its zone. When this list is empty, the belt is healthy." },
  { tab: "Alerts", target: "panel", title: "Alerts",
    text: "One card per problem, with where it is, how serious it is and when it started. Each model alert lists what the model saw that made it raise the alert." },
  { tab: "Predictions", target: "panel", title: "Predictions",
    text: "What the model expects: which part, where on the belt, its failure risk score, and the estimated time left when a failure is predicted." },
  { tab: "Track record", target: "panel", title: "Track record",
    text: "Were the predictions right? After a part is repaired, a repair record says when it really failed. This tab sets that against what the model had said: warned in time, missed, or a false alarm." },
  { tab: "Zone details", target: "panel", title: "Zone details",
    text: "Everything about one zone or the drive: its status, its sensors, and a chart. Pick what to plot with the buttons above the chart; gaps in a line are missing data." },
  { tab: "Data health", target: "panel", title: "Data health",
    text: "Real sensor data is messy. This tab counts the bad input that was caught (missing, impossible or duplicate values) and lists sensors that are silent or stuck." },
  { tab: "Vision", target: "panel", title: "Vision",
    text: "A camera-based check of the belt surface will appear here. It is not part of this run. You can reopen this tour any time with the “Tour” button at the top." },
];

const CARD_W = 360;

export function Tour({ step, onStep, onClose, setTab }: {
  step: number; onStep: (n: number) => void; onClose: () => void; setTab: (t: TabName) => void;
}) {
  const s = STEPS[step];
  const [rect, setRect] = useState<DOMRect | null>(null);

  const measure = useCallback(() => {
    const el = s.target ? document.querySelector<HTMLElement>(`[data-tour="${s.target}"]`) : null;
    setRect(el ? el.getBoundingClientRect() : null);
  }, [s.target]);

  useEffect(() => { setTab(s.tab); }, [s.tab, setTab]);
  useLayoutEffect(() => {
    const el = s.target ? document.querySelector<HTMLElement>(`[data-tour="${s.target}"]`) : null;
    el?.scrollIntoView({ block: "nearest" });
    measure();
    const id = window.setInterval(measure, 300);        // the page keeps updating under the tour
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, true);
    return () => { window.clearInterval(id); window.removeEventListener("resize", measure); window.removeEventListener("scroll", measure, true); };
  }, [measure, s.target, s.tab, step]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight" && step < STEPS.length - 1) onStep(step + 1);
      if (e.key === "ArrowLeft" && step > 0) onStep(step - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [step, onStep, onClose]);

  const vw = window.innerWidth, vh = window.innerHeight, pad = 6;
  const hole = rect && { left: rect.left - pad, top: rect.top - pad, width: rect.width + 2 * pad, height: Math.min(rect.height, vh * 0.6) + 2 * pad };
  // Card goes under the highlighted part if there is room, otherwise above it; centred when there is no target.
  let style: React.CSSProperties = { left: (vw - CARD_W) / 2, top: vh * 0.28 };
  if (hole) {
    const left = Math.max(12, Math.min(hole.left, vw - CARD_W - 12));
    const below = hole.top + hole.height + 12;
    style = below + 210 < vh ? { left, top: below } : { left, bottom: Math.max(12, vh - hole.top + 12) };
  }
  const last = step === STEPS.length - 1;
  return (
    <div className="fixed inset-0 z-50" role="dialog" aria-modal="true" aria-label={`Tour step ${step + 1}: ${s.title}`}>
      {hole
        ? <div className="pointer-events-none absolute rounded-lg ring-2 ring-white" style={{ ...hole, boxShadow: "0 0 0 9999px rgba(0,0,0,0.55)" }} />
        : <div className="absolute inset-0" style={{ background: "rgba(0,0,0,0.55)" }} />}
      <div className="absolute rounded-lg border border-line bg-surface p-4 shadow-xl" style={{ width: Math.min(CARD_W, vw - 24), ...style }}>
        <div className="mb-1 flex items-baseline justify-between gap-3">
          <h2 className="text-base font-semibold">{s.title}</h2>
          <span className="tabular whitespace-nowrap text-xs text-ink3">Step {step + 1} of {STEPS.length}</span>
        </div>
        <p className="text-sm leading-relaxed text-ink2">{s.text}</p>
        <div className="mt-4 flex items-center justify-between">
          <button onClick={onClose} className="text-sm text-ink3 underline-offset-2 hover:underline">{last ? "Close" : "Skip tour"}</button>
          <div className="flex gap-2">
            <button onClick={() => onStep(step - 1)} disabled={step === 0}
              className="rounded border border-line px-3 py-1.5 text-sm hover:bg-raised disabled:opacity-40">Previous</button>
            <button onClick={() => (last ? onClose() : onStep(step + 1))} autoFocus
              className="rounded border border-ink bg-ink px-3 py-1.5 text-sm font-medium text-surface">{last ? "Finish" : "Next"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
