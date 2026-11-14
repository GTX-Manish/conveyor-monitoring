import { useState } from "react";
import { Asset, Layout, STATUS_GLYPH, STATUS_LABEL, STATUS_VAR, Status, StatusReply, hours, pct } from "./api";
import type { Selection } from "./App";

// The loop drawn as a racetrack: carry run on top (zones 1 to 60, left to right), return run below
// (zones 61 to 120, right to left). One cell per 50 m zone, grouped into the 12 splice sections.
const W = 1280, CELL = 15, GAP = 2, SECTION_GAP = 9, H = 44, X0 = 78, Y_TOP = 62, Y_BOTTOM = 148;
const STATUSES: Status[] = ["normal", "abnormal", "predicted_failure", "no_data"];

function cellX(indexOnRun: number): number {           // 0..59 from the left
  return X0 + indexOnRun * (CELL + GAP) + Math.floor(indexOnRun / 10) * SECTION_GAP;
}
const RIGHT = cellX(59) + CELL;

export function RacetrackMap({ layout, status, selection, onSelect }: {
  layout: Layout; status: StatusReply; selection: Selection; onSelect: (s: Selection) => void;
}) {
  const [hover, setHover] = useState<{ asset: Asset; x: number; y: number } | null>(null);
  const byZone = new Map(status.assets.filter((a) => a.zone_id != null).map((a) => [a.zone_id as number, a]));
  const drive = status.assets.find((a) => a.asset === "DRIVE")!;
  const counts = Object.fromEntries(STATUSES.map((s) => [s, status.assets.filter((a) => a.status === s).length]));
  const selectedSection = selection?.kind === "section" ? selection.section
    : selection?.kind === "zone" ? layout.zones[selection.zone - 1].section : null;

  const place = (zone: number) => {
    const carry = zone <= 60;
    const i = carry ? zone - 1 : 120 - zone;            // return run is drawn right to left
    return { x: cellX(i), y: carry ? Y_TOP : Y_BOTTOM };
  };

  return (
    <section className="rounded-lg border border-line p-3" aria-label="Belt map">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-semibold">Belt map: 6 km loop, 120 zones, 12 sections</h2>
        <ul className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink2">
          {STATUSES.map((s) => (
            <li key={s} className="flex items-center gap-1.5">
              <svg width="14" height="14" aria-hidden><Swatch status={s} x={0} y={0} w={14} h={14} /></svg>
              {STATUS_LABEL[s]} <span className="tabular text-ink3">{counts[s]}</span>
            </li>
          ))}
        </ul>
      </div>
      <div className="relative overflow-x-auto">
        <svg viewBox={`0 0 ${W} 232`} className="min-w-[900px]" role="img" aria-label="Status of every zone on the loop">
          <defs>
            <pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
              <rect width="5" height="5" fill="var(--surface-2)" />
              <rect width="2.2" height="5" fill="var(--st-nodata)" />
            </pattern>
          </defs>
          {/* the belt itself, behind the cells */}
          <rect x={X0 - 22} y={Y_TOP + H / 2} width={RIGHT - X0 + 44} height={Y_BOTTOM - Y_TOP} rx={(Y_BOTTOM - Y_TOP) / 2}
            fill="none" stroke="var(--border)" strokeWidth={1.5} />
          <text x={X0 - 30} y={Y_TOP + H / 2 + 4} textAnchor="end" className="fill-ink2 text-[11px]">Carry →</text>
          <text x={X0 - 30} y={Y_BOTTOM + H / 2 + 4} textAnchor="end" className="fill-ink2 text-[11px]">← Return</text>
          <text x={X0} y={18} className="fill-ink3 text-[10px]">Loading (0 m)</text>
          <text x={RIGHT} y={18} textAnchor="end" className="fill-ink3 text-[10px]">Unloading (3000 m)</text>

          {layout.sections.map((sec) => {
            const first = place(sec.zones[0]), last = place(sec.zones[sec.zones.length - 1]);
            const x = Math.min(first.x, last.x), carry = sec.run === "carry";
            const width = 10 * CELL + 9 * GAP, active = selectedSection === sec.section;
            return (
              <g key={sec.section} onClick={() => onSelect({ kind: "section", section: sec.section })} className="cursor-pointer">
                <rect x={x - 3} y={carry ? Y_TOP - 24 : Y_BOTTOM + H + 4} width={width + 6} height={20} rx={4}
                  fill={active ? "var(--surface-2)" : "transparent"} stroke={active ? "var(--text-secondary)" : "transparent"} />
                <text x={x + width / 2} y={carry ? Y_TOP - 10 : Y_BOTTOM + H + 18} textAnchor="middle"
                  className={`text-[11px] ${active ? "fill-ink font-semibold" : "fill-ink2"}`}>
                  {sec.name} · zones {Math.min(...sec.zones)}–{Math.max(...sec.zones)}
                </text>
              </g>
            );
          })}

          {layout.zones.map((z) => {
            const a = byZone.get(z.zone_id);
            if (!a) return null;
            const { x, y } = place(z.zone_id);
            const selected = selection?.kind === "zone" && selection.zone === z.zone_id;
            return (
              <g key={z.zone_id} onClick={() => onSelect({ kind: "zone", zone: z.zone_id })} className="cursor-pointer"
                onMouseEnter={() => setHover({ asset: a, x: x + CELL / 2, y })} onMouseLeave={() => setHover(null)}>
                <Swatch status={a.status} x={x} y={y} w={CELL} h={H} />
                {selected && <rect x={x - 2} y={y - 2} width={CELL + 4} height={H + 4} rx={3} fill="none" stroke="var(--text-primary)" strokeWidth={2} />}
                <rect x={x - GAP / 2} y={y - 4} width={CELL + GAP} height={H + 8} fill="transparent" />
              </g>
            );
          })}

          {/* the drive sits at the head (unloading) end */}
          <g onClick={() => onSelect({ kind: "drive" })} className="cursor-pointer"
            onMouseEnter={() => setHover({ asset: drive, x: RIGHT + 62, y: 96 })} onMouseLeave={() => setHover(null)}>
            <Swatch status={drive.status} x={RIGHT + 30} y={96} w={64} h={40} label="Drive" />
            {selection?.kind === "drive" && <rect x={RIGHT + 28} y={94} width={68} height={44} rx={5} fill="none" stroke="var(--text-primary)" strokeWidth={2} />}
          </g>
        </svg>
        {hover && <MapTooltip hover={hover} layout={layout} />}
      </div>
    </section>
  );
}

/** One status mark. Status is never colour alone: abnormal carries a dot, predicted failure a "!", and
 *  no data is hatched grey. Normal is drawn quieter so the exceptions stand out. */
function Swatch({ status, x, y, w, h, label }: { status: Status; x: number; y: number; w: number; h: number; label?: string }) {
  const fill = status === "no_data" ? "url(#hatch)" : STATUS_VAR[status];
  const onDark = status === "predicted_failure";
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={3} fill={status === "normal" ? "var(--st-normal-quiet)" : fill} />
      {label ? (
        <text x={x + w / 2} y={y + h / 2 + 4} textAnchor="middle" className="text-[12px] font-semibold"
          fill={onDark ? "#fff" : "var(--text-primary)"}>{label}{STATUS_GLYPH[status] ? ` ${STATUS_GLYPH[status]}` : ""}</text>
      ) : (
        STATUS_GLYPH[status] && (
          <text x={x + w / 2} y={y + h / 2 + (h > 20 ? 5 : 4)} textAnchor="middle" className="text-[13px] font-bold"
            fill={onDark ? "#fff" : "#0b0b0b"}>{STATUS_GLYPH[status]}</text>
        )
      )}
    </g>
  );
}

function MapTooltip({ hover, layout }: { hover: { asset: Asset; x: number; y: number }; layout: Layout }) {
  const a = hover.asset;
  const z = a.zone_id ? layout.zones[a.zone_id - 1] : null;
  const bad = Object.entries(a.sensors).filter(([, v]) => v !== "ok");
  return (
    <div className="pointer-events-none absolute z-20 w-60 rounded-md border border-line bg-surface p-2.5 text-xs shadow-lg"
      style={{ left: `min(calc(${(hover.x / W) * 100}% + 12px), calc(100% - 15.5rem))`, top: hover.y > 100 ? 8 : 118 }}>
      <div className="mb-1 text-sm font-semibold">{z ? `Zone ${z.zone_id}` : "Drive"}</div>
      {z && <div className="text-ink2">Section S{z.section} · {z.position_m.toFixed(0)} m · {z.run} run</div>}
      <div className="mt-1.5 flex items-center gap-1.5 font-medium">
        <svg width="12" height="12" aria-hidden><Swatch status={a.status} x={0} y={0} w={12} h={12} /></svg>{STATUS_LABEL[a.status]}
      </div>
      {a.p_abnormal != null && (
        <dl className="tabular mt-1.5 grid grid-cols-[1fr_auto] gap-x-3 text-ink2">
          <dt>Abnormal score</dt><dd className="text-ink">{pct(a.p_abnormal)}</dd>
          <dt>Fails within 24 h</dt><dd className="text-ink">{pct(a.p_fail_24h)}</dd>
          {a.est_hours_to_failure != null && (<><dt>Estimated time to failure</dt><dd className="text-ink">{hours(a.est_hours_to_failure)}</dd></>)}
        </dl>
      )}
      {bad.length > 0 && <div className="mt-1.5 text-ink2">{a.data_quality === "no_data" ? "All five sensors silent" : `Stuck: ${bad.map(([k]) => k.replace(/_/g, " ")).join(", ")}`}</div>}
    </div>
  );
}
