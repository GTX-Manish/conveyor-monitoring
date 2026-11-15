"""HTTP API. Reads go to SQLite through short read-only connections; every write goes through Engine.submit.

    python -m backend.app            # settings from configs/app.yaml
"""
import json
import logging
from contextlib import asynccontextmanager, closing
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common.ids import ASSETS, BELT_SIGNALS, ZONE_SIGNALS, asset_zone

from .engine import Engine, connect

REPO = Path(__file__).resolve().parents[1]
FRONTEND = REPO / "frontend" / "dist"
MAX_POINTS = 500
log = logging.getLogger("backend")


def load_settings(path=REPO / "configs" / "app.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_layout(scenario_dir: Path) -> list:
    """Zone facts from the scenario's public layout table (the only scenario file the backend opens)."""
    df = pd.read_parquet(Path(scenario_dir) / "public" / "layout.parquet")
    return [{"zone_id": int(r.global_zone), "section": int(r.splice_section), "zone_in_section": int(r.zone_in_section),
             "position_m": float(r.position_m), "run": str(r.run), "is_loading": bool(r.is_loading_zone),
             "is_unloading": bool(r.is_unloading_zone), "incline_deg": float(r.incline_deg),
             "curve_radius_m": None if pd.isna(r.curve_radius_m) else float(r.curve_radius_m)}
            for r in df.itertuples()]


def load_models():
    try:
        from model.predict import Models
        return Models.load()
    except Exception as exc:  # the system still ingests and tracks sensor health without a model
        log.warning("no model loaded (%s); live scoring is off", exc)
        return None


class IngestBody(BaseModel):
    # Rows are checked one by one in the engine, so one bad row is rejected and counted, not the whole batch.
    batch: list[Any] = Field(default_factory=list)
    belt: Optional[dict] = None


class ReplayBody(BaseModel):
    action: Optional[str] = None      # 'play' | 'pause' | 'finished'
    speed: Optional[float] = None     # simulated hours per real second


def create_app(settings: Optional[dict] = None, engine: Optional[Engine] = None) -> FastAPI:
    settings = settings or load_settings()
    server = settings["server"]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is None:
            scenario_dir = REPO / server["data_dir"] / server["scenario"]
            app.state.engine = Engine(REPO / server["db_path"], load_layout(scenario_dir), settings,
                                      load_models(), server["scenario"])
        else:
            app.state.engine = engine
        app.state.engine.start()
        yield
        app.state.engine.stop()

    app = FastAPI(title="Conveyor monitoring", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=server["cors_origins"], allow_methods=["*"], allow_headers=["*"])

    def eng() -> Engine:
        return app.state.engine

    def query(sql: str, params=()) -> list:
        with closing(connect(eng().db_path, read_only=True)) as con:
            return [dict(r) for r in con.execute(sql, params).fetchall()]

    # ---------------------------------------------------------------- write side
    @app.post("/api/ingest")
    async def ingest(body: IngestBody):
        # Processed here, on the server's own thread, before the reply is sent: the sender cannot outrun the
        # engine, and no second thread competes for the interpreter (see Engine.submit).
        e = eng()
        e.submit("ingest", {"batch": body.batch, "belt": body.belt})
        return {"accepted": True, "replay": e.replay}

    @app.post("/api/replay")
    def set_replay(body: ReplayBody):
        e = eng()
        change = {}
        if body.speed is not None:
            if not 0 < body.speed <= settings["replay"]["max_speed"]:
                raise HTTPException(422, f"speed must be above 0 and at most {settings['replay']['max_speed']} "
                                         "simulated hours per second")
            change["speed"] = body.speed
        if body.action is not None:
            states = {"play": "playing", "pause": "paused", "finished": "finished"}
            if body.action not in states:
                raise HTTPException(422, "action must be play, pause or finished")
            change["state"] = states[body.action]
        if change:
            e.submit("replay", change)
            e.replay.update(change)   # visible to the replayer at once; the engine writes it to the database
        return get_replay()

    # ---------------------------------------------------------------- read side
    @app.get("/api/health")
    def health():
        e = eng()
        return {"ok": True, "scenario": e.scenario, "replay_state": e.replay["state"],
                "db_rows": e.ingest_totals()["rows"], "model_version": e.models.version if e.models else None}

    @app.get("/api/replay")
    def get_replay():
        e = eng()
        return {"state": e.replay["state"], "speed": e.replay["speed"], "position_ts": e.watermark,
                "max_speed": settings["replay"]["max_speed"]}

    @app.get("/api/layout")
    def layout():
        zones = eng().layout_rows
        sections = {}
        for z in zones:
            s = sections.setdefault(z["section"], {"section": z["section"], "name": f"S{z['section']}",
                                                    "run": z["run"], "zones": []})
            s["zones"].append(z["zone_id"])
        return {"zones": zones, "sections": list(sections.values()), "zone_signals": ZONE_SIGNALS,
                "belt_signals": BELT_SIGNALS}

    @app.get("/api/status")
    def status():
        return eng().status()

    def series(table: str, signal: str, where: str, params: list, start: Optional[int], end: Optional[int]) -> dict:
        e = eng()
        end = end if end is not None else (e.watermark or 0)
        start = start if start is not None else end - 24 * 3600
        if end <= start:
            raise HTTPException(422, "'to' must be after 'from'")
        bucket = max(60, -(-(end - start) // MAX_POINTS // 60) * 60)   # whole minutes, at most 500 points
        rows = query(f"SELECT (ts - ?) / ? AS b, AVG({signal}) AS value, COUNT({signal}) AS n, MAX(run_state) AS rs "
                     f"FROM {table} WHERE {where} ts >= ? AND ts <= ? GROUP BY b ORDER BY b",
                     [start, bucket, *params, start, end])
        return {"signal": signal, "from": start, "to": end, "bucket_seconds": bucket,
                "points": [{"ts": start + r["b"] * bucket, "value": r["value"], "n": r["n"], "run_state": r["rs"]}
                           for r in rows]}

    @app.get("/api/zones/{zone_id}/series")
    def zone_series(zone_id: int, signal: str, start: Optional[int] = Query(None, alias="from"),
                    end: Optional[int] = Query(None, alias="to")):
        if signal not in ZONE_SIGNALS:
            raise HTTPException(422, f"signal must be one of {ZONE_SIGNALS}")
        if not 1 <= zone_id <= len(ASSETS) - 1:
            raise HTTPException(404, "unknown zone")
        return series("readings", signal, "zone_id = ? AND", [zone_id], start, end)

    @app.get("/api/belt/series")
    def belt_series(signal: str, start: Optional[int] = Query(None, alias="from"),
                    end: Optional[int] = Query(None, alias="to")):
        if signal not in BELT_SIGNALS:
            raise HTTPException(422, f"signal must be one of {BELT_SIGNALS}")
        return series("belt_readings", signal, "", [], start, end)

    @app.get("/api/alerts")
    def alerts(state: Optional[str] = "open", limit: int = 200):
        if state not in (None, "open", "closed", "all"):
            raise HTTPException(422, "state must be open, closed or all")
        where = "" if state in (None, "all") else "WHERE state = ?"
        rows = query(f"SELECT * FROM alerts {where} ORDER BY (state='open') DESC, severity DESC, created_ts DESC "
                     "LIMIT ?", ([state] if where else []) + [min(limit, 1000)])
        for r in rows:
            r["evidence"] = json.loads(r["evidence"]) if r["evidence"] else None
        return {"alerts": rows}

    @app.get("/api/predictions")
    def predictions(asset: str, start: Optional[int] = Query(None, alias="from"),
                    end: Optional[int] = Query(None, alias="to")):
        try:
            asset_zone(asset)
        except ValueError:
            raise HTTPException(404, "unknown asset") from None
        end = end if end is not None else (eng().watermark or 0)
        start = start if start is not None else end - 3 * 24 * 3600
        rows = query("SELECT ts, p_abnormal, p_fail_24h, est_hours_to_failure, status FROM predictions "
                     "WHERE asset = ? AND ts >= ? AND ts <= ? ORDER BY ts", (asset, start, end))
        step = max(1, -(-len(rows) // MAX_POINTS))
        return {"asset": asset, "from": start, "to": end, "points": rows[::step]}

    @app.get("/api/sensors/health")
    def sensors_health():
        return {"sensors": query("SELECT h.zone_id, z.section, z.position_m, h.signal, h.state, h.since_ts, "
                                 "h.last_seen_ts, h.last_value FROM sensor_health h JOIN zones z USING (zone_id) "
                                 "WHERE h.state <> 'ok' ORDER BY h.since_ts DESC"),
                "events": query("SELECT * FROM quality_events ORDER BY start_ts DESC LIMIT 100")}

    @app.get("/api/ingest/stats")
    def ingest_stats():
        e = eng()
        return {"totals": e.ingest_totals(), "watermark_ts": e.watermark,
                "recent": query("SELECT * FROM ingest_log ORDER BY batch_id DESC LIMIT 20")}

    # ---------------------------------------------------------------- vision placeholders
    # The camera pipeline is not built yet. These endpoints exist so the dashboard and a later vision
    # component can be wired in without changing the API.
    def not_enabled():
        return {"enabled": False, "reason": "The camera pipeline is not part of this run.", "items": []}

    app.get("/api/vision/status")(not_enabled)
    app.get("/api/vision/detections")(not_enabled)
    app.get("/api/vision/defects")(not_enabled)

    if FRONTEND.is_dir():   # the built dashboard, so a reviewer needs only Python
        app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

        @app.get("/")
        def index():
            return FileResponse(FRONTEND / "index.html")
    return app


def main():
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = load_settings()
    uvicorn.run(create_app(settings), host=settings["server"]["host"], port=settings["server"]["port"],
                workers=1, log_level="warning")


if __name__ == "__main__":
    main()
