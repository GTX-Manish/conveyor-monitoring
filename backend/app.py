"""HTTP API. Reads go to SQLite through short read-only connections; every write goes through Engine.submit.

    python -m backend.app            # settings from configs/app.yaml
"""
import json
import logging
import os
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
from model.alerting import CALL_HORIZON_HOURS
from vision.defects import CONFIRM_LAPS

from .engine import Engine, connect

REPO = Path(__file__).resolve().parents[1]
FRONTEND = REPO / "frontend" / "dist"
MODEL_SUMMARY = REPO / "model" / "artifacts" / "summary.json"
FRESH_SUMMARY = REPO / "model" / "artifacts" / "fresh_test_summary.json"
MAX_POINTS = 500
log = logging.getLogger("backend")


def load_settings(path=REPO / "configs" / "app.yaml") -> dict:
    settings = yaml.safe_load(Path(path).read_text())
    if os.environ.get("CONVEYOR_PORT"):          # lets a second copy run beside one already on the default port
        settings["server"]["port"] = int(os.environ["CONVEYOR_PORT"])
    return settings


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


class WorkOrdersBody(BaseModel):
    orders: list[dict] = Field(default_factory=list)   # {asset, kind, failed_ts, repaired_ts}, sent after a repair


class CapturesBody(BaseModel):
    # {capture_id, ts, station, ground_position_m, belt_coordinate_m, image}; checked one by one in backend/vision.py
    captures: list[Any] = Field(default_factory=list)


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

    @app.post("/api/work_orders")
    async def work_orders(body: WorkOrdersBody):
        """Repair records. Used only to score past predictions; the model never sees them."""
        e = eng()
        for order in body.orders:
            e.submit("work_order", order)
        return {"received": len(body.orders)}

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

    @app.get("/api/outcomes")
    def outcomes():
        """What the model said against what happened, for every case that has been settled."""
        rows = query("SELECT o.*, w.kind AS fault_kind, w.repaired_ts, z.section, z.position_m FROM outcomes o "
                     "LEFT JOIN work_orders w USING (order_id) LEFT JOIN zones z ON z.zone_id = w.zone_id "
                     "ORDER BY COALESCE(o.failed_ts, o.resolved_ts) DESC")
        n = lambda kind: sum(r["kind"] == kind for r in rows)  # noqa: E731
        called = [r for r in rows if r["call_ts"] is not None]
        failures, warned = n("confirmed") + n("early_warning") + n("missed"), n("confirmed") + n("early_warning")
        # Recall: of the parts that failed, how many had an alert open at that moment. Precision: of the alerts
        # that have been settled, how many were followed by a real failure. None until there is a case to count.
        recall = n("confirmed") / failures if failures else None
        precision = warned / (warned + n("false_alarm")) if warned + n("false_alarm") else None
        return {"outcomes": rows,
                "summary": {"failures": failures, "warned": n("confirmed"), "precision": precision, "recall": recall,
                            "missed": n("missed"),
                            "false_alarms": n("false_alarm"), "early_warnings": n("early_warning"),
                            "failure_calls": len(called),
                            "failure_calls_came_true": sum(bool(r["within_horizon"]) for r in called),
                            "call_horizon_hours": CALL_HORIZON_HOURS}}

    @app.get("/api/model/evaluation")
    def model_evaluation():
        """The held-out test figures written by scripts/score.py, shown beside the live track record."""
        path = Path(MODEL_SUMMARY)
        if not path.exists():
            raise HTTPException(404, "no evaluation summary; run: python -m scripts.score")
        return json.loads(path.read_text())

    @app.get("/api/model/fresh-test")
    def model_fresh_test():
        """The fresh test: scenarios never used for training, tuning or diagnosis, scored once with the frozen models.
        This is the headline accuracy the dashboard quotes."""
        path = Path(FRESH_SUMMARY)
        if not path.exists():
            raise HTTPException(404, "no fresh-test summary; run: python -m scripts.score --frozen --fresh --export-summary")
        return json.loads(path.read_text())

    @app.get("/api/ingest/stats")
    def ingest_stats():
        e = eng()
        return {"totals": e.ingest_totals(), "watermark_ts": e.watermark,
                "recent": query("SELECT * FROM ingest_log ORDER BY batch_id DESC LIMIT 20")}

    # ---------------------------------------------------------------- vision
    # Camera captures go through the engine like sensor batches. Defects are belt locations (vision/defects.py):
    # confirmed ones away from splices, and a separate list of everything at a splice, which a person inspects.
    @app.post("/api/captures")
    async def captures(body: CapturesBody):
        result = eng().submit("captures", body.captures)
        return result if result is not None else {"enabled": eng().vision.enabled, "accepted": 0, "error": True}

    def vision_off():
        v = eng().vision
        return None if v.enabled else {"enabled": False, "reason": v.reason}

    def boxes_by_capture(ids: list) -> dict:
        if not ids:
            return {}
        out: dict = {}
        marks = ",".join("?" * len(ids))
        for r in query(f"SELECT capture_id, bbox FROM vision_detections WHERE capture_id IN ({marks}) "
                       "ORDER BY score DESC", ids):
            out.setdefault(r["capture_id"], []).append(json.loads(r["bbox"]))
        return out

    def capture_view(c: dict, boxes: dict) -> dict:
        return {"capture_id": c["capture_id"], "ts": c["ts"], "station": c["station"],
                "ground_position_m": c["ground_position_m"], "belt_coordinate_m": c["belt_coordinate_m"],
                "image_w": c["image_w"], "image_h": c["image_h"], "boxes": boxes.get(c["capture_id"], []),
                "image_url": f"/api/vision/image/{c['capture_id']}"}

    @app.get("/api/vision/status")
    def vision_status():
        v = eng().vision
        counts = query("SELECT COUNT(*) AS captures, COALESCE(SUM(n_detections > 0), 0) AS with_detections, "
                       "COALESCE(SUM(n_detections), 0) AS detections, COALESCE(SUM(status = 'image_missing'), 0) "
                       "AS image_missing FROM captures")[0]
        loc = query("SELECT COALESCE(SUM(confirmed = 1 AND at_splice = 0), 0) AS defects, "
                    "COALESCE(SUM(at_splice), 0) AS at_splice, COALESCE(SUM(confirmed = 0 AND at_splice = 0), 0) "
                    "AS unconfirmed FROM vision_defects")[0]
        stations = query("SELECT station, ground_position_m, COUNT(*) AS captures FROM captures GROUP BY station "
                         "ORDER BY ground_position_m")
        return {"enabled": v.enabled, "reason": v.reason, "threshold": v.threshold() if v.enabled else None,
                "counts": {**counts, **loc}, "stations": stations, "evaluation": v.evaluation(),
                "settings": {k: v.cfg.get(k) for k in ("merge_m", "splice_m", "splice_spacing_m", "belt_length_m")},
                "confirm_laps": CONFIRM_LAPS}

    @app.get("/api/vision/detections")
    def vision_detections(limit: int = 20):
        off = vision_off()
        if off:
            return {**off, "items": []}
        rows = query("SELECT c.*, d.at_splice, d.confirmed FROM captures c LEFT JOIN vision_defects d USING (defect_id) "
                     "WHERE c.n_detections > 0 ORDER BY c.ts DESC, c.capture_id DESC LIMIT ?", (min(max(limit, 1), 200),))
        boxes = boxes_by_capture([r["capture_id"] for r in rows])
        return {"enabled": True, "items": [{**capture_view(r, boxes), "defect_id": r["defect_id"],
                                            "at_splice": bool(r["at_splice"]), "confirmed": bool(r["confirmed"]),
                                            "max_score": max((b["score"] for b in boxes.get(r["capture_id"], [])),
                                                             default=None)} for r in rows]}

    @app.get("/api/vision/defects")
    def vision_defects():
        off = vision_off()
        if off:
            return {**off, "defects": [], "at_splice": [], "unconfirmed": 0}
        rows = query("SELECT * FROM vision_defects ORDER BY laps DESC, defect_id")
        best = {r["capture_id"]: r for r in query(
            "SELECT * FROM captures WHERE capture_id IN (SELECT best_capture_id FROM vision_defects)")}
        boxes = boxes_by_capture(list(best))

        def view(r):
            b = best.get(r["best_capture_id"])
            return {**{k: r[k] for k in ("defect_id", "belt_coordinate_m", "laps", "laps_looked", "n_captures",
                                         "n_detections", "first_ts", "last_ts", "confirmed_ts", "best_score",
                                         "splice_m", "splice_distance_m")},
                    "stations": json.loads(r["stations"]), "confirmed": bool(r["confirmed"]),
                    "at_splice": bool(r["at_splice"]), "best_capture": capture_view(b, boxes) if b else None}
        return {"enabled": True, "defects": [view(r) for r in rows if r["confirmed"] and not r["at_splice"]],
                "at_splice": [view(r) for r in rows if r["at_splice"]],
                "unconfirmed": sum(1 for r in rows if not r["confirmed"] and not r["at_splice"])}

    @app.get("/api/vision/image/{capture_id}")
    def vision_image(capture_id: int):
        rows = query("SELECT image_path FROM captures WHERE capture_id = ?", (capture_id,))
        path = eng().vision.image_path(rows[0]["image_path"]) if rows else None
        if path is None:
            raise HTTPException(404, "no image for this capture")
        return FileResponse(path, media_type="image/jpeg")

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
