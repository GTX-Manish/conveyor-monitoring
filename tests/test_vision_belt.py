"""The looping-belt layer: simulated captures, belt-coordinate maths, de-duplication, splices, scoring, the API."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient

from simulator import vision as sv
from simulator.config import Config, load_config
from simulator.generate import STREAMS, generate, make_streams
from simulator.io import VISION_PUBLIC_COLUMNS, write_scenario
from simulator.scenario_set import config_hash
from vision.defects import CONFIRM_LAPS, CONFIRM_RATIO, DefectTracker, circ_dist

REPO = Path(__file__).resolve().parents[1]
DEMO_CONFIG = REPO / "configs" / "scenarios" / "demo_01.yaml"
SENSOR_FIXTURE = Path(__file__).parent / "fixtures" / "demo_01_sensor_sha256.json"
LAYOUT_FIXTURE = Path(__file__).parent / "fixtures" / "demo_01_layout.json"
SETTINGS = yaml.safe_load((REPO / "configs" / "app.yaml").read_text())


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    cfg = load_config(DEMO_CONFIG)
    scn = generate(cfg)
    root = write_scenario(scn, cfg, tmp_path_factory.mktemp("data"), manifest_extra={"config_hash": config_hash(cfg)})
    return cfg, scn, root


# ------------------------------------------------------------------ the sensors are untouched
def test_vision_stream_is_appended_last():
    assert STREAMS[-1] == "vision" and STREAMS.index("vision") == len(STREAMS) - 1
    old = np.random.SeedSequence(1015).spawn(len(STREAMS) - 1)        # the streams as they were before vision
    new = make_streams(1015)
    for name, ss in zip(STREAMS[:-1], old):
        assert np.random.default_rng(ss).random() == new[name].random(), name


def test_demo_sensor_files_are_byte_identical(demo):
    """The demo's sensor output must not change because of vision: same bytes as before it was added. The layout is
    compared by value: its float64 columns can differ in the last bit between machines (seen on a CI runner), which
    changes its bytes but not the layout."""
    _, _, root = demo
    expected = json.loads(SENSOR_FIXTURE.read_text())["files"]
    got = {name: hashlib.sha256((root / "public" / name).read_bytes()).hexdigest() for name in expected
           if name != "layout.parquet"}
    assert got == {k: v for k, v in expected.items() if k != "layout.parquet"}
    fx = json.loads(LAYOUT_FIXTURE.read_text())
    layout = pd.read_parquet(root / "public" / "layout.parquet")
    assert {c: str(t) for c, t in layout.dtypes.items()} == fx["dtypes"]
    for col, values in fx["columns"].items():
        if layout[col].dtype.kind == "f":
            np.testing.assert_allclose(layout[col].to_numpy(), np.array(values, dtype=float), rtol=0, atol=1e-9,
                                       equal_nan=True, err_msg=col)
        else:
            assert layout[col].tolist() == values, col


# ------------------------------------------------------------------ captures
def test_capture_generation_is_deterministic(tmp_path):
    cfg = Config(scenario="vis", seed=11, days=3)
    a, b = generate(cfg), generate(cfg)
    pd.testing.assert_frame_equal(a.captures, b.captures)
    pd.testing.assert_frame_equal(a.vision_truth, b.vision_truth)
    assert a.vision_locations == b.vision_locations
    other = generate(Config(scenario="vis", seed=12, days=3))
    assert not other.captures["belt_coordinate_m"].equals(a.captures["belt_coordinate_m"])


def test_demo_layout_of_locations(demo):
    _, scn, _ = demo
    locs = scn.vision_locations
    kinds = [x["kind"] for x in locs]
    assert kinds.count("splice") == 12 and 3 <= kinds.count("crack") <= 4 and kinds.count("clean") == sv.N_CLEAN
    splices = sorted(x["belt_coordinate_m"] for x in locs if x["kind"] == "splice")
    assert splices == [500.0 * k for k in range(12)]
    near = [x for x in locs if x["kind"] == "crack" and x["near_splice"]]
    assert len(near) == 1 and circ_dist(near[0]["belt_coordinate_m"], near[0]["splice_m"], 6000.0) <= 1.5
    others = [x["belt_coordinate_m"] for x in locs if x["kind"] != "splice" and not x.get("near_splice")]
    for c in others:                          # every other location is well away from splices and from each other
        assert min(circ_dist(c, s, 6000.0) for s in splices) >= sv.MIN_GAP_M
    seqs = [x["sequence"] for x in locs if x["kind"] == "crack"]
    assert len(set(seqs)) == len(seqs)        # each crack has its own test sequence


def test_public_captures_say_nothing_about_what_they_show(demo):
    _, scn, root = demo
    pub = pd.read_parquet(root / "public" / "captures.parquet")
    assert list(pub.columns) == VISION_PUBLIC_COLUMNS["captures"] == sv.CAPTURE_COLUMNS
    assert pub["image"].str.fullmatch(r"frames/[0-9a-f]{12}\.jpg").all()     # hashed names, no kind in the path
    assert (root / "truth" / "vision_truth.parquet").exists() and not (root / "public" / "vision_truth.parquet").exists()
    truth = pd.read_parquet(root / "truth" / "vision_truth.parquet")
    assert list(truth["capture_id"]) == list(pub["capture_id"]) == list(range(1, len(pub) + 1))


def test_every_demo_image_is_committed_at_the_detector_size(demo):
    from PIL import Image
    _, scn, _ = demo
    for rel in sorted(set(scn.captures["image"])):
        with Image.open(sv.DEMO_DIR / rel) as im:
            assert im.size == sv.FRAME_SIZE, rel          # fits 640 x 384, so the detector only pads
    assert "CC BY-SA 4.0" in (sv.DEMO_DIR / "ATTRIBUTION.md").read_text()


def test_coordinate_maths():
    assert sv.coordinate_under(1500.0, 0.0, 6000.0) == 1500.0
    assert sv.coordinate_under(0.0, 250.0, 6000.0) == 5750.0              # the belt has moved 250 m past station 0
    assert sv.coordinate_under(3000.0, 6000.0 * 7 + 100.0, 6000.0) == pytest.approx(2900.0)
    assert circ_dist(5999.5, 0.3, 6000.0) == pytest.approx(0.8)


def test_captures_follow_the_belt(demo):
    """Each capture's coordinate is the location's own (with 0.5 m noise); a point passes CAM1, CAM2, CAM3 in order on
    one lap; captures only happen while the belt moves; one recorded lap per running day."""
    _, scn, _ = demo
    cap = scn.captures.merge(scn.vision_truth, on="capture_id")
    where = {x["location_id"]: x["belt_coordinate_m"] for x in scn.vision_locations}
    err = np.array([circ_dist(c, where[lid], 6000.0) for c, lid in zip(cap["belt_coordinate_m"], cap["location_id"])])
    assert err.max() < 4 * sv.COORD_NOISE_M and 0.3 < err.mean() < 0.5        # half-normal mean of 0.5 m noise: 0.40
    for (lid, lap), g in cap.groupby(["location_id", "lap"]):
        g = g.sort_values("ts")
        assert list(g["station"]) == ["CAM1", "CAM2", "CAM3"], (lid, lap)
        gaps = np.diff(g["ts"].to_numpy())
        assert (gaps > 250).all() and (gaps < 1125).all()                    # 375 s at 4 m/s, slower while ramping
    stopped = set(scn.belt_wide.loc[scn.belt_wide["run_state"] == "stopped", "timestamp"].astype("int64") // 10**9)
    assert not {int(t) - int(t) % 60 for t in cap["ts"]} & stopped
    assert cap["lap"].max() == len(set(pd.to_datetime(cap["ts"], unit="s").dt.date))   # one lap per day


def test_each_lap_shows_the_crack_frame_advanced_by_15_to_30(demo):
    _, scn, _ = demo
    cat = sv.load_catalog()
    t = scn.vision_truth[scn.vision_truth["kind"] == "crack"]
    for lid, g in t.groupby("location_id"):
        seq = g["source"].iloc[0].split("/")[1]
        frames = cat["sequences"][seq]
        sources = g.drop_duplicates("lap").sort_values("lap")["source"]
        per_lap = sources.map(lambda s: frames.index(int(s.split("/")[2][:-4])))
        steps = np.diff(per_lap.to_numpy()) % len(frames)
        assert ((steps >= 15) & (steps <= 30)).all(), (lid, steps)


# ------------------------------------------------------------------ de-duplication
def test_a_location_merges_across_laps_and_is_never_duplicated():
    rng = np.random.default_rng(0)
    tr = DefectTracker()
    cid = 0
    for lap in range(6):
        for k, station in enumerate(["CAM1", "CAM2", "CAM3"]):
            cid += 1
            tr.add(cid, lap * 86400 + k * 375, station, 1234.0 + rng.normal(0, 0.5), [0.6])
    assert len(tr.locations) == 1
    (loc,) = tr.locations.values()
    assert loc.laps == 6 and loc.confirmed and loc.stations == ["CAM1", "CAM2", "CAM3"] and len(loc.captures) == 18
    assert abs(loc.coordinate - 1234.0) < 0.5 and tr.defects() == [loc]


def test_one_lap_is_not_confirmed_and_looks_are_counted():
    tr = DefectTracker()
    for k, station in enumerate(["CAM1", "CAM2", "CAM3"]):
        tr.add(k + 1, k * 375, station, 2100.0, [0.5])
    assert not tr.defects() and len(tr.unconfirmed()) == 1                  # three cameras, one lap
    tr.add(4, 86400, "CAM1", 2100.3, [])                                     # looked again, nothing seen
    (loc,) = tr.locations.values()
    assert loc.laps == 1 and loc.laps_looked == 2 and loc.coordinate == pytest.approx(2100.0)
    tr.add(5, 2 * 86400, "CAM2", 2099.8, [0.4])
    assert loc.laps == 2 and loc.confirmed and loc.confirmed_ts == 2 * 86400


def test_two_locations_that_drift_together_are_merged():
    tr = DefectTracker()
    tr.add(1, 0, "CAM1", 100.0, [0.5])
    tr.add(2, 86400, "CAM1", 102.5, [0.5])            # 2.5 m away: a second location
    assert len(tr.locations) == 2
    loc, merged = tr.add(3, 2 * 86400, "CAM1", 101.2, [0.5])   # joins the first and moves it to 100.6 m, within 2 m
    assert merged and len(tr.locations) == 1 and loc.laps == 3 and len(loc.captures) == 3


def test_merging_works_across_the_belt_join():
    tr = DefectTracker()
    tr.add(1, 0, "CAM1", 5999.6, [0.5])
    tr.add(2, 86400, "CAM1", 0.4, [0.5])
    (loc,) = tr.locations.values()
    assert loc.laps == 2 and circ_dist(loc.coordinate, 0.0, 6000.0) < 0.5
    assert loc.at_splice and loc.splice_m == 0.0                             # 0 m is a splice


def test_both_splice_cases_land_in_the_inspect_list():
    tr = DefectTracker()
    for lap in range(3):                                                     # a real crack 1.2 m from the splice at 500 m
        tr.add(10 + lap, lap * 86400, "CAM1", 501.2, [0.7])
    tr.add(20, 0, "CAM2", 3000.1, [0.4])                                     # one false hit at the splice at 3000 m
    near, false_hit = sorted(tr.at_splice(), key=lambda x: x.coordinate)
    assert near.splice_m == 500.0 and near.confirmed and near.splice_distance_m == pytest.approx(1.2)
    assert false_hit.splice_m == 3000.0 and not false_hit.confirmed          # never dropped, even after one lap
    assert tr.defects() == [] and tr.unconfirmed() == []
    tr.add(30, 0, "CAM1", 1510.0, [0.5])                                     # 10 m from a splice: an ordinary location
    assert len(tr.unconfirmed()) == 1 and len(tr.at_splice()) == 2


def test_scorer_counts_the_crack_near_a_splice_as_found_and_flagged():
    from scripts.score_vision import score
    locations = [{"location_id": 1, "kind": "splice", "belt_coordinate_m": 500.0},
                 {"location_id": 2, "kind": "crack", "belt_coordinate_m": 501.0, "near_splice": True},
                 {"location_id": 3, "kind": "crack", "belt_coordinate_m": 2620.0, "near_splice": False},
                 {"location_id": 4, "kind": "splice", "belt_coordinate_m": 3000.0},
                 {"location_id": 5, "kind": "clean", "belt_coordinate_m": 4130.0}]
    rows = []
    for lap in (1, 2, 3):
        for lid, coord, image in ((1, 500.0, "splice_quiet"), (2, 501.0, "crack"), (3, 2620.0, "crack"),
                                  (4, 3000.0, "splice_fires"), (5, 4130.0, "clean_fires" if lap == 1 else "quiet")):
            rows.append((len(rows) + 1, lap * 86400 + lid, "CAM1", 0.0, coord, image, lid,
                         next(x["kind"] for x in locations if x["location_id"] == lid), lap, image))
    df = pd.DataFrame(rows, columns=[*sv.CAPTURE_COLUMNS, "location_id", "kind", "lap", "source"])
    scores = {"crack": [0.6, 0.5], "splice_fires": [0.4], "clean_fires": [0.45]}
    r = score(df[sv.CAPTURE_COLUMNS], df[sv.TRUTH_COLUMNS], locations, lambda image: scores.get(image, []))
    assert r["crack_locations_found"] == 2 and r["duplicates"] == 0
    assert r["near_splice_crack"] == {"found_and_flagged": True, "location_id": 2}
    assert r["false_locations"] == 1 and r["false_locations_at_splice"] == 1 and r["false_confirmed_defects"] == 0
    assert r["laps_to_confirm"] == [2, 2]
    sf = r["single_frame_false_detections"]
    assert sf["detections"] == 4 and sf["at_splices"] == 3 and sf["at_clean_locations"] == 1
    assert sf["removed_by_confirmation"] == 1                                # the one-off hit at the clean location


def test_dedup_eval_is_from_the_committed_demo():
    ev = json.loads((REPO / "vision" / "artifacts" / "dedup_eval.json").read_text())
    assert ev["scenario"] == "demo_01" and ev["duplicates"] == 0
    assert ev["near_splice_crack"]["found_and_flagged"] is True
    old, new = ev["side_by_side"]
    assert old["rule"].startswith("old") and new["rule"].startswith("new")
    assert ev["rule_in_use"] == {**ev["rule_in_use"], "confirm_laps": CONFIRM_LAPS, "confirm_ratio": CONFIRM_RATIO}
    assert new["crack_locations_found"] == old["crack_locations_found"]
    assert new["false_confirmed_defects"] <= old["false_confirmed_defects"]


def test_the_rule_in_use_is_the_one_chosen_away_from_the_demo():
    tuning = json.loads((REPO / "vision" / "artifacts" / "dedup_tuning.json").read_text())
    assert tuning["scenario"]["seed"] != load_config(DEMO_CONFIG).seed
    assert tuning["scenario"]["seed"] not in {load_config(p).seed for p in (REPO / "configs" / "scenarios").glob("*.yaml")}
    assert (tuning["choice"]["confirm_laps"], tuning["choice"]["confirm_ratio"]) == (CONFIRM_LAPS, CONFIRM_RATIO)


def test_ratio_rule_drops_a_location_seen_on_too_few_of_its_laps():
    def run(ratio):
        tr = DefectTracker(confirm_laps=2, confirm_ratio=ratio)
        for lap in range(6):
            tr.add(lap + 1, lap * 86400, "CAM1", 2100.0, [0.5] if lap < 2 else [])   # hits on laps 1-2, then nothing
        (loc,) = tr.locations.values()
        return loc
    loc = run(0.5)
    assert (loc.laps, loc.laps_looked) == (2, 6) and not loc.confirmed           # 2 of 6 laps: no longer confirmed
    assert loc.confirmed_ts == 86400                                               # but it was, after lap 2
    assert run(0.0).confirmed                                                      # the old rule keeps it for good
    tr = DefectTracker(confirm_laps=2, confirm_ratio=0.5)
    for lap in range(6):                                                           # a crack seen on 4 of 6 laps
        tr.add(lap + 1, lap * 86400, "CAM1", 2100.0, [0.5] if lap % 3 else [])
    (loc,) = tr.locations.values()
    assert (loc.laps, loc.laps_looked) == (4, 5) and loc.confirmed               # laps looked count from first seen


# ------------------------------------------------------------------ backend
def _client(tmp_path, settings):
    from backend.app import create_app
    from backend.engine import Engine
    from tests.test_backend import LAYOUT
    e = Engine(tmp_path / "v.sqlite", LAYOUT, settings, models=None, scenario="vision")
    return e, TestClient(create_app(settings, engine=e))


def test_captures_endpoint_runs_the_detector_and_keeps_locations(tmp_path, demo):
    pytest.importorskip("onnxruntime")
    _, scn, _ = demo
    crack_lid = next(x["location_id"] for x in scn.vision_locations if x["kind"] == "crack" and not x["near_splice"])
    t = scn.vision_truth
    ids = set(t.loc[t["location_id"] == crack_lid, "capture_id"])
    caps = [{**{k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}}
            for r in scn.captures[scn.captures["capture_id"].isin(ids)].to_dict("records")]
    e, client = _client(tmp_path, SETTINGS)
    with client:
        r = client.post("/api/captures", json={"captures": caps + [caps[0]] + [{"capture_id": "x"}]}).json()
        assert r["enabled"] and r["accepted"] == len(caps) and r["duplicates"] == 1 and r["rejected"] == 1
        bad = {**caps[0], "capture_id": 99999, "image": "../../configs/app.yaml"}     # outside the image folder
        assert client.post("/api/captures", json={"captures": [bad]}).json()["image_missing"] == 1
        assert client.get("/api/vision/image/99999").status_code == 404
        d = client.get("/api/vision/defects").json()
        listed = d["defects"] + d["at_splice"]
        assert len(listed) <= 1                                              # one crack: never two locations
        status = client.get("/api/vision/status").json()
        assert status["enabled"] and status["counts"]["captures"] == len(caps) + 1
        assert status["evaluation"]["license"] == "AGPL-3.0" and status["threshold"] == pytest.approx(0.363)
        if listed:
            best = listed[0]["best_capture"]
            assert best["boxes"] and client.get(best["image_url"]).headers["content-type"] == "image/jpeg"


@pytest.mark.parametrize("change, reason", [
    ({"model": "vision/artifacts/missing.onnx"}, "crack model"),
    ({"images_dir": "data/no_such_folder"}, "images folder"),
    (None, "No vision settings"),
])
def test_vision_is_off_with_a_reason_when_something_is_missing(tmp_path, change, reason):
    settings = json.loads(json.dumps(SETTINGS))
    if change is None:
        settings.pop("vision")
    else:
        settings["vision"].update(change)
    e, client = _client(tmp_path, settings)
    with client:
        status = client.get("/api/vision/status").json()
        assert status["enabled"] is False and reason in status["reason"]
        posted = client.post("/api/captures", json={"captures": [{"capture_id": 1}]}).json()
        assert posted == {"enabled": False, "reason": status["reason"], "accepted": 0}
        assert client.get("/api/vision/defects").json() == {"enabled": False, "reason": status["reason"], "defects": [],
                                                            "at_splice": [], "unconfirmed": 0}
        assert client.get("/api/vision/detections").json()["items"] == []
        assert client.get("/api/health").json()["ok"] is True                # the rest of the system is unaffected
