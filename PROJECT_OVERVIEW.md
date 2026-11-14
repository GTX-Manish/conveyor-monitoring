# Conveyor Belt Monitoring: Project Overview

Status as of 2026-11-14. Deadline: **Nov 17, 2026, 11:59 PM IST**.
Only the simulator has been planned in depth. Everything else below is an overview, to be detailed when we reach it.
The simulator's build spec is in [`SIMULATOR_SPEC.md`](SIMULATOR_SPEC.md).

---

## 1. What this project is

This is the Avartan Labs "AI Systems Engineer" take-home. We build an end-to-end system that watches a very long mining
conveyor belt and predicts failures before they stop production.

The brief asks for five parts, delivered as one GitHub repo:

1. **Design document** (the README): design decisions, why we chose them, what we would do differently.
2. **Backend**: takes in sensor readings and stores them.
3. **Prediction model**: scores belt health from the data.
4. **Vision component**: finds wear and tear in belt images.
5. **Dashboard**: shows the state of the belt.

Hard rules from the brief:

- Sensor data may come **only from a simulator we write ourselves**. There is no real sensor data.
- Belt images must come from a **public dataset**, and the README must say which one and why.
- The repo must include an `ai_usage/` folder with our AI chat logs.
- The dashboard must show, at minimum: the status of every sensor point (normal, abnormal or predicted failure), active
  anomalies (location, type, severity), model predictions (what fails, where, and when), and vision detections.

What the graders care about, in the brief's own emphasis: judgement, clear writing, what we chose to build and what we
chose to skip, how we handle messy input, and how we used AI.

## 2. What we are trying to achieve

A reviewer should be able to run one command and see a belt that looks alive: sensors reporting, some of them
misbehaving, a few faults developing, the model warning about them with a location and a time, and camera images
showing damage that matches. Beyond that demo, the README must show we thought hard about three things:

- **Honesty about simulated data.** The model learns our simulator's rules, so a good score on simulated data does not
  prove it works on a real belt. We say this plainly and build evaluation that is as hard on the model as we can
  (split by whole scenario, hard negatives, a stress scenario, simple baselines to beat).
- **Messy input.** Real sensors drop readings, go silent and freeze. The system must not read silence or a flat line as
  "all is well".
- **The looping belt.** The brief hides a trap: the belt is a loop. A defect on the belt passes every
  camera once per revolution (25 minutes at our size), so the same defect appears again and again. Vision detections
  must be located by **belt coordinate** and merged into one defect, not counted again each lap.

## 3. The belt we simulate

| Property | Value | Why |
|---|---|---|
| Length | **6 km loop: 3 km carry side + 3 km return side** | The brief says multi-kilometre; 3 km per side is realistic for a mining overland conveyor |
| Speed | 4 m/s | One revolution takes 6000 / 4 = 1500 s = **25 min**, which sets how often a belt defect repeats |
| Sensor spacing | one sensor suite every **50 m** = **120 zones** | Industry practice runs from about 1 m to 25 m; one suite per 50 m zone is a defensible zone-level choice. A 25 m run (240 zones) is a later experiment if time allows. |
| Sections | 12 numbered splice sections (S1 to S12) of 500 m, each holding 10 zones | The dashboard drills from section to zone |
| Loading / unloading | loading at 0 m, unloading at 3000 m | Most wear happens there, so cameras go there |
| Cameras | three stations: start (loading), middle, end (unloading), on the carrying face | Matches the plan that most wear is at loading and unloading |
| Duty cycle | 18 h running, 6 h stopped each day | Matches the brief; starts and stops are a source of false alarms |

## 4. The system in one picture

```
 SENSOR PIPELINE                                             VISION PIPELINE
 ---------------                                             ---------------
 Simulator (Python)                                          Public image dataset
   |  readings every minute per zone,                          |
   |  plus belt-wide drive signals                             v
   v                                                         Wear/tear detector
 Backend (FastAPI + SQLite)                                    |  detections with belt coordinate
   |  validates, stores                                        v
   v                                                         De-duplication across laps
 Data-quality check                                            |
   |  gaps, flat lines -> "no data / sensor suspect"           |
   v                                                           |
 Prediction model (LightGBM)                                   |
   |  status, what/where/when of failures                      |
   v                                                           v
 ------------------------ Dashboard (Streamlit) ------------------------
   120-zone loop map · active anomalies · model predictions · vision detections
```

The two pipelines meet only at the dashboard, so each can be demonstrated alone if time runs out.

## 5. The simulator (planned in depth)

Full build instructions are in `SIMULATOR_SPEC.md`. In short:

**It produces**, for each of the 120 zones every simulated minute: vibration RMS, vibration kurtosis, bearing
temperature, acoustic level and belt lateral offset. For the drive, once per minute: motor current, motor temperature,
gearbox oil temperature, belt speed against drive speed, belt tension and throughput. It also produces camera capture
events with a timestamp and a belt coordinate.

**Healthy behaviour.** Every signal is a physical baseline that moves with load, speed, ambient temperature and dust,
plus noise. Getting this right matters because a model that cannot separate a hot afternoon from a failing bearing is
useless.

**Two faults in version 1:**

| Fault | Where | What it looks like | Why it is in |
|---|---|---|---|
| Roller (idler bearing) wearing out | one zone | slow, accelerating rise in temperature against neighbours, with faint early changes in acoustic level and vibration | The best-documented failure, and the one that answers "where" |
| Motor (drive) overheating | the drive | motor and gearbox temperature rise, motor current rises late | The drive cannot see idler faults, so this is a separate signature |

Belt slipping on the drive pulley is dropped for now. It is cheap to add later because its signals already exist in
the baseline.

**Belt tear or damage is not a sensor fault.** It was planned as a third fault that would link sensors to images, and
was removed on 2026-11-14. Public image datasets are unrelated photos of damage: they have no severity progression,
no repeat views of one defect and no sensor context. Sensors and images would have agreed only because we wired them
to agree. Belt damage is handled by the vision pipeline alone (section 6), and the README will say that sensors and
images are not linked.

**Visual defects for the camera.** The simulator places a few defects at fixed belt coordinates, with no effect on
any sensor. Each comes into camera view once per lap, which gives the vision pipeline the looping-belt problem to
solve.

**Hard negatives.** Normal events that look like trouble (a heavy-load shift, a heat wave, a dusty spell, a start-up
transient) appear in every scenario and are labelled as not faults. Without them the model never learns the
difference and its scores look better than they really are.

**Messy data, version 1:** missing readings, whole-suite outages, stuck sensors that freeze at one value, and noise
scaled by dust. These are added after the physics and written to a separate ground-truth file, so equipment faults
and sensor faults can be told apart when scoring.

**Ground truth** lives in separate files that the pipeline never reads: equipment fault events, sensor faults, and
hard negatives.

**Stack:** Python, numpy and scipy, pandas, YAML config, parquet output, pytest. Everything is seeded, so a seed
reproduces a run exactly.

## 6. Everything else (overview only)

### Backend
A FastAPI service with an ingest endpoint that validates incoming readings and stores them in SQLite. A data-quality
layer runs before the model: it flags gaps and flat lines and produces the fourth status, **"no data / sensor
suspect"**. Live mode replays the simulator's output in timestamp order.

**Why this status exists:** a broken sensor can look perfectly healthy, which would make the dashboard say "normal"
when it simply does not know.

### Prediction model
Gradient boosting (LightGBM). Why: it trains in minutes on 2 CPUs, handles tabular features and rare faults well, and
is easy to explain. We deferred comparing a sequence model for lack of time and will say so in the README. Features:
rolling statistics over 1, 6 and 24 hours, differences from neighbouring zones, and corrections for load and ambient
temperature. Scored by whole-scenario split, against two baselines (a fixed temperature threshold and a
neighbour-deviation check).

### Vision
A public conveyor-belt image dataset (candidates: BeltCrack, CBCD, Roboflow sets) and a small pretrained CNN fine-tuned
on CPU, or a detector if the dataset has box labels. Dataset choice and reasons go in the README. The looping-belt
handling is the important part: detections get a belt coordinate, and detections of the same coordinate on later laps
merge into one defect. Splice joints are known landmarks and must not be reported as damage. The simulator's visual
defects (section 5) are each pinned to one dataset image and re-shown every lap, so de-duplication can be scored.
They are not linked to any sensor signal.

### Dashboard
Streamlit with Plotly. The main view is the 120 zones drawn as a loop, coloured by status (normal, abnormal,
predicted failure, no data / sensor suspect), with drill-down from section to zone. Panels for active anomalies, model
predictions and vision detections.

### Deliverables
Repo with code and tests, README (design decisions, vision dataset and why, approach, dashboard screenshots, what we
would do differently), and the `ai_usage/` folder.

## 7. Plan and time

| When | Work |
|---|---|
| Sat Nov 14 | Simulator: faults, hard negatives, messiness, ground truth |
| Sun Nov 15 | Backend, data-quality check, sensor model and evaluation |
| Mon Nov 16 | Vision pipeline and loop handling, dashboard |
| Tue Nov 17 | README, screenshots, `ai_usage/` export, final run-through (submit before 11:59 PM IST) |

## 8. Decisions so far, with reasons

| Decision | Why |
|---|---|
| Two pipelines joined at the dashboard | Matches the brief and lets each be demoed alone |
| Simulator first, vision second | The simulator defines the data contract everything else depends on |
| Summary features per minute, not raw waveforms | Raw kHz vibration is heavy and the fault signal lives in the features anyway |
| Ground truth in separate files | Honest scoring, and separates equipment faults from sensor faults |
| 6 km loop, 50 m zones, 12 sections | See section 3 |
| Gradient boosting for the sensor model | See section 6 |
| Faults: roller and motor; slip dropped | Slip is cheap to add later |
| Belt damage is vision-only, not a sensor fault | No public dataset supports an honest sensor-to-image link; the loop problem is still shown through visual defects at belt coordinates |
| Hard negatives and a fourth status | Honest scores, and a dashboard that says "I do not know" instead of "normal" |
| Messiness limited to missing, outage, stuck, noise | These are the cases a dashboard must visibly handle |

## 9. Honest limits

- The model learns the simulator's rules; accuracy here is not proof of real-world accuracy.
- The research on early-stage bearing faults is weaker than vendor claims suggest (early signs are faint in tests), so
  the simulator keeps early effects small next to noise.
- Numbers for fault effects are assumptions informed by a few papers, not measured values. They are listed in
  `SIMULATOR_SPEC.md` so they can be argued with.
- Camera coverage is partial: three stations see only a short stretch of belt at a time, on one face.
- Sensors and images are not linked. A defect the camera sees has no sensor signature in our data.
