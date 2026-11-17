"""Belt-coordinate de-duplication of crack detections across laps.

A looping belt shows the same crack to the cameras once per lap, and to every station on that lap. This module
turns per-capture detections into one record per belt location:

- every detection in a capture joins the location at that capture's belt coordinate (frames carry no metric
  scale, so all boxes in one capture count as one place on the belt);
- a capture joins the nearest location within MERGE_M (circular distance on the loop); otherwise it starts a new
  one. Two locations that drift within MERGE_M of each other are merged, so a crack is never listed twice;
- a location is confirmed once it has been seen on CONFIRM_LAPS laps. Captures of one location less than
  SAME_LAP_S apart belong to the same lap (the three stations see a point within 750 s at 4 m/s; the next lap
  comes 1500 s later);
- a capture with no detection does not move or create a location, but if it falls on one it counts as a look:
  "seen on 3 of 12 laps" tells a crack that shows most laps from an occasional false hit;
- a location within SPLICE_M of a splice is put in a separate "at splice, inspect" list and is never dropped: a
  splice joint can look like a crack to the model, and a real crack can sit next to a splice.

The backend (live) and scripts/score_vision.py (offline) use this same code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

BELT_LENGTH_M = 6000.0
SPLICE_SPACING_M = 500.0
MERGE_M = 2.0
SPLICE_M = 1.5
CONFIRM_LAPS = 2
SAME_LAP_S = 1125.0          # 0.75 of a lap at the nominal 4 m/s


def circ_dist(a: float, b: float, length: float = BELT_LENGTH_M) -> float:
    d = abs(a - b) % length
    return min(d, length - d)


@dataclass
class Location:
    location_id: int
    length: float
    sx: float = 0.0                          # sum of unit vectors of member capture coordinates
    sy: float = 0.0
    captures: list = field(default_factory=list)        # (capture_id, ts, station, n_detections, best_score)
    looks: list = field(default_factory=list)           # ts of captures here with no detection
    best_capture_id: int | None = None
    best_score: float = -1.0
    splice_m: float | None = None            # the nearest splice, if within SPLICE_M
    splice_distance_m: float | None = None

    @property
    def coordinate(self) -> float:
        a = math.atan2(self.sy, self.sx)
        return (a / (2 * math.pi) * self.length) % self.length

    def add(self, capture_id: int, ts: float, station: str, coord: float, scores: list[float]) -> None:
        a = 2 * math.pi * coord / self.length
        self.sx += math.cos(a)
        self.sy += math.sin(a)
        best = max(scores)
        self.captures.append((capture_id, ts, station, len(scores), best))
        if best > self.best_score:
            self.best_score, self.best_capture_id = best, capture_id

    def lap_starts(self, same_lap_s: float = SAME_LAP_S, with_looks: bool = False) -> list[float]:
        starts: list[float] = []
        for ts in sorted([c[1] for c in self.captures] + (self.looks if with_looks else [])):
            if not starts or ts - starts[-1] >= same_lap_s:
                starts.append(ts)
        return starts

    @property
    def laps_looked(self) -> int:
        """Laps on which a camera looked at this location since it was first seen, with or without a detection."""
        first = self.first_ts
        return len([s for s in self.lap_starts(with_looks=True) if s >= first - SAME_LAP_S])

    @property
    def laps(self) -> int:
        return len(self.lap_starts())

    @property
    def confirmed(self) -> bool:
        return self.laps >= CONFIRM_LAPS

    @property
    def confirmed_ts(self) -> float | None:
        starts = self.lap_starts()
        return starts[CONFIRM_LAPS - 1] if len(starts) >= CONFIRM_LAPS else None

    @property
    def at_splice(self) -> bool:
        return self.splice_m is not None

    @property
    def stations(self) -> list[str]:
        return sorted({c[2] for c in self.captures})

    @property
    def n_detections(self) -> int:
        return sum(c[3] for c in self.captures)

    @property
    def first_ts(self) -> float:
        return min(c[1] for c in self.captures)

    @property
    def last_ts(self) -> float:
        return max(c[1] for c in self.captures)

    def as_dict(self) -> dict:
        return {"defect_id": self.location_id, "belt_coordinate_m": round(self.coordinate, 2), "laps": self.laps,
                "laps_looked": self.laps_looked,
                "stations": self.stations, "n_captures": len(self.captures), "n_detections": self.n_detections,
                "first_ts": int(self.first_ts), "last_ts": int(self.last_ts), "confirmed": self.confirmed,
                "confirmed_ts": None if self.confirmed_ts is None else int(self.confirmed_ts),
                "best_capture_id": self.best_capture_id, "best_score": round(self.best_score, 4),
                "at_splice": self.at_splice, "splice_m": self.splice_m,
                "splice_distance_m": None if self.splice_distance_m is None else round(self.splice_distance_m, 2)}


class DefectTracker:
    def __init__(self, belt_length_m: float = BELT_LENGTH_M, splice_spacing_m: float = SPLICE_SPACING_M,
                 merge_m: float = MERGE_M, splice_m: float = SPLICE_M):
        self.length, self.merge_m, self.splice_m = float(belt_length_m), float(merge_m), float(splice_m)
        n = int(round(self.length / splice_spacing_m))
        self.splices = [k * float(splice_spacing_m) for k in range(n)]
        self.locations: dict[int, Location] = {}
        self.next_id = 1

    def _nearest(self, coord: float, exclude: int | None = None) -> Location | None:
        best, dist = None, None
        for loc in self.locations.values():
            if loc.location_id == exclude:
                continue
            d = circ_dist(coord, loc.coordinate, self.length)
            if d <= self.merge_m and (dist is None or d < dist):
                best, dist = loc, d
        return best

    def _mark_splice(self, loc: Location) -> None:
        d, s = min((circ_dist(loc.coordinate, s, self.length), s) for s in self.splices)
        loc.splice_m, loc.splice_distance_m = (s, d) if d <= self.splice_m else (None, None)

    def add(self, capture_id: int, ts: float, station: str, coord: float, scores: list[float]):
        """Record one capture's detections. Returns (location, ids of locations merged into it). A capture with no
        detection only counts as a look at the location it falls on (and returns (None, []))."""
        if not scores:
            here = self._nearest(coord)
            if here is not None:
                here.looks.append(ts)
            return None, []
        loc = self._nearest(coord)
        if loc is None:
            loc = self.locations[self.next_id] = Location(self.next_id, self.length)
            self.next_id += 1
        loc.add(capture_id, ts, station, coord, scores)
        merged = []
        while (other := self._nearest(loc.coordinate, exclude=loc.location_id)) is not None:
            keep, gone = (loc, other) if loc.location_id < other.location_id else (other, loc)
            keep.sx, keep.sy = keep.sx + gone.sx, keep.sy + gone.sy
            keep.captures += gone.captures
            keep.looks += gone.looks
            if gone.best_score > keep.best_score:
                keep.best_score, keep.best_capture_id = gone.best_score, gone.best_capture_id
            del self.locations[gone.location_id]
            merged.append(gone.location_id)
            loc = keep
        self._mark_splice(loc)
        return loc, merged

    def defects(self) -> list[Location]:
        """Confirmed locations away from splices, most laps first."""
        return sorted((x for x in self.locations.values() if x.confirmed and not x.at_splice),
                      key=lambda x: (-x.laps, x.location_id))

    def at_splice(self) -> list[Location]:
        """Every location at a splice, confirmed or not."""
        return sorted((x for x in self.locations.values() if x.at_splice), key=lambda x: (-x.laps, x.location_id))

    def unconfirmed(self) -> list[Location]:
        return sorted((x for x in self.locations.values() if not x.confirmed and not x.at_splice),
                      key=lambda x: x.location_id)
