"""Direction-aware flow analytics.

tracks -> headings -> (a) per-direction counts, (b) virtual line counts,
(c) learned flow field => road mask + one-way flow zones (road direction).
Assumes a STATIC camera. Image "up" is called N.
"""
from __future__ import annotations

import math
from collections import deque

import cv2
import numpy as np

N_BINS = 8
DIR_NAMES = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
PALETTE = [(60, 76, 231), (0, 165, 255), (60, 200, 60), (200, 200, 0),
           (255, 120, 40), (200, 60, 200), (160, 160, 160), (0, 215, 255)]  # BGR


def angle_to_bin(dx: float, dy: float) -> int:
    ang = math.atan2(-dy, dx) % (2 * math.pi)  # image y points down -> flip
    return int(round(ang / (2 * math.pi / N_BINS))) % N_BINS


def bin_vector(b: int):
    a = b * 2 * math.pi / N_BINS
    return math.cos(a), -math.sin(a)  # image coordinates


def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def seg_intersect(p1, p2, q1, q2) -> bool:
    """Did the step p1->p2 cross (or land on) the line segment q1-q2? Touching counts once."""
    d1, d2 = _ccw(q1, q2, p1), _ccw(q1, q2, p2)
    d3, d4 = _ccw(p1, p2, q1), _ccw(p1, p2, q2)
    return d1 != 0 and d1 * d2 <= 0 and d3 * d4 <= 0


def _inc(d: dict, key: str, cls: str):
    sub = d.setdefault(key, {})
    sub[cls] = sub.get(cls, 0) + 1


class Track:
    def __init__(self, tid, cls, maxlen):
        self.tid, self.cls = tid, cls
        self.pts = deque(maxlen=maxlen)
        self.first = None
        self.dir_counted = False
        self.crossed = set()


class FlowField:
    """Accumulates motion votes per grid cell and per direction bin."""

    def __init__(self, size, cell, min_votes, blur, min_area):
        self.w, self.h = size
        self.cell, self.min_votes, self.blur, self.min_area = cell, min_votes, blur, min_area
        self.gw, self.gh = math.ceil(self.w / cell), math.ceil(self.h / cell)
        self.hist = np.zeros((self.gh, self.gw, N_BINS), np.float32)
        self._state = None

    def add(self, x, y, b):
        gx, gy = int(x // self.cell), int(y // self.cell)
        if 0 <= gx < self.gw and 0 <= gy < self.gh:
            self.hist[gy, gx, b] += 1

    def refresh(self):
        h = np.empty_like(self.hist)
        for b in range(N_BINS):
            h[..., b] = cv2.GaussianBlur(self.hist[..., b], (0, 0), self.blur)
        total, dom = h.sum(2), h.argmax(2)
        mask = (total >= self.min_votes).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        keep = np.zeros(mask.shape, bool)
        zones = []
        for b in range(N_BINS):
            m = ((dom == b) & (mask > 0)).astype(np.uint8)
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for j in range(1, n):
                area = int(stats[j, cv2.CC_STAT_AREA])
                if area < self.min_area:
                    continue
                keep |= lab == j
                zones.append({"dir": b, "name": DIR_NAMES[b], "area_px": area * self.cell ** 2,
                              "centroid": [float(cent[j][0] * self.cell + self.cell / 2),
                                           float(cent[j][1] * self.cell + self.cell / 2)]})
        self._state = (keep, dom, zones)
        return self._state

    @property
    def zones(self):
        return (self._state or self.refresh())[2]

    def _up(self, a):
        return cv2.resize(a, (self.gw * self.cell, self.gh * self.cell),
                          interpolation=cv2.INTER_NEAREST)[: self.h, : self.w]

    def road_mask(self):
        keep = (self._state or self.refresh())[0]
        return self._up(keep.astype(np.uint8)) > 0

    def render(self, frame, alpha=0.35, arrows=True):
        keep, dom, _ = self._state or self.refresh()
        if not keep.any():
            return frame
        col = np.zeros((self.gh, self.gw, 3), np.uint8)
        col[keep] = np.array(PALETTE, np.uint8)[dom[keep]]
        col, sel = self._up(col), self._up(keep.astype(np.uint8)) > 0
        out = frame.copy()
        out[sel] = cv2.addWeighted(frame, 1 - alpha, col, alpha, 0)[sel]
        if arrows:
            stride = max(1, 64 // self.cell)
            for gy in range(stride // 2, self.gh, stride):
                for gx in range(stride // 2, self.gw, stride):
                    if keep[gy, gx]:
                        vx, vy = bin_vector(int(dom[gy, gx]))
                        c = (int(gx * self.cell + self.cell / 2), int(gy * self.cell + self.cell / 2))
                        tip = (int(c[0] + vx * 26), int(c[1] + vy * 26))
                        cv2.arrowedLine(out, c, tip, (255, 255, 255), 2, tipLength=0.4)
        return out


class FlowAnalyzer:
    def __init__(self, fps, size, cfg, lines=None):
        f = cfg["flow"]
        self.fps, self.size = fps, size
        self.win, self.min_heading, self.min_travel = f["heading_window"], f["min_heading_px"], f["min_travel_px"]
        self.anchor = f["anchor"]
        self.rate_window = f["rate_window_s"]
        self.trail = cfg["output"]["trail_len"]
        self.lines = lines or []
        self.tracks: dict[int, Track] = {}
        self.events: list[dict] = []
        self.dir_counts: dict[str, dict[str, int]] = {}
        self.line_counts: dict[str, dict[str, int]] = {}
        self.field = FlowField(size, f["cell_px"], f["min_cell_votes"], f["blur_cells"], f["min_zone_cells"])

    def _anchor(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, y2 if self.anchor == "bottom" else (y1 + y2) / 2)

    def _record(self, t, frame, tid, cls, kind, name):
        self.events.append({"frame": frame, "time_s": round(t, 2), "track_id": tid,
                            "class": cls, "kind": kind, "name": name})

    def update(self, frame_idx, dets):
        """dets: list of (track_id, class_name, (x1,y1,x2,y2)). Returns {track_id: heading_bin|None}."""
        t, heads = frame_idx / self.fps, {}
        for tid, cls, box in dets:
            p = self._anchor(box)
            tr = self.tracks.get(tid)
            if tr is None:
                tr = self.tracks[tid] = Track(tid, cls, max(self.trail, self.win + 1))
                tr.first = p
            prev = tr.pts[-1] if tr.pts else None
            tr.pts.append(p)

            hb = None
            if len(tr.pts) > self.win:
                a = tr.pts[-self.win - 1]
                dx, dy = p[0] - a[0], p[1] - a[1]
                if math.hypot(dx, dy) >= self.min_heading:
                    hb = angle_to_bin(dx, dy)
                    self.field.add(p[0], p[1], hb)
            heads[tid] = hb

            if prev is not None:  # virtual counting lines
                for i, ln in enumerate(self.lines):
                    if i in tr.crossed or not seg_intersect(prev, p, ln["p1"], ln["p2"]):
                        continue
                    tr.crossed.add(i)
                    q1, q2 = ln["p1"], ln["p2"]
                    s = (q2[0] - q1[0]) * (p[1] - prev[1]) - (q2[1] - q1[1]) * (p[0] - prev[0])
                    key = f'{ln["name"]}:{ln["labels"][0] if s > 0 else ln["labels"][1]}'
                    _inc(self.line_counts, key, cls)
                    self._record(t, frame_idx, tid, cls, "line", key)

            if not tr.dir_counted:  # calibration-free per-direction counting
                dx, dy = p[0] - tr.first[0], p[1] - tr.first[1]
                if math.hypot(dx, dy) >= self.min_travel:
                    tr.dir_counted = True
                    name = DIR_NAMES[angle_to_bin(dx, dy)]
                    _inc(self.dir_counts, name, cls)
                    self._record(t, frame_idx, tid, cls, "direction", name)
        return heads

    def total_counted(self):
        return sum(sum(v.values()) for v in self.dir_counts.values())

    def rates(self, t):
        """vehicles/min per direction over the last rate_window seconds."""
        span, lo, out = max(min(self.rate_window, t), 1.0), t - self.rate_window, {}
        for e in reversed(self.events):
            if e["time_s"] < lo:
                break
            if e["kind"] == "direction":
                out[e["name"]] = out.get(e["name"], 0) + 1
        return {k: v * 60.0 / span for k, v in out.items()}

    def per_minute(self):
        buckets: dict[int, dict[str, int]] = {}
        for e in self.events:
            if e["kind"] == "direction":
                b = buckets.setdefault(int(e["time_s"] // 60), {})
                b[e["name"]] = b.get(e["name"], 0) + 1
        return buckets

    def summary(self, n_frames):
        dirs = {k: {**v, "total": sum(v.values())} for k, v in self.dir_counts.items()}
        lines = {k: {**v, "total": sum(v.values())} for k, v in self.line_counts.items()}
        dom = max(dirs, key=lambda k: dirs[k]["total"]) if dirs else None
        return {"frames": n_frames, "duration_s": round(n_frames / self.fps, 2),
                "total_vehicles_counted": self.total_counted(),
                "dominant_direction": dom, "direction_counts": dirs, "line_counts": lines,
                "flow_zones": [{k: v for k, v in z.items() if k != "dir"} for z in self.field.zones]}
