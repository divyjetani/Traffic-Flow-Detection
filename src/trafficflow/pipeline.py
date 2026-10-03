from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import road as roadmod
from .flow import DIR_NAMES, FlowAnalyzer
from .viz import draw_frame


def load_lines(path):
    if not path:
        return []
    data = json.load(open(path))
    return [dict(name=l.get("name", f"L{i + 1}"), p1=tuple(l["p1"]), p2=tuple(l["p2"]),
                 labels=l.get("labels", ["dir1", "dir2"])) for i, l in enumerate(data["lines"])]


def _fps(src):
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps if fps > 1 else 25.0


def _frame_count(src):
    cap = cv2.VideoCapture(src)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(count, 0)


def _parse(r, names):
    dets = []
    if r.boxes is not None and r.boxes.id is not None:
        ids = r.boxes.id.int().cpu().tolist()
        cls = r.boxes.cls.int().cpu().tolist()
        xyxy = r.boxes.xyxy.cpu().numpy()
        dets = [(i, names[c], tuple(float(v) for v in b)) for i, c, b in zip(ids, cls, xyxy)]
    return dets


def run(source, cfg, out_dir="outputs", lines_path=None, show=False, max_frames=None,
        annotated_filename="annotated.mp4",
        progress_callback: Callable[[int, int, np.ndarray], None] | None = None):
    from ultralytics import YOLO

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mc = cfg["model"]
    model = YOLO(mc["weights"])
    names = model.names
    wanted = {n.lower() for n in mc["vehicle_names"]}
    class_ids = [i for i, n in names.items() if n.lower() in wanted]
    if not class_ids:
        raise SystemExit(f"No vehicle classes {sorted(wanted)} in model classes {list(names.values())}")

    src = int(source) if str(source).isdigit() else str(source)
    fps, total_frames, lines = _fps(src), _frame_count(src), load_lines(lines_path)
    if max_frames:
        total_frames = min(total_frames, max_frames) if total_frames else max_frames
    an = writer = first = seg = None
    n = 0
    stream = model.track(source=src, stream=True, persist=True, tracker=mc["tracker"], conf=mc["conf"],
                         iou=mc["iou"], imgsz=mc["imgsz"], classes=class_ids, device=mc["device"], verbose=False)
    for i, r in enumerate(stream):
        frame = r.orig_img
        if an is None:
            h, w = frame.shape[:2]
            an, first = FlowAnalyzer(fps, (w, h), cfg, lines), frame.copy()
            if cfg["output"]["save_video"]:
                writer = cv2.VideoWriter(str(out / annotated_filename), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            if cfg["road"]["use_segformer"]:
                seg = roadmod.segformer_road_mask(frame, cfg["road"]["segformer_model"], mc["device"])
        dets = _parse(r, names)
        heads = an.update(i, dets)
        if i % cfg["flow"]["update_every"] == 0:
            an.field.refresh()
        vis = draw_frame(frame, dets, heads, an, i / fps, seg, cfg)
        if writer:
            writer.write(vis)
        n = i + 1
        if progress_callback:
            progress_callback(n, total_frames, vis)
        if show:
            cv2.imshow("traffic flow (q to quit)", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if i % 100 == 0:
            print(f"frame {i}  counted={an.total_counted()}")
        if max_frames and n >= max_frames:
            break
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    if an is None:
        raise SystemExit("No frames read from source.")
    _save(out, an, first, seg, cfg, n)
    return an


def _save(out, an, first, seg, cfg, n):
    an.field.refresh()
    with open(out / "events.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["frame", "time_s", "track_id", "class", "kind", "name"])
        wr.writeheader()
        wr.writerows(an.events)
    with open(out / "per_minute.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["minute"] + DIR_NAMES)
        for m, d in sorted(an.per_minute().items()):
            wr.writerow([m] + [d.get(k, 0) for k in DIR_NAMES])
    json.dump(an.summary(n), open(out / "summary.json", "w"), indent=2)
    base = roadmod.tint(first, seg) if seg is not None else first
    cv2.imwrite(str(out / "flow_map.png"), an.field.render(base, 0.5))
    print(f"Done. Results in {out}/  (annotated video, flow_map.png, summary.json, events.csv, per_minute.csv)")
