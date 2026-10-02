import cv2
import numpy as np

from . import road as roadmod
from .flow import DIR_NAMES, PALETTE


def draw_hud(img, an, t):
    rows, cols = [f"t={t:6.1f}s   counted={an.total_counted()}"], [(255, 255, 255)]
    rates = an.rates(t)
    for b, name in enumerate(DIR_NAMES):
        c = sum(an.dir_counts.get(name, {}).values())
        if c:
            rows.append(f"{name:>2}: {c:4d}  ({rates.get(name, 0):.0f}/min)")
            cols.append(PALETTE[b])
    for key, d in an.line_counts.items():
        rows.append(f"{key}: {sum(d.values())}")
        cols.append((0, 255, 255))
    h, w = 24 * len(rows) + 10, 270
    roi = img[8:8 + h, 8:8 + w]
    img[8:8 + h, 8:8 + w] = (roi * 0.35).astype(np.uint8)
    for i, (r, c) in enumerate(zip(rows, cols)):
        cv2.putText(img, r, (16, 30 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)


def draw_frame(frame, dets, heads, an, t, seg_mask, cfg):
    out = roadmod.tint(frame, seg_mask) if seg_mask is not None else frame
    out = an.field.render(out, cfg["output"]["overlay_alpha"])
    if out is frame:
        out = frame.copy()
    for ln in an.lines:
        p1, p2 = tuple(map(int, ln["p1"])), tuple(map(int, ln["p2"]))
        cv2.line(out, p1, p2, (0, 255, 255), 2)
        cv2.putText(out, ln["name"], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for tid, cls, (x1, y1, x2, y2) in dets:
        hb = heads.get(tid)
        col = PALETTE[hb] if hb is not None else (200, 200, 200)
        pts = np.array(an.tracks[tid].pts, np.int32).reshape(-1, 1, 2)
        if len(pts) > 1:
            cv2.polylines(out, [pts], False, col, 2)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        label = f"{cls} #{tid}" + (f" {DIR_NAMES[hb]}" if hb is not None else "")
        cv2.putText(out, label, (int(x1), max(12, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    draw_hud(out, an, t)
    return out
