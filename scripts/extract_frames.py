"""Sample frames from your own traffic videos to label (CVAT / Roboflow / Label Studio / labelImg)."""
import argparse
from pathlib import Path

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("videos", nargs="+")
ap.add_argument("--out", default="data/custom/images")
ap.add_argument("--every-sec", type=float, default=1.0)
a = ap.parse_args()

Path(a.out).mkdir(parents=True, exist_ok=True)
for v in a.videos:
    cap = cv2.VideoCapture(v)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step, i, saved = max(1, int(fps * a.every_sec)), 0, 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            cv2.imwrite(f"{a.out}/{Path(v).stem}_{i:06d}.jpg", fr)
            saved += 1
        i += 1
    print(f"{v}: saved {saved} frames")
