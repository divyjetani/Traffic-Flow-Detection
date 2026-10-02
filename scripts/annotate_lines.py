"""Click two points per counting line on the first frame. 'u' undo, 's' save, 'q' quit."""
import argparse
import json

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="video or image")
ap.add_argument("--out", default="configs/lines.json")
a = ap.parse_args()

cap = cv2.VideoCapture(a.source)
ok, frame = cap.read()
assert ok, "cannot read source"
pts = []


def redraw():
    img = frame.copy()
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i + 1], (0, 255, 255), 2)
        cv2.putText(img, f"L{i // 2 + 1}", pts[i], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    for p in pts:
        cv2.circle(img, p, 4, (0, 0, 255), -1)
    cv2.imshow("annotate", img)


def on_mouse(ev, x, y, *_):
    if ev == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        redraw()


cv2.namedWindow("annotate")
cv2.setMouseCallback("annotate", on_mouse)
redraw()
while True:
    k = cv2.waitKey(50) & 0xFF
    if k == ord("u") and pts:
        pts.pop()
        redraw()
    elif k == ord("s"):
        lines = [{"name": f"L{i // 2 + 1}", "p1": list(pts[i]), "p2": list(pts[i + 1]),
                  "labels": ["dir1", "dir2"]} for i in range(0, len(pts) - 1, 2)]
        json.dump({"lines": lines}, open(a.out, "w"), indent=2)
        print(f"saved {len(lines)} line(s) to {a.out} (edit 'labels' to name the two crossing directions)")
        break
    elif k == ord("q"):
        break
cv2.destroyAllWindows()
