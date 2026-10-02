import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights/best_traffic.pt")
ap.add_argument("--data", default="VisDrone.yaml")
ap.add_argument("--imgsz", type=int, default=960)
a = ap.parse_args()
m = YOLO(a.weights).val(data=a.data, imgsz=a.imgsz)
print(f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
