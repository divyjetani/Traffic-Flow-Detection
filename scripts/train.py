"""Fine-tune a YOLO detector on a traffic dataset, then copy best weights to weights/best_traffic.pt.

Examples:
  python scripts/train.py --data VisDrone.yaml                  # aerial/drone views (auto-downloads)
  python scripts/train.py --data data/detrac.yaml               # CCTV views (after scripts/convert_detrac.py)
  python scripts/train.py --data data/traffic.yaml              # your own labelled frames
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="VisDrone.yaml")
ap.add_argument("--model", default="yolo11s.pt", help="pretrained start point (n/s/m/l/x)")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--batch", type=int, default=16, help="-1 = auto")
ap.add_argument("--device", default=None)
ap.add_argument("--name", default="traffic")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()

model = YOLO(a.model)
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            project="runs/train", name=a.name, patience=20, cos_lr=True, close_mosaic=10,
            degrees=0.0, flipud=0.0, fliplr=0.5, resume=a.resume)
best = Path(model.trainer.best)
Path("weights").mkdir(exist_ok=True)
shutil.copy(best, "weights/best_traffic.pt")
print("Best weights ->", "weights/best_traffic.pt")
print(YOLO("weights/best_traffic.pt").val(data=a.data, imgsz=a.imgsz))
