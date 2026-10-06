import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from _metrics import collect_metrics, write_metrics_report

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights/best_traffic.pt")
ap.add_argument("--data", default="VisDrone.yaml")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--report", default="model_metrics.json")
a = ap.parse_args()
model = YOLO(a.weights)
evaluation = collect_metrics(model, data=a.data, imgsz=a.imgsz)
report_path = Path(a.report)
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
training = report.get("training", {})
reported_checkpoint = training.get("checkpoint") if isinstance(training, dict) else None
if not reported_checkpoint or Path(reported_checkpoint).resolve() != Path(a.weights).resolve():
    report["training"] = {"checkpoint": str(a.weights)}
report.update(report_version=1, dataset=a.data, validation_split="val", evaluation=evaluation)
write_metrics_report(report_path, report)
print(json.dumps(evaluation["metrics"], indent=2))
print(f"Metrics report -> {a.report}")
