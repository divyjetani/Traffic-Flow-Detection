import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data import utils as data_utils
import yaml

from _metrics import collect_metrics, write_metrics_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_utils.DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="weights/best_traffic.pt")
ap.add_argument("--data", default="VisDrone.yaml")
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--report", default="model_metrics.json")
ap.add_argument("--training-run", help="run directory containing args.yaml and results.csv")
a = ap.parse_args()
model = YOLO(a.weights)
evaluation = collect_metrics(model, data=a.data, imgsz=a.imgsz)
report_path = Path(a.report)
report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else {}
training = report.get("training", {})
reported_checkpoint = training.get("checkpoint") if isinstance(training, dict) else None
if not reported_checkpoint or Path(reported_checkpoint).resolve() != Path(a.weights).resolve():
    report["training"] = {"checkpoint": str(a.weights)}
if a.training_run:
    run_path = Path(a.training_run)
    run_args = yaml.safe_load((run_path / "args.yaml").read_text(encoding="utf-8"))
    with (run_path / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No completed training epochs found in {run_path / 'results.csv'}")
    report["training"] = {
        "initialized_from": run_args.get("model"),
        "checkpoint": str(a.weights),
        "epochs_requested": run_args.get("epochs"),
        "epochs_completed": len(rows),
        "image_size": run_args.get("imgsz"),
        "batch_size": run_args.get("batch"),
        "dataset_fraction": run_args.get("fraction", 1.0),
        "device": run_args.get("device"),
        "metrics": {
            key: float(value) for key, value in rows[-1].items()
            if key.startswith(("train/", "metrics/"))
        },
    }
report.update(report_version=1, dataset=a.data, validation_split="val", evaluation=evaluation)
write_metrics_report(report_path, report)
print(json.dumps(evaluation["metrics"], indent=2))
print(f"Metrics report -> {a.report}")
