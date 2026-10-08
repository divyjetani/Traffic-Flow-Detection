import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data import utils as data_utils

from _dataset_split import create_visdrone_split
from _metrics import collect_metrics, write_metrics_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
data_utils.DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="VisDrone.yaml")
ap.add_argument("--model", default="yolo11s.pt", help="pretrained start point (n/s/m/l/x)")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--imgsz", type=int, default=960)
ap.add_argument("--batch", type=int, default=16, help="-1 = auto")
ap.add_argument("--fraction", type=float, default=1.0,
                help="fraction of the training split to use (0 < fraction <= 1)")
ap.add_argument("--split-seed", type=int, default=42,
                help="seed for the reproducible VisDrone 70/15/15 split")
ap.add_argument("--device", default=None)
ap.add_argument("--name", default="traffic")
ap.add_argument("--resume", action="store_true")
ap.add_argument("--report", default=str(PROJECT_ROOT / "model_metrics.json"),
                help="JSON file for training details and held-out evaluation metrics")
a = ap.parse_args()

dataset = a.data
split_metadata = None
if a.data == "VisDrone.yaml":
    source_dataset = data_utils.check_det_dataset(a.data)
    split_metadata = create_visdrone_split(
        source_dataset["path"],
        PROJECT_ROOT / "data" / "datasets" / "VisDrone-70-15-15",
        seed=a.split_seed,
    )
    dataset = split_metadata["dataset_yaml"]

model = YOLO(a.model)
model.train(data=dataset, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
            fraction=a.fraction,
            project=str(PROJECT_ROOT / "runs" / "train"), name=a.name, patience=20,
            cos_lr=True, close_mosaic=10,
            degrees=0.0, flipud=0.0, fliplr=0.5, resume=a.resume)
best = Path(model.trainer.best)
trained_weights = PROJECT_ROOT / "weights" / "best_traffic.pt"
trained_weights.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(best, trained_weights)
trained_model = YOLO(str(trained_weights))
evaluation = {
    "validation": collect_metrics(trained_model, data=dataset, imgsz=a.imgsz, split="val"),
}
if split_metadata is not None:
    evaluation["test"] = collect_metrics(
        trained_model, data=dataset, imgsz=a.imgsz, split="test"
    )

training_metrics = {
    str(key): float(value) for key, value in getattr(model.trainer, "metrics", {}).items()
    if isinstance(value, (int, float))
}
report = {
    "report_version": 1,
    "dataset": a.data,
    "dataset_split": split_metadata,
    "validation_split": "val",
    "test_split": "test" if split_metadata is not None else None,
    "training": {
        "initialized_from": a.model,
        "checkpoint": trained_weights.relative_to(PROJECT_ROOT).as_posix(),
        "epochs_requested": a.epochs,
        "epochs_completed": int(model.trainer.epoch + 1),
        "image_size": a.imgsz,
        "batch_size": a.batch,
        "dataset_fraction": a.fraction,
        "device": str(model.trainer.device),
        "metrics": training_metrics,
    },
    "evaluation": evaluation,
}
write_metrics_report(a.report, report)
print("Best weights ->", trained_weights)
print("Metrics report ->", a.report)
print("Validation metrics ->", json.dumps(evaluation["validation"]["metrics"], indent=2))
if "test" in evaluation:
    print("Test metrics ->", json.dumps(evaluation["test"]["metrics"], indent=2))
