"""Shared object-detection metric collection and JSON report output."""
import json
import os
import tempfile
from pathlib import Path


def collect_metrics(model, data, imgsz):
    results = model.val(data=data, imgsz=imgsz, split="val")
    box = results.box
    names = model.names
    per_class = {}
    for index, class_id in enumerate(box.ap_class_index):
        class_id = int(class_id)
        class_name = names[class_id] if isinstance(names, dict) else names[class_id]
        per_class[str(class_name)] = {
            "precision": float(box.p[index]),
            "recall": float(box.r[index]),
            "map50": float(box.ap50[index]),
            "map50_95": float(box.ap[index].mean()),
        }

    return {
        "dataset": str(data),
        "split": "val",
        "image_size": imgsz,
        "metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "per_class": per_class,
        "speed_ms_per_image": {
            str(key): float(value) for key, value in results.speed.items()
        },
    }


def write_metrics_report(path, report):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        os.replace(temporary_path, destination)
    except BaseException:
        Path(temporary_path).unlink(missing_ok=True)
        raise
