"""Create reproducible VisDrone YOLO manifests using only labeled images."""
import random
from pathlib import Path

import yaml


VISDRONE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]
VISDRONE_SOURCE_SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def create_visdrone_split(data_root, output_dir, seed=42):
    """Write image-list manifests with a deterministic 70/15/15 partition."""
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    samples = []
    for source_split in VISDRONE_SOURCE_SPLITS:
        image_dir = data_root / "images" / source_split
        label_dir = data_root / "labels" / source_split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"VisDrone labeled split is missing: {image_dir} or {label_dir}"
            )
        for image in sorted(image_dir.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES:
                label = label_dir / f"{image.stem}.txt"
                if not label.is_file():
                    raise FileNotFoundError(f"Missing YOLO annotation for {image}: {label}")
                samples.append(image.resolve())

    if len(samples) < 3:
        raise ValueError(f"At least 3 labeled images are required; found {len(samples)}")

    random.Random(seed).shuffle(samples)
    train_count = round(len(samples) * 0.70)
    validation_count = round(len(samples) * 0.15)
    test_count = len(samples) - train_count - validation_count
    if min(train_count, validation_count, test_count) < 1:
        raise ValueError("The dataset is too small to populate all three splits.")

    split_paths = {
        "train": samples[:train_count],
        "val": samples[train_count : train_count + validation_count],
        "test": samples[train_count + validation_count :],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, paths in split_paths.items():
        (output_dir / f"{split_name}.txt").write_text(
            "".join(f"{path.as_posix()}\n" for path in paths),
            encoding="utf-8",
        )

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": data_root.as_posix(),
                "train": (output_dir / "train.txt").as_posix(),
                "val": (output_dir / "val.txt").as_posix(),
                "test": (output_dir / "test.txt").as_posix(),
                "names": VISDRONE_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    return {
        "seed": seed,
        "source_splits": list(VISDRONE_SOURCE_SPLITS),
        "total_images": len(samples),
        "train_images": train_count,
        "validation_images": validation_count,
        "test_images": test_count,
        "target_percentages": {"train": 70, "validation": 15, "test": 15},
        "actual_percentages": {
            "train": round(train_count * 100 / len(samples), 2),
            "validation": round(validation_count * 100 / len(samples), 2),
            "test": round(test_count * 100 / len(samples), 2),
        },
        "dataset_yaml": dataset_yaml.as_posix(),
    }
