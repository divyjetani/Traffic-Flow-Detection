import tempfile
import unittest
from pathlib import Path

import yaml

from scripts._dataset_split import create_visdrone_split


class DatasetSplitTests(unittest.TestCase):
    def test_creates_reproducible_labeled_70_15_15_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_root = root / "VisDrone"
            for split in ("train", "val", "test"):
                image_dir = data_root / "images" / split
                label_dir = data_root / "labels" / split
                image_dir.mkdir(parents=True)
                label_dir.mkdir(parents=True)
                for index in range(10):
                    (image_dir / f"{split}_{index}.jpg").touch()
                    (label_dir / f"{split}_{index}.txt").write_text(
                        "3 0.5 0.5 0.2 0.2\n",
                        encoding="utf-8",
                    )

            output_dir = root / "generated-split"
            metadata = create_visdrone_split(data_root, output_dir, seed=7)
            repeated = create_visdrone_split(data_root, output_dir, seed=7)

            self.assertEqual(metadata, repeated)
            self.assertEqual(metadata["total_images"], 30)
            self.assertEqual(metadata["train_images"], 21)
            self.assertEqual(metadata["validation_images"], 4)
            self.assertEqual(metadata["test_images"], 5)
            self.assertEqual(
                metadata["target_percentages"],
                {"train": 70, "validation": 15, "test": 15},
            )
            self.assertEqual(
                sum(metadata["actual_percentages"].values()),
                100,
            )

            split_lists = {
                name: (output_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines()
                for name in ("train", "val", "test")
            }
            self.assertEqual(len(set(sum(split_lists.values(), []))), 30)
            for paths in split_lists.values():
                for image_path in paths:
                    image = Path(image_path)
                    label = data_root / "labels" / image.parent.name / f"{image.stem}.txt"
                    self.assertTrue(label.is_file())

            config = yaml.safe_load((output_dir / "dataset.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config["train"], (output_dir / "train.txt").as_posix())
            self.assertEqual(config["val"], (output_dir / "val.txt").as_posix())
            self.assertEqual(config["test"], (output_dir / "test.txt").as_posix())

    def test_fails_for_missing_annotation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val", "test"):
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
            (root / "images" / "train" / "unlabeled.jpg").touch()

            with self.assertRaisesRegex(FileNotFoundError, "Missing YOLO annotation"):
                create_visdrone_split(root, root / "generated")


if __name__ == "__main__":
    unittest.main()
