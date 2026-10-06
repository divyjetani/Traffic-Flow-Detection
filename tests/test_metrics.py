import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts._metrics import collect_metrics, write_metrics_report


class MetricReportTests(unittest.TestCase):
    def test_collects_overall_per_class_and_speed_metrics(self):
        box = SimpleNamespace(
            ap_class_index=np.array([0]),
            p=np.array([0.8]),
            r=np.array([0.7]),
            ap50=np.array([0.75]),
            ap=np.array([[0.75, 0.5]]),
            mp=0.8,
            mr=0.7,
            map50=0.75,
            map=0.625,
        )
        results = SimpleNamespace(box=box, speed={"inference": 2.5})
        model = SimpleNamespace(names={0: "car"}, val=lambda **_: results)

        report = collect_metrics(model, data="VisDrone.yaml", imgsz=640)

        self.assertEqual(report["metrics"]["map50"], 0.75)
        self.assertEqual(report["per_class"]["car"]["precision"], 0.8)
        self.assertEqual(report["speed_ms_per_image"]["inference"], 2.5)

    def test_writes_valid_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "model_metrics.json"
            write_metrics_report(target, {"evaluation": {"metrics": {"map50": 0.75}}})
            self.assertEqual(
                json.loads(target.read_text(encoding="utf-8"))["evaluation"]["metrics"]["map50"],
                0.75,
            )


if __name__ == "__main__":
    unittest.main()
