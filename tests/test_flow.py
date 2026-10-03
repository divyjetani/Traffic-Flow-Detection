"""Synthetic test of the flow logic (no model / video needed):  python tests/test_flow.py"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficflow.config import load_config  # noqa: E402
from trafficflow.flow import FlowAnalyzer, FlowField  # noqa: E402


def make():
    cfg = load_config()
    cfg["flow"]["min_cell_votes"] = 1.0  # only 5 synthetic cars per direction -> fewer votes than real traffic
    lines = [dict(name="A", p1=(300, 0), p2=(300, 480), labels=["Westbound", "Eastbound"])]
    return FlowAnalyzer(10.0, (640, 480), cfg, lines)


def test_directions_lines_and_zones():
    an = make()
    for f in range(90):
        dets = []
        for k in range(5):
            xe, xw = -k * 40 + f * 10, 640 + k * 40 - f * 10
            if 0 <= xe < 640:
                dets.append((k + 1, "car", (xe - 10, 80, xe + 10, 100)))
            if 0 <= xw < 640:
                dets.append((k + 11, "truck", (xw - 10, 200, xw + 10, 220)))
        an.update(f, dets)
    assert an.dir_counts["E"] and an.dir_counts["W"], an.dir_counts
    assert sum(an.dir_counts["E"].values()) == 5 and sum(an.dir_counts["W"].values()) == 5
    assert sum(an.line_counts["A:Eastbound"].values()) == 5, an.line_counts
    assert sum(an.line_counts["A:Westbound"].values()) == 5, an.line_counts
    names = {z["name"] for z in an.field.zones}
    assert {"E", "W"} <= names, names
    assert an.field.road_mask().any()
    s = an.summary(90)
    assert s["total_vehicles_counted"] == 10
    print("OK", s["direction_counts"], s["line_counts"], [z["name"] for z in s["flow_zones"]])


def test_render_hides_flow_arrows_by_default():
    import numpy as np

    field = FlowField((640, 480), cell=16, min_votes=0.1, blur=1.5, min_area=1)
    field.add(100, 100, 0)
    field.refresh()
    with patch("trafficflow.flow.cv2.arrowedLine") as draw_arrow:
        field.render(np.zeros((480, 640, 3), dtype=np.uint8))
    draw_arrow.assert_not_called()


if __name__ == "__main__":
    test_directions_lines_and_zones()
    test_render_hides_flow_arrows_by_default()
