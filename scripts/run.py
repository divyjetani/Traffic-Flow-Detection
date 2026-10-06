"""Run detection + tracking + direction-aware flow analysis on a video/webcam/RTSP stream."""
import _path  # noqa: F401
import argparse

from trafficflow.config import load_config
from trafficflow.models import MODEL_CHOICES, default_model_choice, resolve_model_weights
from trafficflow.pipeline import run

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="video path, webcam index (0), or RTSP/HTTP URL")
ap.add_argument("--config", default=None)
ap.add_argument("--model-choice", choices=MODEL_CHOICES, default=None,
                help="choose trained or pretrained weights; prompts when omitted")
ap.add_argument("--weights", default=None, help="advanced override, e.g. weights/best_traffic.pt")
ap.add_argument("--lines", default=None, help="counting lines JSON (see scripts/annotate_lines.py)")
ap.add_argument("--out", default="outputs")
ap.add_argument("--show", action="store_true")
ap.add_argument("--max-frames", type=int, default=None)
ap.add_argument("--segformer", action="store_true", help="also segment road with SegFormer (ground-level cams)")
ap.add_argument("--anchor", choices=["bottom", "center"], default=None)
a = ap.parse_args()

choice = a.model_choice
if choice is None and a.weights is None:
    print("Choose the detector model:")
    print(f"  1. {MODEL_CHOICES['trained']}")
    print(f"  2. {MODEL_CHOICES['pretrained']}")
    answer = input(f"Select 1 or 2 [default {1 if default_model_choice() == 'trained' else 2}]: ").strip()
    if not answer:
        choice = default_model_choice()
    elif answer == "1":
        choice = "trained"
    elif answer == "2":
        choice = "pretrained"
    else:
        ap.error("Select 1 or 2.")

cfg = load_config(a.config)
if a.weights:
    cfg["model"]["weights"] = a.weights
else:
    cfg["model"]["weights"] = resolve_model_weights(choice)
if a.segformer:
    cfg["road"]["use_segformer"] = True
if a.anchor:
    cfg["flow"]["anchor"] = a.anchor
run(a.source, cfg, a.out, a.lines, a.show, a.max_frames)
