# Traffic Flow Detection (Deep Learning)

Detects vehicles with **YOLO**, tracks them with **ByteTrack**, and measures **traffic flow per road direction**.
It also **learns the road and its flow directions** from the traffic itself (no road labels needed), and can
optionally add a **SegFormer road segmentation** for ground-level cameras.

```
video -> YOLO detect -> ByteTrack IDs -> headings (8 compass bins) -> per-direction counts + vehicles/min
                                                   \-> flow field -> road mask + one-way flow zones + arrows
                                      virtual counting lines (optional, 2-way counts per line)
```

## Outputs (`outputs/`)
| File | What |
|---|---|
| `annotated.mp4` | boxes, IDs, trails, direction colours, road/flow overlay with arrows, live HUD (counts, veh/min) |
| `flow_map.png` | learned road + direction map on the first frame |
| `summary.json` | totals, per-class, per-direction, per-line counts, dominant direction, flow zones |
| `events.csv` | every counting event (time, track id, class, direction/line) |
| `per_minute.csv` | counts per direction per minute |

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # install a CUDA build of torch first if you have an NVIDIA GPU
python tests/test_flow.py                           # sanity test of the flow logic (no model needed)
```

## Run (choose your detector)

The web app shows a detector selector when you upload a video. **My trained traffic model**
is selected by default. It is unavailable until training creates
`weights/best_traffic.pt`; in that case, explicitly select **Pretrained YOLO11s (COCO)** if
you want to analyze a video before training. The app never silently falls back to pretrained
weights.

The command-line runner prompts for the same choice, or accepts it explicitly:
```bash
python scripts/run.py --source data/videos/traffic.mp4 --model-choice trained
python scripts/run.py --source data/videos/traffic.mp4 --model-choice pretrained
```
Omit `--model-choice` to be prompted:
```bash
python scripts/run.py --source data/videos/traffic.mp4
python scripts/run.py --source 0                                   # webcam
python scripts/run.py --source rtsp://user:pass@ip/stream          # IP camera
python scripts/run.py --source traffic.mp4 --anchor center         # drone / top-down footage
python scripts/run.py --source traffic.mp4 --segformer             # + road segmentation (pip install transformers)
```
Use `--model-choice trained` or `--model-choice pretrained` to choose from the command line.
The pretrained option uses `yolo11s.pt` (COCO); the trained option uses the fine-tuned
VisDrone checkpoint created below.

## Web app
On Windows, run `run` from Command Prompt at the repository root (or `.\run.bat` from PowerShell).
The launcher creates/activates `.venv` as needed, installs missing dependencies, starts the web app,
and opens the upload page in your browser. Alternatively, install the dependencies above and run:
```bash
python app.py
```
Open http://127.0.0.1:5000, upload a video, and watch annotated frames update live with a frame
counter and percentage progress when the source video reports its duration. The original and
processed videos appear side by side; the downloadable result is saved in `outputs/` as
`<original-name>_annoted.mp4`. If that name already exists, a unique suffix is added so earlier
results are preserved. The app uses FFmpeg, when available, to encode browser-playable H.264 video.
Uploaded temporary video files are removed after processing.

### Counting lines (optional, for a specific road section)
```bash
python scripts/annotate_lines.py --source data/videos/traffic.mp4 --out configs/lines.json
# edit "labels" in the JSON, e.g. ["Northbound","Southbound"]; first label = crossing with positive cross-product
python scripts/run.py --source data/videos/traffic.mp4 --lines configs/lines.json
```

## Train / fine-tune your own model

The default command downloads the public VisDrone dataset automatically on first run and
fine-tunes YOLO11s on its training split. It then evaluates the custom checkpoint on the
held-out validation split. Dataset files are stored under `data/datasets/VisDrone/` and
ignored by Git. The trained inference choice loads these fine-tuned weights, not the original
COCO checkpoint.
```bash
python scripts/train.py --data VisDrone.yaml                       # drone view; dataset auto-downloads
python scripts/convert_detrac.py --images <DETRAC-train-data> --ann <DETRAC-Train-Annotations-XML>
python scripts/train.py --data data/detrac.yaml                    # CCTV view
python scripts/extract_frames.py my.mp4 && # label in CVAT/Roboflow, export YOLO, fill data/traffic.yaml
python scripts/train.py --data data/traffic.yaml --epochs 80
python scripts/evaluate.py --weights weights/best_traffic.pt --data data/traffic.yaml
python scripts/run.py --source my.mp4 --weights weights/best_traffic.pt
```
Training writes the single machine-readable [`model_metrics.json`](./model_metrics.json)
report. It records the dataset and split, fine-tuning settings, completed epochs, overall
precision, recall, mAP50 and mAP50-95, per-class metrics, and inference speed. These are
object-detection metrics rather than a single classification-style accuracy score. To refresh
measurements for an existing checkpoint, run `scripts/evaluate.py` with the same dataset and
image size.

### Latest trained-model validation

Run `python scripts/train.py` to create `model_metrics.json`. Use that report as the source
of truth for the measured metrics of the latest checkpoint; values are generated from the
actual held-out validation run rather than guessed or hard-coded.

## Where to get data
| Need | Source |
|---|---|
| Aerial vehicle detection | VisDrone: https://github.com/VisDrone/VisDrone-Dataset (auto-downloaded by `VisDrone.yaml`) |
| CCTV detection + tracking | UA-DETRAC: https://detrac-db.rit.albany.edu/ (also mirrored on Kaggle; check the license) |
| Dashcam / driving | BDD100K: https://bdd-data.berkeley.edu/ |
| Traffic cameras, city scale | AI City Challenge: https://www.aicitychallenge.org/ |
| Ready labelled sets | Roboflow Universe (search "vehicle detection"): https://universe.roboflow.com/ |
| Road segmentation | Cityscapes (used by the SegFormer weights): https://www.cityscapes-dataset.com/ |
| Test videos | Pexels / Pixabay "traffic" stock footage; any fixed-camera clip works |

## Docs / papers
- Ultralytics: https://docs.ultralytics.com (modes: train, val, track) 
- ByteTrack: https://arxiv.org/abs/2110.06864
- SegFormer (Hugging Face): https://huggingface.co/docs/transformers/model_doc/segformer
- YOLO dataset format: https://docs.ultralytics.com/datasets/detect/

## How "road direction" works
1. Each tracked vehicle's heading over the last `heading_window` frames is binned into 8 directions (image up = N).
2. Every heading votes into a grid cell. Cells with enough votes = **road**; each cell's dominant direction = **road direction**.
3. Connected cells with the same direction become **flow zones** (e.g. the two carriageways of a highway get different colours and arrows).
4. Each vehicle is counted once per direction after it travels `min_travel_px`; `vehicles/min` uses a sliding window.
The flow map improves as more traffic passes, so run on a few minutes of footage.

## Tuning (`configs/default.yaml`)
| Problem | Change |
|---|---|
| Missing small/far cars | raise `imgsz`, lower `conf`, or fine-tune |
| Double counts / ID switches | raise `min_travel_px`, try `tracker: botsort.yaml` |
| Road map patchy | lower `min_cell_votes`, raise `blur_cells`, run longer video |
| Road map too wide/noisy | raise `min_cell_votes` or `min_zone_cells` |
| Top-down camera | `--anchor center` |

## Limitations
- Static camera assumed (moving dashcam footage needs ego-motion compensation).
- Directions are in image space (not map north). Speed in km/h needs a pixel-to-metre calibration (homography), not included.
- Heavy occlusion/night scenes need fine-tuning for good accuracy.

## Structure
```
configs/   default.yaml, lines_example.json
src/trafficflow/  flow.py (counting, flow field, zones) pipeline.py viz.py road.py config.py
scripts/   run.py train.py evaluate.py annotate_lines.py extract_frames.py convert_detrac.py
data/      traffic.yaml (template), README.md      weights/  outputs/  tests/test_flow.py
```
