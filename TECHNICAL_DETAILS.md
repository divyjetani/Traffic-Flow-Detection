# Technical details

This document describes the current implementation of Traffic Flow Detection: how video is processed, how detections become traffic counts, how the learned flow map is built, what each script does, and how to interpret the generated files.

## 1. Project purpose and scope

The project analyzes video from a file, webcam, or network stream. It uses a YOLO object detector through Ultralytics, associates detections over time with an Ultralytics tracker (ByteTrack by default), then derives vehicle directions and traffic statistics from the tracked motion.

The application does not require road-lane labels or a pre-drawn counting line for its automatic direction counts. It can optionally count crossings of user-defined virtual lines and optionally apply a SegFormer road segmentation mask.

The current implementation assumes a stationary camera. Direction names describe movement in image coordinates, not compass directions on a geographic map. It does not estimate physical speed, perform camera-motion compensation, or calibrate pixels to metres.

## 2. Processing architecture

For each frame, the processing path is:

1. **Load configuration.** `scripts/run.py` loads `configs/default.yaml` unless a different YAML file is supplied.
2. **Initialize YOLO.** `src/trafficflow/pipeline.py` loads the configured weights and finds the model class IDs whose names match `model.vehicle_names`.
3. **Open and track the source.** Ultralytics `YOLO.track` processes the video as a stream with persistent track IDs and the configured tracker, confidence, IoU, image size, device, and selected vehicle classes.
4. **Extract tracked vehicle data.** Results with track IDs are converted to tuples of `(track_id, class_name, (x1, y1, x2, y2))`. Untracked detections are not used by the flow analyzer.
5. **Update motion analytics.** `FlowAnalyzer` converts each tracked bounding box to an anchor point, estimates its heading, records a one-time automatic direction count when it has travelled far enough, tests configured virtual lines, and adds heading votes to the flow field.
6. **Refresh and draw the flow overlay.** At the configured interval, the flow field is converted into a road mask and direction zones. The visualization adds the flow map, optional segmentation tint, lines, bounding boxes, IDs, trails, direction labels, and a HUD.
7. **Write outputs.** At the end of processing, the pipeline writes event and per-minute CSV files, a JSON summary, and a still flow map. An annotated video is also written when enabled.

In short:

```text
video frame
  -> YOLO vehicle detections
  -> ByteTrack / configured tracker IDs
  -> anchor-point history per track
  -> 8-bin headings
       -> one automatic count per track
       -> optional one crossing count per configured line
       -> accumulated grid-cell votes
  -> direction zones and annotated frame
  -> video and analytics files
```

## 3. Source layout

| Path | Purpose |
|---|---|
| `src/trafficflow/flow.py` | Direction bins, track history, line-crossing test, automatic counts, flow-field accumulation, zone extraction, overlay rendering, and summary calculations. |
| `src/trafficflow/pipeline.py` | End-to-end video processing, Ultralytics integration, source/FPS setup, CSV/JSON/image output. |
| `src/trafficflow/viz.py` | Frame annotations and live heads-up display. |
| `src/trafficflow/road.py` | Optional SegFormer inference and road-mask tinting. |
| `src/trafficflow/config.py` | YAML configuration loading. |
| `src/trafficflow/__init__.py` | Package version (`1.0.0`). |
| `scripts/run.py` | Command-line entry point for video, webcam, and stream inference. |
| `scripts/annotate_lines.py` | Mouse-based tool to define line segments on the first video/image frame. |
| `scripts/train.py` | YOLO fine-tuning and promotion of the best checkpoint to `weights/best_traffic.pt`. |
| `scripts/evaluate.py` | YOLO validation and mAP50 / mAP50-95 reporting. |
| `scripts/extract_frames.py` | Periodic frame extraction for manual annotation. |
| `scripts/convert_detrac.py` | UA-DETRAC XML-to-YOLO conversion with sequence-based train/validation splitting. |
| `scripts/_path.py` | Adds the project `src` directory to Python's import path for scripts. |
| `configs/default.yaml` | Runtime model, flow, road segmentation, and output settings. |
| `configs/lines_example.json` | Example virtual-line JSON format. |
| `data/traffic.yaml` | Template dataset configuration for custom YOLO-labelled data. |
| `tests/test_flow.py` | Synthetic logic test for directions, line counts, zones, road mask, and summary counts; does not load YOLO or video. |
| `requirements.txt` | Python runtime dependencies. |

Runtime output is written to `outputs/` by default. Input clips in `inputs/`, generated output, virtual-environment files, training runs, downloaded or trained weight files, and several generated datasets are excluded by `.gitignore`.

## 4. Configuration

The defaults live in `configs/default.yaml`. `load_config()` reads that file relative to the package location, so the default config is found even when the CLI is launched from another working directory. Paths passed as CLI arguments (such as video, weights, lines, and output paths) are otherwise interpreted by the process working directory.

### Model settings

| Key | Meaning |
|---|---|
| `model.weights` | Ultralytics model/checkpoint path or model name. The default is `yolo11s.pt`; Ultralytics can download this pretrained checkpoint when needed. |
| `model.imgsz` | Inference image size. Larger images can help with small/distant vehicles at increased compute cost. |
| `model.conf` | Detection confidence threshold. |
| `model.iou` | IoU parameter passed to `YOLO.track` for detection/NMS behavior. |
| `model.device` | `null` lets Ultralytics choose; values such as `"cpu"`, `0`, or `"cuda:0"` select a device. |
| `model.tracker` | Ultralytics tracker configuration, defaulting to `bytetrack.yaml`; `botsort.yaml` is another suggested option. |
| `model.vehicle_names` | Class names to retain. The code matches these names case-insensitively against the model's class-name map. A model whose class names do not overlap this list exits with an explicit error. |

### Flow settings

| Key | Meaning |
|---|---|
| `flow.anchor` | Point used to represent a vehicle. `bottom` uses the bounding-box bottom-centre (ground contact proxy); `center` uses its centre (often more suitable for top-down footage). |
| `flow.heading_window` | Number of frame steps used to calculate a recent heading from the stored anchor history. |
| `flow.min_heading_px` | Minimum displacement across the heading window before a heading is accepted. Smaller movement is treated as jitter/no heading. |
| `flow.min_travel_px` | Minimum displacement from a track's first observed anchor before the track contributes one automatic direction count. |
| `flow.cell_px` | Width and height, in image pixels, of one flow-field grid cell. |
| `flow.min_cell_votes` | Minimum smoothed heading-vote total for a cell to qualify as road in the flow field. |
| `flow.blur_cells` | Gaussian smoothing sigma applied to each direction's grid of votes. |
| `flow.min_zone_cells` | Minimum connected-component area in grid cells for a directional zone to be retained. |
| `flow.update_every` | Number of frames between flow-field refreshes during processing. The final state is refreshed again before saving. |
| `flow.rate_window_s` | Look-back interval used for the live vehicles-per-minute estimate. |

### Road and output settings

`road.use_segformer` is false by default. If enabled, `road.segformer_model` selects the Hugging Face model. `output.save_video` controls annotated MP4 creation, `output.trail_len` limits stored/drawn anchor history, and `output.overlay_alpha` controls the flow overlay opacity.

The runtime CLI can override weights, enable SegFormer, and set the anchor. Other values are changed by supplying a custom YAML file with `--config`.

## 5. Detection, tracking, and anchors

`pipeline.run()` imports Ultralytics lazily and instantiates `YOLO` using the configured weights. It determines the IDs of target classes by comparing model class names to the configured vehicle names, then passes those IDs into `model.track`.

The tracker is responsible for associating detections across frames. The flow code does not implement object association itself; it relies on stable tracker IDs. A detection result is included only when `r.boxes` and `r.boxes.id` are present. For each retained object, the pipeline keeps its ID, class name, and pixel-space XYXY bounding box.

The analyzer maps a box `(x1, y1, x2, y2)` to:

```text
anchor_x = (x1 + x2) / 2
anchor_y = y2                         when anchor = "bottom"
anchor_y = (y1 + y2) / 2              when anchor = "center"
```

The first anchor seen for an ID is retained as its starting point. Subsequent points are stored in a bounded deque whose capacity is at least `heading_window + 1` and at least `trail_len`. An ID's track state includes its point history, whether its automatic direction count has been recorded, and which virtual lines it has crossed.

Track identity quality directly affects counts. An ID switch can split a vehicle into multiple tracks; an ID reused for a different object can combine two paths. Tune the tracker or detection settings when this occurs.

## 6. Direction estimation and automatic counts

### Eight image-space headings

The analyzer estimates a recent displacement from the current anchor to the anchor `heading_window` frame steps earlier:

```text
dx = current_x - earlier_x
dy = current_y - earlier_y
```

If the Euclidean displacement is at least `min_heading_px`, the motion is converted to one of eight bins, in this order:

| Bin | Label | Approximate image-space direction |
|---:|---|---|
| 0 | E | right |
| 1 | NE | up and right |
| 2 | N | up |
| 3 | NW | up and left |
| 4 | W | left |
| 5 | SW | down and left |
| 6 | S | down |
| 7 | SE | down and right |

The implementation uses `atan2(-dy, dx)` because image Y increases downwards. Each bin is 45 degrees wide, with the bin selected by rounding to the nearest direction. Therefore, `N` means image-up regardless of the camera's geographic orientation.

Headings are frame-based rather than time-normalized. `heading_window` specifies the number of frames and `min_heading_px` specifies pixels, so changing frame rate, resolution, camera zoom, or object scale changes the physical meaning of these settings.

### One automatic count per track

Automatic direction counting is based on net displacement from the first observed anchor, not a virtual road gate. Once the total displacement reaches `min_travel_px`, the track is counted exactly once in the direction bin from its initial point to its current point. The event is assigned the class name stored for the current detection, and the track is marked as already counted.

This is a calibration-free count of vehicles that travel far enough within the observed video. It is not a lane-entry/exit count: vehicles that leave before reaching the threshold are not counted, and the count may include a vehicle that was already in view at the start. A track's automatic direction count is not repeated if it reverses or later moves into another direction.

## 7. Virtual-line counts

Line counts are an independent optional counter. Each configured line is a finite segment with two endpoints. On consecutive detections for a track, the analyzer tests whether the movement segment intersects the configured line segment. Each track is counted at most once for each line, regardless of later recrossings.

The sign of a 2D cross product between the configured line vector and the vehicle movement selects one of the line's two labels:

```text
s = (line_dx * movement_dy) - (line_dy * movement_dx)
label = labels[0] if s > 0 else labels[1]
```

Use a consistent endpoint order and inspect the first test crossing to determine which label corresponds to each travel direction. The crossing result is recorded under a key such as `A:Northbound`. If labels are omitted, the loader defaults to `["dir1", "dir2"]`.

Example:

```json
{
  "lines": [
    {
      "name": "A",
      "p1": [100, 400],
      "p2": [900, 400],
      "labels": ["Northbound", "Southbound"]
    }
  ]
}
```

`scripts/annotate_lines.py` displays the first frame and collects pairs of mouse clicks. Press `u` to undo one point, `s` to save all complete pairs, or `q` to quit without saving. It saves default labels (`dir1`, `dir2`); edit the resulting JSON to give them meaningful names. The annotation tool does not infer direction labels or validate the line against traffic.

## 8. Learned road mask and directional flow zones

`FlowField` divides the frame into a grid of `cell_px` square cells. Grid dimensions are rounded up to cover the frame. Each accepted heading update adds one vote to the current anchor's cell and heading bin. A moving track can vote repeatedly as it advances through cells; this is an accumulated motion-occupancy map, not a semantic road detector.

When refreshed:

1. Each of the eight direction-vote grids is Gaussian-blurred using `blur_cells`.
2. The vote totals and dominant direction are computed for every cell.
3. Cells whose total reaches `min_cell_votes` form the initial road mask.
4. A 3-by-3 morphological close fills small gaps in the mask.
5. The mask is divided by dominant direction. For each direction, 8-connected components smaller than `min_zone_cells` are discarded.
6. Kept components become flow zones. Zone area is reported as grid-cell count times `cell_px ** 2`; centroids are converted from grid coordinates to image-pixel coordinates.

The result is a traffic-learned visualization: regions with enough observed motion appear as road, and regions are colored by their locally dominant motion bin. If there has not been enough traffic or the thresholds are too strict, the map can be empty or patchy. The `flow_map.png` image and `flow_zones` summary reflect the final accumulated state, not necessarily a stable road geometry.

The rendering upsamples grid cells to image resolution using nearest-neighbour interpolation, blends the corresponding direction color into qualifying pixels, and draws periodically spaced white direction arrows. Since the grid uses ceiling dimensions, the upsampled grid can extend beyond the source frame; the rendered mask is cropped back to the configured frame dimensions.

### Optional SegFormer road tint

When enabled, `road.segformer_road_mask()` loads a Hugging Face SegFormer model and processor on first use, converts the BGR frame to RGB, runs inference without gradients, resizes logits to the input frame size, and selects semantic class ID `0` as road (the configured model is a Cityscapes-finetuned SegFormer). The model and processor are cached in-process by model name.

In the current video pipeline this segmentation mask is computed once, from the first frame, then used as a fixed tint on subsequent frames. That matches the static-camera assumption but is not frame-by-frame semantic segmentation. This option requires PyTorch and `transformers`, which are not installed by the base requirements file.

## 9. Live HUD and frame rendering

`viz.draw_frame()` composes each annotated frame in this order:

1. Apply the optional fixed SegFormer road tint.
2. Blend the current learned flow field over the frame.
3. Draw configured counting-line segments and their names.
4. Draw each tracked bounding box and its stored trail.
5. Color boxes and trails using the most recent valid heading bin; use grey if no heading is available.
6. Add class name, track ID, and direction label where available.
7. Draw the HUD with elapsed time, automatic count total, per-direction totals and rates, and per-line totals.

The current frame is displayed in an OpenCV window only when `--show` is supplied. Pressing `q` stops processing and the pipeline continues to release resources and save the results collected so far.

## 10. Generated files and data meaning

By default, outputs are written under `outputs/`. Existing output files with the same names are overwritten on a later run.

| File | Contents |
|---|---|
| `annotated.mp4` | Annotated video encoded using OpenCV's `mp4v` codec when `output.save_video` is true. |
| `flow_map.png` | First frame with the optional road tint and final learned flow-field overlay. |
| `events.csv` | One row per automatic direction count or virtual-line crossing. Columns: `frame`, `time_s`, `track_id`, `class`, `kind`, `name`. `kind` is `direction` or `line`; `name` is a direction such as `SW` or a labeled line such as `A:Northbound`. |
| `per_minute.csv` | Automatic direction counts bucketed by elapsed minute. Columns are `minute`, then the eight direction labels. Line-crossing events are not included. |
| `summary.json` | Frame count and duration, automatic total, dominant direction, class subtotals for each direction and line, and final flow-zone area/centroid information. |

`direction_counts` and `line_counts` in the JSON include per-class counts plus a `total` for each direction/key. `dominant_direction` is the direction with the highest automatic direction-count total, or `null` if no automatic counts were recorded. `total_vehicles_counted` is the sum of automatic direction counts; it does not include line crossings.

The live vehicles-per-minute rate counts automatic direction events in the last `rate_window_s`. It scales the number of events in that interval to a per-minute rate, using elapsed time rather than dividing by a full minute when the video has run for less than a minute. The per-minute CSV instead uses fixed elapsed-time buckets beginning at minute zero.

`events.csv` records the count decision frame/time, not every detection. It is an event log, not a trajectory log. The pipeline reports progress to the console every 100 frames and prints the output location at completion.

## 11. Command-line usage

Create and activate an environment, then install `requirements.txt`. On Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tests/test_flow.py
```

Run a video:

```powershell
python scripts/run.py --source inputs\1.mp4
```

Other supported forms:

```powershell
python scripts/run.py --source 0 --show
python scripts/run.py --source "rtsp://camera-address/stream" --show
python scripts/run.py --source inputs\1.mp4 --weights weights\best_traffic.pt
python scripts/run.py --source inputs\1.mp4 --config configs\default.yaml --lines configs\lines_example.json
python scripts/run.py --source inputs\1.mp4 --segformer --anchor center --max-frames 1000
python scripts/run.py --source inputs\1.mp4 --out outputs\experiment-1
```

`--source` is required. A value containing only digits is converted to an integer camera index; other values are treated as a path or URL. `--show` displays the preview, `--max-frames` limits how many results are processed, and `--out` selects the output directory. `--weights`, `--segformer`, and `--anchor` override the corresponding configuration values. A source whose FPS metadata is absent, zero, or at most 1 is treated as 25 FPS. The source is opened once to read FPS and then opened by the Ultralytics tracking iterator.

## 12. Training and model evaluation

The repository uses Ultralytics YOLO for both training and evaluation. Pretrained COCO weights can be used directly for common vehicle classes. Fine-tuning is useful when camera angle, object scale, weather, or domain differs substantially from the pretrained data.

### Custom dataset format

`data/traffic.yaml` is a template. Its example layout is:

```text
data/custom/
  images/
    train/
    val/
  labels/
    train/
    val/
```

Each image has a same-stem text label file. Each non-empty label line follows YOLO detection format:

```text
class_id x_center y_center width height
```

The four box values are normalized to the image width/height and are between 0 and 1. Class IDs correspond to the ordered `names` mapping in the dataset YAML. The template maps IDs 0-4 to car, van, bus, truck, and motorcycle.

### Training script

`scripts/train.py` accepts `--data`, `--model`, `--epochs`, `--imgsz`, `--batch`, `--device`, `--name`, and `--resume`. Defaults are VisDrone, `yolo11s.pt`, 60 epochs, image size 960, batch size 16, run name `traffic`, and no resume. It trains under `runs/train/<name>`, uses patience 20, cosine learning-rate scheduling, closes mosaic augmentation for the last 10 epochs, disables rotation/vertical flips, and enables horizontal flips.

After training, it copies Ultralytics' best checkpoint to `weights/best_traffic.pt` and runs validation on that checkpoint. The copy overwrites an existing checkpoint with that name.

### Evaluation script

`scripts/evaluate.py` accepts `--weights`, `--data`, and `--imgsz`. Defaults are `weights/best_traffic.pt`, `VisDrone.yaml`, and image size 960. It runs Ultralytics validation and prints mAP50 and mAP50-95. Evaluation quality depends on the dataset YAML and split being representative of the intended deployment footage.

### Frame extraction and UA-DETRAC conversion

`scripts/extract_frames.py` accepts one or more videos, an output directory, and an interval in seconds. It samples by frame index using the video's FPS (falling back to 25 FPS), writes JPEGs, and does not generate labels. The extracted images must be labelled with a compatible tool before training.

`scripts/convert_detrac.py` converts UA-DETRAC frame annotations from XML to normalized YOLO boxes. It samples every `--step` frame (default 5), finds images in either `<images>/<sequence>` or `<images>/Insight-<sequence>`, and assigns whole XML sequences to train or validation to avoid splitting frames from one sequence across both sets. The last portion of sorted XML files is used for validation, with at least one validation sequence requested. Vehicle types map to car, van, bus, and a catch-all `others` class represented as `truck`. It writes converted data under `data/detrac_yolo` by default and generates `data/detrac.yaml`.

## 13. Dependencies and optional components

`requirements.txt` declares:

- `ultralytics` for YOLO inference, ByteTrack/BoT-SORT integration, training, and evaluation.
- `opencv-python` for video capture/writing, image transforms, drawing, and display.
- `numpy` for flow grids and array operations.
- `pyyaml` for runtime configuration.
- `lapx` for tracker assignment support.

PyTorch is generally installed as part of the Ultralytics dependency graph, but users with NVIDIA GPUs may need to install a CUDA-compatible PyTorch build for their environment. `transformers` is commented as optional and is needed for SegFormer. The baseline flow test is synthetic and does not need a model checkpoint or input video.

## 14. Testing

Run the synthetic test with:

```powershell
python tests/test_flow.py
```

The test adds `src/` to `sys.path`, lowers `min_cell_votes` for its small synthetic sample, simulates five right-moving cars and five left-moving trucks, and checks:

- Both E and W direction counts are populated with five tracks each.
- The named line's two crossing directions each count five tracks.
- E and W flow zones are present.
- The learned road mask contains pixels.
- The summary reports ten automatically counted vehicles.

The test exercises flow logic but not YOLO weight loading, tracker behavior, real video codecs, webcam/RTSP access, optional SegFormer, or end-to-end output writing.

## 15. Important assumptions and limitations

- **Static camera:** Camera movement causes apparent motion to be treated as vehicle motion. There is no ego-motion correction.
- **Image coordinates:** Direction labels are relative to the image. They are not geographic bearings.
- **No physical speed:** Motion is measured in pixels. Converting it to metres or km/h requires scene calibration and is not implemented.
- **Track-dependent counting:** ID switches or ID reuse can result in missed or duplicate counts.
- **Minimum travel threshold:** Automatic counts require the track to move at least `min_travel_px` from its first observation.
- **Heading noise and scale:** Frame rate, image resolution, box jitter, camera zoom, and anchor choice all affect pixel thresholds.
- **Learned, not semantic, road geometry:** The flow map grows from observed heading votes and can be sparse, noisy, or dominated by early traffic.
- **SegFormer is fixed to the initial image:** In this pipeline it does not recompute a mask for every video frame.
- **Vehicle class matching is name-based:** A custom model needs class names matching one or more configured `vehicle_names`.
- **Line direction depends on endpoints:** Reversing `p1` and `p2` changes which movement sign maps to each label.
- **Video codec availability:** MP4 writing depends on the codecs available in the installed OpenCV build and system.

## 16. Tuning guide

| Symptom | First settings or actions to try |
|---|---|
| Small or distant vehicles are missed | Increase `model.imgsz`, lower `model.conf` cautiously, or fine-tune on representative footage. |
| Counts appear too early or are noisy | Increase `flow.min_travel_px`; verify tracker IDs and anchor choice. |
| Track direction flickers or is missing | Increase `flow.heading_window` or lower `flow.min_heading_px` carefully. |
| Flow map is sparse or takes too long to appear | Lower `flow.min_cell_votes`, reduce `flow.min_zone_cells`, or process more footage. |
| Flow map is noisy or covers irrelevant pixels | Increase `flow.min_cell_votes` / `flow.min_zone_cells`; confirm input has a static camera. |
| Top-down/drone anchor is misleading | Set `flow.anchor` to `center` or pass `--anchor center`. |
| Vehicle IDs switch or merge | Try `botsort.yaml`, adjust detector thresholds, or use a more suitable fine-tuned detector. |
| Segmentation overlay is not appropriate | Disable SegFormer or choose a segmentation model suited to the scene and its label set. |
| A line assigns labels backwards | Swap its labels or reverse the line endpoint order, then verify with a test clip. |

Tune one parameter at a time on a representative clip and compare `events.csv`, `summary.json`, and the annotated video. Thresholds expressed in pixels are camera- and resolution-dependent.
