"""Local web interface for uploading and annotating traffic videos."""
from __future__ import annotations

import sys
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trafficflow.config import load_config  # noqa: E402
from trafficflow.models import MODEL_CHOICES, TRAINED_WEIGHTS, default_model_choice, resolve_model_weights  # noqa: E402
from trafficflow.pipeline import run  # noqa: E402

INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "outputs"
ALLOWED_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict[str, Any]] = {}
jobs_condition = threading.Condition()


def _update_progress(job_id: str, frame: int, total_frames: int, image) -> None:
    preview_width = min(image.shape[1], 640)
    preview_height = max(1, round(image.shape[0] * preview_width / image.shape[1]))
    preview = cv2.resize(image, (preview_width, preview_height))
    encoded, buffer = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not encoded:
        raise RuntimeError("Could not encode the annotated preview frame.")

    with jobs_condition:
        job = jobs[job_id]
        job["frame"] = frame
        job["total_frames"] = total_frames
        job["progress"] = round(min(frame / total_frames * 100, 100), 1) if total_frames else None
        job["preview"] = buffer.tobytes()
        job["preview_sequence"] += 1
        jobs_condition.notify_all()


def _process_video(job_id: str, input_path: Path, output_dir: Path, output_name: str,
                   model_choice: str = "pretrained") -> None:
    with jobs_condition:
        jobs[job_id]["status"] = "processing"
        jobs_condition.notify_all()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_name = f".{Path(output_name).stem}_opencv.mp4"
        config = load_config()
        config["model"]["weights"] = resolve_model_weights(model_choice)
        run(
            str(input_path),
            config,
            str(output_dir),
            annotated_filename=raw_name,
            progress_callback=lambda frame, total, image: _update_progress(
                job_id, frame, total, image
            ),
        )
        with jobs_condition:
            jobs[job_id]["status"] = "encoding"
            jobs_condition.notify_all()
        raw_path = output_dir / raw_name
        output_path = output_dir / output_name
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                subprocess.run(
                    [
                        ffmpeg, "-y", "-i", str(raw_path), "-an",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(output_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
                raw_path.unlink(missing_ok=True)
            except (OSError, subprocess.SubprocessError) as exc:
                app.logger.warning(
                    "Could not transcode annotated video to browser-friendly H.264: %s", exc
                )
                raw_path.replace(output_path)
        else:
            app.logger.warning(
                "ffmpeg is unavailable; saving the OpenCV MP4 without browser-compatible transcoding."
            )
            raw_path.replace(output_path)
        with jobs_condition:
            jobs[job_id]["status"] = "completed"
            jobs_condition.notify_all()
    except Exception as exc:
        app.logger.exception("Video processing failed for job %s", job_id)
        with jobs_condition:
            jobs[job_id].update(status="failed", error=str(exc))
            jobs_condition.notify_all()
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/")
def index():
    return render_template(
        "index.html",
        model_choices=MODEL_CHOICES,
        default_model=default_model_choice(),
        trained_model_available=TRAINED_WEIGHTS.is_file(),
    )


@app.post("/api/jobs")
def create_job():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify(error="Choose a video file to upload."), 400

    original_name = secure_filename(video.filename)
    extension = Path(original_name).suffix.lower()
    if not original_name or extension not in ALLOWED_EXTENSIONS:
        return jsonify(error="Upload a supported video file (MP4, MOV, AVI, MKV, M4V, or WebM)."), 400

    model_choice = request.form.get("model_choice", default_model_choice())
    if model_choice not in MODEL_CHOICES:
        return jsonify(error="Choose either the trained or pretrained detector."), 400
    try:
        resolve_model_weights(model_choice)
    except FileNotFoundError as exc:
        return jsonify(error=str(exc)), 409

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    input_path = INPUT_DIR / f"{job_id}{extension}"
    output_name = f"{Path(original_name).stem}_annoted.mp4"

    try:
        video.save(input_path)
    except OSError as exc:
        app.logger.exception("Could not save uploaded video")
        return jsonify(error=f"Could not save the uploaded video: {exc}"), 500

    with jobs_condition:
        reserved_names = {job["output_name"] for job in jobs.values()}
        output_path = OUTPUT_DIR / output_name
        if output_name in reserved_names or output_path.exists():
            output_name = f"{Path(original_name).stem}_annoted_{job_id[:8]}.mp4"
        jobs[job_id] = {
            "status": "queued",
            "model_choice": model_choice,
            "output_name": output_name,
            "frame": 0,
            "total_frames": 0,
            "progress": 0,
            "preview": None,
            "preview_sequence": 0,
        }
    executor.submit(_process_video, job_id, input_path, OUTPUT_DIR, output_name, model_choice)
    return jsonify(job_id=job_id, model_choice=model_choice), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with jobs_condition:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="Processing job not found."), 404
        result = {
            key: job.get(key)
            for key in ("status", "output_name", "model_choice", "frame", "total_frames", "progress", "error")
        }

    if result["status"] == "completed":
        result["video_url"] = f"/api/jobs/{job_id}/video"
        result["download_url"] = f"/api/jobs/{job_id}/download"
    return jsonify(result)


@app.get("/api/jobs/<job_id>/preview.jpg")
def preview_video(job_id: str):
    with jobs_condition:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="Processing job not found."), 404
        image = job["preview"]
    if image is None:
        return jsonify(error="No annotated frame is available yet."), 404
    return Response(image, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/jobs/<job_id>/video")
def stream_video(job_id: str):
    return _send_job_video(job_id, as_attachment=False)


@app.get("/api/jobs/<job_id>/download")
def download_video(job_id: str):
    return _send_job_video(job_id, as_attachment=True)


def _send_job_video(job_id: str, as_attachment: bool):
    with jobs_condition:
        job = jobs.get(job_id)
        if job is None or job["status"] != "completed":
            return jsonify(error="The annotated video is not ready."), 404
        output_path = OUTPUT_DIR / job["output_name"]

    if not output_path.is_file():
        return jsonify(error="The annotated video could not be found."), 404
    return send_file(output_path, mimetype="video/mp4", as_attachment=as_attachment,
                     download_name=job["output_name"])


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="The uploaded video exceeds the 2 GB limit."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
