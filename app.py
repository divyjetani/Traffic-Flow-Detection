"""Local web interface for uploading and annotating traffic videos."""
from __future__ import annotations

import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from trafficflow.config import load_config  # noqa: E402
from trafficflow.pipeline import run  # noqa: E402

INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "outputs"
ALLOWED_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict[str, str]] = {}
jobs_lock = threading.Lock()


def _process_video(job_id: str, input_path: Path, output_dir: Path, output_name: str) -> None:
    with jobs_lock:
        jobs[job_id]["status"] = "processing"

    try:
        run(
            str(input_path),
            load_config(),
            str(output_dir),
            annotated_filename=output_name,
        )
        with jobs_lock:
            jobs[job_id]["status"] = "completed"
    except Exception as exc:
        app.logger.exception("Video processing failed for job %s", job_id)
        with jobs_lock:
            jobs[job_id].update(status="failed", error=str(exc))
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/jobs")
def create_job():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify(error="Choose a video file to upload."), 400

    original_name = secure_filename(video.filename)
    extension = Path(original_name).suffix.lower()
    if not original_name or extension not in ALLOWED_EXTENSIONS:
        return jsonify(error="Upload a supported video file (MP4, MOV, AVI, MKV, M4V, or WebM)."), 400

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    input_path = INPUT_DIR / f"{job_id}{extension}"
    output_dir = OUTPUT_DIR / job_id
    output_name = f"{Path(original_name).stem}_annoted.mp4"

    try:
        video.save(input_path)
    except OSError as exc:
        app.logger.exception("Could not save uploaded video")
        return jsonify(error=f"Could not save the uploaded video: {exc}"), 500

    with jobs_lock:
        jobs[job_id] = {
            "status": "queued",
            "output_name": output_name,
        }
    executor.submit(_process_video, job_id, input_path, output_dir, output_name)
    return jsonify(job_id=job_id), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="Processing job not found."), 404
        result = dict(job)

    if result["status"] == "completed":
        result["video_url"] = f"/api/jobs/{job_id}/video"
        result["download_url"] = f"/api/jobs/{job_id}/download"
    return jsonify(result)


@app.get("/api/jobs/<job_id>/video")
def stream_video(job_id: str):
    return _send_job_video(job_id, as_attachment=False)


@app.get("/api/jobs/<job_id>/download")
def download_video(job_id: str):
    return _send_job_video(job_id, as_attachment=True)


def _send_job_video(job_id: str, as_attachment: bool):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None or job["status"] != "completed":
            return jsonify(error="The annotated video is not ready."), 404
        output_path = OUTPUT_DIR / job_id / job["output_name"]

    if not output_path.is_file():
        return jsonify(error="The annotated video could not be found."), 404
    return send_file(output_path, mimetype="video/mp4", as_attachment=as_attachment,
                     download_name=job["output_name"])


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify(error="The uploaded video exceeds the 2 GB limit."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, threaded=True)
