import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from unittest.mock import patch

import app
import numpy as np


class WebAppTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.temp_dir.name) / "inputs"
        self.output_dir = Path(self.temp_dir.name) / "outputs"
        self.client = app.app.test_client()
        app.jobs.clear()
        self.input_patch = patch.object(app, "INPUT_DIR", self.input_dir)
        self.output_patch = patch.object(app, "OUTPUT_DIR", self.output_dir)
        self.input_patch.start()
        self.output_patch.start()
        self.addCleanup(self.input_patch.stop)
        self.addCleanup(self.output_patch.stop)
        self.addCleanup(self.temp_dir.cleanup)
        self.addCleanup(app.jobs.clear)

    def test_rejects_non_video_upload(self):
        response = self.client.post(
            "/api/jobs",
            data={"video": (tempfile.SpooledTemporaryFile(), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_creates_downloadable_named_output(self):
        def process_video(_source, _config, output_dir, **kwargs):
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / kwargs["annotated_filename"]).write_bytes(b"mp4")
            kwargs["progress_callback"](5, 10, np.zeros((24, 32, 3), dtype=np.uint8))

        with patch.object(app, "run", side_effect=process_video), patch.object(
            app.shutil, "which", return_value=None
        ), patch.object(
            app.executor, "submit", side_effect=lambda function, *args: function(*args)
        ):
            response = self.client.post(
                "/api/jobs",
                data={"video": (tempfile.SpooledTemporaryFile(), "rush hour.mp4")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 202)
        job_id = response.get_json()["job_id"]
        status = self.client.get(f"/api/jobs/{job_id}").get_json()
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["frame"], 5)
        self.assertEqual(status["total_frames"], 10)
        self.assertEqual(status["progress"], 50.0)
        self.assertEqual(status["output_name"], "rush_hour_annoted.mp4")
        self.assertTrue((self.output_dir / status["output_name"]).is_file())
        preview_response = self.client.get(f"/api/jobs/{job_id}/preview.jpg?frame=5")
        self.assertEqual(preview_response.content_type, "image/jpeg")
        self.assertTrue(preview_response.data.startswith(b"\xff\xd8"))
        preview_response.close()
        video_response = self.client.get(status["video_url"])
        self.assertEqual(video_response.data, b"mp4")
        video_response.close()
        download_response = self.client.get(status["download_url"])
        self.assertIn("attachment", download_response.headers["Content-Disposition"])
        download_response.close()

    def test_duplicate_upload_names_get_distinct_saved_outputs(self):
        def process_video(_source, _config, output_dir, **kwargs):
            (Path(output_dir) / kwargs["annotated_filename"]).write_bytes(b"mp4")

        with patch.object(app, "run", side_effect=process_video), patch.object(
            app.shutil, "which", return_value=None
        ), patch.object(
            app.executor, "submit", side_effect=lambda function, *args: function(*args)
        ):
            responses = [
                self.client.post(
                    "/api/jobs",
                    data={"video": (tempfile.SpooledTemporaryFile(), "rush hour.mp4")},
                    content_type="multipart/form-data",
                )
                for _ in range(2)
            ]

        self.assertTrue(all(response.status_code == 202 for response in responses))
        statuses = [
            self.client.get(f"/api/jobs/{response.get_json()['job_id']}").get_json()
            for response in responses
        ]
        output_names = [status["output_name"] for status in statuses]
        self.assertEqual(len(set(output_names)), 2)
        self.assertTrue(all((self.output_dir / name).is_file() for name in output_names))

    def test_processing_transcodes_output_for_browser_playback(self):
        job_id = "browser-playback"
        input_path = self.input_dir / "input.mp4"
        input_path.parent.mkdir(parents=True)
        input_path.write_bytes(b"input")
        self.output_dir.mkdir(parents=True)
        raw_path = self.output_dir / ".rush_hour_annoted_opencv.mp4"
        output_path = self.output_dir / "rush_hour_annoted.mp4"
        app.jobs[job_id] = {"status": "queued", "output_name": output_path.name}

        def fake_run(_source, _config, output_dir, **kwargs):
            (Path(output_dir) / kwargs["annotated_filename"]).write_bytes(b"opencv")

        def fake_ffmpeg(command, **_kwargs):
            Path(command[-1]).write_bytes(b"h264")

        with patch.object(app, "run", side_effect=fake_run), patch.object(
            app.shutil, "which", return_value="ffmpeg"
        ), patch.object(app.subprocess, "run", side_effect=fake_ffmpeg):
            app._process_video(job_id, input_path, self.output_dir, output_path.name)

        self.assertEqual(output_path.read_bytes(), b"h264")
        self.assertFalse(raw_path.exists())
        self.assertEqual(app.jobs[job_id]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
