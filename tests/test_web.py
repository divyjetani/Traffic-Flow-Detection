import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app


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

        with patch.object(app, "run", side_effect=process_video), patch.object(
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
        self.assertEqual(status["output_name"], "rush_hour_annoted.mp4")
        self.assertTrue((self.output_dir / job_id / status["output_name"]).is_file())
        video_response = self.client.get(status["video_url"])
        self.assertEqual(video_response.data, b"mp4")
        video_response.close()
        download_response = self.client.get(status["download_url"])
        self.assertIn("attachment", download_response.headers["Content-Disposition"])
        download_response.close()


if __name__ == "__main__":
    unittest.main()
