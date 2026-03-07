import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from api import app


class PipelineApiTests(unittest.TestCase):
    def test_pipeline_job_runs_stage1_and_reports_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        with sample_path.open("rb") as f:
            response = client.post(
                "/api/pipeline/jobs",
                files={"file_input": ("sample.png", f, "image/png")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("job_id", payload)
        job_id = payload["job_id"]

        deadline = time.time() + 10
        job_payload = None
        while time.time() < deadline:
            poll = client.get(f"/api/pipeline/jobs/{job_id}")
            self.assertEqual(poll.status_code, 200)
            job_payload = poll.json()
            if job_payload["status"] in {"completed", "failed"}:
                break
            time.sleep(0.1)

        self.assertIsNotNone(job_payload)
        assert job_payload is not None
        self.assertEqual(job_payload["status"], "completed")
        self.assertEqual(job_payload["current_stage"], "stage1_input_normalization")
        self.assertEqual(len(job_payload["manifest"]["stages"]), 1)
        artifact_names = {item["name"] for item in job_payload["artifacts"]}
        self.assertIn("stage1_gray.png", artifact_names)
        self.assertIn("stage1_normalization_summary.json", artifact_names)


if __name__ == "__main__":
    unittest.main()
