import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import app
import numpy as np


class PipelineApiTests(unittest.TestCase):
    def test_pipeline_job_runs_stage2_and_reports_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "text_regions": [
                    {
                        "id": "ocr_000001",
                        "text": "P-1001",
                        "normalized_text": "P-1001",
                        "class": "line_number",
                        "confidence": 0.91,
                        "bbox": {"x_min": 10, "y_min": 20, "x_max": 90, "y_max": 40},
                        "rotation": 0,
                        "reading_direction": "horizontal",
                        "legibility": "clear",
                    }
                ],
            },
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "tile_count": 1,
                "raw_detection_count": 1,
                "merged_region_count": 1,
                "exception_candidate_count": 0,
                "slice_height": 1600,
                "slice_width": 1600,
                "overlap_height_ratio": 0.2,
                "overlap_width_ratio": 0.2,
            },
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "2", "ocr_route": "easyocr"},
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
            self.assertEqual(job_payload["current_stage"], "stage2_ocr_discovery")
            self.assertEqual(job_payload["ocr_route"], "easyocr")
            self.assertEqual(len(job_payload["manifest"]["stages"]), 2)
            self.assertEqual(job_payload["manifest"]["ocr_route"], "easyocr")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage1_gray.png", artifact_names)
            self.assertIn("stage1_normalization_summary.json", artifact_names)
            self.assertIn("stage2_ocr_regions.json", artifact_names)
            self.assertIn("stage2_ocr_summary.json", artifact_names)
            self.assertIn("stage2_ocr_exception_candidates.json", artifact_names)

    def test_pipeline_job_rejects_missing_ocr_route(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        with sample_path.open("rb") as f:
            response = client.post(
                "/api/pipeline/jobs",
                files={"file_input": ("sample.png", f, "image/png")},
                data={"stop_after": "2"},
            )

        self.assertEqual(response.status_code, 422)

    def test_pipeline_job_rejects_invalid_ocr_route(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        with sample_path.open("rb") as f:
            response = client.post(
                "/api/pipeline/jobs",
                files={"file_input": ("sample.png", f, "image/png")},
                data={"stop_after": "2", "ocr_route": "bad-route"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("ocr_route", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
