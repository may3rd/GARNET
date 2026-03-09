import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import numpy as np

try:
    from api import app
except ModuleNotFoundError as exc:
    if exc.name == "pdf2image":
        app = None
    else:
        raise


@unittest.skipIf(app is None, "pdf2image is not installed in this test environment")
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

    def test_pipeline_job_runs_stage4_and_reports_object_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "tile_count": 1,
                "raw_detection_count": 0,
                "merged_region_count": 0,
                "exception_candidate_count": 0,
                "slice_height": 1600,
                "slice_width": 1600,
                "overlap_height_ratio": 0.2,
                "overlap_width_ratio": 0.2,
            },
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "objects": [
                    {
                        "id": "obj_000001",
                        "class_name": "valve",
                        "confidence": 0.88,
                        "bbox": {"x_min": 5, "y_min": 6, "x_max": 20, "y_max": 30},
                        "source_model": "ultralytics",
                        "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
                    }
                ],
            },
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 1,
                "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_line_number_fusion_result = {
            "line_numbers_payload": {"line_numbers": [], "rejected": []},
            "summary": {"matched_line_number_count": 0},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch(
            "garnet.pid_extractor.run_line_number_fusion_stage", return_value=fake_line_number_fusion_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "4", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage4_line_number_fusion")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage4_objects.json", artifact_names)
            self.assertIn("stage4_objects_summary.json", artifact_names)
            self.assertIn("stage4_objects_overlay.png", artifact_names)
            self.assertIn("stage4_line_numbers.json", artifact_names)
            self.assertIn("stage4_line_number_summary.json", artifact_names)

    def test_pipeline_job_runs_stage5_and_reports_pipe_mask_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "mask_pixel_count": 42,
                "source_artifacts": [
                    "stage1_gray.png",
                    "stage2_ocr_regions.json",
                    "stage4_objects.json",
                ],
            },
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "5", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage5_pipe_mask")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage5_pipe_mask.png", artifact_names)
            self.assertIn("stage5_pipe_mask_overlay.png", artifact_names)
            self.assertIn("stage5_pipe_mask_summary.json", artifact_names)

    def test_pipeline_job_runs_stage6_and_reports_pipe_seal_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 42},
        }
        fake_pipe_seal_result = {
            "sealed_mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 40},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result), patch(
            "garnet.pid_extractor.run_pipe_seal_stage", return_value=fake_pipe_seal_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "6", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage6_morphological_sealing")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage6_pipe_mask_sealed.png", artifact_names)
            self.assertIn("stage6_pipe_mask_sealed_overlay.png", artifact_names)
            self.assertIn("stage6_pipe_mask_sealed_summary.json", artifact_names)

    def test_pipeline_job_runs_stage7_and_reports_skeleton_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 42},
        }
        fake_pipe_seal_result = {
            "sealed_mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 40},
        }
        fake_skeleton_result = {
            "skeleton_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "skeleton_pixel_count": 15},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result), patch(
            "garnet.pid_extractor.run_pipe_seal_stage", return_value=fake_pipe_seal_result
        ), patch("garnet.pid_extractor.run_pipe_skeleton_stage", return_value=fake_skeleton_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "7", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage7_skeleton_generation")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage7_pipe_skeleton.png", artifact_names)
            self.assertIn("stage7_pipe_skeleton_overlay.png", artifact_names)
            self.assertIn("stage7_pipe_skeleton_summary.json", artifact_names)

    def test_pipeline_job_runs_stage8_and_reports_node_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 42},
        }
        fake_pipe_seal_result = {
            "sealed_mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 40},
        }
        fake_skeleton_result = {
            "skeleton_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "skeleton_pixel_count": 15},
        }
        fake_node_result = {
            "endpoint_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_count": 4, "junction_count": 1},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result), patch(
            "garnet.pid_extractor.run_pipe_seal_stage", return_value=fake_pipe_seal_result
        ), patch("garnet.pid_extractor.run_pipe_skeleton_stage", return_value=fake_skeleton_result), patch(
            "garnet.pid_extractor.run_pipe_node_stage", return_value=fake_node_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "8", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage8_skeleton_node_detection")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage8_endpoints.png", artifact_names)
            self.assertIn("stage8_junctions.png", artifact_names)
            self.assertIn("stage8_nodes_overlay.png", artifact_names)
            self.assertIn("stage8_node_summary.json", artifact_names)

    def test_pipeline_job_runs_stage9_and_reports_node_cluster_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 42},
        }
        fake_pipe_seal_result = {
            "sealed_mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 40},
        }
        fake_skeleton_result = {
            "skeleton_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "skeleton_pixel_count": 15},
        }
        fake_node_result = {
            "endpoint_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_count": 4, "junction_count": 1},
        }
        fake_cluster_result = {
            "endpoint_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "clusters_payload": {"clusters": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_cluster_count": 2, "junction_cluster_count": 1},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result), patch(
            "garnet.pid_extractor.run_pipe_seal_stage", return_value=fake_pipe_seal_result
        ), patch("garnet.pid_extractor.run_pipe_skeleton_stage", return_value=fake_skeleton_result), patch(
            "garnet.pid_extractor.run_pipe_node_stage", return_value=fake_node_result
        ), patch("garnet.pid_extractor.run_pipe_node_cluster_stage", return_value=fake_cluster_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "9", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage9_node_clustering")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage9_endpoint_clusters.png", artifact_names)
            self.assertIn("stage9_junction_clusters.png", artifact_names)
            self.assertIn("stage9_node_clusters_overlay.png", artifact_names)
            self.assertIn("stage9_node_clusters.json", artifact_names)
            self.assertIn("stage9_node_cluster_summary.json", artifact_names)

    def test_pipeline_job_runs_stage10_and_reports_edge_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 42},
        }
        fake_pipe_seal_result = {
            "sealed_mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 40},
        }
        fake_skeleton_result = {
            "skeleton_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "skeleton_pixel_count": 15},
        }
        fake_node_result = {
            "endpoint_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_count": 4, "junction_count": 1},
        }
        fake_cluster_result = {
            "endpoint_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "clusters_payload": {"clusters": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_cluster_count": 2, "junction_cluster_count": 1},
        }
        fake_edge_result = {
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "edges_payload": {"edges": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "edge_count": 0},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result), patch(
            "garnet.pid_extractor.run_pipe_seal_stage", return_value=fake_pipe_seal_result
        ), patch("garnet.pid_extractor.run_pipe_skeleton_stage", return_value=fake_skeleton_result), patch(
            "garnet.pid_extractor.run_pipe_node_stage", return_value=fake_node_result
        ), patch("garnet.pid_extractor.run_pipe_node_cluster_stage", return_value=fake_cluster_result), patch(
            "garnet.pid_extractor.run_pipe_edge_stage", return_value=fake_edge_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "10", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage10_edge_tracing")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage10_pipe_edges_overlay.png", artifact_names)
            self.assertIn("stage10_pipe_edges.json", artifact_names)
            self.assertIn("stage10_pipe_edge_summary.json", artifact_names)

    def test_pipeline_job_runs_stage11_and_reports_junction_review_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 42},
        }
        fake_pipe_seal_result = {
            "sealed_mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 40},
        }
        fake_skeleton_result = {
            "skeleton_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "skeleton_pixel_count": 15},
        }
        fake_node_result = {
            "endpoint_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_count": 4, "junction_count": 1},
        }
        fake_cluster_result = {
            "endpoint_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "clusters_payload": {"clusters": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_cluster_count": 2, "junction_cluster_count": 1},
        }
        fake_edge_result = {
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "edges_payload": {"edges": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "edge_count": 0},
        }
        fake_junction_review_result = {
            "confirmed_junction_image": np.zeros((50, 100), dtype=np.uint8),
            "unresolved_junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "junctions_payload": {"confirmed_junctions": [], "unresolved_junctions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "confirmed_junction_count": 0, "unresolved_junction_count": 0},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result), patch(
            "garnet.pid_extractor.run_pipe_seal_stage", return_value=fake_pipe_seal_result
        ), patch("garnet.pid_extractor.run_pipe_skeleton_stage", return_value=fake_skeleton_result), patch(
            "garnet.pid_extractor.run_pipe_node_stage", return_value=fake_node_result
        ), patch("garnet.pid_extractor.run_pipe_node_cluster_stage", return_value=fake_cluster_result), patch(
            "garnet.pid_extractor.run_pipe_edge_stage", return_value=fake_edge_result
        ), patch("garnet.pid_extractor.run_pipe_junction_stage", return_value=fake_junction_review_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "11", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage11_junction_review")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage11_confirmed_junctions.png", artifact_names)
            self.assertIn("stage11_unresolved_junctions.png", artifact_names)
            self.assertIn("stage11_junction_review_overlay.png", artifact_names)
            self.assertIn("stage11_junctions.json", artifact_names)
            self.assertIn("stage11_junction_review_summary.json", artifact_names)

    def test_pipeline_job_runs_stage12_and_reports_graph_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 42},
        }
        fake_pipe_seal_result = {
            "sealed_mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 40},
        }
        fake_skeleton_result = {
            "skeleton_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "skeleton_pixel_count": 15},
        }
        fake_node_result = {
            "endpoint_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_count": 4, "junction_count": 1},
        }
        fake_cluster_result = {
            "endpoint_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "clusters_payload": {"clusters": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_cluster_count": 2, "junction_cluster_count": 1},
        }
        fake_edge_result = {
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "edges_payload": {"edges": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "edge_count": 0},
        }
        fake_junction_review_result = {
            "confirmed_junction_image": np.zeros((50, 100), dtype=np.uint8),
            "unresolved_junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "junctions_payload": {"confirmed_junctions": [], "unresolved_junctions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "confirmed_junction_count": 0, "unresolved_junction_count": 0},
        }
        fake_attachment_result = {
            "attachments_payload": {"accepted": [], "rejected": []},
            "summary": {"accepted_attachment_count": 0},
        }
        fake_text_attachment_result = {
            "attachments_payload": {"accepted": [], "rejected": []},
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"accepted_attachment_count": 0},
        }
        fake_graph_result = {
            "graph_payload": {"nodes": [], "edges": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "node_count": 0, "edge_count": 0},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result), patch(
            "garnet.pid_extractor.run_pipe_seal_stage", return_value=fake_pipe_seal_result
        ), patch("garnet.pid_extractor.run_pipe_skeleton_stage", return_value=fake_skeleton_result), patch(
            "garnet.pid_extractor.run_pipe_node_stage", return_value=fake_node_result
        ), patch("garnet.pid_extractor.run_pipe_node_cluster_stage", return_value=fake_cluster_result), patch(
            "garnet.pid_extractor.run_pipe_edge_stage", return_value=fake_edge_result
        ), patch("garnet.pid_extractor.run_pipe_junction_stage", return_value=fake_junction_review_result), patch(
            "garnet.pid_extractor.run_pipe_equipment_attachment_stage", return_value=fake_attachment_result
        ), patch(
            "garnet.pid_extractor.run_pipe_text_attachment_stage", return_value=fake_text_attachment_result
        ), patch(
            "garnet.pid_extractor.run_pipe_graph_stage", return_value=fake_graph_result
        ):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "12", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage12_graph_assembly")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage12_equipment_attachments.json", artifact_names)
            self.assertIn("stage12_equipment_attachment_summary.json", artifact_names)
            self.assertIn("stage12_text_attachments.json", artifact_names)
            self.assertIn("stage12_text_attachment_summary.json", artifact_names)
            self.assertIn("stage12_text_attachment_overlay.png", artifact_names)
            self.assertIn("stage12_graph.json", artifact_names)
            self.assertIn("stage12_graph_summary.json", artifact_names)

    def test_pipeline_job_runs_stage13_and_reports_qa_artifacts(self) -> None:
        client = TestClient(app)
        sample_path = Path(__file__).resolve().parents[1] / "sample.png"

        fake_ocr_result = {
            "regions_payload": {"image_id": "sample.png", "pass_type": "sheet", "text_regions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet"},
            "exception_candidates": [],
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_detection_result = {
            "objects_payload": {"image_id": "sample.png", "pass_type": "sheet", "objects": []},
            "summary": {
                "image_id": "sample.png",
                "pass_type": "sheet",
                "route": "ultralytics",
                "object_count": 0,
                "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt",
            },
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
        }
        fake_pipe_mask_result = {
            "mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 42},
        }
        fake_pipe_seal_result = {
            "sealed_mask_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "mask_pixel_count": 40},
        }
        fake_skeleton_result = {
            "skeleton_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "skeleton_pixel_count": 15},
        }
        fake_node_result = {
            "endpoint_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_count": 4, "junction_count": 1},
        }
        fake_cluster_result = {
            "endpoint_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "junction_cluster_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "clusters_payload": {"clusters": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "endpoint_cluster_count": 2, "junction_cluster_count": 1},
        }
        fake_edge_result = {
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "edges_payload": {"edges": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "edge_count": 0},
        }
        fake_junction_review_result = {
            "confirmed_junction_image": np.zeros((50, 100), dtype=np.uint8),
            "unresolved_junction_image": np.zeros((50, 100), dtype=np.uint8),
            "overlay_image": np.zeros((50, 100, 3), dtype=np.uint8),
            "junctions_payload": {"confirmed_junctions": [], "unresolved_junctions": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "confirmed_junction_count": 0, "unresolved_junction_count": 0},
        }
        fake_graph_result = {
            "graph_payload": {"nodes": [], "edges": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "node_count": 0, "edge_count": 0},
        }
        fake_qa_result = {
            "anomaly_report": {"connected_component_count": 0},
            "review_queue": {"items": []},
            "summary": {"image_id": "sample.png", "pass_type": "sheet", "review_queue_count": 0},
        }

        with patch("garnet.pid_extractor.run_easyocr_sahi", return_value=fake_ocr_result), patch(
            "garnet.pid_extractor.run_object_detection_sahi", return_value=fake_detection_result
        ), patch("garnet.pid_extractor.run_pipe_mask_stage", return_value=fake_pipe_mask_result), patch(
            "garnet.pid_extractor.run_pipe_seal_stage", return_value=fake_pipe_seal_result
        ), patch("garnet.pid_extractor.run_pipe_skeleton_stage", return_value=fake_skeleton_result), patch(
            "garnet.pid_extractor.run_pipe_node_stage", return_value=fake_node_result
        ), patch("garnet.pid_extractor.run_pipe_node_cluster_stage", return_value=fake_cluster_result), patch(
            "garnet.pid_extractor.run_pipe_edge_stage", return_value=fake_edge_result
        ), patch("garnet.pid_extractor.run_pipe_junction_stage", return_value=fake_junction_review_result), patch(
            "garnet.pid_extractor.run_pipe_graph_stage", return_value=fake_graph_result
        ), patch("garnet.pid_extractor.run_pipe_graph_qa_stage", return_value=fake_qa_result):
            with sample_path.open("rb") as f:
                response = client.post(
                    "/api/pipeline/jobs",
                    files={"file_input": ("sample.png", f, "image/png")},
                    data={"stop_after": "13", "ocr_route": "easyocr"},
                )

            self.assertEqual(response.status_code, 200)
            job_id = response.json()["job_id"]

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
            self.assertEqual(job_payload["current_stage"], "stage13_graph_qa")
            artifact_names = {item["name"] for item in job_payload["artifacts"]}
            self.assertIn("stage13_graph_anomalies.json", artifact_names)
            self.assertIn("stage13_review_queue.json", artifact_names)
            self.assertIn("stage13_graph_qa_summary.json", artifact_names)

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
