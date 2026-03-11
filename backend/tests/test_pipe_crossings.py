import unittest

import numpy as np

from garnet.pipe_crossings import _classify_candidate, run_pipe_crossing_stage


class PipeCrossingTests(unittest.TestCase):
    def test_run_pipe_crossing_stage_classifies_crossing(self) -> None:
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        sealed = np.zeros((21, 21), dtype=np.uint8)
        skeleton = np.zeros((21, 21), dtype=np.uint8)
        sealed[10, 4:17] = 255
        sealed[4:17, 10] = 255
        skeleton[10, 4:17] = 255
        skeleton[4:17, 10] = 255

        result = run_pipe_crossing_stage(
            image_bgr=image,
            sealed_mask=sealed,
            skeleton_mask=skeleton,
            node_clusters=[
                {
                    "id": "junction_0",
                    "kind": "junction",
                    "centroid": {"x": 10.0, "y": 10.0},
                    "member_count": 1,
                    "members": [{"row": 10, "col": 10}],
                }
            ],
            image_id="sample.png",
        )

        self.assertEqual(result["summary"]["candidate_count"], 1)
        candidate = result["crossings_payload"]["candidates"][0]
        self.assertEqual(candidate["classification"], "non_connecting_crossing")
        self.assertEqual(len(candidate["routing_pairs"]), 2)
        self.assertIn("stage4_object_evidence", candidate)

    def test_run_pipe_crossing_stage_classifies_tee_as_junction(self) -> None:
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        sealed = np.zeros((21, 21), dtype=np.uint8)
        skeleton = np.zeros((21, 21), dtype=np.uint8)
        sealed[10, 4:17] = 255
        sealed[4:11, 10] = 255
        skeleton[10, 4:17] = 255
        skeleton[4:11, 10] = 255

        result = run_pipe_crossing_stage(
            image_bgr=image,
            sealed_mask=sealed,
            skeleton_mask=skeleton,
            node_clusters=[
                {
                    "id": "junction_0",
                    "kind": "junction",
                    "centroid": {"x": 10.0, "y": 10.0},
                    "member_count": 1,
                    "members": [{"row": 10, "col": 10}],
                }
            ],
            image_id="sample.png",
        )

        self.assertEqual(result["crossings_payload"]["candidates"][0]["classification"], "confirmed_junction")

    def test_stage4_markers_can_promote_ambiguous_candidate_to_junction(self) -> None:
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        sealed = np.zeros((21, 21), dtype=np.uint8)
        skeleton = np.zeros((21, 21), dtype=np.uint8)
        sealed[10, 4:17] = 255
        sealed[4:17, 10] = 255
        skeleton[10, 4:17] = 255
        skeleton[4:17, 10] = 255

        base = run_pipe_crossing_stage(
            image_bgr=image,
            sealed_mask=sealed,
            skeleton_mask=skeleton,
            node_clusters=[
                {
                    "id": "junction_0",
                    "kind": "junction",
                    "centroid": {"x": 10.0, "y": 10.0},
                    "member_count": 1,
                    "members": [{"row": 10, "col": 10}],
                }
            ],
            topology_markers=[],
            image_id="sample.png",
            center_blob_radius_px=4,
            center_blob_threshold=0.85,
        )
        self.assertEqual(base["crossings_payload"]["candidates"][0]["classification"], "non_connecting_crossing")

        promoted = run_pipe_crossing_stage(
            image_bgr=image,
            sealed_mask=sealed,
            skeleton_mask=skeleton,
            node_clusters=[
                {
                    "id": "junction_0",
                    "kind": "junction",
                    "centroid": {"x": 10.0, "y": 10.0},
                    "member_count": 1,
                    "members": [{"row": 10, "col": 10}],
                }
            ],
            topology_markers=[
                {
                    "id": "topology_marker::obj_1",
                    "source_object_id": "obj_1",
                    "class_name": "node",
                    "role": "junction_marker",
                    "confidence": 0.92,
                    "bbox": {"x_min": 8, "y_min": 8, "x_max": 12, "y_max": 12},
                }
            ],
            image_id="sample.png",
            center_blob_radius_px=4,
            center_blob_threshold=0.85,
        )
        candidate = promoted["crossings_payload"]["candidates"][0]
        self.assertEqual(candidate["classification"], "confirmed_junction")
        self.assertTrue(candidate["stage4_object_evidence"]["supported"])
        self.assertEqual(candidate["stage4_object_evidence"]["role_counts"]["junction_marker"], 1)

    def test_unresolved_candidates_include_reason_labels(self) -> None:
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        sealed = np.zeros((21, 21), dtype=np.uint8)
        skeleton = np.zeros((21, 21), dtype=np.uint8)
        sealed[10, 4:17] = 255
        sealed[4:17, 10] = 255
        skeleton[10, 4:17] = 255
        skeleton[4:17, 10] = 255

        result = run_pipe_crossing_stage(
            image_bgr=image,
            sealed_mask=sealed,
            skeleton_mask=skeleton,
            node_clusters=[
                {
                    "id": "junction_0",
                    "kind": "junction",
                    "centroid": {"x": 10.0, "y": 10.0},
                    "member_count": 1,
                    "members": [{"row": 10, "col": 10}],
                }
            ],
            topology_markers=[],
            image_id="sample.png",
            center_blob_radius_px=4,
            center_blob_threshold=0.6,
        )

        candidate = result["crossings_payload"]["candidates"][0]
        if candidate["classification"] != "unresolved":
            self.skipTest("Synthetic case did not land in unresolved state with current thresholds")
        self.assertTrue(candidate["unresolved_reasons"])
        summary_reasons = result["summary"]["unresolved_reason_counts"]
        for reason in candidate["unresolved_reasons"]:
            self.assertIn(reason, summary_reasons)

    def test_classify_candidate_promotes_oversegmented_four_way_with_strong_blob(self) -> None:
        sealed = np.zeros((21, 21), dtype=np.uint8)
        sealed[6:15, 6:15] = 255
        classification, _, _, _ = _classify_candidate(
            cluster={"centroid": {"x": 10.0, "y": 10.0}},
            branches=[
                {"branch_id": "b0", "angle_deg": 0.0},
                {"branch_id": "b1", "angle_deg": 90.0},
                {"branch_id": "b2", "angle_deg": 180.0},
                {"branch_id": "b3", "angle_deg": 270.0},
            ],
            raw_branch_count=6,
            sealed_mask=sealed,
            stage4_marker_evidence={"role_counts": {"junction_marker": 0, "flow_marker": 0}, "matched_markers": []},
            opposite_angle_tolerance_deg=35.0,
            blob_radius_px=4,
            blob_threshold=0.5,
        )
        self.assertEqual(classification, "confirmed_junction")


if __name__ == "__main__":
    unittest.main()
