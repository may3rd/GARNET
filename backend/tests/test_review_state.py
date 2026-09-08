import json
import tempfile
import unittest
from pathlib import Path

from garnet.review_state import (
    build_stage4_line_numbers_from_review_state,
    empty_review_state,
    load_review_state,
    save_review_state,
)


class ReviewStateTests(unittest.TestCase):
    def test_empty_review_state_contains_all_buckets(self) -> None:
        payload = empty_review_state("job_123", {"image_path": "sample.png"})
        self.assertEqual(payload["job_id"], "job_123")
        self.assertEqual(payload["image_path"], "sample.png")
        self.assertEqual(
            sorted(payload["workspace_objects"].keys()),
            [
                "stage12_instrument_attachment",
                "stage12_line_attachment",
                "stage4_instrument",
                "stage4_line_number",
                "stage4_object",
                "stage6_line_association",
            ],
        )

    def test_save_and_load_review_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"image_path": "sample.png"}
            path = save_review_state(
                tmp,
                {
                    "items": [
                        {
                            "item_id": "stage4_line_number:line_number_000001",
                            "bucket": "stage4_line_number",
                            "decision": "accepted",
                        }
                    ],
                    "workspace_objects": {
                        "stage4_object": [{"Object": "gate_valve"}],
                        "stage4_line_number": [{"Object": "line_number"}],
                        "stage6_line_association": [{"Object": "line_number"}],
                    },
                },
                manifest,
            )
            self.assertTrue(path.exists())
            loaded = load_review_state(tmp, manifest)
            self.assertEqual(loaded["image_path"], "sample.png")
            self.assertEqual(loaded["items"][0]["decision"], "accepted")
            self.assertEqual(loaded["workspace_objects"]["stage4_object"][0]["Object"], "gate_valve")
            self.assertEqual(loaded["workspace_objects"]["stage4_line_number"][0]["Object"], "line_number")
            self.assertEqual(loaded["workspace_objects"]["stage6_line_association"][0]["Object"], "line_number")

    def test_save_review_state_rejects_invalid_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_review_state(
                    tmp,
                    {
                        "items": [{"item_id": "bad", "bucket": "nope", "decision": "accepted"}],
                        "workspace_objects": {},
                    },
                )

    def test_build_stage4_line_numbers_returns_none_when_no_hitl_review_done(self) -> None:
        # empty_review_state() always pre-populates "stage4_line_number" as [],
        # even when no reviewer ever touched it. stage6_trace_associations
        # must not treat that as "reviewer rejected everything" and overwrite
        # the real stage4_line_number_fusion output with an empty payload.
        with tempfile.TemporaryDirectory() as tmp:
            result = build_stage4_line_numbers_from_review_state(tmp, {"image_path": "sample.png"})
            self.assertIsNone(result)

    def test_build_stage4_line_numbers_returns_payload_when_reviewed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = {"image_path": "sample.png"}
            save_review_state(
                tmp,
                {
                    "items": [],
                    "workspace_objects": {
                        "stage4_line_number": [{"Text": "L-100", "review_state": "accepted"}],
                    },
                },
                manifest,
            )
            result = build_stage4_line_numbers_from_review_state(tmp, manifest)
            self.assertIsNotNone(result)
            self.assertEqual(len(result["line_numbers"]), 1)


if __name__ == "__main__":
    unittest.main()
