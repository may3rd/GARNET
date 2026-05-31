import json
import tempfile
import unittest
from pathlib import Path

from garnet.review_state import empty_review_state, load_review_state, save_review_state


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
                        "stage4_line_number": [{"Object": "line_number"}],
                    },
                },
                manifest,
            )
            self.assertTrue(path.exists())
            loaded = load_review_state(tmp, manifest)
            self.assertEqual(loaded["image_path"], "sample.png")
            self.assertEqual(loaded["items"][0]["decision"], "accepted")
            self.assertEqual(loaded["workspace_objects"]["stage4_line_number"][0]["Object"], "line_number")

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


if __name__ == "__main__":
    unittest.main()
