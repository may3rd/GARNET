import json
import tempfile
import unittest
from pathlib import Path

from garnet.reviewed_outputs import generate_reviewed_outputs


class ReviewedOutputsTests(unittest.TestCase):
    def test_generate_reviewed_outputs_filters_rejected_line_and_instrument_attachments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "stage12_graph.json").write_text(
                json.dumps(
                    {
                        "image_id": "sample.png",
                        "pass_type": "sheet",
                        "nodes": [
                            {"id": "n1", "type": "endpoint", "position": {"x": 0, "y": 0}, "member_count": 1, "review_state": "accepted"},
                            {"id": "n2", "type": "endpoint", "position": {"x": 10, "y": 0}, "member_count": 1, "review_state": "accepted"},
                        ],
                        "edges": [
                            {
                                "id": "e1",
                                "source": "n1",
                                "target": "n2",
                                "pixel_length": 10,
                                "polyline": [],
                                "review_state": "provisional",
                                "line_texts": [{"region_id": "line_number_1", "text": "L-1", "normalized_text": "L-1"}],
                                "instrument_tags": [{"region_id": "instrument_tag_1", "text": "PI-1001", "normalized_text": "PI-1001"}],
                            }
                        ],
                        "unresolved_junction_ids": [],
                        "equipment_attachments": [],
                        "text_attachments": [{"region_id": "line_number_1", "text": "L-1", "normalized_text": "L-1", "edge_id": "e1"}],
                        "instrument_tag_attachments": [{"region_id": "instrument_tag_1", "text": "PI-1001", "normalized_text": "PI-1001", "edge_id": "e1"}],
                    }
                ),
                encoding="utf-8",
            )
            (base / "stage_review_state.json").write_text(
                json.dumps(
                    {
                        "job_id": "job1",
                        "image_path": "sample.png",
                        "version": 1,
                        "updated_at": 1.0,
                        "items": [
                            {"item_id": "stage12_line_attachment:line_number_1", "bucket": "stage12_line_attachment", "entity_id": "line_number_1", "decision": "rejected"},
                            {"item_id": "stage4_instrument:instrument_tag_1", "bucket": "stage4_instrument", "entity_id": "instrument_tag_1", "decision": "rejected"},
                        ],
                        "workspace_objects": {
                            "stage4_line_number": [],
                            "stage4_instrument": [],
                            "stage12_line_attachment": [],
                            "stage12_instrument_attachment": [],
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = generate_reviewed_outputs(base)

            self.assertEqual(result["graph_summary"]["accepted_text_attachment_count"], 0)
            self.assertEqual(result["graph_summary"]["accepted_instrument_tag_attachment_count"], 0)
            self.assertEqual(result["graph_payload"]["edges"][0]["line_texts"], [])
            self.assertEqual(result["graph_payload"]["edges"][0]["instrument_tags"], [])
            self.assertTrue((base / "stage12_graph_reviewed.json").exists())
            self.assertTrue((base / "stage13_review_queue_reviewed.json").exists())


if __name__ == "__main__":
    unittest.main()
