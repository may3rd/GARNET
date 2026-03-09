import unittest

from garnet.pipe_text_attachment import run_pipe_text_attachment_stage


class PipeTextAttachmentTests(unittest.TestCase):
    def test_run_pipe_text_attachment_stage_attaches_line_number_to_edge(self) -> None:
        text_regions = [
            {
                "id": "ocr_1",
                "text": "3\"-PL-25-002013",
                "normalized_text": "3\"-PL-25-002013",
                "class": "line_number",
                "bbox": {"x_min": 5, "y_min": 5, "x_max": 25, "y_max": 15},
            }
        ]
        edges = [
            {
                "id": "edge_1",
                "source": "n1",
                "target": "n2",
                "polyline": [
                    {"row": 10, "col": 0},
                    {"row": 10, "col": 30},
                ],
            }
        ]

        result = run_pipe_text_attachment_stage(
            image_id="sample.png",
            text_regions=text_regions,
            edges=edges,
            max_distance_px=20.0,
        )

        self.assertEqual(result["summary"]["candidate_count"], 1)
        self.assertEqual(result["summary"]["accepted_attachment_count"], 1)
        self.assertEqual(len(result["attachments_payload"]["accepted"]), 1)


if __name__ == "__main__":
    unittest.main()
