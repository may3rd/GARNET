import unittest
from copy import deepcopy

from garnet.pid_extractor import (
    LINE_NUMBER_REVIEW_ASSUMPTION,
    _mark_line_number_review_state,
    simulate_line_number_hitl_for_missing_traces,
    build_stage11_line_number_review_payload,
)


class Stage11LineNumberHitlTests(unittest.TestCase):
    def test_mark_line_number_review_state_marks_accepted_as_human_assumed(self) -> None:
        association = {"id": "ln1", "trace_id": "trace_a"}

        result = _mark_line_number_review_state(association, accepted=True)

        self.assertEqual(result["review_state"], "accepted")
        self.assertEqual(result["review_source"], "human_assumed")
        self.assertFalse(result["review_required"])
        self.assertEqual(result["id"], "ln1")

    def test_mark_line_number_review_state_marks_rejected_as_needs_review(self) -> None:
        association = {"id": "ln2", "reason": "distance_over_threshold"}

        result = _mark_line_number_review_state(association, accepted=False)

        self.assertEqual(result["review_state"], "needs_review")
        self.assertEqual(result["review_source"], "system")
        self.assertTrue(result["review_required"])

    def test_build_stage11_line_number_review_payload(self) -> None:
        payload, summary = build_stage11_line_number_review_payload(
            image_id="synthetic.png",
            accepted=[{"id": "ln1", "trace_id": "trace_a", "review_state": "accepted"}],
            rejected=[{"id": "ln2", "reason": "distance_over_threshold", "review_state": "needs_review"}],
            traces_without_line_number=["trace_b"],
        )

        self.assertEqual(payload["image_id"], "synthetic.png")
        self.assertEqual(payload["review_assumption"], LINE_NUMBER_REVIEW_ASSUMPTION)
        self.assertEqual(payload["accepted"][0]["id"], "ln1")
        self.assertEqual(payload["needs_review"][0]["id"], "ln2")
        self.assertEqual(payload["traces_without_line_number"], ["trace_b"])
        self.assertEqual(summary["accepted_count"], 1)
        self.assertEqual(summary["needs_review_count"], 1)
        self.assertEqual(summary["trace_without_line_number_count"], 1)

    def test_simulate_line_number_hitl_assigns_missing_traces_deterministically(self) -> None:
        edges = [
            {"trace_id": "trace_a", "attachments": {"line_numbers": [{"id": "line_existing"}]}},
            {"trace_id": "trace_b", "attachments": {}},
            {"trace_id": "trace_c", "attachments": {}},
        ]
        reviewed_line_numbers = [
            {"id": "line_1", "text": "1-A", "review_state": "accepted"},
            {"id": "line_2", "text": "2-B", "review_state": "accepted"},
        ]

        first_edges = deepcopy(edges)
        second_edges = deepcopy(edges)
        first = simulate_line_number_hitl_for_missing_traces(first_edges, reviewed_line_numbers)
        second = simulate_line_number_hitl_for_missing_traces(second_edges, reviewed_line_numbers)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertTrue(all(item["review_source"] == "human_simulated" for item in first))
        self.assertEqual({item["trace_id"] for item in first}, {"trace_b", "trace_c"})
        self.assertTrue(all(edge.get("attachments", {}).get("line_numbers") for edge in first_edges))


if __name__ == "__main__":
    unittest.main()
