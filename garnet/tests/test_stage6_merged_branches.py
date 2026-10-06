"""Stage 6 against Stage 5c output: a run 5c absorbed ("merged") is neither an
edge nor a skipped branch, and the payload names the branch file it used."""
import unittest

from garnet.stage5b.trace_associations import build_trace_associations


def _seg(x1, y1, x2, y2):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


class MergedBranchTests(unittest.TestCase):
    def test_two_merged_fragments_give_one_edge_zero_skipped(self):
        branch_payload = {"branches": {
            "br_a": {"status": "traced", "segments": [_seg(0, 50, 100, 50), _seg(100, 50, 200, 50)],
                     "merged_from": ["br_b"]},
            "br_b": {"status": "merged", "merged_into": "br_a", "segments": [_seg(100, 50, 200, 50)]},
        }}
        result = build_trace_associations(
            image_id="t", objects=[], trace_payload={}, branch_payload=branch_payload,
            ports_payload={}, line_numbers=[], instrument_tags=[],
            equipment_port_max_distance_px=20, inline_object_max_distance_px=20,
            text_max_distance_px=40, instrument_max_distance_px=40, arrow_max_distance_px=20,
            image_bgr=None, branch_source="stage5c_branch_trace_results.json",
        )
        s = result["trace_association_summary"]
        p = result["trace_associations_payload"]
        self.assertEqual(s["branch_trace_count"], 1)
        self.assertEqual(s["skipped_branch_count"], 0)
        self.assertEqual(s["merged_branch_count"], 1)
        self.assertEqual(p["merged_branches"][0]["merged_into"], "br_a")
        self.assertEqual(p["branch_source"], "stage5c_branch_trace_results.json")
        self.assertEqual(s["branch_source"], "stage5c_branch_trace_results.json")


if __name__ == "__main__":
    unittest.main()
