import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.pipe_sheet_merge import (
    resolve_merge_pairs,
    MergeResult,
    CrossSheetEdge,
    MergeIssue,
)


def _graph(doc_id: str, edges: list[dict]) -> dict:
    return {"schema_version": "graph_v1", "document": {"doc_id": doc_id}, "edges": edges}


def _edge(edge_id: str, ref_type: str, ref_value: str, direction: str, exit_terminal: str = "source") -> dict:
    return {
        "id": edge_id,
        "source": "n1",
        "target": "n2",
        "off_page_connector": {
            "reference_type": ref_type,
            "reference_value": ref_value,
            "direction": direction,
            "exit_terminal": exit_terminal,
            "local_edge_id": edge_id,
        },
    }


class ResolveMergePairsTests(unittest.TestCase):
    def test_two_sheets_one_pair_merged(self):
        g1 = _graph("SHEET-1", [_edge("e1", "sheet", "A-3", "output")])
        g2 = _graph("SHEET-A3", [_edge("e2", "sheet", "A-3", "input", "destination")])

        result = resolve_merge_pairs([g1, g2])

        self.assertEqual(len(result.cross_sheet_edges), 1)
        edge = result.cross_sheet_edges[0]
        self.assertEqual(edge.reference_value, "A-3")
        self.assertEqual(sorted(edge.sheets), ["SHEET-1", "SHEET-A3"])
        self.assertEqual(edge.status, "merged")
        self.assertEqual(edge.direction_a, "output")
        self.assertEqual(edge.direction_b, "input")
        self.assertEqual(len(result.merge_issues), 0)
        self.assertEqual(len(result.per_sheet_resolved), 2)
        for s in result.per_sheet_resolved:
            self.assertEqual(s.resolved_count, 1)
            self.assertEqual(s.dangling_count, 0)

    def test_bidirectional_pairs_with_output_input(self):
        g1 = _graph("S1", [_edge("e1", "sheet", "X-1", "bidirectional")])
        g2 = _graph("S2", [_edge("e2", "sheet", "X-1", "output")])

        result = resolve_merge_pairs([g1, g2])
        self.assertEqual(len(result.cross_sheet_edges), 1)
        self.assertEqual(result.cross_sheet_edges[0].reference_value, "X-1")

    def test_ambiguous_more_than_two_sheets(self):
        g1 = _graph("S1", [_edge("e1", "sheet", "B-2", "output")])
        g2 = _graph("S2", [_edge("e2", "sheet", "B-2", "input", "destination")])
        g3 = _graph("S3", [_edge("e3", "sheet", "B-2", "input", "destination")])

        result = resolve_merge_pairs([g1, g2, g3])

        self.assertEqual(len(result.cross_sheet_edges), 0)
        self.assertEqual(len(result.merge_issues), 1)
        issue = result.merge_issues[0]
        self.assertEqual(issue.type, "ambiguous_merge")
        self.assertIn("B-2", issue.issue_id)
        self.assertEqual(sorted(issue.sheets_involved), ["S1", "S2", "S3"])

    def test_dangling_one_sheet_only(self):
        g1 = _graph("S1", [_edge("e1", "sheet", "Z-9", "output")])

        result = resolve_merge_pairs([g1])

        self.assertEqual(len(result.cross_sheet_edges), 0)
        self.assertEqual(len(result.merge_issues), 1)
        issue = result.merge_issues[0]
        self.assertEqual(issue.type, "dangling_connector")
        self.assertIn("Z-9", issue.issue_id)
        self.assertEqual(issue.sheets_involved, ["S1"])
        sheet_stat = next(s for s in result.per_sheet_resolved if s.doc_id == "S1")
        self.assertEqual(sheet_stat.resolved_count, 0)
        self.assertEqual(sheet_stat.dangling_count, 1)

    def test_intra_sheet_duplicate(self):
        g1 = _graph("S1", [
            _edge("e1", "sheet", "C-1", "output"),
            _edge("e2", "sheet", "C-1", "input", "destination"),
        ])

        result = resolve_merge_pairs([g1])

        self.assertEqual(len(result.cross_sheet_edges), 0)
        self.assertEqual(len(result.merge_issues), 1)
        issue = result.merge_issues[0]
        self.assertEqual(issue.type, "intra_sheet_duplicate")

    def test_direction_conflict_both_output(self):
        g1 = _graph("S1", [_edge("e1", "sheet", "D-1", "output")])
        g2 = _graph("S2", [_edge("e2", "sheet", "D-1", "output")])

        result = resolve_merge_pairs([g1, g2])

        self.assertEqual(len(result.cross_sheet_edges), 0)
        self.assertEqual(len(result.merge_issues), 1)
        self.assertEqual(result.merge_issues[0].type, "direction_conflict")

    def test_edges_without_off_page_connector_are_ignored(self):
        g1 = _graph("S1", [
            _edge("e1", "sheet", "E-1", "output"),
            {"id": "internal", "source": "n1", "target": "n2"},  # no off_page_connector
        ])
        g2 = _graph("S2", [_edge("e2", "sheet", "E-1", "input", "destination")])

        result = resolve_merge_pairs([g1, g2])

        self.assertEqual(len(result.cross_sheet_edges), 1)
        self.assertEqual(result.cross_sheet_edges[0].reference_value, "E-1")

    def test_unknown_doc_id_falls_back(self):
        g1 = _graph("S1", [_edge("e1", "sheet", "F-1", "output")])
        g2 = _graph("S2", [_edge("e2", "sheet", "F-1", "input", "destination")])
        # Also test truly empty doc_ids — both normalize to UNKNOWN, treated as
        # same-sheet, which correctly becomes an INTRA_SHEET_DUPLICATE.
        g3 = _graph(None, [_edge("e3", "sheet", "F-2", "output")])  # type: ignore
        g4 = _graph("  ", [_edge("e4", "sheet", "F-2", "input", "destination")])  # type: ignore

        result = resolve_merge_pairs([g1, g2, g3, g4])

        # g1+g2 have real doc_ids → merged
        self.assertEqual(len(result.cross_sheet_edges), 1)
        self.assertEqual(result.cross_sheet_edges[0].reference_value, "F-1")
        # g3+g4 both normalize to UNKNOWN doc_id → INTRA_SHEET_DUPLICATE
        self.assertEqual(len(result.merge_issues), 1)
        self.assertEqual(result.merge_issues[0].type, "intra_sheet_duplicate")

    def test_reference_type_pid(self):
        g1 = _graph("S1", [_edge("e1", "pid", "P-101", "output")])
        g2 = _graph("S2", [_edge("e2", "pid", "P-101", "input", "destination")])

        result = resolve_merge_pairs([g1, g2])

        self.assertEqual(len(result.cross_sheet_edges), 1)
        edge = result.cross_sheet_edges[0]
        self.assertEqual(edge.reference_type, "pid")
        self.assertEqual(edge.merge_key, ("pid", "P-101"))

    def test_virtual_edge_id_is_deterministic(self):
        g1 = _graph("B", [_edge("e1", "sheet", "X-1", "output")])
        g2 = _graph("A", [_edge("e2", "sheet", "X-1", "input", "destination")])

        result = resolve_merge_pairs([g1, g2])

        edge = result.cross_sheet_edges[0]
        # Edge ID uses sorted sheet order so (B,A) and (A,B) produce the same ID
        self.assertEqual(edge.sheets, ["A", "B"])
        self.assertEqual(edge.id, "xs::A::X-1::B")

    def test_merge_result_to_dict(self):
        g1 = _graph("S1", [_edge("e1", "sheet", "G-1", "output")])
        g2 = _graph("S2", [_edge("e2", "sheet", "G-1", "input", "destination")])

        d = resolve_merge_pairs([g1, g2]).to_dict()

        self.assertEqual(d["schema_version"], "graph_v2")
        self.assertEqual(len(d["cross_sheet_edges"]), 1)
        self.assertEqual(d["cross_sheet_edges"][0]["reference_value"], "G-1")
        self.assertEqual(len(d["merge_issues"]), 0)

    def test_issue_to_dict(self):
        g1 = _graph("S1", [_edge("e1", "sheet", "H-1", "output")])
        result = resolve_merge_pairs([g1])
        d = result.to_dict()

        self.assertEqual(len(d["merge_issues"]), 1)
        issue = d["merge_issues"][0]
        self.assertEqual(issue["type"], "dangling_connector")
        self.assertIn("H-1", issue["issue_id"])


if __name__ == "__main__":
    unittest.main()