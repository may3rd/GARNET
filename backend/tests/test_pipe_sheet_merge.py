import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.pipe_sheet_merge import (
    resolve_merge_pairs,
    MergeResult,
    CrossSheetEdge,
    MergeIssue,
    _normalize_connector_key,
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


def _system_edge(edge_id: str, target_sheet: str, connector_key: str | None) -> dict:
    edge = _edge(edge_id, "drawing", target_sheet, "bidirectional")
    edge["off_page_connector"]["target_sheet_reference"] = target_sheet
    if connector_key is not None:
        edge["off_page_connector"]["connector_key"] = connector_key
    return edge


class ResolveMergePairsTests(unittest.TestCase):
    def test_strict_merge_requires_unique_reciprocal_sheet_references_and_line_key(self):
        g1 = _graph("DWG-100", [_system_edge("e1", "DWG-200", "10-P-100-A")])
        g2 = _graph("DWG-200", [_system_edge("e2", "DWG-100", " 10-p-100-a ")])

        result = resolve_merge_pairs([g1, g2], strict=True)

        self.assertEqual(len(result.cross_sheet_edges), 1)
        edge = result.cross_sheet_edges[0].to_dict()
        self.assertEqual(edge["connector_key"], "10-P-100-A")
        self.assertEqual(edge["match_method"], "automatic")
        self.assertEqual(edge["sheets"], ["DWG-100", "DWG-200"])
        self.assertEqual(result.merge_issues, [])

    def test_strict_merge_reports_missing_connector_key(self):
        g1 = _graph("DWG-100", [_system_edge("e1", "DWG-200", None)])
        g2 = _graph("DWG-200", [_system_edge("e2", "DWG-100", "10-P-100-A")])

        result = resolve_merge_pairs([g1, g2], strict=True)

        self.assertEqual(result.cross_sheet_edges, [])
        self.assertIn("missing_connector_key", {issue.type for issue in result.merge_issues})

    def test_strict_merge_reports_non_reciprocal_reference(self):
        g1 = _graph("DWG-100", [_system_edge("e1", "DWG-200", "10-P-100-A")])
        g2 = _graph("DWG-200", [_system_edge("e2", "DWG-300", "10-P-100-A")])
        g3 = _graph("DWG-300", [])

        result = resolve_merge_pairs([g1, g2, g3], strict=True)

        self.assertEqual(result.cross_sheet_edges, [])
        self.assertIn("non_reciprocal_reference", {issue.type for issue in result.merge_issues})

    def test_strict_merge_applies_override_then_manual_pair(self):
        g1 = _graph("DWG-100", [_system_edge("e1", "bad OCR", "10-P-100-A")])
        g2 = _graph("DWG-200", [_system_edge("e2", "also bad", "wrong")])

        result = resolve_merge_pairs(
            [g1, g2],
            strict=True,
            connector_overrides={
                "DWG-100::e1": {"target_sheet_id": "DWG-200", "connector_key": "10-P-100-A"},
                "DWG-200::e2": {"target_sheet_id": "DWG-100", "connector_key": "10-P-100-A"},
            },
            manual_pairs=[{"left_connector_id": "DWG-100::e1", "right_connector_id": "DWG-200::e2"}],
        )

        self.assertEqual(len(result.cross_sheet_edges), 1)
        self.assertEqual(result.cross_sheet_edges[0].to_dict()["match_method"], "manual")
        self.assertEqual(result.merge_issues, [])

    def test_strict_merge_does_not_guess_when_duplicate_line_tags_are_ambiguous(self):
        g1 = _graph("DWG-100", [_system_edge("e1", "DWG-200", "10-P-100-A")])
        g2 = _graph("DWG-200", [
            _system_edge("e2", "DWG-100", "10-P-100-A"),
            _system_edge("e3", "DWG-100", "10-P-100-A"),
        ])

        result = resolve_merge_pairs([g1, g2], strict=True)

        self.assertEqual(result.cross_sheet_edges, [])
        self.assertIn("ambiguous_match", {issue.type for issue in result.merge_issues})

    def test_strict_merge_output_is_deterministic_for_reversed_graph_order(self):
        g1 = _graph("DWG-100", [_system_edge("edge-z", "DWG-200", "10-P-100-A")])
        g2 = _graph("DWG-200", [_system_edge("edge-a", "DWG-100", "10-P-100-A")])

        forward = resolve_merge_pairs([g1, g2], strict=True).to_dict()
        reverse = resolve_merge_pairs([g2, g1], strict=True).to_dict()

        self.assertEqual(forward, reverse)

    def test_strict_merge_keeps_connector_direction_bidirectional(self):
        g1 = _graph("DWG-100", [_system_edge("e1", "DWG-200", "10-P-100-A")])
        g2 = _graph("DWG-200", [_system_edge("e2", "DWG-100", "10-P-100-A")])
        g1["edges"][0]["off_page_connector"]["direction"] = "output"
        g2["edges"][0]["off_page_connector"]["direction"] = "input"

        edge = resolve_merge_pairs([g1, g2], strict=True).cross_sheet_edges[0].to_dict()

        self.assertEqual(edge["direction_pair"], ("bidirectional", "bidirectional"))

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

    def test_duplicate_unknown_doc_ids_are_rejected(self):
        g3 = _graph(None, [_edge("e3", "sheet", "F-2", "output")])  # type: ignore
        g4 = _graph("  ", [_edge("e4", "sheet", "F-2", "input", "destination")])  # type: ignore

        with self.assertRaisesRegex(ValueError, "duplicate document.doc_id"):
            resolve_merge_pairs([g3, g4])

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

    def test_normalize_connector_key_extracts_line_number(self):
        """OCR-noisy renderings of the same line number canonicalize to one key."""
        self.assertEqual(_normalize_connector_key("2NAS-25-003004-B2A2-NI"), "25-003004")
        self.assertEqual(_normalize_connector_key('1I,"-NAS-25-003004-82A2-NI'), "25-003004")
        self.assertEqual(_normalize_connector_key("3\"-PL-26-003008-N2A1-NI"), "26-003008")

    def test_normalize_connector_key_falls_back_to_cleaned_text(self):
        """Keys without a line-number pattern keep their cleaned text."""
        self.assertEqual(_normalize_connector_key("10-P-100-A"), "10-p-100-a")
        self.assertEqual(_normalize_connector_key(""), "")

    def test_merge_resolves_reciprocal_connectors_with_ocr_noisy_keys(self):
        """Two sheets whose connectors reference each other and share a line
        number (rendered with OCR noise) resolve to a cross-sheet edge."""
        g1 = _graph(
            "25-0002",
            [
                {
                    "id": "e1",
                    "off_page_connector": {
                        "connector_key": "2NAS-25-003004-B2A2-NI",
                        "target_sheet_reference": "25-0003",
                        "reference_type": "sheet",
                    },
                }
            ],
        )
        g2 = _graph(
            "25-0003",
            [
                {
                    "id": "e2",
                    "off_page_connector": {
                        "connector_key": '1I,"-NAS-25-003004-82A2-NI',
                        "target_sheet_reference": "25-0002",
                        "reference_type": "sheet",
                    },
                }
            ],
        )

        d = resolve_merge_pairs([g1, g2], strict=True).to_dict()

        self.assertEqual(len(d["cross_sheet_edges"]), 1)
        self.assertEqual(len(d["merge_issues"]), 0)


if __name__ == "__main__":
    unittest.main()
