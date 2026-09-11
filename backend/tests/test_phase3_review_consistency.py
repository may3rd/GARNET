import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from garnet.graph_export_adapter import build_graph_v1_payload
from garnet.pid_extractor import PIDPipeline
from garnet.stage10_process_exports import build_stage10_process_exports
from garnet.stage9_review_decisions import apply_stage9_review_decisions


def _line(line_id: str, text: str, *, review_state: str | None = None) -> dict:
    record = {
        "id": line_id,
        "source_object_id": f"ocr::{line_id}",
        "display_text": text,
        "normalized_text": text,
    }
    if review_state is not None:
        record["review_state"] = review_state
    return record


def _correction_fixture() -> dict:
    return {
        "image_id": "synthetic.png",
        "nodes": [],
        "line_number_catalog": [_line("line_new", "NEW-LINE")],
        "edges": [
            {
                "id": "edge_target",
                "source": "n1",
                "target": "n2",
                "trace_length_px": 80,
                "line_numbers": [_line("line_old", "OLD-LINE")],
                "effective_line_number_ids": ["line_old"],
                "effective_line_numbers": [_line("line_old", "OLD-LINE")],
                "direct_line_number_ids": ["line_old"],
                "line_number_assignment_state": "direct",
                "attachments": {
                    "line_numbers": [_line("line_old", "OLD-LINE")],
                    "inline_objects": [
                        {"id": "valve_1", "source_object_id": "valve_1", "class_name": "gate valve"}
                    ],
                },
                "polyline": [{"x": 0, "y": 0}, {"x": 80, "y": 0}],
            },
            {
                "id": "edge_catalog",
                "source": "n2",
                "target": "n3",
                "trace_length_px": 20,
                "line_numbers": [_line("line_new", "NEW-LINE")],
                "effective_line_number_ids": ["line_new"],
                "effective_line_numbers": [_line("line_new", "NEW-LINE")],
                "attachments": {"line_numbers": [_line("line_new", "NEW-LINE")]},
                "polyline": [{"x": 80, "y": 0}, {"x": 100, "y": 0}],
            },
        ],
        "line_groups": [
            {"line_number_id": "line_old", "edge_ids": ["edge_target"]},
            {"line_number_id": "line_new", "edge_ids": ["edge_catalog"]},
        ],
    }


class Phase3ReviewConsistencyTests(unittest.TestCase):
    def _apply_correction(self) -> dict:
        review_id = "review::line_number_conflict::edge_target"
        return apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload=_correction_fixture(),
            review_items_payload={"review_items": [{"id": review_id, "category": "line_number_conflict"}]},
            decisions_payload={
                "decisions": [
                    {
                        "review_item_id": review_id,
                        "decision": "set_line_number",
                        "line_number_id": "line_new",
                        "edge_ids": ["edge_target"],
                        "reviewer": "reviewer-1",
                    }
                ]
            },
        )

    def test_correction_resolves_existing_catalog_record_and_preserves_observed_evidence(self):
        edge = next(edge for edge in self._apply_correction()["corrected_graph_payload"]["edges"] if edge["id"] == "edge_target")

        self.assertEqual(edge["effective_line_number_ids"], ["line_new"])
        self.assertEqual(edge["direct_line_number_ids"], ["line_new"])
        self.assertIn(edge["line_number_assignment_state"], {"direct", "human_reviewed"})
        self.assertNotIn(edge["line_number_assignment_state"], {"inferred", "conflict", "missing"})
        self.assertEqual(edge["effective_line_numbers"][0]["display_text"], "NEW-LINE")
        self.assertEqual(edge["line_numbers"][0]["display_text"], "NEW-LINE")
        self.assertEqual(
            [record["id"] for record in edge.get("direct_line_numbers", [])],
            edge["direct_line_number_ids"],
        )
        self.assertEqual(
            [record["id"] for record in edge.get("inferred_line_numbers", [])],
            edge.get("inferred_line_number_ids", []),
        )
        observed = edge["attachments"]["observed_line_numbers"]
        old = next(record for record in observed if record["id"] == "line_old")
        self.assertEqual(old["display_text"], "OLD-LINE")
        self.assertNotEqual(old.get("review_state"), "accepted")
        self.assertEqual(edge["attachments"]["line_numbers"][0]["review_state"], "accepted")

    def test_correction_rebuilds_line_groups_without_stale_membership(self):
        graph = self._apply_correction()["corrected_graph_payload"]
        groups = {group["line_number_id"]: set(group["edge_ids"]) for group in graph["line_groups"]}

        self.assertNotIn("edge_target", groups.get("line_old", set()))
        self.assertIn("edge_target", groups["line_new"])
        self.assertIn("edge_catalog", groups["line_new"])

    def test_corrected_graph_v1_keeps_reviewed_line_semantics_and_attachments(self):
        graph = self._apply_correction()["corrected_graph_payload"]
        payload = build_graph_v1_payload(stage12_graph=graph, image_dimensions={"width": 120, "height": 40})
        edge = next(edge for edge in payload["edges"] if edge["id"] == "edge_target")

        self.assertEqual(edge["line_number_ids"], ["line_new"])
        self.assertEqual(edge["line_numbers"][0]["display_text"], "NEW-LINE")
        self.assertEqual(edge["attachments"]["line_numbers"][0]["display_text"], "NEW-LINE")
        self.assertEqual(edge["line_number_review_state"], "human_reviewed")

    def test_stage10_inline_mto_uses_corrected_line_text(self):
        graph = self._apply_correction()["corrected_graph_payload"]
        item = build_stage10_process_exports(image_id="synthetic.png", corrected_graph_payload=graph)["inline_mto_payload"]["items"][0]

        self.assertEqual(item["line_number_ids"], ["line_new"])
        self.assertEqual(item["line_number_texts"], ["NEW-LINE"])
        self.assertEqual(item["line_number_assignment_state"], "selected")

    def test_top_level_only_original_line_record_is_preserved_as_observed(self):
        graph = _correction_fixture()
        target = graph["edges"][0]
        target["attachments"].pop("line_numbers")
        target["line_numbers"] = [_line("line_old", "OLD-LINE")]

        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload=graph,
            review_items_payload={"review_items": [{"id": "review::top-level", "category": "line_number_conflict"}]},
            decisions_payload={
                "decisions": [{
                    "review_item_id": "review::top-level",
                    "decision": "set_line_number",
                    "line_number_id": "line_new",
                    "edge_ids": ["edge_target"],
                    "reviewer": "reviewer-1",
                }]
            },
        )
        corrected = result["corrected_graph_payload"]["edges"][0]

        self.assertEqual(corrected["line_numbers"][0]["id"], "line_new")
        self.assertEqual(corrected["line_numbers"][0]["display_text"], "NEW-LINE")
        observed_old = next(record for record in corrected["attachments"]["observed_line_numbers"] if record["id"] == "line_old")
        self.assertEqual(observed_old["display_text"], "OLD-LINE")
        self.assertNotEqual(observed_old.get("review_state"), "accepted")

    def test_multiple_decisions_same_edge_keep_first_observed_and_sync_final_views(self):
        graph = _correction_fixture()
        graph["edges"].append({
            "id": "edge_second_catalog",
            "line_numbers": [_line("line_final", "FINAL-LINE")],
            "effective_line_numbers": [_line("line_final", "FINAL-LINE")],
        })
        review_items = {
            "review_items": [
                {"id": "review::first", "category": "line_number_conflict"},
                {"id": "review::second", "category": "line_number_conflict"},
            ]
        }
        decisions = {
            "decisions": [
                {"review_item_id": "review::first", "decision": "set_line_number", "line_number_id": "line_new", "edge_ids": ["edge_target"]},
                {"review_item_id": "review::second", "decision": "set_line_number", "line_number_id": "line_final", "edge_ids": ["edge_target"]},
            ]
        }

        result = apply_stage9_review_decisions(
            image_id="synthetic.png",
            graph_payload=graph,
            review_items_payload=review_items,
            decisions_payload=decisions,
        )
        edge = next(edge for edge in result["corrected_graph_payload"]["edges"] if edge["id"] == "edge_target")

        self.assertEqual(edge["line_numbers"][0]["id"], "line_final")
        self.assertEqual(edge["direct_line_number_ids"], ["line_final"])
        self.assertEqual(edge["effective_line_number_ids"], ["line_final"])
        self.assertEqual([record["id"] for record in edge["direct_line_numbers"]], ["line_final"])
        self.assertEqual([record["id"] for record in edge["effective_line_numbers"]], ["line_final"])
        observed_ids = [record["id"] for record in edge["attachments"]["observed_line_numbers"]]
        self.assertEqual(observed_ids, ["line_old"])

    def test_runner_stage9_writes_consistent_corrected_graph_and_graph_v1(self):
        review_id = "review::line_number_conflict::edge_target"
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            image_path = out_dir / "synthetic.png"
            image_path.write_bytes(b"synthetic image placeholder")
            graph = _correction_fixture()
            (out_dir / "stage7_graph.json").write_text(json.dumps(graph), encoding="utf-8")
            (out_dir / "stage8_review_items.json").write_text(
                json.dumps({"review_items": [{"id": review_id, "category": "line_number_conflict"}]}),
                encoding="utf-8",
            )
            (out_dir / "stage8_review_decisions.json").write_text(
                json.dumps(
                    {
                        "decisions": [
                            {
                                "review_item_id": review_id,
                                "decision": "set_line_number",
                                "line_number_id": "line_new",
                                "edge_ids": ["edge_target"],
                                "reviewer": "reviewer-1",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            for name, payload in {
                "stage4_objects": {"objects": []},
                "stage4_line_numbers": {"line_numbers": []},
                "stage4_instrument_tags": {"instrument_tags": []},
                "stage1_normalization_summary": {"dimensions": {"width": 120, "height": 40}},
            }.items():
                (out_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

            pipeline = PIDPipeline(str(image_path), output_dir=out_dir)
            pipeline.stage9_apply_review_decisions()

            corrected = json.loads((out_dir / "stage9_corrected_graph.json").read_text(encoding="utf-8"))
            public = json.loads((out_dir / "stage7b_graph_v1.json").read_text(encoding="utf-8"))
            corrected_edge = next(edge for edge in corrected["edges"] if edge["id"] == "edge_target")
            public_edge = next(edge for edge in public["edges"] if edge["id"] == "edge_target")
            self.assertEqual(corrected_edge["effective_line_number_ids"], ["line_new"])
            self.assertEqual(public_edge["line_number_ids"], ["line_new"])
            self.assertEqual(public_edge["line_numbers"][0]["display_text"], "NEW-LINE")

    def test_runner_stage11_overlay_reads_corrected_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            image_path = out_dir / "synthetic.png"
            image_path.write_bytes(b"synthetic image placeholder")
            old_graph = {"image_id": "synthetic.png", "nodes": [], "edges": [], "revision": "old"}
            corrected_graph = {"image_id": "synthetic.png", "nodes": [], "edges": [], "revision": "corrected"}
            (out_dir / "stage7_graph.json").write_text(json.dumps(old_graph), encoding="utf-8")
            (out_dir / "stage9_corrected_graph.json").write_text(json.dumps(corrected_graph), encoding="utf-8")
            pipeline = PIDPipeline(str(image_path), output_dir=out_dir)
            pipeline.image_bgr = np.zeros((4, 4, 3), dtype=np.uint8)
            captured = {}
            with patch("garnet.trace_graph_builder.render_stage12_graph_overlay", side_effect=lambda image, graph: captured.setdefault("graph", graph) or image), patch.object(pipeline, "_save_img"):
                pipeline.stage11_connection_overlay()
            self.assertEqual(captured["graph"], corrected_graph)


if __name__ == "__main__":
    unittest.main()
