"""Stage 10's released envelope must resolve every route reference.

The Stage 9 corrected graph has no top-level inline-object catalog (inline
objects live on edge attachments) and gives an equipment port the same id as
its graph node. Both used to make build_downstream_export raise, but only
once a release gate actually passed -- a blocked gate redacts the graph
before validation, so the defect stayed hidden behind the gate.
"""

import unittest

from garnet.versioned_export import build_downstream_export

GATE = {"release_ready": True, "status": "ready", "blocking_count": 0}

GRAPH = {
    "image_id": "sheet.png",
    "nodes": [
        {"id": "equipment::sheet.png::V-1::port::xy_10_20", "type": "equipment_port"},
        {"id": "junction::xy::80::20", "type": "tee_junction"},
    ],
    "ports": [
        {
            "id": "equipment::sheet.png::V-1::port::xy_10_20",
            "equipment_id": "equipment::sheet.png::V-1",
            "source_node_id": "equipment::sheet.png::V-1::port::xy_10_20",
        }
    ],
    "equipment": [{"id": "equipment::sheet.png::V-1", "class_name": "vessel"}],
    "edges": [
        {
            "id": "trace::V-1:port_01::part_001",
            "source": "equipment::sheet.png::V-1::port::xy_10_20",
            "target": "junction::xy::80::20",
            "flow_direction_state": "forward",
            "segments": [{"x1": 10, "y1": 20, "x2": 80, "y2": 20}],
            "polyline": [{"x": 10, "y": 20}, {"x": 80, "y": 20}],
            "attachments": {
                "inline_objects": [
                    {"id": "obj_000001", "source_object_id": "obj_000001", "class_name": "gate valve"}
                ]
            },
        }
    ],
}


class TestReleasedExportCatalogs(unittest.TestCase):
    def test_released_export_resolves_attachment_only_inline_objects(self) -> None:
        export = build_downstream_export(GRAPH, release_gate=GATE, scope="page")

        self.assertTrue(export["release_ready"])
        graph = export["graph"]
        self.assertEqual([item["id"] for item in graph["inline_objects"]], ["obj_000001"])
        # The route's reference and the catalog id are the same string, so the
        # dangling_reference check passes by construction rather than by luck.
        self.assertEqual(graph["routes"][0]["inline_object_ids"], ["obj_000001"])
        self.assertEqual(len(graph["nodes"]), 2)
        self.assertEqual(len(graph["ports"]), 1)

    def test_port_sharing_its_node_id_is_not_ambiguous(self) -> None:
        # source_node_id declares the identity; without the exemption this
        # raises "used by multiple typed collections".
        build_downstream_export(GRAPH, release_gate=GATE, scope="page")

        undeclared = {
            **GRAPH,
            "ports": [{"id": "equipment::sheet.png::V-1::port::xy_10_20", "equipment_id": "equipment::sheet.png::V-1"}],
        }
        with self.assertRaises(ValueError) as ctx:
            build_downstream_export(undeclared, release_gate=GATE, scope="page")
        self.assertIn("multiple typed collections", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
