import unittest

from garnet.trace_associations import build_trace_associations


def _connection_object(obj_id: str) -> dict:
    return {"id": obj_id, "class_name": "page connection"}


def _trace_payload(traces: dict) -> dict:
    return traces


class Stage6LineNumberAttachmentTests(unittest.TestCase):
    """Line-number attachment must respect label orientation.

    Real failure on drawing 25-0004 (Test-00003): the label 3"-CUL-25-003001
    (wide, horizontal) sits in the corridor between a vertical descent that
    merely clips its corner (8px) and the horizontal connector line it names
    (11px away). Pure nearest-segment matching attached it to the vertical
    cross-sheet trace, drawing the number on the wrong pipe in the
    stage6_trace_association_overlay.
    """

    def _build(self, line_numbers: list[dict]):
        trace_payload = {
            "trace_v": {
                "source_obj_id": "obj_900",
                "segments": [{"x1": 732, "y1": 527, "x2": 732, "y2": 1279, "direction": "DOWN", "length_px": 752}],
                "terminal_type": "tee_junction",
                "terminal_x": 733,
                "terminal_y": 1275,
            },
            "trace_h": {
                "source_obj_id": "obj_901",
                "segments": [{"x1": 370, "y1": 1275, "x2": 1202, "y2": 1275, "direction": "RIGHT", "length_px": 832}],
                "terminal_type": "equipment",
                "terminal_x": 1690,
                "terminal_y": 1587,
            },
        }
        return build_trace_associations(
            image_id="synthetic.png",
            objects=[_connection_object("obj_900"), _connection_object("obj_901")],
            trace_payload=trace_payload,
            branch_payload={"branches": {}},
            ports_payload={},
            line_numbers=line_numbers,
            instrument_tags=[],
            equipment_port_max_distance_px=16.0,
            inline_object_max_distance_px=24.0,
            text_max_distance_px=100.0,
            instrument_max_distance_px=90.0,
            arrow_max_distance_px=45.0,
        )

    def test_horizontal_label_prefers_horizontal_segment_over_closer_perpendicular(self) -> None:
        label = {
            "id": "line_000006",
            "normalized_text": '3"-CUL-25-003001-81A2-NI',
            "bbox": {"x_min": 387, "y_min": 1229, "x_max": 724, "y_max": 1264},
        }

        result = self._build([label])

        accepted = result["associations"]["line_numbers"]["accepted"]
        # The other connection trace may receive a simulated fill number; the
        # real attachment must land on the horizontal run.
        real = [item for item in accepted if item.get("source") != "simulated_hitl"]
        self.assertEqual(len(real), 1)
        self.assertEqual(real[0]["trace_id"], "trace_h")
        attached = result["trace_edges"][1]["attachments"]["line_numbers"]
        self.assertEqual([item["id"] for item in attached], ["line_000006"])

    def test_ambiguous_square_label_keeps_pure_nearest_segment(self) -> None:
        # Near-square label: no orientation signal, nearest wins (trace_v, 8px).
        label = {
            "id": "line_square",
            "normalized_text": '3"-CUL-25-003001-81A2-NI',
            "bbox": {"x_min": 680, "y_min": 1229, "x_max": 730, "y_max": 1264},
        }

        result = self._build([label])

        accepted = result["associations"]["line_numbers"]["accepted"]
        real = [item for item in accepted if item.get("source") != "simulated_hitl"]
        self.assertEqual(len(real), 1)
        self.assertEqual(real[0]["trace_id"], "trace_v")

    def test_horizontal_label_attaches_to_horizontal_run_when_no_vertical_nearby(self) -> None:
        # Sanity: a plain horizontal label over a horizontal line is unchanged.
        label = {
            "id": "line_plain",
            "normalized_text": '5"-CUL-25-004002-81A2-NI',
            "bbox": {"x_min": 767, "y_min": 485, "x_max": 1115, "y_max": 520},
        }
        trace_payload = {
            "trace_h": {
                "source_obj_id": "obj_901",
                "segments": [{"x1": 731, "y1": 527, "x2": 3143, "y2": 527, "direction": "LEFT", "length_px": 2412}],
                "terminal_type": "tee_junction",
                "terminal_x": 3143,
                "terminal_y": 527,
            },
        }

        result = build_trace_associations(
            image_id="synthetic.png",
            objects=[_connection_object("obj_901")],
            trace_payload=trace_payload,
            branch_payload={"branches": {}},
            ports_payload={},
            line_numbers=[label],
            instrument_tags=[],
            equipment_port_max_distance_px=16.0,
            inline_object_max_distance_px=24.0,
            text_max_distance_px=100.0,
            instrument_max_distance_px=90.0,
            arrow_max_distance_px=45.0,
        )

        accepted = result["associations"]["line_numbers"]["accepted"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["trace_id"], "trace_h")


if __name__ == "__main__":
    unittest.main()