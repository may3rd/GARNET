import json
import math
import tempfile
import unittest
from pathlib import Path

from garnet.graph.build_graph import build_graph

LN = '4"-CUL-25-002007-B1A2-NI'


def _ln(x, y, text=LN):
    return {"text": text, "review_state": "accepted", "projected_xy": [x, y]}


def _trace(tid, poly, ttype, src="page connection", **kw):
    return {"trace_id": tid, "trace_kind": "port", "source_obj_id": tid, "source_obj_type": src,
            "port": {"x": poly[0][0], "y": poly[0][1], "direction": "RIGHT"}, "terminal_type": ttype,
            "terminal_obj_id": None, "terminal_xy": poly[-1], "polyline": poly, "status": "ok",
            "attachments": kw.get("att", {})}


def _run(traces, objects=(), ln_assign=None):
    with tempfile.TemporaryDirectory() as d:
        if ln_assign is not None:
            (Path(d) / "ln_assign.json").write_text(json.dumps({"sheet": "x-y-0009", "assignments": ln_assign}))
        (Path(d) / "x-y-0009_objects.json").write_text(json.dumps({"objects": list(objects)}))
        (Path(d) / "stage6_trace_associations.json").write_text(json.dumps({"trace_edges": traces}))
        return build_graph("x-y-0009", d, d)


class TestBuildGraph(unittest.TestCase):
    def test_tee_splits_target_and_orients(self):
        a = _trace("A", [[0, 0], [600, 0]], "dead_end",
                   att={"line_numbers": [_ln(100, 0), _ln(500, 0)]})
        b = _trace("B", [[300, 400], [300, 5]], "tee_junction", att={"line_numbers": [_ln(300, 200)]})
        g = _run([a, b])
        tees = [n for n in g["nodes"] if n["kind"] == "tee"]
        pipes = [e for e in g["edges"] if e["kind"] == "pipe"]
        self.assertEqual(g["sheets"], ["x-y-0009"])  # full stem, node ids keep the short form
        self.assertEqual(len(tees), 1)
        self.assertEqual(len(pipes), 3)
        for e in pipes:
            self.assertEqual(e["line_core"], "25-002007")
            for p, q in zip(e["polyline"], e["polyline"][1:]):
                self.assertLessEqual(math.dist(p, q), 200)
        # A's pieces run u->v: first starts at the connector (x=0), second ends at the dead end (x=600)
        ea = sorted((e for e in pipes if e["trace_ids"] == ["A"]), key=lambda e: e["polyline"][0][0])
        self.assertEqual([e["polyline"][0][0] for e in ea], [0.0, 300.0])
        self.assertEqual(ea[1]["polyline"][-1][0], 600.0)
        self.assertEqual(sum(tees[0]["id"] in (e["u"], e["v"]) for e in pipes), 3)
        self.assertEqual([f["kind"] for f in g["flags"] if f["kind"] == "dead_end"], ["dead_end"])
        self.assertEqual(g["line_list"][0]["line_core"], "25-002007")
        self.assertEqual(len(g["line_list"][0]["edges"]), 3)
        self.assertEqual([e["line_core"] for e in ea], ["25-002007"] * 2)  # label on one half, both pieces carry it

    def test_connector_text_and_direction(self):
        a = _trace("A", [[0, 0], [300, 0]], "dead_end")
        g = _run([a], [{"id": "A", "class_name": "page connection", "Text": "X-1", "Direction": "To"}])
        n = [n for n in g["nodes"] if n["kind"] == "page_connection"][0]
        self.assertEqual((n["id"], n["text"], n["direction"]), ("0009/PC/X-1@1", "X-1", "to"))

    def test_inline_hit_deduped_against_detector_object(self):
        def v(i, x):
            return {"id": i, "class_name": "gate valve", "projected_xy": [x, 0]}
        a = _trace("A", [[0, 0], [300, 0]], "dead_end",
                   att={"inline_objects": [v("obj_1", 100), v("A:hit_001", 90), v("A:hit_002", 250)]})
        e = [e for e in _run([a])["edges"] if e["kind"] == "pipe"][0]
        self.assertEqual(e["inline_objects"], ["gate valve#A:hit_002", "gate valve#obj_1"])  # unmatched hit stays

    def test_vision_assignment_overrides_stage6(self):
        a = _trace("A", [[0, 0], [300, 0]], "dead_end", att={"line_numbers": [dict(_ln(50, 0), id="line_number_0001")]})
        b = _trace("B", [[0, 100], [300, 100]], "dead_end")
        mv = [{"id": "line_number_0001", "choice": "B", "trace_id": "B", "xy": [60, 100], "via": "leader_arrow", "why": ""}]
        by = {e["trace_ids"][0]: e for e in _run([a, b], ln_assign=mv)["edges"] if e["kind"] == "pipe"}
        self.assertEqual((by["B"]["line_core"], by["B"]["line_number_source"]), ("25-002007", "vision"))
        self.assertEqual(by["A"]["line_core"], "")
        g = _run([a, b], ln_assign=[dict(mv[0], choice="none", trace_id=None, xy=None, why="points at nothing")])
        self.assertEqual([f["why"] for f in g["flags"] if f["kind"] == "ln_unassigned"], ["points at nothing"])
        g = _run([a, b], ln_assign=[])  # file present but label omitted: Stage 6 pick kept and flagged
        self.assertEqual([f["kind"] for f in g["flags"] if f["kind"] == "ln_not_reviewed"], ["ln_not_reviewed"])
        self.assertEqual({e["line_number_source"] for e in g["edges"] if e["trace_ids"] == ["A"]}, {"geometry"})

    def test_multiple_line_numbers_flagged(self):
        a = _trace("A", [[0, 0], [300, 0]], "dead_end",
                   att={"line_numbers": [_ln(50, 0), _ln(150, 0), _ln(250, 0, '3"-PL-25-002008-B1A2-NI')]})
        g = _run([a])
        f = [f for f in g["flags"] if f["kind"] == "multiple_line_numbers"]
        self.assertEqual(len(f), 1)
        self.assertIn("25-002008", f[0]["why"])
        self.assertEqual([e for e in g["edges"] if e["kind"] == "pipe"][0]["line_core"], "25-002007")

    def test_missing_stage6_raises(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(FileNotFoundError):
            build_graph("x-y-0009", d, d)


if __name__ == "__main__":
    unittest.main()
