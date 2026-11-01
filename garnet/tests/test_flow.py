import unittest

import numpy as np

from garnet.graph.flow import infer_flow, read_arrow_vector


def _g(nodes, edges):
    ns = [{"id": i, "kind": k, **kw} for i, k, kw in nodes]
    es = [{"id": f"E{n}", "kind": "pipe", "u": u, "v": v, "trace_ids": [t],
           "polyline": [[0, 0], [100, 0]] if len(p) == 0 else p} for n, (u, v, t, p) in enumerate(edges, 1)]
    return {"nodes": ns, "edges": es, "flags": []}


def _flow(g):
    return {e["id"]: (e["flow"], e["flow_source"]) for e in g["edges"]}


def _arrow(tid, xy, vec):
    return {"id": "a1", "trace_id": tid, "projected_xy": xy, "flow_direction_evidence": {"vector": vec}}


class TestFlow(unittest.TestCase):
    def test_arrow_on_middle_piece_orients_all_pieces(self):
        g = _g([("A", "page_connection", {}), ("T", "tee", {}), ("B", "page_connection", {}), ("C", "page_connection", {})],
               [("A", "T", "t1", [[0, 0], [100, 0]]), ("T", "B", "t1", [[100, 0], [200, 0]]),
                ("C", "T", "t2", [[100, 90], [100, 0]])])
        s6 = {"associations": {"flow_arrows": {"accepted": [_arrow("t1", [150, 0], [-1, 0])]}}}
        f = _flow(infer_flow(g, s6))
        self.assertEqual(f["E1"], ("reverse", "propagated"))
        self.assertEqual(f["E2"], ("reverse", "arrow"))

    def test_connector_to_orients_into_connector(self):
        g = _g([("A", "port", {}), ("PC", "page_connection", {"direction": "to"})], [("A", "PC", "t", [])])
        g2 = _g([("PC", "page_connection", {"direction": "to"}), ("A", "port", {})], [("PC", "A", "t", [])])
        self.assertEqual(_flow(infer_flow(g, {}))["E1"], ("forward", "connector"))
        self.assertEqual(_flow(infer_flow(g2, {}))["E1"], ("reverse", "connector"))

    def test_pump_top_port_out_bottom_in(self):
        g = _g([("P", "equipment", {"type": "pump", "tag": "P-1"}), ("top", "port", {"equipment": "P", "xy_px": [0, 10]}),
                ("bot", "port", {"equipment": "P", "xy_px": [0, 90]}), ("X", "end", {}), ("Y", "end", {})],
               [("top", "X", "t1", []), ("Y", "bot", "t2", [])])
        f = _flow(infer_flow(g, {}))
        self.assertEqual(f["E1"], ("forward", "pump"))   # out of the top port
        self.assertEqual(f["E2"], ("forward", "pump"))   # into the bottom port

    def test_tee_conservation_decides_exactly_one(self):
        g = _g([("A", "page_connection", {"direction": "from"}), ("T", "tee", {}), ("B", "page_connection", {"direction": "to"}),
                ("C", "end", {}), ("D", "end", {})],
               [("A", "T", "t1", []), ("T", "B", "t2", []), ("T", "C", "t3", []), ("D", "C", "t4", [])])
        f = _flow(infer_flow(g, {}))
        self.assertEqual(f["E1"], ("forward", "connector"))   # in to tee
        self.assertEqual(f["E2"], ("forward", "connector"))   # out of tee
        self.assertEqual(f["E3"][0], "unknown")               # in 1, out 1, unknown 1: not decidable
        self.assertEqual(f["E4"][0], "unknown")               # end node never decides
        g = _g([("A", "page_connection", {"direction": "from"}), ("T", "tee", {}), ("B", "end", {})],
               [("A", "T", "t1", []), ("T", "B", "t2", [])])
        self.assertEqual(_flow(infer_flow(g, {}))["E2"], ("forward", "propagated"))  # all in -> unknown is out

    def test_arrow_vs_connector_conflict(self):
        g = _g([("A", "port", {}), ("PC", "page_connection", {"direction": "to"})], [("A", "PC", "t", [[0, 0], [100, 0]])])
        s6 = {"associations": {"flow_arrows": {"accepted": [_arrow("t", [50, 0], [-1, 0])]}}}
        g = infer_flow(g, s6)
        self.assertEqual(_flow(g)["E1"], ("unknown", "conflict"))
        f = [x for x in g["flags"] if x["kind"] == "flow_conflict"]
        self.assertEqual(len(f), 1)
        self.assertTrue("arrow" in f[0]["why"] and "connector" in f[0]["why"])

    def _split(self):
        return _g([("A", "port", {}), ("T", "tee", {}), ("B", "port", {})],
                  [("A", "T", "t1", [[0, 0], [100, 0]]), ("T", "B", "t1", [[100, 0], [200, 0]])])

    def test_vision_assignment_orients_both_pieces(self):
        for d, want in (("a_to_b", "forward"), ("b_to_a", "reverse")):
            f = _flow(infer_flow(self._split(), {}, assign={"assignments": [{"id": "t1", "direction": d, "evidence": "other"}]}))
            self.assertEqual(f, {"E1": (want, "vision"), "E2": (want, "vision")})
        g = infer_flow(self._split(), {}, assign={"assignments": [{"id": "t1", "direction": "unknown", "why": "occluded"}]})
        self.assertEqual({e["flow"] for e in g["edges"]}, {"unknown"})
        self.assertTrue(all("occluded" in f["why"] for f in g["flags"] if f["kind"] == "flow_unknown"))

    def test_vision_contradicting_arrow_is_conflict(self):
        s6 = {"associations": {"flow_arrows": {"accepted": [_arrow("t1", [50, 0], [1, 0])]}}}  # arrow: A -> B
        g = infer_flow(self._split(), s6, assign={"assignments": [{"id": "t1", "direction": "b_to_a", "why": "check valve"}]})
        self.assertEqual({e["flow_source"] for e in g["edges"]}, {"conflict"})
        w = [f["why"] for f in g["flags"] if f["kind"] == "flow_conflict"][0]
        self.assertTrue("arrow" in w and "vision" in w)

    def test_raster_arrow_tip_up(self):
        img = np.full((40, 24, 3), 255, np.uint8)
        img[:, 11:13] = 0                      # shaft
        for r in range(8, 32):                 # triangle widening downward: tip at the top
            half = (r - 8) // 2 + 1
            img[r, 12 - half:12 + half] = 0
        box = {"x_min": 0, "y_min": 0, "x_max": 24, "y_max": 40}
        self.assertEqual(read_arrow_vector(img, box), [0.0, -1.0])
        self.assertEqual(read_arrow_vector(img[::-1], box), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
