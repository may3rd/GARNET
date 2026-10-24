import copy, unittest

from garnet.eval.score import score

S = "0002A"
G = {"sheet": "14780-8120-25-25-0002A", "verified_by": "t", "verified_on": "2026-01-01",
     "connections": [{"a": "V-1/N2", "b": "P-1/N1", "line_core": "25-000001"},
                     {"a": "V-1/N1", "b": "PC/25-0003", "line_core": "25-000002"}],
     "line_numbers": ['4"-CUL-25-000001-B1A2-NI', '3"-CUL-25-000002-B1A2-NI'],
     "inline": [{"line_core": "25-000001", "class": "valve", "count": 2}],
     "cross_sheet": [], "not_connected": [{"a": "P-1/N1", "b": "PC/25-0003", "why": "separate"}]}


def nd(i, kind, **k):
    return {"id": f"{S}/{i}", "kind": kind, **k}


def pe(i, u, v, ln, core, inl=()):
    return {"id": i, "kind": "pipe", "u": f"{S}/{u}", "v": f"{S}/{v}", "line_number": ln,
            "line_core": core, "inline_objects": list(inl), "flow": "forward"}


L1, L2 = '4"-CUL-25-000001-B1A2-NI', '3"-CUL-25-000002-B1A2-NI'
GRAPH = {"nodes": [nd("V-1", "equipment"), nd("V-1/N1", "port"), nd("V-1/N2", "port"), nd("P-1/N1", "port"),
                   nd("T1", "tee"), nd("PC/25-0003@1", "page_connection"), nd("PC/25-0003@2", "page_connection")],
         "edges": [{"id": "h1", "kind": "has_port", "u": f"{S}/V-1", "v": f"{S}/V-1/N1"},
                   {"id": "h2", "kind": "has_port", "u": f"{S}/V-1", "v": f"{S}/V-1/N2"},
                   pe("e1", "V-1/N2", "T1", L1, "25-000001", ["valve#1"]),
                   pe("e2", "T1", "P-1/N1", L1, "25-000001", ["valve#2"]),
                   pe("e3", "V-1/N1", "PC/25-0003@2", L2, "25-000002")]}


class T(unittest.TestCase):
    def test_perfect(self):
        r = score(GRAPH, [G])
        self.assertTrue(r["pass"], r)
        for m in r["metrics"].values():
            self.assertIn(m["value"], (1.0, None, 0))
        self.assertEqual(r["metrics"]["cross_sheet"]["value"], None)  # connector @2 matched golden PC/25-0003

    def test_mutation_fails(self):
        g = copy.deepcopy(GRAPH)
        g["edges"] = [e for e in g["edges"] if e["id"] not in ("e1", "e2")]
        g["edges"].append(pe("bad", "P-1/N1", "PC/25-0003@1", L2, "25-000002"))
        r = score(g, [G])
        self.assertFalse(r["pass"])
        self.assertLess(r["metrics"]["connection_recall"]["value"], 1)
        self.assertEqual(r["metrics"]["false_joins"]["value"], 1)
        self.assertEqual(len(r["false_joins"]), 1)

    def test_equipment_path_not_connected(self):
        gold = {**G, "connections": [{"a": "V-1/N1", "b": "V-1/N2", "line_core": "25-000001"}],
                "not_connected": [], "line_numbers": [], "inline": []}
        g = copy.deepcopy(GRAPH)
        g["edges"] = g["edges"][:2]  # only has_port edges
        r = score(g, [gold])
        self.assertEqual(r["metrics"]["connection_recall"]["value"], 0.0)
        self.assertFalse(r["pass"])

    def test_missing_terminal(self):
        gold = {**G, "connections": G["connections"] + [{"a": "V-9/N1", "b": "P-1/N1", "line_core": "25-000009"}]}
        r = score(GRAPH, [gold])
        self.assertEqual([m["terminal"] for m in r["missing_terminals"]], [f"{S}/V-9/N1"])
        self.assertLess(r["metrics"]["connection_recall"]["value"], 1)

    def _flow_case(self, flow):
        def e(i, u, v):
            return {**pe(i, u, v, L1, "25-000001"), "flow": flow}
        g = {"nodes": [nd("P-1/N2", "port"), nd("P-2/N2", "port"), nd("T1", "tee"), nd("O", "page_connection")],
             "edges": [e("a", "P-1/N2", "T1"), e("b", "P-2/N2", "T1"), e("c", "T1", "O")]}
        gold = {**G, "line_numbers": [], "inline": [],
                "connections": [{"a": "P-1/N2", "b": "O", "line_core": "25-000001"},
                                {"a": "O", "b": "P-1/N2", "line_core": "25-000001"},
                                {"a": "P-1/N2", "b": "P-2/N2", "line_core": "25-000001"}],
                "not_connected": [{"a": "P-1/N2", "b": "P-2/N2", "why": "both discharge to header"}]}
        return score(g, [gold]), g

    def test_directed_flow(self):
        r, _ = self._flow_case("forward")
        self.assertEqual(r["flow_known_share"], 1.0)
        self.assertEqual(r["metrics"]["false_joins"]["value"], 0)
        by = {m["golden"]: m for m in r["misses"] if m["metric"] == "connection_recall"}
        self.assertNotIn(f"{S}/P-1/N2 -> {S}/O", by)                       # p1 -> O found
        self.assertEqual(by[f"{S}/O -> {S}/P-1/N2"]["found_undirected"], True)   # O -> p1 not found
        self.assertEqual(by[f"{S}/O -> {S}/P-1/N2"]["found_directed"], False)
        self.assertIn(f"{S}/P-1/N2 -> {S}/P-2/N2", by)                     # p1 -> q1 not found

    def test_unknown_flow_gives_false_join(self):
        r, _ = self._flow_case("unknown")
        self.assertEqual(r["flow_known_share"], 0.0)
        self.assertEqual(r["metrics"]["false_joins"]["value"], 1)
        self.assertFalse(r["pass"])


if __name__ == "__main__":
    unittest.main()
