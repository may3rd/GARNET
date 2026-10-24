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
            "line_core": core, "inline_objects": list(inl)}


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


if __name__ == "__main__":
    unittest.main()
