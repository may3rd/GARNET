import unittest

from garnet.graph.link_sheets import link_sheets

A, B = "14780-8120-25-25-0002A", "14780-8120-25-25-0003"


def sheet(stem, other_ref, spec, hops=1):
    """spec: list of (name, line_number, direction). Connector -> [mid ->] pipe -> end."""
    g = {"sheets": [stem], "nodes": [], "edges": [], "line_list": [], "equipment_connectivity": [], "flags": []}
    s = stem[-4:]
    for name, ln, d in spec:
        c, t = f"{s}/{name}", f"{s}/{name}t"
        g["nodes"] += [{"id": c, "kind": "page_connection", "sheet": stem, "text": other_ref, "direction": d},
                       {"id": t, "kind": "end", "sheet": stem}]
        if hops == 2:
            m = f"{s}/{name}m"
            g["nodes"].append({"id": m, "kind": "port", "sheet": stem})
            g["edges"] += [{"id": f"{s}/{name}e0", "kind": "pipe", "u": c, "v": m},
                           {"id": f"{s}/{name}e1", "kind": "pipe", "u": m, "v": t, "line_number": ln}]
        elif ln:
            g["edges"].append({"id": f"{s}/{name}e", "kind": "pipe", "u": c, "v": t, "line_number": ln})
    return g


def ln(size, n):
    return f'{size}"-CUL-25-{n}-B1A2-NI'


class T(unittest.TestCase):
    def test_size_change(self):
        m = link_sheets([sheet(A, "25-0003", [("c1", ln(4, "002007"), "to")]),
                         sheet(B, "25-0002A", [("c1", ln(3, "002007"), "from")])])
        self.assertEqual(len(m["cross_sheet"]["links"]), 1)
        ev = m["cross_sheet"]["links"][0]["evidence"]
        self.assertEqual((ev["reference_consistent"], ev["direction_consistent"]), (True, True))

    def test_cartesian(self):
        spec = [(f"c{i}", ln(4, f"00200{i}"), "") for i in (1, 2, 3)]
        m = link_sheets([sheet(A, "25-0003", spec), sheet(B, "25-0002A", spec)])
        links = m["cross_sheet"]["links"]
        self.assertEqual(len(links), 3)
        for e in links:
            self.assertEqual(e["u"][-2:], e["v"][-2:])

    def test_tie(self):
        m = link_sheets([sheet(A, "25-0003", [("c1", ln(4, "002007"), "")]),
                         sheet(B, "25-0002A", [("c1", ln(4, "002007"), ""), ("c2", ln(4, "002007"), "")])])
        self.assertEqual(m["cross_sheet"]["links"], [])
        self.assertEqual(sum(f["kind"] == "ambiguous_cross_sheet" for f in m["flags"]), 1)

    def test_hops_and_no_core(self):
        a = sheet(A, "25-0003", [("c1", ln(4, "002007"), "to")], hops=2)
        b = sheet(B, "25-0002A", [("c1", ln(4, "002007"), "from"), ("c2", "", "")])
        m = link_sheets([a, b])
        self.assertEqual(len(m["cross_sheet"]["links"]), 1)
        self.assertEqual([u["id"] for u in m["cross_sheet"]["unmatched"]], ["0003/c2"])
        self.assertTrue(any(f["kind"] == "connector_without_line" for f in m["flags"]))


if __name__ == "__main__":
    unittest.main()
