"""Merge per-sheet Contract D graphs; join page connectors across sheets by line core."""
import argparse
import json
import re
from collections import defaultdict

from garnet.graph.contracts import SCHEMA, line_core, short_sheet

_RANK = {True: 2, None: 1, False: 0}


def _core_of(e):
    return e.get("line_core") or line_core(e.get("line_number", ""))


def _connector_core(nid, adj, nodes):
    """Core from incident pipe edges, else walk <=3 edge hops through non tee/end nodes."""
    seen, frontier = {nid}, [nid]
    for _ in range(3):
        nxt = []
        for n in frontier:
            for e in adj[n]:
                if _core_of(e):
                    return _core_of(e)
                o = e["v"] if e["u"] == n else e["u"]
                if o not in seen and nodes.get(o, {}).get("kind") not in ("tee", "end"):
                    seen.add(o)
                    nxt.append(o)
        frontier = nxt
    return ""


def _ref(text, stem):
    """(consistent, note): connector text names the other sheet's stem by trailing tokens."""
    t = [x for x in re.split(r"-", text or "") if x]
    s = stem.split("-")
    if not t or len(t) > len(s):
        return False, ""
    if t == s[-len(t):]:
        return True, ""
    strip = lambda ts: ts[:-1] + [re.sub(r"[A-Za-z]+$", "", ts[-1])]
    if strip(t) == strip(s[-len(t):]):
        return True, "letter_suffix_mismatch"
    return False, ""


def _dir(a, b):
    a, b = a.get("direction", ""), b.get("direction", "")
    if not a or not b:
        return None
    return {a, b} == {"to", "from"}


def link_sheets(graphs):
    out = {"schema": SCHEMA, "sheets": [], "nodes": [], "edges": [], "line_list": [],
           "equipment_connectivity": [], "flags": []}
    for g in graphs:
        for k in ("sheets", "nodes", "edges", "line_list", "equipment_connectivity", "flags"):
            out[k] += g.get(k, [])
    for k in ("nodes", "edges"):
        ids = [x["id"] for x in out[k]]
        assert len(ids) == len(set(ids)), f"duplicate {k} ids"
    nodes = {n["id"]: n for n in out["nodes"]}
    adj = defaultdict(list)
    for e in out["edges"]:
        if e.get("kind") == "pipe":
            adj[e["u"]].append(e)
            adj[e["v"]].append(e)

    stems = {short_sheet(s): s for s in out["sheets"]}
    stem_of = lambda n: stems.get(short_sheet(n.get("sheet", "")), n.get("sheet", ""))
    flags, unmatched = out["flags"], []
    conns, by_core = [], defaultdict(list)  # ponytail: O(n^2) per core group, fine for a few sheets
    for n in out["nodes"]:
        if n.get("kind") != "page_connection":
            continue
        core = _connector_core(n["id"], adj, nodes)
        if not core:
            flags.append({"kind": "connector_without_line", "where": n["id"],
                          "why": "no line core within 3 hops"})
            unmatched.append({"id": n["id"], "line_core": "", "why": "connector_without_line"})
            continue
        conns.append(n)
        by_core[core].append(n)
    core_of = {n["id"]: c for c, ns in by_core.items() for n in ns}

    def ev(a, b):
        r, note = _ref(a.get("text", ""), stem_of(b))
        r2, _ = _ref(b.get("text", ""), stem_of(a))
        d = _dir(a, b)
        return r and r2, d, note

    best, why = {}, {}
    for a in conns:
        cands = [b for b in by_core[core_of[a["id"]]] if stem_of(b) != stem_of(a)]
        if not cands:
            why[a["id"]] = "no_counterpart"
            continue
        sc = {b["id"]: (_RANK[ev(a, b)[0]], _RANK[ev(a, b)[1]]) for b in cands}
        top = max(sc.values())
        win = [b for b in cands if sc[b["id"]] == top]
        if len(win) > 1:
            why[a["id"]] = "ambiguous"
            flags.append({"kind": "ambiguous_cross_sheet", "where": a["id"],
                          "why": f"core {core_of[a['id']]} candidates: " + ", ".join(b["id"] for b in win)})
        else:
            best[a["id"]] = win[0]
    # ponytail: ref consistency is checked on both ends' texts at once (and), weaker than per-end reporting

    links, done = [], set()
    for a in conns:
        b = best.get(a["id"])
        if b and best.get(b["id"], {}).get("id") == a["id"] and a["id"] not in done:
            done |= {a["id"], b["id"]}
            r, d, note = ev(a, b)
            e = {"id": f"X/{len(links) + 1}", "kind": "cross_sheet_continues", "u": a["id"], "v": b["id"],
                 "line_core": core_of[a["id"]], "evidence": {"reference_consistent": r, "direction_consistent": d}}
            if a.get("text"):
                e["line_number"] = a["text"]
            if note:
                e["evidence"]["reference_note"] = note
            links.append(e)
    for a in conns:
        if a["id"] not in done:
            w = why.get(a["id"], "not_mutual")
            unmatched.append({"id": a["id"], "line_core": core_of[a["id"]], "why": w})
            if w != "ambiguous":
                flags.append({"kind": "unmatched_connector", "where": a["id"], "why": w})
    out["edges"] += links
    out["cross_sheet"] = {"links": links, "unmatched": unmatched}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gs = [json.load(open(p)) for p in a.graphs]
    m = link_sheets(gs)
    mk = lambda v: {True: "✓", False: "✗", None: "?"}[v]
    for e in m["cross_sheet"]["links"]:
        ev = e["evidence"]
        print(f"{e['u']} <-> {e['v']} core={e['line_core']} ref={mk(ev['reference_consistent'])} "
              f"dir={mk(ev['direction_consistent'])}")
    print(f"{len(m['cross_sheet']['links'])} links, {len(m['cross_sheet']['unmatched'])} unmatched, "
          f"{sum(f['kind'] == 'ambiguous_cross_sheet' for f in m['flags'])} ambiguous")
    json.dump(m, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
