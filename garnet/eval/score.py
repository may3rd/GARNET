"""Score a Contract D graph against Contract E golden files. See garnet/graph/contracts.py."""
import argparse, json, os, re, sys
from collections import deque

from garnet.graph.contracts import line_core, short_sheet

# (threshold, direction): ">=" higher is better, "<=" lower is better
DEFAULTS = {"connection_recall": 0.90, "connection_precision": 0.95, "false_joins": 0,
            "line_number_text": 0.95, "line_assignment": 0.90, "inline_count": 0.90,
            "cross_sheet": 1.0}
_PASSTHRU = ("tee", "end")
_AT = re.compile(r"@\d+$")


def _key(nid):
    return _AT.sub("", nid)


def score(graph, goldens, thresholds=None):
    th = {**DEFAULTS, **(thresholds or {})}
    nodes = {n["id"]: n for n in graph["nodes"]}
    by_key = {}
    for nid in nodes:
        by_key.setdefault(_key(nid), []).append(nid)
    adj = {}  # reachability graph: pipe + cross_sheet_continues; has_port is deliberately absent
    pipe_edges = [e for e in graph["edges"] if e.get("kind") == "pipe"]
    cross_edges = [e for e in graph["edges"] if e.get("kind") == "cross_sheet_continues"]
    for e in pipe_edges + cross_edges:
        adj.setdefault(e["u"], []).append((e["v"], e))
        adj.setdefault(e["v"], []).append((e["u"], e))
    # ponytail: ids only, goldens carry no pixel positions so no position fallback; add one if goldens gain xy_px

    def bfs(starts):
        """-> {terminal node: (dist, path edges)}; expands only through tee/end nodes."""
        seen = {s: (0, []) for s in starts}
        q = deque(starts)
        res = {}
        while q:
            u = q.popleft()
            if u not in starts and nodes.get(u, {}).get("kind") not in _PASSTHRU:
                res[u] = seen[u]
                continue
            for v, e in adj.get(u, []):
                if v not in seen:
                    seen[v] = (seen[u][0] + 1, seen[u][1] + [e])
                    q.append(v)
        return res

    reach_cache = {}

    def reach(k):  # golden key -> {terminal D node: (dist, path)}, best over all matching nodes
        if k not in reach_cache:
            out = {}
            for s in by_key.get(k, []):
                for n, r in bfs([s]).items():
                    if n not in out or r[0] < out[n][0]:
                        out[n] = r
            reach_cache[k] = out
        return reach_cache[k]

    def keys_of(res):
        return sorted({_key(n) for n in res})

    conns, ncs, lns, inls, cross = [], [], [], [], []
    for g in goldens:
        s = short_sheet(g["sheet"])
        q = lambda i: f"{s}/{i}"
        conns += [(q(c["a"]), q(c["b"]), c.get("line_core", "")) for c in g.get("connections", [])]
        ncs += [(q(c["a"]), q(c["b"])) for c in g.get("not_connected", [])]
        lns += g.get("line_numbers", [])
        inls += g.get("inline", [])
        cross += [(c["a"], c["b"], c.get("line_core", "")) for c in g.get("cross_sheet", [])]

    misses, missing, false_joins, extra = [], {}, [], []

    def need(k, why):
        if k not in by_key and k not in missing:
            missing[k] = {"terminal": k, "needed_by": why}
        return k in by_key

    metrics = {}

    def put(key, num, den, val=None):
        v = val if val is not None else (None if den == 0 else num / den)
        t = th[key]
        ok = True if v is None else (v <= t if key == "false_joins" else v >= t)
        metrics[key] = {"value": v, "threshold": t, "pass": ok}

    # connection recall + line assignment
    found = []
    for a, b, core in conns:
        ok_a, ok_b = need(a, f"{a} -- {b}"), need(b, f"{a} -- {b}")
        r = reach(a) if ok_a and ok_b else {}
        hit = [n for n in r if _key(n) == b]
        if hit:
            found.append((a, b, core, min(hit, key=lambda n: r[n][0])))
        else:
            misses.append({"metric": "connection_recall", "golden": f"{a} -- {b}",
                           "d_had": f"{a} reaches {keys_of(r)}" if ok_a and ok_b else "terminal missing"})
    put("connection_recall", len(found), len(conns))
    bad = 0
    for a, b, core, n in found:
        path = reach(a)[n][1]
        if not any((e.get("line_core") or line_core(e.get("line_number", ""))) == core for e in path):
            bad += 1
            misses.append({"metric": "line_assignment", "golden": f"{a} -- {b} core {core}",
                           "d_had": sorted({e.get("line_core") or line_core(e.get("line_number", "")) for e in path})})
    put("line_assignment", len(found) - bad, len(found))

    # precision
    gterms = {k for a, b, _ in conns for k in (a, b)} | {k for a, b in ncs for k in (a, b)}
    pairs = {frozenset(p[:2]) for p in conns} | {frozenset(p[:2]) for p in cross}
    num = den = 0
    for t in sorted(gterms & set(by_key)):
        for k in {_key(n) for n in reach(t)} & gterms:
            if k == t:
                continue
            den += 1
            if frozenset((t, k)) in pairs:
                num += 1
            else:
                misses.append({"metric": "connection_precision", "golden": f"{t} -- {k}",
                               "d_had": "reachable in D but golden does not list a connection"})
    put("connection_precision", num, den)

    # false joins
    for a, b in ncs:
        if a in by_key and b in by_key and any(_key(n) == b for n in reach(a)):
            false_joins.append({"golden": f"{a} !-- {b}", "d_had": "reachable via pipe edges"})
    put("false_joins", 0, 0, val=len(false_joins))

    # line number text
    dl = {e.get("line_number") for e in pipe_edges}
    for ln in lns:
        if ln not in dl:
            misses.append({"metric": "line_number_text", "golden": ln, "d_had": "not among D edge line_numbers"})
    put("line_number_text", sum(ln in dl for ln in lns), len(lns))

    # inline counts
    okc = 0
    for r in inls:
        c = sum(1 for e in pipe_edges
                if (e.get("line_core") or line_core(e.get("line_number", ""))) == r["line_core"]
                for o in e.get("inline_objects", []) if o.startswith(r["class"] + "#"))
        if c == r["count"]:
            okc += 1
        else:
            misses.append({"metric": "inline_count", "golden": f"{r['line_core']} {r['class']} x{r['count']}",
                           "d_had": c})
    put("inline_count", okc, len(inls))

    # cross sheet
    cpairs = {frozenset((_key(e["u"]), _key(e["v"]))) for e in cross_edges}
    gc = {frozenset(p[:2]) for p in cross}
    for p in cross:
        if frozenset(p[:2]) not in cpairs:
            misses.append({"metric": "cross_sheet", "golden": f"{p[0]} -- {p[1]}", "d_had": "no cross_sheet_continues edge"})
    gk = {k for p in cross for k in p[:2]}
    for p in cpairs - gc:
        if p <= gk:
            extra.append(" -- ".join(sorted(p)))
    nf = sum(frozenset(p[:2]) in cpairs for p in cross)
    put("cross_sheet", nf, len(cross) + len(extra))

    return {"pass": all(m["pass"] for m in metrics.values()), "metrics": metrics, "misses": misses,
            "false_joins": false_joins, "missing_terminals": list(missing.values()), "extra_links": extra}


def contact_sheet(result, graph, image_path, out_path):
    import cv2, numpy as np
    img = cv2.imread(image_path)
    if img is None:
        return
    nodes = {_key(n["id"]): n for n in graph["nodes"] if n.get("xy_px")}
    tiles = []
    for label, text in [(m["golden"], m["metric"]) for m in result["misses"]] + \
                       [(f["golden"], "false_join") for f in result["false_joins"]]:
        pts = [nodes[k]["xy_px"] for k in re.split(r" !?-- ", label.split(" core ")[0]) if k in nodes]
        if not pts:
            continue
        cx, cy = (int(sum(p[i] for p in pts) / len(pts)) for i in (0, 1))
        pad = cv2.copyMakeBorder(img, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        t = pad[cy:cy + 400, cx:cx + 400].copy()  # padded by 200 so centre (cx,cy) -> (cx+200,cy+200)
        for p in pts:
            cv2.circle(t, (int(p[0]) - cx + 200, int(p[1]) - cy + 200), 8, (0, 0, 255), 2)
        cv2.putText(t, f"{text}: {label}"[:48], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        tiles.append(t)
    if not tiles:
        return
    tiles = tiles[:12]  # ponytail: first 12 problems only
    cols = min(3, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.full((400, 400, 3), 255, np.uint8))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(out_path, np.vstack(rows))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--golden", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--image")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args(argv)
    try:
        graph = json.load(open(a.graph))
        goldens = [json.load(open(p)) for p in a.golden]
        th = {k: float(v) for k, v in (s.split("=", 1) for s in a.set)}
        assert all(k in DEFAULTS for k in th), f"unknown metric in --set; known: {sorted(DEFAULTS)}"
        res = score(graph, goldens, th)
    except (OSError, ValueError, KeyError, AssertionError) as e:
        print(f"bad input: {e}", file=sys.stderr)
        return 2
    json.dump(res, open(a.out, "w"), indent=2)
    for k, m in res["metrics"].items():
        v = "null" if m["value"] is None else f"{m['value']:.3f}"
        print(f"{k:22s} {v:>7s}  thr {m['threshold']}  {'ok' if m['pass'] else 'FAIL'}")
    print("PASS" if res["pass"] else "FAIL")
    if a.image:
        contact_sheet(res, graph, a.image, os.path.join(os.path.dirname(a.out) or ".", "score_contact.png"))
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
