#!/usr/bin/env python3
"""P&ID pipe tracer: skeleton-graph core, terminal attachment, labelling.

  python trace_pid2.py --image s.jpg --detection d.json --equipment eq.json \
      [--exclude ex.json] [--text t.json] --sheet 25-0007 --out out/

Thins the pipe mask to a 1px skeleton and walks its topology directly
(skelgraph.py) instead of directed pixel-stepping: every skeleton pixel is
either a node (endpoint/branch) or on exactly one edge. Gaps are bridged
collinearly, false 4-way crossings are split into pass-through lines,
terminals are snapped onto the graph, and degree-2 chains are contracted
into whole runs. Emits routes, a graph, an overlay, and a coverage figure
(mask pixels explained by some edge).
"""
import argparse, json, math, os, time
import cv2, numpy as np
import skelgraph as sg

TERMINAL_CLASSES = {"page connection": "PC", "utility connection": "UC",
                    "sampling point": "SP", "pressure relief valve": "PSV",
                    "connection": "OS"}      # OSBL battery-limit line into the sheet
TEXT_CLASSES = {"line number", "instrument tag", "instrument dcs"}
INLINE_CLASSES = {"gate valve", "globe valve", "check valve", "ball valve",
                  "butterfly valve", "control valve", "reducer", "strainer",
                  "spectacle blind"}


# --- assignment helpers ------------------------------------------------------
def segments(poly):
    return [(a, b) for a, b in zip(poly, poly[1:]) if a != b]


def assign_inline(edges, inline, max_dist=40):
    """An inline symbol (valve, reducer, ...) sits ON the line it belongs to
    -- nearest point on any edge's (simplified) polyline, not the parallel-
    corridor test assign_labels uses for line-number text beside the line."""
    for e in edges:
        e["inline_objects"] = []
    for o in inline:
        x1, y1, x2, y2 = o["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best, bd = None, max_dist
        for e in edges:
            for a, b in segments(e["polyline"]):
                ax, ay = a; bx, by = b
                dx, dy = bx - ax, by - ay
                L2 = dx * dx + dy * dy
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - ax) * dx + (cy - ay) * dy) / L2))
                px, py = ax + t * dx, ay + t * dy
                d = math.hypot(cx - px, cy - py)
                if d < bd:
                    bd, best = d, e
        if best is not None:
            best["inline_objects"].append(o["id"])


def assign_labels(runs, labels, max_perp=90):
    """A line number belongs to the run it is drawn ALONGSIDE and PARALLEL to.

    Nearest-centroid swaps labels between parallel runs (two PSV outlets, a bank
    of headers). Orientation first, then a narrow perpendicular corridor with
    along-axis overlap, is what the draughtsman actually meant.
    """
    for r in runs:
        r["line_numbers"] = []
    for L in labels:
        x1, y1, x2, y2 = L["box"]
        horiz = (x2 - x1) >= (y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best, bd = None, max_perp
        for r in runs:
            for a, b in segments(r["polyline"]):
                if horiz != (abs(b[0] - a[0]) > abs(b[1] - a[1])):
                    continue
                if horiz:
                    lo, hi = sorted((a[0], b[0]))
                    if not (lo - 20 <= cx <= hi + 20):
                        continue
                    perp = abs(cy - a[1])
                else:
                    lo, hi = sorted((a[1], b[1]))
                    if not (lo - 20 <= cy <= hi + 20):
                        continue
                    perp = abs(cx - a[0])
                if perp < bd:
                    bd, best = perp, r
        if best is not None:
            best["line_numbers"].append(L["text"])


def arrow_flow(runs, arrows, gray):
    """Flow from arrowheads only. Walk order is a tracing artefact, never flow."""
    for r in runs:
        r["flow"] = "unknown"
    for A in arrows:
        x1, y1, x2, y2 = A["box"]
        crop = gray[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        ink = (crop < 128).astype(float)
        if ink.sum() == 0:
            continue
        horiz = (x2 - x1) >= (y2 - y1)
        prof = ink.sum(axis=0) if horiz else ink.sum(axis=1)
        half = len(prof) // 2
        lo, hi = prof[:half].sum(), prof[half:].sum()
        if abs(lo - hi) / (lo + hi) < 0.15:
            continue
        sign = 1 if hi < lo else -1                      # tip is the light half
        vec = (sign, 0) if horiz else (0, sign)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        best, bd = None, 45
        for r in runs:
            for a, b in segments(r["polyline"]):
                d = abs(cx - a[0]) + abs(cy - a[1]) if a[0] == b[0] else abs(cy - a[1]) + 0
                d = min(abs(cx - a[0]) + abs(cy - a[1]), abs(cx - b[0]) + abs(cy - b[1]))
                if d < bd:
                    bd, best = d, (r, a, b)
        if best:
            r, a, b = best
            dot = vec[0] * (b[0] - a[0]) + vec[1] * (b[1] - a[1])
            r["flow"] = "forward" if dot > 0 else "reverse" if dot < 0 else r["flow"]


def coverage(mask, runs):
    """Mask pixels explained by some run. The one number that shows a miss."""
    cov = np.zeros_like(mask)
    for r in runs:
        pts = np.array(r["polyline"], np.int32)
        if len(pts) >= 2:
            cv2.polylines(cov, [pts], False, 1, 7)
    tot = int(mask.sum())
    return round(100.0 * int(((cov > 0) & (mask > 0)).sum()) / max(tot, 1), 1)


# --- main --------------------------------------------------------------------
def _run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True); ap.add_argument("--detection", required=True)
    ap.add_argument("--equipment", required=True); ap.add_argument("--exclude")
    ap.add_argument("--text"); ap.add_argument("--sheet", default="")
    ap.add_argument("--out", default="out"); ap.add_argument("--min-run", type=int, default=25)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    det = json.load(open(a.detection)); objs = det["objects"]
    equip = json.load(open(a.equipment))
    exclude = json.load(open(a.exclude)) if a.exclude else []
    text = json.load(open(a.text)) if a.text else {}
    def lab(oid):
        v = text.get(oid)
        return v if isinstance(v, dict) else ({"text": v} if v else {})
    bx = lambda o: (o["Left"], o["Top"], o["Left"] + o["Width"], o["Top"] + o["Height"])
    oid = lambda o: f"{o['Object'].replace(' ', '_')}_{o['ObjectID']}"

    gray = cv2.imread(a.image, cv2.IMREAD_GRAYSCALE)
    H, W = gray.shape
    assert (W, H) == (det["image_width"], det["image_height"]), \
        f"frame mismatch: image {W}x{H} vs detection {det['image_width']}x{det['image_height']}"

    ink = (cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] > 0).astype(np.uint8)
    runs_k = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, 41), np.uint8)) | \
             cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((41, 1), np.uint8))
    for x1, y1, x2, y2 in exclude:
        ink[max(0, y1):y2, max(0, x1):x2] = 0; runs_k[max(0, y1):y2, max(0, x1):x2] = 0
    for o in objs:
        if o["Object"] in TEXT_CLASSES:
            x1, y1, x2, y2 = bx(o); ink[max(0, y1 - 4):y2 + 4, max(0, x1 - 4):x2 + 4] = 0
    ink |= runs_k
    for e in equip:
        x1, y1, x2, y2 = e["bbox"]; ink[y1 + 2:y2 - 2, x1 + 2:x2 - 2] = 0
    # Fragmentation is fine now: the walker jumps gaps, so min_run only sheds specks.
    n, cc, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.array([i > 0 and max(st[i, 2], st[i, 3]) >= a.min_run for i in range(n)])
    mask = keep[cc].astype(np.uint8)

    terms = []
    for o in objs:
        if o["Object"] in TERMINAL_CLASSES:
            k = oid(o)
            terms.append({"id": f"{TERMINAL_CLASSES[o['Object']]}-{o['ObjectID']}",
                          "kind": o["Object"], "box": bx(o), "object_id": k, **lab(k)})
    for e in equip:
        for i, (px, py) in enumerate(e["ports"], 1):
            terms.append({"id": f"{e['tag']}:p{i}", "kind": "equipment port",
                          "equipment": e["tag"], "box": (px - 10, py - 10, px + 10, py + 10)})

    inline = [{"id": f"{o['Object'].replace(' ','_')}_{o['ObjectID']}", "class": o["Object"],
               "box": bx(o)} for o in objs if o["Object"] in INLINE_CLASSES]
    labels = [{"box": bx(o), "text": lab(f"line_number_{o['ObjectID']}").get("text")
               or f"line_number_{o['ObjectID']}"} for o in objs if o["Object"] == "line number"]
    arrows = [{"box": bx(o)} for o in objs if o["Object"] == "arrow"]

    # what can legitimately interrupt a line: erased text, symbols, equipment bodies
    gapmask = np.zeros_like(mask)
    for o in objs:
        if o["Object"] in TEXT_CLASSES | INLINE_CLASSES | {"arrow", "node", "instrument logic"}:
            x1, y1, x2, y2 = bx(o); gapmask[max(0, y1-6):y2+6, max(0, x1-6):x2+6] = 1
    for e in equip:
        x1, y1, x2, y2 = e["bbox"]; gapmask[max(0, y1-6):y2+6, max(0, x1-6):x2+6] = 1
    gapmask |= cv2.dilate(mask, np.ones((9, 9), np.uint8))      # crossing lines

    _t0 = time.time()
    nodes, edges, n_bridges = sg.build_graph(mask, gapmask)
    n_raw_edges = len(edges)
    sg.classify_nodes(nodes, edges)
    term_ids = {t["id"] for t in terms}
    attached, not_attached = sg.attach_terminals(nodes, edges, terms)
    edges = sg.contract_degree2(nodes, edges, term_ids)
    for e in edges:
        e["length_px"] = int(round(e["length_px"]))

    # B. straight-line segmentation: which physical stretch of pipe a line
    # number labels. NOT connectivity -- a tee stops a segment even though
    # the branch is still connected to everything the header reaches.
    segs = sg.extract_lines(nodes, edges, term_ids)
    for r in segs:
        r["length_px"] = int(round(r["length_px"]))
    assign_labels(segs, labels)
    assign_inline(segs, inline)
    arrow_flow(segs, arrows, gray)
    cov = coverage(mask, segs)

    # A. connectivity = graph reachability. Prune to the real pipe network
    # (drop self-loops, strip non-terminal dead-end spurs to fixpoint,
    # re-contract), then it's components + terminal-to-terminal shortest
    # paths on what's left.
    net_nodes, net_edges = sg.prune_to_network(nodes, edges, term_ids)
    components, connections = sg.network_connections(net_nodes, net_edges, term_ids)
    print(f"[trace {time.time()-_t0:.1f}s, nodes={len(nodes)} raw_edges={n_raw_edges} "
          f"bridges={n_bridges} contracted_edges={len(edges)} -> segments={len(segs)} "
          f"net_nodes={len(net_nodes)} net_edges={len(net_edges)} connections={len(connections)}]")

    multi_term_nodes = {n for c in components if len(c["terminals"]) >= 2 for n in c["terminals"]}
    unreached = [t["id"] for t in terms if t["id"] not in multi_term_nodes]
    # there is no separate "seeding" step any more (attach_terminals snaps
    # directly onto the graph); terminals_without_seed / signal_only_terminals
    # keep their JSON keys for schema compatibility but now report the same
    # thing: could not be attached to the graph at all.
    no_seed = not_attached

    out = {"sheet": a.sheet, "image": os.path.basename(a.image), "width": W, "height": H,
           "coverage_pct": cov,
           "equipment": [{"tag": e["tag"], "bbox": e["bbox"]} for e in equip],
           "terminals": [{k: v for k, v in t.items()} for t in terms],
           "raw_run_count": n_raw_edges,
           "routes": [{"from": c["from"], "to": c["to"],
                       "from_kind": next((t["kind"] for t in terms if t["id"] == c["from"]), "junction"),
                       "to_kind": next((t["kind"] for t in terms if t["id"] == c["to"]), "junction"),
                       "path_px_len": int(round(c["length_px"])), "from_runs": c["edges"],
                       "jumps": c["bridges"], "hops": c["hops"],
                       "polyline": c["polyline"]} for c in connections],
           "segments": [{"from": r["from"], "to": r["to"], "end_reason": r["end_reason"],
                        "path_px_len": r["length_px"], "line_numbers": r["line_numbers"],
                        "inline_objects": r["inline_objects"], "flow": r["flow"],
                        "jumps": r["bridges"], "edges": r["edges"]} for r in segs],
           "graph": {"nodes": [{"id": nid, "x": nd["xy"][0], "y": nd["xy"][1], "degree": nd["degree"]}
                                for nid, nd in nodes.items()],
                     "edges": [{"id": e["id"], "u": e["u"], "v": e["v"],
                                "length_px": e["length_px"], "bridge": e["bridges"]} for e in edges]},
           "terminals_not_reached": unreached, "terminals_without_seed": no_seed,
           "signal_only_terminals": no_seed}
    json.dump(out, open(f"{a.out}/connectivity.json", "w"), indent=1)

    ov = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    fnt, fs = cv2.FONT_HERSHEY_SIMPLEX, max(0.5, W / 3500); th = max(2, W // 1200)
    rng = np.random.default_rng(7)
    for r in segs:
        col = tuple(int(v) for v in rng.integers(30, 230, 3))
        cv2.polylines(ov, [np.array(r["polyline"], np.int32)], False, col, th * 3)
    for nd in nodes.values():
        if nd["degree"] >= 3:
            x, y = nd["xy"]
            cv2.circle(ov, (int(x), int(y)), th * 3, (255, 0, 255), -1)
    for e in equip:
        x1, y1, x2, y2 = e["bbox"]
        cv2.rectangle(ov, (x1, y1), (x2, y2), (255, 0, 0), th * 2)
        cv2.putText(ov, e["tag"], (x1, max(14, y1 - 10)), fnt, fs, (255, 0, 0), th)
    for t in terms:
        x1, y1, x2, y2 = t["box"]
        c = (0, 140, 255) if t["id"] in unreached else (0, 150, 0)
        cv2.circle(ov, ((x1 + x2) // 2, (y1 + y2) // 2), th * 4, c, -1)
        cv2.putText(ov, f'{t["id"]} {t.get("text","")}'.strip(), (x1, max(14, y1 - 10)), fnt, fs, c, th)
    cv2.imwrite(f"{a.out}/overlay.png", ov)
    cv2.imwrite(f"{a.out}/pipe_mask.png", mask * 255)

    with open(f"{a.out}/connectivity.mmd", "w") as f:
        f.write("flowchart LR\n")
        nid = lambda s: str(s).replace("-", "_").replace(":", "_").replace("@", "_")
        for e in equip:
            f.write(f'  {nid(e["tag"])}["{e["tag"]}"]\n')
        for r in out["routes"]:
            f.write(f'  {nid(r["from"])} --> {nid(r["to"])}\n')

    tt = {t["id"] for t in terms}
    t2t = [r for r in segs if r["from"] in tt and r["to"] in tt]
    print(f"segments={len(segs)} terminal-to-terminal={len(t2t)} coverage={cov}%")
    print(f"terminals={len(terms)} attached={len(attached)} not_reached={unreached}")
    for r in sorted(segs, key=lambda r: -r["path_px_len"])[:26]:
        m = "*" if (r["from"] in tt and r["to"] in tt) else " "
        print(f" {m}{str(r['from']):<17} -> {str(r['to']):<20} len={r['path_px_len']:5} "
              f"jmp={r['jumps']} {r['flow']:8} {','.join(r['line_numbers'])[:46]}")


if __name__ == "__main__":
    _run()
