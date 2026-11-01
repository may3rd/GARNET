"""Give every pipe edge a flow direction (forward = u->v, reverse = v->u, unknown).

Evidence, in order: arrows, off-page connector direction, pump nozzles; then propagation along a
trace and through tees by flow conservation. Two sources disagreeing -> `conflict`, never a silent pick.
All evidence is kept per Stage 6 trace: the pieces of a split trace are oriented along the trace, so they
share one sign.
"""
from __future__ import annotations

import math
from collections import defaultdict

DOT_MIN = 0.35        # |cos| between arrow vector and pipe tangent below which the arrow is ignored
IMBALANCE_MIN = 0.30  # raster head reader: mass imbalance between the two halves of the arrow box


def read_arrow_vector(image, bbox):
    """Unit axis vector (x right, y down) an arrowhead points along, or None.

    Stage 6 arrows are filled triangles with a 1-2 px shaft through them (and sometimes a crossing pipe in the
    crop). Take rows along the box's long axis, find the longest run of rows wider than the shaft (min row width + 3):
    that is the head; the tip is the end of the run with the narrower row, needing a 2x width ratio.
    The vendored flow_direction._raster_arrow uses the whole-crop centroid instead, which scores 0.00-0.18
    against its 0.15 threshold on these ~22x34 px arrows, so it resolves almost none.
    # ponytail: axis-aligned filled heads only; diagonal or hollow heads stay unresolved (None).
    """
    if image is None or not bbox:
        return None
    x0, y0, x1, y1 = (int(round(bbox[k])) for k in ("x_min", "y_min", "x_max", "y_max"))
    crop = image[max(0, y0):y1, max(0, x0):x1]
    if crop.size == 0:
        return None
    dark = (crop.mean(axis=2) if crop.ndim == 3 else crop) < 128
    vertical = dark.shape[0] >= dark.shape[1]
    w = (dark.sum(axis=1) if vertical else dark.sum(axis=0)).tolist()
    shaft = min((x for x in w if x), default=0)  # thick pipes have a 6 px shaft
    best, i = (0, 0), 0
    while i < len(w):  # longest run of head-wide rows
        if w[i] >= shaft + 3:
            j = i
            while j + 1 < len(w) and w[j + 1] >= shaft + 3:
                j += 1
            best = max(best, (j - i + 1, i))
            i = j + 1
        else:
            i += 1
    n, start = best
    if n < 8:
        return None
    first, last = w[start] - shaft, w[start + n - 1] - shaft
    if min(first, last) * 2 > max(first, last):
        return None
    sgn = -1.0 if first < last else 1.0  # narrow first row => tip at the start (up / left)
    return [0.0, sgn] if vertical else [sgn, 0.0]


def _tangent(poly, xy):
    best = (1e18, None)
    for p, q in zip(poly, poly[1:]):
        dx, dy = q[0] - p[0], q[1] - p[1]
        seg = dx * dx + dy * dy
        if seg == 0:
            continue
        f = max(0.0, min(1.0, ((xy[0] - p[0]) * dx + (xy[1] - p[1]) * dy) / seg))
        d = math.dist(xy, (p[0] + dx * f, p[1] + dy * f))
        if d < best[0]:
            n = math.sqrt(seg)
            best = (d, (dx / n, dy / n))
    return best


def infer_flow(graph, stage6, objects=None, image=None, assign=None):
    """Set flow / flow_source / flow_evidence on every pipe edge; add flow_conflict / flow_unknown flags.

    `assign` is the parsed flow_assign.json (vision agent): per Stage 6 trace a_to_b | b_to_a | unknown.
    It is one more source, applied before propagation, so it can feed tee conservation or conflict.
    """
    nodes = {n["id"]: n for n in graph["nodes"]}
    pipes = [e for e in graph["edges"] if e["kind"] == "pipe"]
    by_trace = defaultdict(list)
    for e in pipes:
        by_trace[e["trace_ids"][0]].append(e)
    ev = defaultdict(list)  # trace id -> [(sign along trace, source, text, edge id)]

    def add(e, sign, src, text):
        ev[e["trace_ids"][0]].append((sign, src, text, e["id"]))

    # a. arrows
    for a in ((stage6.get("associations") or {}).get("flow_arrows") or {}).get("accepted", []):
        pieces = by_trace.get(a.get("trace_id"))
        if not pieces:
            continue
        # ours first: Stage 6's own raster vector picked the wrong axis on 4 of 4 arrows it resolved on 0007
        vec = read_arrow_vector(image, a.get("bbox")) or (a.get("flow_direction_evidence") or {}).get("vector")
        if not vec:
            continue
        xy = a.get("projected_xy") or [(a["bbox"]["x_min"] + a["bbox"]["x_max"]) / 2, (a["bbox"]["y_min"] + a["bbox"]["y_max"]) / 2]
        d, (e, tan) = min(((_tangent(e["polyline"], xy)[0], (e, _tangent(e["polyline"], xy)[1])) for e in pieces),
                          key=lambda r: r[0])
        if tan is None:
            continue
        dot = vec[0] * tan[0] + vec[1] * tan[1]
        if abs(dot) >= DOT_MIN:
            add(e, 1 if dot > 0 else -1, "arrow", f"arrow {a['id']} points {'along' if dot > 0 else 'against'} {e['id']}")

    # b. off-page connectors: "to" = sink (flow into it), "from" = source
    for e in pipes:
        for end, nid in (("u", e["u"]), ("v", e["v"])):
            n = nodes[nid]
            d = n.get("direction")
            if n["kind"] in ("page_connection", "utility_connection") and d in ("to", "from"):
                into = d == "to"
                add(e, 1 if (end == "v") == into else -1, "connector", f"{n['id']} direction={d}")

    # c. pumps: [ASSUMPTION] top port (smallest y) is the discharge, the others are suction
    # ponytail: holds for vertical-nozzle pumps drawn like these sheets; a side-suction/top-discharge mix-up
    # or horizontal pump flips it, and a pump with one port gets only a discharge guess.
    ports = defaultdict(list)
    for n in graph["nodes"]:
        if n["kind"] == "port" and "pump" in (nodes.get(n.get("equipment"), {}).get("type") or "").lower():
            ports[n["equipment"]].append(n)
    for eq, ps in ports.items():
        ps.sort(key=lambda n: (n["xy_px"][1], n["xy_px"][0]))
        for i, p in enumerate(ps):
            out = i == 0
            for e in pipes:
                for end, nid in (("u", e["u"]), ("v", e["v"])):
                    if nid == p["id"]:
                        add(e, 1 if (end == "u") == out else -1, "pump",
                            f"{p['id']} is {'discharge (top)' if out else 'suction'} of {nodes[eq].get('tag')}")

    # c2. vision: A is the trace's start, every piece is oriented along the trace, so a_to_b = forward on all pieces
    unknown_why = {}
    for a in (assign or {}).get("assignments", []):
        pieces = by_trace.get(a.get("id"))
        if not pieces:
            continue
        if a.get("direction") in ("a_to_b", "b_to_a"):
            for e in pieces:
                add(e, 1 if a["direction"] == "a_to_b" else -1, "vision",
                    f"vision {a['direction']} via {a.get('evidence', 'other')}: {a.get('why', '')}")
        else:
            unknown_why[a["id"]] = a.get("why", "")

    # d. propagation to fixpoint (trace sharing is built in: evidence is keyed by trace)
    def state(tid):
        signs = {s for s, *_ in ev.get(tid, [])}
        return "conflict" if len(signs) > 1 else (next(iter(signs)) if signs else 0)

    changed = True
    while changed:
        changed = False
        for n in graph["nodes"]:
            if n["kind"] != "tee":
                continue  # `end` and every other node kind never decide anything
            inc = [e for e in pipes if n["id"] in (e["u"], e["v"])]
            known, unk = [], []
            for e in inc:
                st = state(e["trace_ids"][0])
                (known if st in (1, -1) else unk).append((e, st))
            if len(unk) != 1 or not known or unk[0][1] == "conflict":
                continue
            # flow into the tee: +1 on an edge ending at it, -1 on an edge starting at it
            ins = {(st == 1) == (e["v"] == n["id"]) for e, st in known}
            if len(ins) != 1:
                continue
            e = unk[0][0]
            want_in = not ins.pop()  # known all in -> unknown goes out
            add(e, 1 if (e["v"] == n["id"]) == want_in else -1, "propagated",
                f"tee {n['id']} conservation: known edges all {'out' if want_in else 'in'}")
            changed = True

    for tid, es in by_trace.items():
        st = state(tid)
        if st == "conflict":
            seen = "; ".join(f"{src}: {'forward' if s > 0 else 'reverse'} ({txt})" for s, src, txt, _ in ev[tid])
            graph["flags"].append({"kind": "flow_conflict", "where": tid, "why": seen})
        for e in es:
            if st == "conflict":
                e.update(flow="unknown", flow_source="conflict", flow_evidence=seen)
            elif st:
                own = [x for x in ev[tid] if x[3] == e["id"]]
                src, txt = (own[0][1], own[0][2]) if own else ("propagated", f"same trace {tid}")
                e.update(flow="forward" if st > 0 else "reverse", flow_source=src, flow_evidence=txt)
            else:
                e.update(flow="unknown", flow_source="none", flow_evidence="")
                why = f"no flow evidence for {e['u']} -> {e['v']}"
                if tid in unknown_why:
                    why += f"; vision could not tell: {unknown_why[tid]}"
                graph["flags"].append({"kind": "flow_unknown", "where": e["id"], "why": why})
    return graph
