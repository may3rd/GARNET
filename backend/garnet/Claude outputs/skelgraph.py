#!/usr/bin/env python3
"""Skeleton-graph pipe tracer core.

thin -> classify (endpoint/interior/branch by neighbour count) -> cluster
branch pixels into nodes -> walk interior chains into edges -> simplify ->
bridge gaps -> resolve false 4-way crossings -> attach terminals -> contract
degree-2 chains. The raster already encodes the topology (a skeleton pixel
has 1, 2 or >=3 skeleton neighbours); this walks that structure once instead
of re-deriving it with a directed pixel-stepper.

Edge dict fields: id, u, v, polyline [(x,y),...] (simplified corner points),
length_px (true pixel-path length), dir_u/dir_v (R/L/U/D, direction leaving
that end into the edge -- cached from the raw walk, survives simplification),
bridges (count of gap-bridge edges folded into this one), src (underlying
skelgraph edge ids merged into this one).
"""
import heapq
import math
from collections import Counter, deque
import cv2
import numpy as np
from skimage.morphology import skeletonize

D = {"R": (1, 0), "L": (-1, 0), "D": (0, 1), "U": (0, -1)}
BACK = {"R": "L", "L": "R", "U": "D", "D": "U"}
NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _dir_from_points(pts, max_px=15.0):
    """Snap the direction leaving pts[0] toward pts[-1] to R/L/U/D, using
    only the first ~max_px pixels of arc length (the true local heading)."""
    x0, y0 = pts[0]
    acc, px, py = 0.0, x0, y0
    for x, y in pts[1:]:
        acc += math.hypot(x - px, y - py)
        px, py = x, y
        if acc >= max_px:
            break
    dx, dy = px - x0, py - y0
    if dx == 0 and dy == 0:
        return "R"
    return ("R" if dx > 0 else "L") if abs(dx) >= abs(dy) else ("D" if dy > 0 else "U")


def _set_degrees(nodes, edges):
    deg = Counter()
    for e in edges:
        deg[e["u"]] += 1
        deg[e["v"]] += 1
    for n in nodes:
        nodes[n]["degree"] = deg.get(n, 0)


# --------------------------------------------------------------------------
def build_graph(mask, gapmask=None, simplify_eps=2.0,
                 bridge_max_gap=130, bridge_free_gap=12, bridge_perp_tol=4):
    """mask: uint8 0/1. Returns (nodes, edges) -- steps 1-7 plus bridging (6)."""
    skel = skeletonize(mask > 0).astype(np.uint8)
    H, W = skel.shape
    if gapmask is None:
        gapmask = np.zeros_like(mask)

    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    cnt = cv2.filter2D(skel, cv2.CV_8U, k, borderType=cv2.BORDER_CONSTANT) * skel

    branch_mask = ((cnt >= 3) & (skel > 0)).astype(np.uint8)
    nodes = {}

    # ponytail: the walk below is pure-Python over ~80k skeleton pixels --
    # fine per the brief, but only once every lookup is a dict/set membership
    # test, not a numpy 2D scalar index (that alone was the 30x slowdown).
    node_at = {}                                 # (y, x) -> node idx, node pixels only
    seeds = []                                   # (y, x, idx) pixels to fire walk_from
    # np.where(labels == lbl) per label rescans the whole (17M px) image once
    # per branch cluster -- vectorize instead: one pass to find all branch
    # pixels and gather their labels. Cluster on the branch mask DILATED by
    # 6px, not the raw mask: a thick valve/reducer stroke skeletonizes into
    # several separate branch blobs a few px apart, which otherwise become
    # separate nodes joined by near-duplicate parallel micro-edges. Centroids
    # come from the UNDILATED pixels only, so the node still sits on the
    # real junction, not smeared toward the dilation halo.
    dilated = cv2.dilate(branch_mask, np.ones((13, 13), np.uint8))
    ncomp, dlabels = cv2.connectedComponentsWithStats(dilated, connectivity=8)[:2]
    labels = dlabels * branch_mask               # zero outside real branch pixels
    if ncomp > 1:
        ys, xs = np.where(labels > 0)
        idxs = (labels[ys, xs].astype(np.int64) - 1)
        counts = np.bincount(idxs, minlength=ncomp - 1)
        sumx = np.bincount(idxs, weights=xs.astype(np.float64), minlength=ncomp - 1)
        sumy = np.bincount(idxs, weights=ys.astype(np.float64), minlength=ncomp - 1)
        for idx in range(ncomp - 1):
            if counts[idx] == 0:
                continue
            nodes[f"sk{idx}"] = {"xy": (float(sumx[idx] / counts[idx]), float(sumy[idx] / counts[idx])), "_idx": idx}
        for y, x, idx in zip(ys.tolist(), xs.tolist(), idxs.tolist()):
            node_at[(y, x)] = idx
            seeds.append((y, x, idx))
    ys, xs = np.where((cnt <= 1) & (skel > 0))          # endpoints (1) + isolated specks (0)
    for y, x in zip(ys.tolist(), xs.tolist()):
        idx = len(nodes)
        nid = f"sk{idx}"
        nodes[nid] = {"xy": (float(x), float(y)), "_idx": idx}
        node_at[(y, x)] = idx
        seeds.append((y, x, idx))
    idx2id = {v["_idx"]: k for k, v in nodes.items()}

    interior_pts = set(map(tuple, np.argwhere((cnt == 2) & (skel > 0)).tolist()))
    skel_pts = set(node_at) | interior_pts
    consumed = set()
    edges = []
    direct_seen = set()

    def make_edge(u, v, path):
        length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:]))
        eid = f"g{len(edges)}"
        e = {"id": eid, "u": u, "v": v, "polyline": path,
             "length_px": length, "bridges": 0, "src": [eid],
             "dir_u": _dir_from_points(path), "dir_v": _dir_from_points(path[::-1])}
        edges.append(e)

    def walk_from(y, x, i):
        """Node pixel (y,x) of node index i: chase every unclaimed skeleton
        neighbour into an edge (direct node-node, or a walked interior chain)."""
        for dy, dx in NB8:
            p = (y + dy, x + dx)
            if p not in skel_pts:
                continue
            j = node_at.get(p)
            if j is not None:
                if j == i:
                    continue
                key = frozenset(((y, x), p))
                if key in direct_seen:
                    continue
                direct_seen.add(key)
                make_edge(idx2id[i], idx2id[j], [(x, y), (p[1], p[0])])
                continue
            if p not in interior_pts or p in consumed:
                continue
            path = [(x, y), (p[1], p[0])]
            consumed.add(p)
            prev, cur = (y, x), p
            end_idx = None
            while True:
                cy, cx = cur
                nxt = None
                for ddy, ddx in NB8:
                    q = (cy + ddy, cx + ddx)
                    if q == prev or q not in skel_pts:
                        continue
                    nxt = q
                    break
                if nxt is None:
                    break                       # anomaly: dead end mid-chain
                j2 = node_at.get(nxt)
                if j2 is not None:
                    path.append((nxt[1], nxt[0]))
                    end_idx = j2
                    break
                if nxt in consumed:
                    break                       # anomaly: already claimed
                consumed.add(nxt)
                path.append((nxt[1], nxt[0]))
                prev, cur = cur, nxt
            if end_idx is not None:
                make_edge(idx2id[i], idx2id[end_idx], path)

    for y, x, i in seeds:                      # every member pixel of every node fires --
        walk_from(y, x, i)                     # a branch cluster's arms leave from different pixels

    # ponytail: deduplicate only geometrically-coincident parallel edges.
    # When a cluster has N pixels, the walk fires N times and can create
    # N edges between the same pair of nodes. BUT: on a P&ID, two distinct
    # pipes often connect the same junctions (bypass around a valve, parallel
    # headers, double-line runs). Drop an edge only if an already-kept edge
    # with the same (u,v) pair has coincident geometry:
    # - length within 2% of each other, AND
    # - midpoints within 6 px.
    # Anything else is a real parallel run and must be kept.
    def _polyline_midpoint(poly):
        """Point at half the polyline's arc length."""
        if len(poly) <= 1:
            return poly[0]
        total = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(poly, poly[1:]))
        target, acc = total / 2.0, 0.0
        for a, b in zip(poly, poly[1:]):
            seg_len = math.hypot(b[0]-a[0], b[1]-a[1])
            if acc + seg_len >= target:
                t = (target - acc) / seg_len if seg_len > 0 else 0
                return (a[0] + t*(b[0]-a[0]), a[1] + t*(b[1]-a[1]))
            acc += seg_len
        return poly[-1]

    def _edges_coincident(e1, e2, len_tol=0.02, mid_tol=6):
        """Same (u,v) edges are coincident only if lengths differ by <=2% and midpoints <=6px apart."""
        r = e2["length_px"] / e1["length_px"] if e1["length_px"] > 0 else 1.0
        if r < 1.0:
            r = 1.0 / r
        if r > 1.0 + len_tol:
            return False
        m1, m2 = _polyline_midpoint(e1["polyline"]), _polyline_midpoint(e2["polyline"])
        return math.hypot(m1[0]-m2[0], m1[1]-m2[1]) <= mid_tol

    edges_before_dedup = len(edges)
    seen_pairs = {}
    deduped_edges = []
    genuine_parallel = 0  # count of (u,v) pairs with 2+ distinct edges
    for e in edges:
        pair = (e["u"], e["v"]) if e["u"] <= e["v"] else (e["v"], e["u"])
        if pair not in seen_pairs:
            seen_pairs[pair] = []
            deduped_edges.append(e)
            seen_pairs[pair].append(e)
        else:
            if not any(_edges_coincident(e, kept) for kept in seen_pairs[pair]):
                deduped_edges.append(e)
                seen_pairs[pair].append(e)
                if len(seen_pairs[pair]) == 2:
                    genuine_parallel += 1
    edges[:] = deduped_edges
    edges_after_dedup = len(edges)
    # Store for reporting
    if not hasattr(build_graph, '_dedup_stats'):
        build_graph._dedup_stats = (edges_before_dedup, edges_after_dedup, genuine_parallel)

    # leftover: closed loops touching no node at all -> pick one pixel as a
    # node and let the walk consume the rest of that ring as a self-loop.
    for p0 in list(interior_pts - consumed):
        if p0 in consumed:
            continue                            # already swept up by an earlier p0's walk
        idx = len(nodes)
        nid = f"sk{idx}"
        nodes[nid] = {"xy": (float(p0[1]), float(p0[0])), "_idx": idx}
        idx2id[idx] = nid
        node_at[p0] = idx
        skel_pts.add(p0)
        interior_pts.discard(p0)
        walk_from(p0[0], p0[1], idx)

    # every skeleton pixel is now either a node pixel or a consumed interior
    # pixel -- exactly once. this is the pixel-accounting guarantee the whole
    # design leans on; check it here, always, not just in the test harness.
    n_node_px = len(node_at)
    n_interior_px = len(interior_pts)
    n_consumed_px = len(consumed)
    assert n_consumed_px == n_interior_px, \
        f"pixel accounting: {n_interior_px - n_consumed_px} interior px never consumed"
    assert n_node_px + n_consumed_px == int(skel.sum()), \
        f"pixel accounting: node+edge px ({n_node_px + n_consumed_px}) != skeleton px ({int(skel.sum())})"

    for n in nodes.values():
        n.pop("_idx", None)
        n["xy"] = (n["xy"][0], n["xy"][1])
    _set_degrees(nodes, edges)

    # step 5: simplify (dir_u/dir_v already cached from the raw walk above)
    for e in edges:
        poly = e["polyline"]
        if len(poly) > 2:
            arr = np.array(poly, np.int32).reshape(-1, 1, 2)
            simp = cv2.approxPolyDP(arr, simplify_eps, False).reshape(-1, 2)
            e["polyline"] = [tuple(p) for p in simp.tolist()]

    bridge_count = _bridge_gaps(nodes, edges, gapmask, bridge_max_gap, bridge_free_gap, bridge_perp_tol)
    _set_degrees(nodes, edges)
    return nodes, edges, bridge_count


def _bridge_gaps(nodes, edges, gapmask, max_gap, free_gap, perp_tol):
    """Step 6: connect degree-1 endpoints across an unexplained-but-short (or
    gapmask-explained) collinear gap. Greedy nearest match, symmetric."""
    inc = {}
    for e in edges:
        inc.setdefault(e["u"], []).append((e, "u"))
        inc.setdefault(e["v"], []).append((e, "v"))
    deg1 = sorted(n for n, es in inc.items() if len(es) == 1)
    outgoing = {}
    for n in deg1:
        e, end = inc[n][0]
        outgoing[n] = BACK[e["dir_u"] if end == "u" else e["dir_v"]]

    H, W = gapmask.shape
    used, added = set(), 0
    for a in deg1:
        if a in used:
            continue
        da = outgoing[a]
        ax, ay = nodes[a]["xy"]
        dx, dy = D[da]
        best, bestd = None, None
        for b in deg1:
            if b == a or b in used or outgoing[b] != BACK[da]:
                continue
            bx, by = nodes[b]["xy"]
            if dx:
                forward, perp = (bx - ax) * dx, abs(by - ay)
            else:
                forward, perp = (by - ay) * dy, abs(bx - ax)
            if forward <= 0 or perp > perp_tol or forward > max_gap:
                continue
            if forward > free_gap:
                mx, my = int((ax + bx) / 2), int((ay + by) / 2)
                if not (0 <= my < H and 0 <= mx < W and gapmask[my, mx]):
                    continue
            if best is None or forward < bestd:
                best, bestd = b, forward
        if best is not None:
            used.add(a); used.add(best)
            bx, by = nodes[best]["xy"]
            bid = f"b{added}"
            edges.append({"id": bid, "u": a, "v": best,
                          "polyline": [(int(ax), int(ay)), (int(bx), int(by))],
                          "length_px": bestd, "bridges": 1, "src": [bid],
                          "dir_u": da, "dir_v": outgoing[best]})
            added += 1
    return added


# --------------------------------------------------------------------------
def classify_nodes(nodes, edges):
    """Step 8: split a degree-4 node whose 4 arms are exactly one each of
    R/L/U/D (two collinear opposite pairs) into 2 pass-through edges -- a
    drawn crossing, not a real junction. A degree-4 node that doesn't pair
    (a real cross-tee) is left untouched.

    ponytail: only the clean {R,L,U,D} pattern is split; a rarer
    {R,R,L,L}-style double-parallel crossing is left as one real node --
    add positional pairing (match by lateral offset) if that shows up.

    Fixpoint, one split at a time, re-deriving incidence from the CURRENT
    edge list each round -- two crossings directly adjacent to each other
    (real in this data, not the rare case it looks like) would otherwise
    both get processed off one stale snapshot, each creating a new edge
    that points at the other's about-to-be-removed node id.
    """
    n_split = 0
    changed = True
    while changed:
        changed = False
        inc = {}
        for e in edges:
            inc.setdefault(e["u"], []).append((e, "u"))
            inc.setdefault(e["v"], []).append((e, "v"))
        for n, arms in inc.items():
            if len(arms) != 4:
                continue
            if any(e["u"] == e["v"] for e, _ in arms):
                continue                     # a self-loop spur double-counts as 2 fake arms -- not a crossing
            dirs = [e["dir_u"] if end == "u" else e["dir_v"] for e, end in arms]
            if sorted(dirs) != sorted(D):
                continue
            by_dir = {d: e for (e, _), d in zip(arms, dirs)}
            remove_ids, add_edges = set(), []
            for a, b in (("R", "L"), ("U", "D")):
                e1, e2 = by_dir[a], by_dir[b]
                far1 = e1["v"] if e1["u"] == n else e1["u"]
                far2 = e2["v"] if e2["u"] == n else e2["u"]
                dir1 = e1["dir_v"] if e1["u"] == n else e1["dir_u"]
                dir2 = e2["dir_v"] if e2["u"] == n else e2["dir_u"]
                p1 = list(reversed(e1["polyline"])) if e1["u"] == n else e1["polyline"]
                p2 = e2["polyline"] if e2["u"] == n else list(reversed(e2["polyline"]))
                add_edges.append({"id": f"x{n_split}_{len(add_edges)}", "u": far1, "v": far2,
                                  "polyline": p1 + p2[1:],
                                  "length_px": e1["length_px"] + e2["length_px"],
                                  "dir_u": dir1, "dir_v": dir2,
                                  "bridges": e1.get("bridges", 0) + e2.get("bridges", 0),
                                  "src": e1.get("src", []) + e2.get("src", [])})
                remove_ids.add(id(e1)); remove_ids.add(id(e2))
            edges[:] = [e for e in edges if id(e) not in remove_ids] + add_edges
            nodes.pop(n, None)
            n_split += 1
            changed = True
            break                            # re-derive inc before touching another node
    _set_degrees(nodes, edges)
    return n_split


def _closest_on_seg(px, py, a, b):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return cx, cy, (px - cx) ** 2 + (py - cy) ** 2, t


def attach_terminals(nodes, edges, terminals, snap_node=25, snap_edge=40):
    """Step 10: snap each terminal to the nearest node (renaming it), or
    split the nearest edge at the nearest point on it and insert a new
    terminal node there. Terminal node ids are the terminal's own id."""
    node_ids = list(nodes)
    node_xy = np.array([nodes[n]["xy"] for n in node_ids], float) if node_ids else np.zeros((0, 2))
    attached, not_attached = [], []
    for t in terminals:
        x1, y1, x2, y2 = t["box"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        tid = t["id"]
        if len(node_ids):
            d = np.hypot(node_xy[:, 0] - cx, node_xy[:, 1] - cy)
            j = int(np.argmin(d))
            if d[j] <= snap_node:
                old = node_ids[j]
                if old != tid:
                    nodes[tid] = nodes.pop(old)
                    for e in edges:
                        if e["u"] == old:
                            e["u"] = tid
                        if e["v"] == old:
                            e["v"] = tid
                    node_ids[j] = tid
                attached.append(tid)
                continue

        best = None
        for e in edges:
            poly = e["polyline"]
            for i, (a, b) in enumerate(zip(poly, poly[1:])):
                px, py, d2, tt = _closest_on_seg(cx, cy, a, b)
                if best is None or d2 < best[0]:
                    best = (d2, e, i, (px, py))
        if best is not None and best[0] <= snap_edge ** 2:
            _, e, i, (px, py) = best
            poly = e["polyline"]
            pt = (int(round(px)), int(round(py)))
            head = poly[:i + 1] + [pt]
            tail = [pt] + poly[i + 1:]
            len_head = sum(math.hypot(q[0]-p[0], q[1]-p[1]) for p, q in zip(head, head[1:]))
            len_tail = sum(math.hypot(q[0]-p[0], q[1]-p[1]) for p, q in zip(tail, tail[1:]))
            edges.remove(e)
            edges.append({"id": e["id"] + "a", "u": e["u"], "v": tid, "polyline": head,
                          "length_px": len_head, "dir_u": e["dir_u"],
                          "dir_v": _dir_from_points(list(reversed(head))),
                          "bridges": e.get("bridges", 0), "src": e.get("src", [])})
            edges.append({"id": e["id"] + "b", "u": tid, "v": e["v"], "polyline": tail,
                          "length_px": len_tail, "dir_u": _dir_from_points(tail),
                          "dir_v": e["dir_v"], "bridges": e.get("bridges", 0),
                          "src": e.get("src", [])})
            nodes[tid] = {"xy": pt}
            node_ids.append(tid)
            node_xy = np.vstack([node_xy, [pt]]) if node_xy.size else np.array([pt], float)
            attached.append(tid)
        else:
            not_attached.append(tid)
    _set_degrees(nodes, edges)
    return attached, not_attached


def contract_degree2(nodes, edges, terminal_ids):
    """Step 9: merge the two edges at any non-terminal degree-2 node,
    concatenating polylines, to fixpoint."""
    changed = True
    while changed:
        changed = False
        inc = {}
        for e in edges:
            inc.setdefault(e["u"], []).append(e)
            inc.setdefault(e["v"], []).append(e)
        for node, es in inc.items():
            if node in terminal_ids or len(es) != 2:
                continue
            a, b = es
            if a is b:
                continue
            pa = a["polyline"] if a["v"] == node else list(reversed(a["polyline"]))
            pb = b["polyline"] if b["u"] == node else list(reversed(b["polyline"]))
            ua = a["u"] if a["v"] == node else a["v"]
            vb = b["v"] if b["u"] == node else b["u"]
            if ua == vb:
                continue                       # would collapse to a bare self-loop; leave it
            merged = {"id": a["id"], "u": ua, "v": vb, "polyline": pa + pb[1:],
                      "length_px": a["length_px"] + b["length_px"],
                      "dir_u": a["dir_u"] if a["v"] == node else a["dir_v"],
                      "dir_v": b["dir_v"] if b["u"] == node else b["dir_u"],
                      "bridges": a.get("bridges", 0) + b.get("bridges", 0),
                      "src": a.get("src", []) + b.get("src", [])}
            ida, idb = id(a), id(b)
            edges = [e for e in edges if id(e) not in (ida, idb)] + [merged]
            nodes.pop(node, None)
            changed = True
            break
    edges[:] = edges
    _set_degrees(nodes, edges)
    return edges


# --------------------------------------------------------------------------
def _heading_vec(pts, max_px=15.0):
    """Real (dx, dy) heading leaving pts[0] toward pts[-1], first ~max_px of
    arc length -- unlike _dir_from_points this is NOT snapped to R/L/U/D, so
    diagonal runs get a real angle."""
    x0, y0 = pts[0]
    acc, px, py = 0.0, x0, y0
    for x, y in pts[1:]:
        acc += math.hypot(x - px, y - py)
        px, py = x, y
        if acc >= max_px:
            break
    return (px - x0, py - y0)


def _leaving(edge, node):
    """Heading vector leaving `node` into `edge` (node is edge['u'] or ['v'])."""
    return _heading_vec(edge["polyline"] if edge["u"] == node else list(reversed(edge["polyline"])))


def _angle_between(v1, v2):
    a = abs(math.atan2(v1[1], v1[0]) - math.atan2(v2[1], v2[0])) % (2 * math.pi)
    return min(a, 2 * math.pi - a)


def _oriented(edge, from_node):
    return edge["polyline"] if edge["u"] == from_node else list(reversed(edge["polyline"]))


def extract_lines(nodes, edges, terminal_ids, straight_max_deg=45.0):
    """Post-contract, every node is degree-1, degree>=3, or a terminal, and
    elbows are just bends inside a polyline -- so a "line" the user's six
    tracing rules describe is just: from a terminal, keep taking the
    straightest arm at every junction until another terminal or a dead end.

    Returns a list of run dicts: id, from, to, end_reason, polyline,
    edges (edge ids), bridges, length_px. end_reason is one of
    terminal / dead_end / no_straight_arm / loop.
    """
    by_id = {e["id"]: e for e in edges}
    inc = {}
    for e in edges:
        inc.setdefault(e["u"], []).append(e)
        inc.setdefault(e["v"], []).append(e)
    max_rad = math.radians(straight_max_deg)

    queued = set()                          # (node, edge_id) already used as a walk origin
    work = deque()
    for t in terminal_ids:
        for e in inc.get(t, []):
            key = (t, e["id"])
            if key not in queued:
                queued.add(key)
                work.append((t, e))

    runs = []
    while work:
        origin, first_edge = work.popleft()
        poly = _oriented(first_edge, origin)
        edge_ids = [first_edge["id"]]
        prev_edge = first_edge
        node = first_edge["v"] if first_edge["u"] == origin else first_edge["u"]
        end_reason = None
        while True:
            if node in terminal_ids or len(inc.get(node, [])) == 1:
                end_reason = "terminal" if node in terminal_ids else "dead_end"
                break
            arms = [oe for oe in inc[node] if oe["id"] != prev_edge["id"]]
            arrival = tuple(-c for c in _leaving(prev_edge, node))   # reverse of "leaving node back into prev_edge"
            best, best_ang = None, None
            for oe in arms:
                ang = _angle_between(arrival, _leaving(oe, node))
                if best is None or ang < best_ang:
                    best, best_ang = oe, ang
            chosen = best if (best is not None and best_ang <= max_rad) else None
            for oe in arms:                 # every arm not chosen is a branch: queue it
                if oe is chosen:
                    continue
                key = (node, oe["id"])
                if key not in queued:
                    queued.add(key)
                    work.append((node, oe))
            if chosen is None:
                end_reason = "no_straight_arm"
                break
            if chosen["id"] in edge_ids:
                end_reason = "loop"
                break
            poly = poly[:-1] + _oriented(chosen, node)
            edge_ids.append(chosen["id"])
            prev_edge = chosen
            node = chosen["v"] if chosen["u"] == node else chosen["u"]

        run_edges = [by_id[eid] for eid in edge_ids]
        runs.append({"id": f"r{len(runs)}", "from": origin, "to": node, "end_reason": end_reason,
                     "polyline": poly, "edges": edge_ids,
                     "bridges": sum(e.get("bridges", 0) for e in run_edges),
                     "length_px": sum(e["length_px"] for e in run_edges)})
    return runs


# --------------------------------------------------------------------------
def prune_to_network(nodes, edges, terminal_ids):
    """Connectivity != straight segments: a branch teeing into a header IS
    connected to everything that header reaches. This prunes the contracted
    graph down to the real pipe network so that question can be answered by
    plain reachability -- drop self-loops, strip every non-terminal degree-1
    node (leader lines, annotation strokes, dimension ticks, symbol spurs)
    to fixpoint (no length threshold: they simply dead-end once their spur
    is gone), then re-contract (pruning creates new degree-2 nodes).
    Operates on copies; the inputs are left untouched.
    """
    net_nodes = {k: dict(v) for k, v in nodes.items()}
    net_edges = [dict(e) for e in edges if e["u"] != e["v"]]
    changed = True
    while changed:
        deg = Counter()
        for e in net_edges:
            deg[e["u"]] += 1
            deg[e["v"]] += 1
        drop = {n for n in net_nodes if n not in terminal_ids and deg.get(n, 0) <= 1}
        changed = bool(drop)
        if changed:
            net_edges = [e for e in net_edges if e["u"] not in drop and e["v"] not in drop]
            for n in drop:
                net_nodes.pop(n, None)
    net_edges = contract_degree2(net_nodes, net_edges, terminal_ids)
    return net_nodes, net_edges


def network_connections(nodes, edges, terminal_ids, max_full_pairs=12, k_nearest=5):
    """Connected components of the (pruned) network, and terminal-to-terminal
    shortest paths (Dijkstra on length_px) within each component. A component
    with more than max_full_pairs terminals emits only each terminal's
    nearest k_nearest rather than every pair, deduped on the unordered pair.

    Returns (components, connections):
      components: [{"nodes": [id,...], "terminals": [id,...]}, ...]
      connections: [{"from", "to", "edges": [id,...], "length_px", "bridges"}, ...]
    """
    by_id = {e["id"]: e for e in edges}
    adj = {}
    for e in edges:
        adj.setdefault(e["u"], []).append((e["v"], e))
        adj.setdefault(e["v"], []).append((e["u"], e))

    seen, components = set(), []
    for n in nodes:
        if n in seen:
            continue
        comp, dq = [], deque([n])
        seen.add(n)
        while dq:
            cur = dq.popleft()
            comp.append(cur)
            for nb, _ in adj.get(cur, []):
                if nb not in seen:
                    seen.add(nb)
                    dq.append(nb)
        components.append({"nodes": comp, "terminals": sorted(t for t in comp if t in terminal_ids)})

    connections, emitted_pairs = [], set()
    for comp in components:
        terms = comp["terminals"]
        if len(terms) < 2:
            continue
        cap = None if len(terms) <= max_full_pairs else k_nearest
        for src in terms:
            dist, prev_edge, prev_node, visited = {src: 0.0}, {}, {}, set()
            heap = [(0.0, src)]
            while heap:
                d, u = heapq.heappop(heap)
                if u in visited:
                    continue
                visited.add(u)
                for v, e in adj.get(u, []):
                    nd = d + e["length_px"]
                    if v not in dist or nd < dist[v]:
                        dist[v] = nd
                        prev_edge[v] = e
                        prev_node[v] = u
                        heapq.heappush(heap, (nd, v))
            reach = sorted((t for t in terms if t != src and t in dist), key=lambda t: dist[t])
            if cap is not None:
                reach = reach[:cap]
            for t in reach:
                pair = frozenset((src, t))
                if pair in emitted_pairs:
                    continue
                emitted_pairs.add(pair)
                path_edges = []
                cur = t
                while cur != src:
                    e = prev_edge[cur]
                    path_edges.append(e["id"])
                    cur = prev_node[cur]
                path_edges.reverse()

                # Build polyline by concatenating edge polylines along the path
                polyline = []
                for i, eid in enumerate(path_edges):
                    e = by_id[eid]
                    if i == 0:
                        polyline.extend(e["polyline"])
                    else:
                        polyline.extend(e["polyline"][1:])  # skip first point to avoid duplication

                connections.append({"from": src, "to": t, "edges": path_edges,
                                    "length_px": dist[t], "hops": len(path_edges),
                                    "bridges": sum(by_id[eid].get("bridges", 0) for eid in path_edges),
                                    "polyline": polyline})
    return components, connections


if __name__ == "__main__":
    # ponytail self-check: a synthetic + and X on a small canvas
    m = np.zeros((60, 60), np.uint8)
    cv2.line(m, (5, 30), (55, 30), 1, 3)      # horizontal through-line
    cv2.line(m, (30, 5), (30, 25), 1, 3)      # a tee stub from above (degree 3 at ~30,30)
    m2 = np.zeros((60, 60), np.uint8)
    cv2.line(m2, (5, 10), (55, 10), 1, 3)     # a clean crossing
    cv2.line(m2, (30, 0), (30, 20), 1, 3)

    # Bypass test: two distinct paths between the same two junctions.
    # Draw a horizontal line and a detour around it (top and bottom).
    m_bypass = np.zeros((80, 80), np.uint8)
    cv2.line(m_bypass, (10, 40), (70, 40), 1, 3)      # main path
    cv2.line(m_bypass, (10, 20), (10, 60), 1, 3)      # left connector
    cv2.line(m_bypass, (10, 20), (70, 20), 1, 3)      # top bypass
    cv2.line(m_bypass, (70, 20), (70, 60), 1, 3)      # right connector
    cv2.line(m_bypass, (70, 60), (10, 60), 1, 3)      # bottom bypass
    nodes, edges, nb = build_graph(m, np.zeros_like(m))
    assert nb == 0
    classify_nodes(nodes, edges)
    edges = contract_degree2(nodes, edges, set())
    degs = sorted(n["degree"] for n in nodes.values())
    assert 3 in degs, f"expected a degree-3 tee node, got degrees {degs}"

    nodes2, edges2, _ = build_graph(m2, np.zeros_like(m2))
    classify_nodes(nodes2, edges2)
    edges2 = contract_degree2(nodes2, edges2, set())
    kinds = sorted(n["degree"] for n in nodes2.values())
    assert 4 not in kinds, f"crossing should have been split, degrees {kinds}"
    assert len(edges2) == 2, f"crossing should yield 2 pass-through edges, got {len(edges2)}"

    # extract_lines: a T -- horizontal through-line, perpendicular stub, a
    # terminal at each of the 3 tips. The through-line must come out as one
    # straight terminal-to-terminal run; the stub, arriving at 90 degrees,
    # must fail the straight-arm test.
    terms = [{"id": "TL", "box": (3, 28, 7, 32)}, {"id": "TR", "box": (53, 28, 57, 32)},
             {"id": "TS", "box": (28, 3, 32, 7)}]
    attached, not_attached = attach_terminals(nodes, edges, terms)
    assert not not_attached, f"terminals not attached: {not_attached}"
    edges = contract_degree2(nodes, edges, {"TL", "TR", "TS"})
    runs = extract_lines(nodes, edges, {"TL", "TR", "TS"})
    straight = [r for r in runs if {r["from"], r["to"]} == {"TL", "TR"} and r["end_reason"] == "terminal"]
    assert straight, f"expected a straight TL<->TR run, got {[(r['from'], r['to'], r['end_reason']) for r in runs]}"
    assert any(r["end_reason"] == "no_straight_arm" for r in runs), \
        "the 90-degree stub should have failed the straight-arm test somewhere"
    covered = {eid for r in runs for eid in r["edges"]}
    assert covered == {e["id"] for e in edges}, "every edge should appear in at least one run"

    # prune_to_network / network_connections: the T's stub is NOT a straight
    # continuation (extract_lines correctly stops there) but it IS connected
    # -- all 3 terminals must land in one component with 3 pairwise routes.
    net_nodes, net_edges = prune_to_network(nodes, edges, {"TL", "TR", "TS"})
    comps, conns = network_connections(net_nodes, net_edges, {"TL", "TR", "TS"})
    multi = [c for c in comps if len(c["terminals"]) >= 2]
    assert len(multi) == 1 and set(multi[0]["terminals"]) == {"TL", "TR", "TS"}, \
        f"expected TL/TR/TS in one component, got {[c['terminals'] for c in comps]}"
    assert len(conns) == 3, f"expected 3 pairwise connections, got {len(conns)}: {conns}"

    # Bypass correctness: two geometrically-distinct paths between the same nodes must both survive.
    nodes_by, edges_by, _ = build_graph(m_bypass, np.zeros_like(m_bypass))
    classify_nodes(nodes_by, edges_by)
    edges_by = contract_degree2(nodes_by, edges_by, set())
    # Count edges by (u,v) pair; at least one pair should have 2 edges (the parallel paths).
    pair_counts = {}
    for e in edges_by:
        pair = (e["u"], e["v"]) if e["u"] <= e["v"] else (e["v"], e["u"])
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    multi_edges = [c for c in pair_counts.values() if c >= 2]
    assert multi_edges, f"bypass: expected at least one (u,v) pair with >=2 distinct paths, got {pair_counts}"

    print("skelgraph self-check OK:", "tee degrees", degs, "crossing edges", len(edges2),
          "extract_lines runs", len(runs), "network connections", len(conns),
          "bypass parallel edges", multi_edges)
