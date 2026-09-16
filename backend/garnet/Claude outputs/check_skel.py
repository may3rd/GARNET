#!/usr/bin/env python3
"""Validation harness for skelgraph.py against the real v3_0002A mask.

No source sheet or detection JSON survive from the session, only the mask
and the real terminals list (v3_0002A/connectivity.json). gapmask here is
an approximation: dilate(mask, 9x9) only, so it covers crossings but NOT
the erased-text/symbol boxes the real gapmask in trace_pid2.py also
includes -- so bridging may be a bit more conservative here than in
production. Noted, not tuned around.
"""
import json, time
from collections import Counter
import cv2, numpy as np
import skelgraph as sg
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from trace_pid2 import coverage  # reuse the real coverage() unchanged

t0 = time.time()
mask = (cv2.imread("v3_0002A/pipe_mask.png", cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
conn = json.load(open("v3_0002A/connectivity.json"))
terminals = conn["terminals"]

# Use recovered gapmask instead of crippled approximation
gapmask_old = cv2.dilate(mask, np.ones((9, 9), np.uint8))  # old approximation
gapmask = np.load("gapmask_recovered.npy").astype(np.uint8)  # recovered from overlay

nodes, edges, nbridge = sg.build_graph(mask, gapmask)
t1 = time.time()
# Extract dedup stats if available
dedup_before, dedup_after, genuine_parallel = getattr(sg.build_graph, '_dedup_stats', (None, None, None))
n_edges_before = len(edges)

sg.classify_nodes(nodes, edges)
attached, not_attached = sg.attach_terminals(nodes, edges, terminals)
term_ids = {t["id"] for t in terminals}
edges = sg.contract_degree2(nodes, edges, term_ids)
t2 = time.time()

# pixel accounting (every skeleton pixel is a node or on exactly one edge) is
# asserted unconditionally inside skelgraph.build_graph itself -- it already
# ran, without raising, as part of the build above. Report the raw counts.
from skimage.morphology import skeletonize
total_skel_px = int(skeletonize(mask > 0).sum())
print(f"[pixel accounting OK] skeleton px={total_skel_px}")
if dedup_before is not None:
    print(f"[dedup step] edges_before={dedup_before} edges_after={dedup_after} "
          f"(removed={dedup_before-dedup_after}) genuine_parallel_pairs={genuine_parallel}")

cov = coverage(mask, edges)
n_edges_after = len(edges)
attached_ct, not_attached_ct = len(attached), len(not_attached)

print(f"[time] build={t1-t0:.2f}s graph_ops={t2-t1:.2f}s total={t2-t0:.2f}s")
print(f"nodes={len(nodes)} edges_before_contract={n_edges_before} "
      f"edges_after_contract={n_edges_after} bridges={nbridge}")
print(f"coverage={cov}%")
print(f"terminals attached={attached_ct}/{len(terminals)} not_attached={not_attached}")

longest = sorted(edges, key=lambda e: -e["length_px"])[:20]
print("20 longest edges:")
for e in longest:
    print(f"  {e['u']:>10} -> {e['v']:<10} len={e['length_px']:.0f}")

assert cov >= 90.0, f"coverage {cov}% below 90% -- structural bug in edge tracing"
assert t2 - t0 < 30, f"runtime {t2-t0:.1f}s over budget"
print("CHECKS PASSED")

# --- render skel_check.png ---------------------------------------------
H, W = mask.shape
ov = cv2.cvtColor((mask * 120).astype(np.uint8), cv2.COLOR_GRAY2BGR)
rng = np.random.default_rng(3)
for e in edges:
    col = tuple(int(v) for v in rng.integers(40, 255, 3))
    pts = np.array(e["polyline"], np.int32)
    if len(pts) >= 2:
        cv2.polylines(ov, [pts], False, col, 3)
for nd in nodes.values():
    x, y = nd["xy"]
    cv2.circle(ov, (int(x), int(y)), 6, (0, 0, 255), -1)
scale = min(1.0, 2500 / W)
ov_small = cv2.resize(ov, (int(W * scale), int(H * scale)))
cv2.imwrite("skel_check.png", ov_small)
print("wrote skel_check.png")

# --- extract_lines: the graph walk that turns edges into actual lines ------
t3 = time.time()
runs = sg.extract_lines(nodes, edges, term_ids)
t4 = time.time()
print(f"\n[extract_lines] {t4-t3:.2f}s -> runs={len(runs)}")

tt = term_ids
t2t = [r for r in runs if r["from"] in tt and r["to"] in tt]
t2t_terms = {r["from"] for r in t2t} | {r["to"] for r in t2t}
t2t_terms &= tt
print(f"terminal-to-terminal runs={len(t2t)}  "
      f"terminals appearing in >=1 t2t run={len(t2t_terms)}/{len(terminals)}")
missing_from_t2t = sorted(tt - t2t_terms)
if missing_from_t2t:
    print(f"  terminals with no terminal-to-terminal run: {missing_from_t2t}")

reasons = Counter(r["end_reason"] for r in runs)
print("end_reason breakdown:", dict(reasons))

print("25 longest runs:")
for r in sorted(runs, key=lambda r: -r["length_px"])[:25]:
    print(f"  {str(r['from']):>12} -> {str(r['to']):<12} len={r['length_px']:5} "
          f"bridges={r['bridges']} reason={r['end_reason']}")

# --- render lines_check.png (runs, not raw edges; t2t runs drawn thicker) --
ov2 = cv2.cvtColor((mask * 120).astype(np.uint8), cv2.COLOR_GRAY2BGR)
rng2 = np.random.default_rng(11)
for r in runs:
    col = tuple(int(v) for v in rng2.integers(40, 255, 3))
    pts = np.array(r["polyline"], np.int32)
    thick = 7 if (r["from"] in tt and r["to"] in tt) else 3
    if len(pts) >= 2:
        cv2.polylines(ov2, [pts], False, col, thick)
for nd in nodes.values():
    x, y = nd["xy"]
    cv2.circle(ov2, (int(x), int(y)), 6, (0, 0, 255), -1)
ov2_small = cv2.resize(ov2, (int(W * scale), int(H * scale)))
cv2.imwrite("lines_check.png", ov2_small)
print("wrote lines_check.png")

# --- network connectivity: the real test -----------------------------------
t5 = time.time()
net_nodes, net_edges = sg.prune_to_network(nodes, edges, term_ids)
components, connections = sg.network_connections(net_nodes, net_edges, term_ids)
t6 = time.time()

print(f"\n[network connectivity] {t6-t5:.2f}s")
print(f"pruned graph: nodes={len(net_nodes)} edges={len(net_edges)} "
      f"(from {len(nodes)} nodes, {len(edges)} edges)")
print(f"connected components={len(components)}")

# Key metric: how many of 33 terminals sit in a component with >=2 terminals
multi_term_nodes = set()
for c in components:
    if len(c["terminals"]) >= 2:
        multi_term_nodes.update(c["terminals"])
print(f"\nterminals in multi-terminal components: {len(multi_term_nodes)}/33")

if len(multi_term_nodes) < 33:
    isolated = sorted(term_ids - multi_term_nodes)
    print(f"  isolated terminals: {isolated}")
    for iso in isolated:
        for c in components:
            if iso in c["terminals"]:
                print(f"    {iso}: alone in component with nodes {c['nodes']}")

print(f"\ntotal terminal-to-terminal routes: {len(connections)}")

# 25 longest connections
if connections:
    longest = sorted(connections, key=lambda c: -c["length_px"])[:25]
    print("25 longest terminal-to-terminal connections:")
    for c in longest:
        nh = len(c.get("edges", []))
        print(f"  {c['from']:>12} -> {c['to']:<12} len={c['length_px']:6.0f} "
              f"hops={nh} bridges={c['bridges']}")

# --- render net_check.png ---------------------------------------------------
H, W = mask.shape
# assign each component a colour
comp_colours = {}
rng3 = np.random.default_rng(42)
for i, comp in enumerate(components):
    comp_colours[i] = tuple(int(v) for v in rng3.integers(50, 220, 3))

# render: network edges in component colours
ov_net = cv2.cvtColor((mask * 120).astype(np.uint8), cv2.COLOR_GRAY2BGR)
by_id = {e["id"]: e for e in net_edges}
for i, comp in enumerate(components):
    col = comp_colours[i]
    for eid in [ee["id"] for ee in net_edges for nn in [ee["u"], ee["v"]] if nn in comp["nodes"]]:
        if eid in by_id:
            e = by_id[eid]
            pts = np.array(e["polyline"], np.int32)
            if len(pts) >= 2:
                cv2.polylines(ov_net, [pts], False, col, 2)

# mark terminals with their id, colour by component membership
fnt, fs = cv2.FONT_HERSHEY_SIMPLEX, 0.4
for t in terminals:
    tid = t["id"]
    x1, y1, x2, y2 = t["box"]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for i, comp in enumerate(components):
        if tid in comp["terminals"]:
            col = comp_colours[i]
            cv2.circle(ov_net, (cx, cy), 5, col, -1)
            cv2.putText(ov_net, tid, (cx + 8, cy - 8), fnt, fs, col, 1)
            break

scale = min(1.0, 2500 / W)
ov_net_small = cv2.resize(ov_net, (int(W * scale), int(H * scale)))
cv2.imwrite("net_check.png", ov_net_small)
print(f"wrote net_check.png")
