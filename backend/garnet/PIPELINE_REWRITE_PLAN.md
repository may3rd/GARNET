# GARNET Pipeline Rewrite Plan — Continuity-Aware Architecture

## Context

The 10 continuity rules (pipe-continuity-rules.md) define what a "valid" P&ID pipe graph looks like.
The current pipeline was built stage-by-stage without an explicit continuity contract. As a result:

| Violation Count | Root Cause | Stage |
|-----------------|-----------|-------|
| R1: 282 dead-end stubs | Stage 10 closes edges without checking topology | S10 |
| R2: 230 orphan stubs | Stage 12 connects edges without geometry proximity check | S12 |
| R7: 103 floating segments | Stage 10 creates edges from unvalidated endpoints | S10 |
| R8: 551 geometric gaps | Stage 10 doesn't check alignment at branch points; Stage 11 doesn't bridge gaps | S10+S11 |
| R3: 26 chain breaks | Stage 10 closes segments at wrong locations | S10 |
| R5: 19 T-junction missing | Stage 9 clustering misses degree-2 endpoints | S9 |
| R9: 15 degree anomalies | Stage 11 junction review doesn't enforce degree targets | S11 |

**Total fixable upstream: ~1,200 of 1,233 violations**
by improving stages 9, 10, 11, 12.

---

## Architecture Shift

### Before: Pipeline builds graph, Stage 14 validates
```
S8→S9→S10→S10b→S10c→S10d→S11→S12→S13→S14(check only)→S15
```

### After: Every stage enforces continuity contract; Stage 14 is authoritative reporter + new validator
```
S8→S9→S10→S10b→S10c→S10d→S11→S12→S13→S14(contract enforcer + reporter)→S15
```

**Core principle:** The continuity rules are the shared contract. Each stage
imports and calls `check_continuity_rule_N()` from `pipe_continuity_checker.py`
at its own decision points. Stage 14 remains the canonical QA layer.

---

## Stage-by-Stage Rewrite

### Stage 8 — Skeleton Node Detection
**Current behavior:** Detects endpoints (degree-1 skeleton pixels) and junctions
(degree ≥ 3) from the skeleton image using local neighborhood analysis.
Produces 764 endpoints + 1840 raw junctions.

**Continuity contract needed:** Nodes should be classified by expected degree:
- `dead_end` = degree 1 in final graph (goes to equipment nozzle or sheet break)
- `tee_candidate` = degree 3
- `cross_candidate` = degree 4
- `bend` = degree 2

**Changes:**
1. Add `expected_degree` metadata to each node at detection time.
2. Flag nodes where skeleton pixel neighborhood suggests degree-2 but current logic
   classifies them as endpoint — these are "hidden bends" (R5/R9 source).
3. Output: node clusters with degree hint → passes to Stage 9.

**Files to change:** `pipe_nodes.py`
**New function:** `_classify_node_degree_from_skeleton(skeleton_patch, pixel_pos) -> DegreeHint`

---

### Stage 9 — Node Clustering
**Current behavior:** Groups nearby skeleton endpoints/junctions into cluster nodes.
Reduces 764+1840 raw nodes to 436 junctions + 703 endpoints.

**Continuity contract needed:** Clusters must reflect valid topology:
- Cluster with expected degree 2 → must connect to exactly 2 neighbors
- Cluster with expected degree 3 → must have 3 incident edges
- Cluster merging should NOT combine what should be separate junctions

**Changes:**
1. Add cluster-degree validation: after clustering, check if the centroid node's
   incident edge count matches the expected degree from Stage 8 hints.
2. If cluster is "overmerged" (degree >> expected), split it — this fixes R5
   (T-junction missing) at source.
3. Add "alignment check": cluster endpoints that are geometrically aligned
   within threshold should be same cluster (fixes R8 geometric gaps before they form).

**Files to change:** `pipe_node_clusters.py`
**New function:** `_validate_cluster_degree(clusters, degree_hints) -> split_recommendations`

---

### Stage 10 — Edge Tracing
**Current behavior:** Traces skeleton paths between node clusters to produce
polylines. 783 edges. Edge closes when it reaches another node cluster.
**Problem:** R1 (282 dead stubs) + R7 (103 floating) + R8 (551 gaps) originate here.
Stage 10 creates edges endpoint-to-endpoint without checking:
- Is this endpoint near another edge (should connect, not terminate)?
- Is this edge going to a true dead-end or a gap in the graph?
- Is this a branch that should attach to the parent pipe at a tee?

**Continuity contract needed:** Every edge terminal should be validated before
the edge is "closed."

**Changes:**
1. **Near-edge detection at tracing time:** Before closing an edge at endpoint A,
   check if endpoint A is within `GAP_THRESHOLD_PX` (e.g., 20px) of any existing
   edge's polyline. If yes → don't close, instead create a T-junction connection
   to the existing edge. This directly fixes R8 (geometric gaps).
2. **Provisional vs. validated terminal:** Mark edge terminals as `provisional`
   when created, then run a quick sanity pass after all edges are traced:
   - If both terminals are `provisional` and edge is very short (< 30px),
     this is likely a tracing artifact → mark for Stage 11 review.
3. **Branch attachment guard:** When tracing creates a new edge that starts
   near an existing edge's midspan (not at a node), flag as "midspan branch"
   → requires T-junction insertion before Stage 11. This fixes R2 (orphan stubs).
4. **Alignment check at endpoints:** When an edge terminates at an endpoint,
   check if another endpoint is aligned within 5px — if yes, these should
   be one continuous pipe (R8 gaps).

**Files to change:** `pipe_edges.py`, `pipe_edge_connectivity.py`
**New function:** `_detect_near_edge_at_endpoint(endpoint_pos, all_polyline_segments, threshold=20) -> bool`
**New function:** `_should_close_edge_at_terminal(endpoint, all_edges) -> CloseAction(STAY_OPEN|CLOSE|SPLIT_PARENT)`

---

### Stage 10b — Polyline Simplification
**Current behavior:** Reduces polyline point count while preserving shape.

**Continuity contract:** Simplification must not break geometric continuity.
- A polyline with a sharp kink (angle > 150°) at a non-junction location
  should NOT be simplified away — it's a legitimate bend.
- Post-simplification check: verify simplified polyline still passes within
  3px of all inline node positions → otherwise preserve the node.

**Files to change:** `polyline_simplify.py`
**New:** Post-simplification validation pass.

---

### Stage 10c — Edge Direction (Arrow Assignment)
**Current behavior:** Assigns `flow_direction` to edges based on arrow detection.
6 edges have arrow assignments.

**Continuity contract:** R10 (arrow direction QA) is already a QA check —
no structural change needed here. However:
- Add warning when an edge has multiple arrows pointing in different directions → R6.
- Mark edges with arrow + short length (< 50px) as high-priority for Stage 11
  review (likely branch stub).

**Files to change:** `edge_direction.py` (add multi-arrow conflict detection).

---

### Stage 10d — Edge Split (Inline Nodes)
**Current behavior:** Splits edges at detected inline components (valves, instruments).

**Continuity contract:** R3 (segment chain breaks) — flanges are part of the
same segment. Inline components should NOT be treated as segment breaks unless
they are explicit flanged breaks in the source drawing.

**Changes:**
1. Add `is_explicit_flange_break` flag: only split if the inline component
   is classified as a flange in the symbol detection.
2. Valves/instruments are passthrough — don't split, just attach metadata.
3. Post-split validation: verify that the two resulting sub-edges are both
   ≥ 5px or flag as a chain integrity violation (R3).

**Files to change:** `edge_split.py`
**New:** `_is_flange_break(inline_component_class) -> bool`

---

### Stage 11 — Junction Review
**Current behavior:** 331 junctions confirmed, 42 unresolved, 63 non-connecting
crossings. Junction decision uses geometric alignment (opposite vectors).

**Continuity contract needed:** R9 (15 degree anomalies) come from junctions
accepted when degree doesn't match type.

**Changes:**
1. **Degree enforcement before accepting junction:** If junction has degree 1
   → reject as dead-end stub (R9 error). If degree > 4 → flag as high-degree
   anomaly (R9 warning).
2. **R8 gap bridging:** 63 non-connecting crossings are exactly the R8 gap
   candidates. At junction review, any crossing where two pipe centerlines
   are aligned within 5px should be connected — don't leave them as separate
   crossing objects.
3. **T-junction validation:** A confirmed tee should have exactly 3 edges.
   After confirming, verify degree == 3. If degree != 3, keep as "provisional
   junction" → passes to Stage 12 for further resolution.

**Files to change:** `pipe_junctions.py`, junction decision logic in `pipe_edge_connectivity.py`
**New function:** `_enforce_junction_degree(junction_node_id, incident_edges) -> Decision(ACCEPT|REJECT|PROVISIONAL)`

---

### Stage 12 — Graph Assembly
**Current behavior:** Builds the final networkX graph. 792 edges, 1157 nodes.
493 of 792 edges are `provisional` (unresolved terminals) — the main source of R1/R7.

**Continuity contract needed:** R2 (230 orphan stubs) — branches that don't
attach to parent.

**Changes:**
1. **Branch attachment validation:** When selecting candidate links,
   verify that the branch edge's terminal is within `CONNECTION_THRESHOLD_PX`
   (5px) of the parent edge's centerline. If not → reject as orphan stub (R2).
2. **Use Stage 14 continuity checker as a sub-step:** Before finalizing the
   graph, run `check_continuity()` on the assembled graph as an internal
   validation pass. Any R1/R7 violations → mark as `provisional` with
   specific rule violation reason.
3. **Equipment attachment refinement:** 9 equipment attachments, 5 utility
   connections. Validate that every edge connected to equipment_terminal
   has its endpoint within 5px of the nozzle anchor. If not → flag R4
   (overlap/underlap at equipment).
4. **Page connector validation:** 5 page connectors. Edges connected to
   `connection_terminal` should terminate exactly at the connector anchor.
   Any gap → R8 warning.

**Files to change:** `pipe_graph.py`, `pipe_edge_connectivity.py`
**New:** Call `check_continuity()` as internal validation before saving stage12_graph.json.

---

### Stage 13 — Graph QA (unchanged)
Current `pipe_graph_qa.py` does articulation points, isolated nodes, crossings,
unresolved terminals — these remain valid QA checks.
No continuity contract changes needed here.

---

### Stage 14 — Continuity Checker (NEW — this stage)
**Role:** Authoritative reporter of all continuity violations.
Not just a checker — becomes the canonical source of truth for graph quality.

**Changes from current implementation:**
1. Run `check_continuity()` as the primary rule engine.
2. Cross-reference with Stage 13's articulation/isolated node data —
   violations that overlap with Stage 13 findings should be deduplicated
   and merged into a single review item.
3. Output `stage14_continuity_summary.json` with per-rule breakdown and
   severity counts.
4. Output `stage14_continuity_violations_overlay.png` with color-coded
   markers: red=error, yellow=warning, labeled by rule number.
5. **Rule priority mapping:**
   - R1/R7 → combined into "orphan terminal" review queue item
   - R8 → "geometric alignment gap" review queue item
   - R2 → "branch attachment" review queue item

**Files to change:** `run_continuity_checker_stage.py`, `pipe_continuity_checker.py`

---

### Stage 15 — Recovery Loop (unchanged)
No changes needed — the recovery engine should already handle the improved
Stage 14 output.

---

## Summary: Scope of Rewrite

| Stage | Effort | Priority | Violations Fixed |
|-------|--------|----------|-----------------|
| S10 edge tracing | High | 🔴 High | R1, R7, R8 (936) |
| S11 junction review | Medium | 🔴 High | R8, R9 (566) |
| S12 graph assembly | Medium | 🟡 Medium | R2, R4 (235) |
| S9 node clustering | Low | 🟡 Medium | R5 (19) |
| S10d edge split | Low | 🟢 Low | R3 (26) |
| S10c edge direction | Low | 🟢 Low | R6 (1) |
| S14 continuity checker | Low | 🔴 High | Already done — refine output |

**Total violations fixable: ~1,200 of 1,233**

---

## Implementation Sequence

**Phase 1 — Quick wins (S10d, S10c, S9):** Add degree hints and degree validation.
Low effort, fixes 46 violations (R3, R5, R6).

**Phase 2 — Core fix (S10):** Near-edge detection at tracing time + alignment check.
High effort, fixes 936 violations (R1, R7, R8).

**Phase 3 — Connection refinement (S11, S12):** Junction degree enforcement +
branch attachment validation. Medium effort, fixes 800+ (R2, R8, R9).

**Phase 4 — Stage 14 enhancement:** Merge Stage 13 QA + Stage 14 continuity
into unified review queue. Low effort, improves human review experience.

---

## Test Strategy

After each phase:
1. Run full pipeline on fullrun_04 (Test-00008.jpg) — same image for comparison.
2. Compare `stage14_violations.json` counts before/after.
3. Target: reduce total violations from 1,233 while keeping rules unchanged.
4. Cross-check with `stage13_review_queue.json` — items should decrease
   as upstream stages produce cleaner graphs.

---

## Key Files

- Rules definition: `garnet/pipe-continuity-rules.md`
- Rule engine: `garnet/pipe_continuity_checker.py`
- Pipeline integration: `garnet/run_continuity_checker_stage.py`
- Main pipeline: `garnet/pid_extractor.py`
- Per-stage files: `pipe_nodes.py`, `pipe_node_clusters.py`, `pipe_edges.py`,
  `pipe_edge_connectivity.py`, `edge_split.py`, `edge_direction.py`,
  `pipe_junctions.py`, `pipe_graph.py`

---

*Generated: 2026-05-06*
*Context: Test image Test-00008.jpg — 1,233 continuity violations (254 errors, 979 warnings)*