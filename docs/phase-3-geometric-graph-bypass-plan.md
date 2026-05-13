# Phase 3 — Direct Polylines-to-Graph Bypass

## Problem Statement

**The geometric pipeline produces 1,373 fragments for Test-00008 but only 128 H+V endpoint junctions are detected.** The remaining 1,152 segments have no detected junction involvement — they are fragments that don't connect to anything.

The root cause: the geometric pipeline renders segments to a binary mask, then runs Stages 6–10 (skeletonization → edge tracing). This loses the segment directionality advantage and creates the same fragmentation problem Fix 1 identified.

**Phase 3 bypasses Stages 6–11 entirely**, processing geometric segments directly to build the graph.

---

## Architecture Overview

```
Stage 5 (geometric)                  New Phase 3 module
┌─────────────────────┐               ┌──────────────────────────────────────────────┐
│ stage5_geometric   │──────────────▶│ geometric_graph_builder.py                   │
│ _segments.json      │               │                                              │
│ [1373 segments]     │               │  Phase A: Segment chaining                  │
│ {x1,y1,x2,y2,       │               │    → ~200-400 runs (was 1373 segments)     │
│  length, area_parent}               │                                              │
└─────────────────────┘               │  Phase B: Junction detection                │
                                       │    → T-junctions, L-corners, X-crossings   │
                                       │    → ~200-400 nodes (was 1139 clusters)   │
                                       │                                              │
                                       │  Phase C: Graph assembly                   │
                                       │    → Run endpoint → nearest junction        │
                                       │    → ~200-400 edges                         │
                                       └──────────────────────────────────────────────┘
                                                               │
                                                               ▼
                                       ┌──────────────────────────────────────────────┐
                                       │ stage12_graph_assembly (existing)            │
                                       │  → Equipment/connection/text attachments     │
                                       │  → Edge terminals + connectivity            │
                                       │  → NetworkX graph → stage12_graph.json      │
                                       └──────────────────────────────────────────────┘
```

**Key insight:** Segment chaining + proximity-based junction detection replaces the entire Stages 6–11 chain.

---

## Phase A — Segment Chaining

### Why Chaining?

1,373 geometric segments → chaining collinear runs → ~200–400 runs.

The skeleton-based edge tracer implicitly chains skeleton pixels. We need to do this explicitly for segments.

### Algorithm

**Orientation:** Classify each segment as H or V:
```python
angle = atan2(y2-y1, x2-x1) % 180
HV_THRESHOLD = 45  # angles ≤45° or ≥135° = horizontal
```

**H-chain:** Sort H segments by y, then by x. Merge if:
- Same y (within `CHAIN_ALIGN_TOLERANCE_PX = 3`)
- x-ranges overlap or are within `CHAIN_GAP_PX = 8`

**V-chain:** Sort V segments by x, then by y. Merge if:
- Same x (within `CHAIN_ALIGN_TOLERANCE_PX = 3`)
- y-ranges overlap or are within `CHAIN_GAP_PX = 8`

**Output:** Each run = `{id, orientation, x1, y1, x2, y2, length, segments: [seg_ids]}`

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `CHAIN_ALIGN_TOLERANCE_PX` | 3 | Allow slight angle drift in "straight" runs |
| `CHAIN_GAP_PX` | 8 | Max gap between segments to still chain |
| `MIN_RUN_LENGTH_PX` | 25 | Minimum run length (same as `MIN_SEGMENT_LENGTH_PX`) |

---

## Phase B — Junction Detection

### Approach: Proximity Clustering of Run Endpoints

For each run, compute its two endpoints. Cluster endpoints that are within `JUNCTION_PROXIMITY_PX = 15` of each other.

### Algorithm

```
1. Collect all run endpoints: (x, y, run_id, is_start: bool)
2. Build a spatial hash grid (cell size = JUNCTION_PROXIMITY_PX)
3. For each endpoint, find nearby endpoints in adjacent cells
4. Cluster nearby endpoints using single-linkage clustering
5. Each cluster = one junction
6. For each junction, classify by run count:
   - 2 runs = L-corner (90° turn)
   - 3 runs = T-junction
   - 4 runs = X-crossing
   - 1 run = dead end (terminal)
```

### Angle-Based Sub-Classification

For L-corners and T-junctions, compute the angle between connected runs to verify 90° turn:
```
L-corner: |angle_diff - 90°| ≤ 15°
T-junction: one run perpendicular to two collinear runs
```

### Output: Junction Node

```python
{
    "id": "geo_junction_0",
    "type": "junction",  # or "terminal" for dead ends
    "junction_subtype": "L" | "T" | "X" | "terminal",
    "position": {"x": float, "y": float},  # centroid of endpoint cluster
    "connected_runs": ["run_0", "run_1", ...],
    "review_state": "provisional",  # no unresolved concept in geometric bypass
}
```

### Parameters

| Parameter | Value | Rationale |
|---|---|---|
| `JUNCTION_PROXIMITY_PX` | 15 | Same as `ENDPOINT_MERGE_PX` from inpaint pipeline |
| `ANGLE_TOLERANCE_DEG` | 15 | Accept 90° ± 15° as valid L/T |

---

## Phase C — Graph Assembly

### Build Edges from Runs

For each run:
1. Find the junction at each endpoint (if any)
2. If both endpoints connect to junctions → edge between those two junctions
3. If one endpoint connects to junction, other is dead end → edge from junction to terminal node

**Run-to-node mapping:**
```python
def find_nearest_junction(endpoint, junctions, max_dist=JUNCTION_PROXIMITY_PX):
    best = None
    for j in junctions:
        dist = hypot(endpoint.x - j.position.x, endpoint.y - j.position.y)
        if dist <= max_dist:
            if best is None or dist < best.dist:
                best = j
    return best
```

### Terminal Nodes

Dead-end runs (one or both endpoints with no nearby junction) become terminal nodes:
```python
{
    "id": f"geo_terminal_{n}",
    "type": "terminal",
    "position": {"x": float, "y": float},
}
```

### Edge Structure

```python
{
    "id": f"geo_edge_{n}",
    "source": node_id,  # junction or terminal
    "target": node_id,  # junction or terminal
    "pixel_length": run.length,
    "polyline": [
        {"row": y1, "col": x1},
        {"row": y2, "col": x2}
    ],
    "flow_direction": None,  # N/A for geometric — arrow stages still run downstream
    "review_state": "provisional",
}
```

### Graph Output

Matches `stage12_graph.json` format exactly:
```python
{
    "nodes": [junction_nodes + terminal_nodes],
    "edges": [geo_edges],
    "unresolved_junction_ids": [],  # geometric bypass has no unresolved concept
    "crossings": [],  # crossings → junctions in geometric path
}
```

---

## Integration: `stage12_geometric_graph_assembly`

Add a new stage method to `pid_extractor.py`:

```python
def stage12_geometric_graph_assembly(self) -> None:
    """
    Bypass Stages 6–11: build graph directly from Stage 5 geometric segments.
    Reuses all Stage 12 attachment stages (equipment, connection, text, etc.)
    """
    # Phase A: chain segments into runs
    segments_payload = self._load_json_artifact("stage5_geometric_segments")
    runs = chain_geometric_segments(segments_payload["segments"])

    # Phase B: detect junctions from run endpoints
    junctions = detect_junctions_from_runs(runs)

    # Phase C: build graph
    geo_graph = build_graph_from_runs_and_junctions(runs, junctions)

    # Save intermediate artifacts
    self._save_json("phase3_runs", {"runs": runs})
    self._save_json("phase3_junctions", {"junctions": junctions})
    self._save_json("phase3_graph", geo_graph["graph_payload"])

    # Reuse existing Stage 12 attachment stages
    # (they take edges + objects → attachment decisions)
    # ... existing attachment logic unchanged ...
```

### Dispatcher Logic

In `pid_extractor.py`, add a new dispatcher after Stage 5:
```python
# After stage5_dispatcher
if self.cfg.use_geometric_line_detection:
    self.stage12_geometric_graph_assembly()  # skips 6-11
else:
    self.stage6_pipe_mask_sealing()
    self.stage7_pipe_skeleton()
    # ... stages 8-12 as normal ...
```

Or more surgical: keep `use_geometric_line_detection` behavior where Stage 5 produces both mask AND segments, but for `geometric_bypass=True`, skip stages 6-11 and call the new phase 3 module.

---

## New File: `backend/garnet/geometric_graph_builder.py`

### Functions

```python
def chain_geometric_segments(segments: list[dict]) -> list[dict]:
    """
    Phase A: Chain collinear/near-collinear segments into runs.
    Returns list of runs with orientation, endpoints, length, and member segments.
    """

def detect_junctions_from_runs(runs: list[dict]) -> list[dict]:
    """
    Phase B: Cluster run endpoints → junctions.
    Returns list of junction nodes with subtype (L/T/X/terminal).
    """

def build_graph_from_runs_and_junctions(
    runs: list[dict],
    junctions: list[dict],
) -> dict:
    """
    Phase C: Assemble NetworkX graph from runs and junctions.
    Returns graph_payload + summary matching stage12_graph format.
    """
```

---

## Key Differences from Skeleton-Based Pipeline

| Aspect | Skeleton pipeline | Geometric bypass |
|---|---|---|
| Segments | Skeleton pixels | Explicit line segments |
| Chaining | Done by skeletonization + edge trace | Done by collinear chaining |
| Junction detection | Branch angle analysis on skeleton clusters | Proximity clustering on run endpoints |
| Edge direction | Requires arrow matching | Not yet available (upstream stages handle) |
| Fragmentation | ~63% provisional edges | Expected ~50% provisional (chaining should help) |

---

## Expected Outcome

| Metric | Rectangular (current) | Geometric + Phase 3 (target) |
|---|---|---|
| Edge count | 783 | ~300–500 (after chaining) |
| Node count | 1,139 | ~200–400 |
| Provisional rate | 63% | TBD (estimate: 40–50%) |
| Flow direction | Via arrow matching | Not available until Stage 10c equivalent |
| Gap coverage | 27.0% | TBD |

---

## Constraints

- Output format must match `stage12_graph.json` exactly for downstream stages (13, 14, 16)
- Attachment stages (equipment, connection, text) must work unchanged — they rely on edge objects with `id`, `source`, `target`, `polyline`
- Do NOT modify skeleton-based pipeline (Stages 6–11) — Phase 3 is an additive alternative path
- All new functions require unit tests

---

## Verification

```bash
cd /Volumes/Ginnungagap/maetee/Code/GARNET/backend

# Run geometric bypass pipeline
python -m garnet.pid_extractor \
  --image test/ppcl/Test-00008.jpg \
  --out output/phase3_test \
  --stop-after 12 \
  --geometric \
  --ocr-route ocrmac

# Compare with rectangular baseline
python3 -c "
import json

# Phase 3 result
with open('output/phase3_test/stage12_graph_summary.json') as f:
    ph3 = json.load(f)
print('=== PHASE 3 (geometric bypass) ===')
print(f'  nodes: {ph3[\"node_count\"]}')
print(f'  edges: {ph3[\"edge_count\"]}')
print(f'  components: {ph3[\"connected_component_count\"]}')

# Rectangular baseline
with open('/Volumes/Ginnungagap/maetee/Code/GARNET/autoresearch/graph_loop/tmp_eval/Test-00008/stage12_graph_summary.json') as f:
    rect = json.load(f)
print('\n=== RECTANGULAR (baseline) ===')
print(f'  nodes: {rect[\"node_count\"]}')
print(f'  edges: {rect[\"edge_count\"]}')
print(f'  components: {rect[\"connected_component_count\"]}')
"
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Chaining produces wrong merges (pipe A merged with pipe B) | Medium | Strict alignment tolerance (3px), gap limit (8px) |
| Junction proximity clustering creates spurious junctions | Medium | Require ≥2 runs to form a junction; singletons = terminals |
| Geometric segments still fragmented after chaining | High | May need iterative chaining (chain → detect junctions → re-chain) |
| Attachment stages fail on geometric edges | Low | Edge structure identical (id, source, target, polyline) |
| Terminal nodes not handled by QA/continuity stages | Low | QA/continuity use graph structure, not node type |

---

## Implementation Order

1. **Phase A** — segment chaining (isolated, testable)
2. **Phase B** — junction detection from runs (isolated, testable)
3. **Phase C** — graph assembly (integrates A+B, testable)
4. **Integration** — new stage method in `pid_extractor.py`
5. **QA comparison** — A/B test on 3 images vs rectangular pipeline
6. **Documentation** — update `fix-1-stage5-inpainting-plan.md` with Phase 3 results