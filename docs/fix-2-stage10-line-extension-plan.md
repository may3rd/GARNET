# Fix 2 — Stage 10 Line Extension: Skeleton Endpoint Ray-Casting for T-Junction Recovery

## Problem

**493 of 783 edges (63%) are provisional** in Test-00008 — flagged because they connect to unresolved terminal positions. The root cause is **T-junctions where one branch is missing from the detected skeleton**.

The crossing classifier (`pipe_crossings.py` `_classify_candidate`) uses angle-pair heuristics:
- 4-way candidates with 2 opposite pairs → `non_connecting_crossing`
- Junction blob score check for tie-breaking

For some T-junctions, the 4th branch (entering at ~180° from the T-stem) is partially absorbed by the skeleton break. The classifier sees 4 branches at angles like `[0, 71.565, 180, 251.565]` — which has two valid opposite pairs `(0, 180)` and `(71.565, 251.565)` — and classifies as crossing. But visually, one branch should actually be the T-stem, not the crossing leg.

The DFS tracer in `pipe_edges.py` follows the crossing map to route through crossings by aiming at branch centroids. When a crossing has no valid pair (T-junction vs crossing ambiguity), the tracer bails out at the crossing pixel — the missing branch from the T-stem is never traced.

## Fix Approach: Post-Trace Ray-Casting

Add a new post-processing phase in `pipe_edges.py` that:
1. Identifies skeleton endpoints — pixels where the skeleton terminates at the boundary of a traced edge
2. For each endpoint, casts rays in up to 4 directions (primary axis + perpendicular)
3. If a ray hits skeleton pixels within `RAY_MAX_DISTANCE_PX`, creates a new edge to the nearest node

This is added as a final step in `run_pipe_edge_stage`, after `_trace_edges` but before the continuity check.

## Implementation

### New Function: `_extend_endpoints_with_raycasting`

```python
def _extend_endpoints_with_raycasting(
    skeleton: np.ndarray,
    edges: list[dict[str, Any]],
    node_clusters: list[dict[str, Any]],
    *,
    ray_max_distance_px: int = 30,
    min_extension_length_px: int = 10,
) -> list[dict[str, Any]]:
    """
    Post-trace extension: for each edge endpoint that terminates at a skeleton
    pixel with no further neighbors (dead end), cast rays in the edge's exit
    direction to find nearby skeleton segments. If found within threshold,
    create a new edge to the nearest node.
    """
    # Build node centroid map
    node_centroids = {
        str(c["id"]): (float(c["centroid"]["y"]), float(c["centroid"]["x"]))
        for c in node_clusters
        if c.get("kind") == "junction"
    }
    
    # Build set of skeleton pixels covered by existing edges
    covered = set()
    for edge in edges:
        for pt in edge.get("polyline", []):
            covered.add((int(pt["row"]), int(pt["col"])))
    
    new_edges: list[dict[str, Any]] = []
    
    for edge in edges:
        polyline = edge.get("polyline", [])
        if len(polyline) < 2:
            continue
        
        # Check both endpoints
        for side_prompt in ["start", "end"]:
            endpoint = polyline[0] if side_prompt == "start" else polyline[-1]
            ep_row, ep_col = int(endpoint["row"]), int(endpoint["col"])
            
            # Find neighbors of endpoint in skeleton
            neighbors = _neighbors((ep_row, ep_col), skeleton)
            live_neighbors = [n for n in neighbors if n not in covered]
            
            if live_neighbors:
                continue  # Edge continues, not a dead end
            
            # This is a dead-end endpoint — find exit direction from polyline
            if side_prompt == "start":
                exit_pt = polyline[1]
            else:
                exit_pt = polyline[-2]
            
            dx = exit_pt["col"] - ep_col
            dy = exit_pt["row"] - ep_row
            length = math.hypot(dx, dy)
            if length < 1:
                continue
            
            # Normalize direction
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Cast ray from endpoint in the exit direction
            hit = _cast_ray((ep_row, ep_col), (dy_norm, dx_norm), skeleton, max_distance=ray_max_distance_px)
            
            if hit is None:
                continue
            
            hit_row, hit_col = hit
            hit_distance = math.hypot(hit_row - ep_row, hit_col - ep_col)
            if hit_distance < min_extension_length_px:
                continue
            
            # Find nearest node to hit point
            nearest_node_id = None
            nearest_dist = None
            for node_id, (node_row, node_col) in node_centroids.items():
                dist = math.hypot(hit_row - node_row, hit_col - node_col)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_node_id = node_id
            
            if nearest_node_id is None or nearest_node_id == edge["source"] or nearest_node_id == edge["target"]:
                continue
            
            # Build new edge polyline: existing edge + extension ray
            extended_polyline = polyline.copy()
            if side_prompt == "start":
                extended_polyline = [{"row": ep_row, "col": ep_col}] + extended_polyline
            else:
                extended_polyline = extended_polyline + [{"row": ep_row, "col": ep_col}]
            
            new_edge_id = f"{edge['source']}__{nearest_node_id}_ext_{len(polyline)}"
            new_edges.append({
                "id": new_edge_id,
                "source": edge["source"],
                "target": nearest_node_id,
                "polyline": extended_polyline,
                "pixel_length": len(extended_polyline),
                "extension": True,
            })
    
    return new_edges


def _cast_ray(
    start: Point,
    direction: tuple[float, float],
    skeleton: np.ndarray,
    max_distance: int,
) -> Point | None:
    """Cast a ray from start in direction, return first skeleton pixel hit or None."""
    row, col = start
    dy, dx = direction
    for step in range(1, max_distance + 1):
        probe_row = int(round(row + dy * step))
        probe_col = int(round(col + dx * step))
        if probe_row < 0 or probe_col < 0:
            return None
        if probe_row >= skeleton.shape[0] or probe_col >= skeleton.shape[1]:
            return None
        if skeleton[probe_row, probe_col] > 0:
            return (probe_row, probe_col)
    return None
```

### Integration Point

In `run_pipe_edge_stage`:
```python
def run_pipe_edge_stage(...) -> dict[str, Any]:
    edges = _trace_edges(...)
    
    # ─── Phase 3: Ray-casting T-junction recovery ───────────────────
    extended_edges = _extend_endpoints_with_raycasting(
        skeleton_mask,
        edges,
        node_clusters,
        ray_max_distance_px=30,
        min_extension_length_px=10,
    )
    
    # Deduplicate: don't add if equivalent edge already exists
    existing_pairs = {frozenset((e["source"], e["target"])) for e in edges}
    for new_edge in extended_edges:
        pair = frozenset((new_edge["source"], new_edge["target"]))
        if pair not in existing_pairs:
            edges.append(new_edge)
            existing_pairs.add(pair)
    
    # ─── Phase 2 continuity check (runs after extension) ─────────
    continuity_result = run_post_trace_continuity_check(...)
    ...
```

### Parameters

| Parameter | Default | Rationale |
|---|---|---|
| `ray_max_distance_px` | 30 | T-junction stubs are typically 10–25px |
| `min_extension_length_px` | 10 | Filter noise, only real extensions |

### Validation

1. **Skeleton component count** — should remain the same (no artificial merging)
2. **Edge count** — should increase by ~10–30 edges per image (T-junction recovery)
3. **Provisional edge rate** — should drop (more edges now connected to real nodes)
4. **Visual overlay** — new extended edges shown in cyan on overlay
5. **A/B test** — run on Test-00008, Test-00005, Test-00001; compare `edge_count`, `provisional_edge_count` before/after

## Expected Impact

- **+15–25 edges** recovered per image (based on ~40 unresolved crossings)
- **Provisional edge rate** should drop from 63% to ~55–60%
- Combined with Fix 1 (inpainting): target 83% → 88% validated

## Constraints

- Do NOT modify `_trace_edges` or `_trace_from_pixel`
- Do NOT modify `pipe_crossings.py` — this fix is orthogonal to the crossing classifier
- Only add new functions; existing behavior must be preserved
- All existing tests must pass

## Files Modified

- `backend/garnet/pipe_edges.py` — add `_extend_endpoints_with_raycasting` and `_cast_ray`
- `backend/garnet/tests/test_pipe_edges.py` — add tests for ray-casting extension

## Verification

```bash
cd /Volumes/Ginnungagap/maetee/Code/GARNET/backend
python -m garnet.pid_extractor \
  --image test/ppcl/Test-00008.jpg \
  --out output/fix2_test \
  --stop-after 10 \
  --ocr-route ocrmac

# Check edge count and provisional rate
python3 -c "
import json
with open('output/fix2_test/stage10_pipe_edge_summary.json') as f:
    s = json.load(f)
print(f'edge_count: {s[\"edge_count\"]}')
with open('output/fix2_test/stage12_edge_terminals.json') as f:
    t = json.load(f)
provisional = sum(1 for e in t['edge_terminals'] if e.get('provisional_due_to_unresolved_terminal'))
print(f'provisional: {provisional}/{len(t[\"edge_terminals\"])} = {provisional/len(t[\"edge_terminals\"])*100:.1f}%')
"
```