# Punch List

## Stage 6 Remaining Improvements

These are not blockers for starting Stage 7 if human-in-the-loop review will handle missing or weak data later.

1. Add per-association confidence/status
   - Add explicit statuses such as `exact`, `near`, `weak`, and `rejected` for each accepted/rejected association.
   - Preserve distance, threshold, source artifact, and matching rule so human review can prioritize weak links.

2. Parse line numbers into structured fields
   - Keep raw/normalized line text.
   - Add parsed candidates for size, service, sequence, spec/class, insulation/tracing, and suffix where detectable.
   - Attach parse confidence and unresolved parse tokens.

3. Add trace grouping hints
   - Group traces that likely belong to the same process line by line number, connected terminals, branch relation, direction continuity, and shared equipment/page connector context.
   - Keep these as hints only; Stage 7 should still preserve trace-level provenance.

4. Preserve explicit reused/skipped trace references
   - Keep `skipped_existing_trace` records as non-physical sources with `reused_trace_id`.
   - Ensure downstream stages do not count skipped traces as physical pipe edges.
   - Keep skipped port/node records available for review and equipment-port completeness checks.

5. Add QA severity classification
   - `blocking`: invalid geometry, no terminal where one is required, disconnected equipment, malformed trace.
   - `review`: missing line number, weak association, multiple line candidates, ambiguous terminal, dead end requiring review.
   - `info`: skipped duplicate, reused path, accepted exact match.

## Stage 7 Plan: Build Process Graph From Stage 6

### Goal
Build the next graph assembly stage directly from `stage6_trace_associations.json`, instead of the older `stage5_geometric_segments` / Phase 3 geometric path.

### Assumptions / constraints
- Stage 6 is the source of truth for the geometric-route Stage 7.
- Human review will later resolve missing line numbers and weak associations.
- Skipped/reused traces should be represented as metadata, not counted as duplicate physical edges.
- Existing Stage 7 artifact names should be preserved where practical to avoid breaking downstream export and UI flows.

### Current state
- `stage6_trace_associations.json` contains `trace_edges`, per-edge `attachments`, accepted/rejected associations, and unresolved QA lists.
- Current `stage7_geometric_graph_assembly()` still builds from `stage5_geometric_segments.json`, Phase 3 runs, and geometric edge builders.
- Current geometric pipeline route already runs Stage 5b then Stage 6 before Stage 7.

### Decision
Use Stage 6 as the input contract for Stage 7 graph assembly.

Reason:
- Stage 6 already has traced paths, terminals, equipment ports, inline objects, line numbers, instrument tags, flow arrows, and QA lists.
- Reusing old geometric-segment artifacts would discard the recent path-tracing fixes and duplicate-suppression logic.

### Implementation plan
1. Add a Stage 7 loader for Stage 6
   - Read `stage6_trace_associations.json`.
   - Validate required keys: `trace_edges`, `associations`, `unresolved`.
   - Reject or warn when no physical trace edges exist.

2. Normalize Stage 6 trace edges into graph edges
   - Include only physical traces with non-empty `segments`.
   - Exclude `skipped_existing_trace` from physical edge count.
   - Preserve skipped/reused records in graph metadata/review artifacts.
   - Carry `trace_id`, source object, source port, terminal, polyline, turns, hits, and attachments.

3. Create graph nodes
   - Create source nodes for equipment ports, page connections, branch starts, and object ports.
   - Create terminal nodes for equipment, page connections, instruments, tee junctions, branch connections, dead ends, and sheet edges.
   - Merge node positions within a small tolerance when they represent the same physical point.
   - Preserve original trace provenance on each node.

4. Build connectivity
   - Connect source node to terminal node for each physical trace edge.
   - Use tee/branch terminal metadata to connect branch edges at shared junction nodes.
   - Use `reused_trace_id` metadata to mark equivalent/reverse sources without duplicating edges.

5. Attach process semantics
   - Edge attributes: line number candidates, inline objects/valves, instruments, flow arrows, terminal types, source/terminal objects.
   - Node attributes: equipment/page connector/instrument identity, source detections, associated ports.

6. Emit Stage 7 artifacts
   - `stage7_graph.json`
   - `stage7_graph_summary.json`
   - `stage7_trace_edge_nodes.json`
   - `stage7_line_groups.json` as provisional grouping hints if simple grouping is included.
   - `stage7_review_queue.json` for missing/weak data.
   - `stage7_graph_overlay.png` if practical in the same pass.

7. Add QA/review output
   - Missing line numbers.
   - Dead ends.
   - Skipped/reused traces.
   - Weak or rejected associations.
   - Duplicate/near-duplicate nodes.
   - Unconnected equipment ports.

### Tests to run
- `cd backend && /Users/maetee/Code/GARNET/.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py`
- `cd backend && ./run_stage5b_only.sh /Users/maetee/Code/GARNET/backend/output_debug/Test-00001`
- Regenerate Stage 6 for Test-00001.
- Run the new Stage 7 for Test-00001 first.
- Then run Stage 7 for `Test-00001` through `Test-00009`.
- Check summaries for edge count, node count, skipped trace count, missing line number count, and review item count.

### Risks / edge cases
- Too-aggressive node merging can hide real nearby ports.
- Too-conservative node merging can fragment tees into disconnected nodes.
- Missing line numbers should not block graph creation, but must remain visible for review.
- Skipped/reused traces must not create duplicate physical pipe edges.
- Branch traces that end at the same tee from different directions must share a junction node.

## Stage 7 Detailed Design

### Stage 7 input contract
Primary input:
- `stage6_trace_associations.json`

Required top-level keys:
- `image_id`
- `trace_source`
- `trace_edges`
- `associations`
- `unresolved`

Required per physical trace edge:
- `trace_id`
- `trace_kind`
- `source_obj_id`
- `source_obj_type`
- `port`
- `terminal_type`
- `terminal_obj_id`
- `terminal_xy`
- `segments`
- `polyline`
- `trace_length_px`
- `status`
- `attachments`

Physical edge rule:
- Include trace edges with non-empty `segments`.
- Exclude records with `status = skipped_existing_trace` or no segments from physical graph edges.
- Preserve skipped/reused records in review and source-port metadata.

### Stage 7 graph schema proposal

`stage7_graph.json`:

```json
{
  "schema_version": "stage7_trace_graph_v1",
  "image_id": "Test-00001.jpg",
  "trace_source": "stage6",
  "nodes": [],
  "edges": [],
  "line_groups": [],
  "review_queue": [],
  "metadata": {}
}
```

Node shape:

```json
{
  "id": "node::tee::obj_000054",
  "type": "tee_junction",
  "position": {"x": 792, "y": 1634},
  "source_refs": ["obj_000054"],
  "trace_refs": ["obj_000197", "branch_000001"],
  "port_refs": [],
  "terminal_refs": [],
  "review_status": "ok"
}
```

Edge shape:

```json
{
  "id": "edge::obj_000197",
  "trace_id": "obj_000197",
  "source_node_id": "node::source::obj_000197",
  "target_node_id": "node::tee::obj_000054",
  "polyline": [[208, 978], [793, 979], [792, 1634]],
  "length_px": 1240,
  "line_number_candidates": [],
  "inline_objects": [],
  "instrument_tags": [],
  "flow_arrows": [],
  "terminal_type": "tee_junction",
  "terminal_obj_id": "obj_000054",
  "trace_kind": "port",
  "review_status": "review"
}
```

Skipped/reused source shape:

```json
{
  "id": "skipped::equip_1_vessel:port_03",
  "source_obj_id": "equip_1_vessel",
  "port_index": 3,
  "reused_trace_id": "obj_000194",
  "reason": "source_reached_by_existing_trace",
  "physical_edge_created": false
}
```

### Node creation rules

1. Source node from trace start
   - If source is equipment: node type `equipment_port`.
   - If source is page/utility/connection: node type `page_connection` or `connection` based on `source_obj_type`.
   - If source is branch: node type `branch_start`.
   - Position comes from `edge.port.x/y`.

2. Terminal node from trace terminal
   - `equipment` terminal -> node type `equipment` or `equipment_port` when terminal point is a known equipment port.
   - `page_connection` terminal -> node type `page_connection`.
   - `instrument_tag` terminal -> node type `instrument`.
   - `tee_junction` terminal -> node type `tee_junction`.
   - `branch_connection` terminal -> node type `tee_junction` or `branch_connection`.
   - `dead_end` terminal -> node type `dead_end`.
   - `sheet_edge` terminal -> node type `sheet_edge`.

3. Node merge rule
   - Merge by stable object id when available, e.g. same `terminal_obj_id` for a tee node.
   - Else merge by spatial bucket within a small tolerance.
   - Use different tolerances by type:
     - tee/branch junction: 12 px
     - equipment port: 16 px
     - page connector: 20 px
     - dead end: 8 px
   - Do not merge two equipment ports on the same equipment unless same point or explicit reused/skipped metadata indicates equivalence.

4. Provenance rule
   - Every node keeps `source_refs`, `trace_refs`, and `port_refs` where applicable.
   - Never drop the original trace id that caused the node.

### Edge creation rules

1. One physical edge per included Stage 6 trace edge.
2. Edge id is `edge::<trace_id>`.
3. Source and target node ids come from the node creation rules.
4. Edge carries Stage 6 attachments directly:
   - `line_numbers`
   - `inline_objects`
   - `instrument_tags`
   - `flow_arrows`
   - `equipment_ports`
   - `terminals`
5. Edge does not include skipped/reused trace records as geometry.
6. Edge review status:
   - `ok` if terminal is known and at least one strong semantic association exists where expected.
   - `review` if missing line number, dead end, weak/multiple associations, or ambiguous terminal.
   - `blocking` only for malformed geometry or missing required endpoint.

### Line group rules

Initial grouping should be conservative.

Group traces into `stage7_line_groups.json` using:
1. Same accepted line number association.
2. Shared connected node or terminal.
3. Same or compatible flow direction evidence.
4. Branch traces connected to the grouped line via tee junctions.

Line group shape:

```json
{
  "id": "line_group::3-CUL-25-002013-B1A2-NI",
  "line_number_raw": "3-CUL-25-002013-B1A2-NI",
  "line_number_normalized": "3-CUL-25-002013-B1A2-NI",
  "trace_ids": ["obj_000190", "obj_000193"],
  "node_ids": [],
  "equipment_ids": [],
  "inline_object_ids": [],
  "instrument_tag_ids": [],
  "review_status": "review",
  "warnings": []
}
```

If no line number exists:
- Create provisional groups by connectivity only.
- Use id pattern `line_group::unassigned::<component_id>`.
- Add review item `missing_line_number`.

### Review queue rules

`stage7_review_queue.json` should be the human-in-the-loop input surface.

Review item shape:

```json
{
  "id": "review::missing_line_number::obj_000197",
  "severity": "review",
  "category": "missing_line_number",
  "trace_id": "obj_000197",
  "node_id": null,
  "message": "Trace has no accepted line number association.",
  "suggested_action": "Assign line number or mark not required.",
  "evidence": {}
}
```

Initial categories:
- `missing_line_number`
- `dead_end_trace`
- `skipped_existing_trace`
- `weak_association`
- `multiple_line_candidates`
- `unattached_line_number`
- `unattached_instrument_tag`
- `unconnected_equipment_port`
- `ambiguous_terminal`
- `malformed_trace_geometry`

Severity policy:
- `blocking`: graph cannot safely use this trace/node.
- `review`: graph can proceed, but process engineer must review.
- `info`: no action required unless auditing provenance.

### Stage 7 artifact list

Core artifacts:
- `stage7_graph.json`
- `stage7_graph_summary.json`
- `stage7_trace_edge_nodes.json`
- `stage7_line_groups.json`
- `stage7_line_group_summary.json`
- `stage7_review_queue.json`
- `stage7_review_queue_summary.json`
- `stage7_graph_overlay.png`

Compatibility artifacts to preserve or map:
- `stage7_equipment_attachments.json`
- `stage7_connection_attachments.json`
- `stage7_edge_terminals.json`
- `stage7_text_attachments.json`
- `stage7_instrument_tag_attachments.json`

These can initially be derived from Stage 6 associations instead of recomputing from old geometric edges.

### Stage 7 summary fields

`stage7_graph_summary.json` should include:
- `node_count`
- `edge_count`
- `physical_trace_edge_count`
- `skipped_trace_count`
- `line_group_count`
- `unassigned_line_group_count`
- `equipment_node_count`
- `tee_junction_node_count`
- `dead_end_node_count`
- `page_connection_node_count`
- `review_item_count`
- `blocking_review_item_count`
- `missing_line_number_trace_count`
- `dead_end_trace_count`
- `source_artifacts`

### Implementation phases

Phase 7A: Minimal graph from Stage 6
- Add helper functions in `pid_extractor.py` or a new module such as `backend/garnet/trace_graph_builder.py`.
- Build nodes and edges from Stage 6 only.
- Emit graph, summary, node mapping, review queue.
- Do not attempt advanced line grouping yet.

Phase 7B: Compatibility outputs
- Derive existing Stage 7 attachment artifacts from Stage 6 associations.
- Keep downstream `stage7b_graph_export()` working.
- Confirm `graph_export_adapter.py` can consume the new `stage7_graph.json` fields.

Phase 7C: Provisional line groups
- Group by accepted line number first.
- Add unassigned connectivity groups second.
- Emit review items for missing/ambiguous line numbers.

Phase 7D: Overlay and QA refinement
- Draw graph nodes/edges with review colors.
- Add visible labels for line groups and problematic traces.
- Add counts by QA category.

### Acceptance criteria for first Stage 7 pass

Minimum acceptable first pass:
- Stage 7 runs on `Test-00001` using only Stage 6 input.
- `stage7_graph.json` contains physical edges from Stage 6 traces and no duplicate skipped edges.
- `stage7_graph_summary.json` reports node/edge/review counts.
- `stage7_review_queue.json` includes missing line numbers and dead ends.
- `stage7b_graph_export()` still runs or has a documented compatibility gap.

Full validation pass:
- Stage 7 runs on `Test-00001` through `Test-00009`.
- No runtime errors.
- Skipped/reused traces are not counted as physical edges.
- Known Stage 5b endpoints remain represented in graph nodes.
- Human review queue contains all Stage 6 unresolved items.

### First implementation target

Implement Phase 7A only first.

Recommended first code slice:
1. Add `backend/garnet/trace_graph_builder.py`.
2. Implement `build_trace_graph_from_stage6(payload, image_id)`.
3. Update `stage7_geometric_graph_assembly()` to use Stage 6 when `stage6_trace_associations.json` exists.
4. Emit only:
   - `stage7_graph.json`
   - `stage7_graph_summary.json`
   - `stage7_trace_edge_nodes.json`
   - `stage7_review_queue.json`
   - `stage7_review_queue_summary.json`
5. Run on `Test-00001` and inspect the graph payload before adding compatibility artifacts.

## Production HITL Additions

These are deferred for first production hardening, not part of the current Stage 6-10 implementation.

### Stage 3: Major Equipment Bounding Box HITL

Add a new Stage 3 review step for human-created or human-corrected major equipment bounding boxes.

Purpose:
- Ensure vessels, columns, exchangers, pumps, compressors, and other major equipment have reliable physical extents before port detection and topology association.
- Allow reviewers to add missing major equipment boxes, resize bad boxes, merge duplicate boxes, and assign stable equipment IDs/tags.

Expected output candidates:
- `stage3_equipment_review.json`
- `stage3_equipment_overlay.png`
- downstream equipment objects promoted into Stage 4/5 context.

### Stage 4: Object Detection HITL

Add HITL review for object detection and semantic text classes.

Review scope:
- detected inline objects and fittings
- line numbers
- instrument tags
- equipment tags
- page/utility connections
- false positives and missing objects

Required reviewer actions:
- accept/reject object detections
- correct class names
- edit bounding boxes
- add missed objects
- correct OCR text for line numbers and tags

Expected output candidates:
- `stage4_object_review.json`
- `stage4_line_number_review.json`
- `stage4_instrument_tag_review.json`
- reviewed objects become the source for Stage 5/5b and later associations.

### Stage 5: Pipe/Port Geometry HITL

Add HITL review for pipe mask and object port geometry before path tracing.

Review scope:
- add/remove/edit object ports
- remove invalid detected pipe lines
- mark non-pipe drawing lines/text artifacts
- correct pipe gaps or bridge/crossing ambiguity where needed
- optionally force known connection directions for equipment ports

Required reviewer actions:
- add missing ports on equipment, inline objects, page connections, and instruments
- delete invalid ports
- correct port direction and snap point
- mask out invalid pipe segments/noise
- annotate explicit keep/remove regions for pipe geometry

Expected output candidates:
- `stage5_port_review.json`
- `stage5_pipe_geometry_review.json`
- `stage5_review_overlay.png`
- reviewed ports and pipe geometry become authoritative inputs for Stage 5b tracing.
