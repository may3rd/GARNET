# Unified Pipeline HITL Review Design

## Goal

Replace the pipeline summary-first review flow with a single review workspace where the user edits the evidence that drives Stage 5, Stage 5b, and Stage 6 in one canvas.

The workspace is the human-in-the-loop source of truth for:

- Stage 3 equipment bounding boxes
- Stage 4 detected objects
- Stage 5 equipment/object ports
- Stage 5b traced paths and branch paths
- Stage 6 line number associations
- Manual additions, deletions, and invalidations

## Product Behavior

After a full pipeline run, the frontend should continue into the unified Review page instead of forcing the user into a summary page. The summary/artifact page can remain available as a secondary QA/details view.

The Review page shows the original drawing with toggleable overlays:

- equipment boxes
- detected objects
- connection ports
- traced paths
- branch traced paths
- line numbers and instrument tags
- Stage 6 trace-to-line associations
- QA warnings or stale indicators

The user can edit, add, delete, accept, or reject entities in a single UI. When the user changes equipment, connection objects, ports, or trace-related objects, the frontend updates the canvas immediately and schedules a backend recompute after 500 ms of inactivity. A visible `Recompute now` button bypasses the debounce.

## Execution Model

Use backend recompute. Do not port the Python tracer to TypeScript.

Reasoning:

- The Python path tracer is the authoritative implementation and has many tuned edge cases.
- A TypeScript tracer would create duplicate behavior and validation drift.
- Backend recompute keeps frontend review output aligned with batch/full-pipeline output.

Initial implementation should rerun the affected Stage 5 to Stage 6 slice as a safe baseline:

1. Save current draft review state.
2. Derive reviewed Stage 3 and Stage 4 artifacts from the draft state.
3. Recompute Stage 5 connection ports, Stage 5b traces, Stage 5b branch traces, and Stage 6 trace associations.
4. Return refreshed layer payloads to the frontend.

Later optimization can add route-level dirty recompute with trace caching, but the first implementation should prioritize correctness and UX consistency.

## Review Workspace State

Add one normalized draft artifact:

`review_workspace_state.json`

Suggested shape:

```json
{
  "version": 1,
  "updated_at": 0,
  "objects": [],
  "equipment": [],
  "manual_ports": [],
  "deleted_entities": [],
  "line_association_overrides": [],
  "trace_overrides": []
}
```

This avoids splitting review truth across localStorage, review-state decisions, and individual stage artifacts.

Canonical artifacts are still produced from the workspace state:

- `stage3_equipment_bboxes.json`
- `stage4_objects.json`
- `stage5_connection_ports.json`
- `stage5b_trace_results.json`
- `stage5b_branch_trace_results.json`
- `stage6_trace_associations.json`
- `stage6_line_number_review.json`

## Backend API

Add draft workspace endpoints:

- `GET /api/pipeline/jobs/{job_id}/review-workspace`
- `PUT /api/pipeline/jobs/{job_id}/review-workspace`
- `POST /api/pipeline/jobs/{job_id}/review-workspace/recompute`
- `POST /api/pipeline/jobs/{job_id}/review-workspace/commit`

`recompute` accepts the draft workspace state and a scope, initially `stage5_to_6`. It writes or updates draft artifacts and returns layer payloads plus stage statuses.

`commit` persists the reviewed workspace as canonical artifacts and marks Stage 7+ stale.

## Frontend Architecture

Create a dedicated unified review component rather than continuing to enlarge `PipelineResultsView`.

Proposed components:

- `PipelineReviewWorkspaceView`
- `ReviewCanvasLayers`
- `ReviewLayerPanel`
- `ReviewEntityInspector`
- `ReviewRecomputeStatus`

The existing `CanvasView` can be reused for object editing, but it needs a richer layer model because traced paths and ports are not rectangular detections.

## Frontend Data Flow

1. Load pipeline job and artifacts.
2. Load or initialize `review_workspace_state.json` from Stage 3/4/5/5b/6 artifacts.
3. Render all review layers in one canvas.
4. User edits local workspace state.
5. Frontend schedules recompute after 500 ms idle.
6. Backend recomputes Stage 5 to Stage 6 and returns updated layers.
7. User confirms/commits, which persists canonical artifacts and invalidates Stage 7+.

## Error Handling

- Recompute failures should not discard local edits.
- Show last successful recompute time and current dirty state.
- If recompute fails, keep the old trace layers visible but mark them stale.
- `Recompute now` should be disabled while a recompute is in flight unless cancellation is implemented.

## First Implementation Slice

1. Add backend workspace state artifact helpers and API endpoints.
2. Add backend recompute endpoint that reruns Stage 5, Stage 5b, and Stage 6 from reviewed Stage 3/4 inputs.
3. Add frontend unified Review Workspace route/view.
4. Render existing Stage 3/4/5/5b/6 artifacts as layers.
5. Add edit support for equipment and Stage 4 objects first.
6. Add debounced recompute and visible `Recompute now`.
7. Add commit action that persists canonical artifacts and marks Stage 7+ stale.

## Deferred Work

- Incremental route-level trace recompute.
- TypeScript preview tracer.
- Advanced conflict resolution for duplicate traces.
- Multi-user review locking.
- Full QA review overlays beyond Stage 6.
