# Unified Pipeline HITL Review Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a single pipeline review workspace where users edit equipment/objects/ports/traces/line associations and trigger debounced backend recompute for Stage 5, Stage 5b, and Stage 6.

**Architecture:** The frontend owns one draft review workspace state and renders all reviewable entities as canvas layers. The backend remains the authoritative computation engine: it derives reviewed Stage 3/4 artifacts from the workspace state, reruns Stage 5 to Stage 6, and returns refreshed layer payloads. The first slice does full Stage 5 to Stage 6 recompute on each debounced update; route-level incremental recompute is deferred.

**Tech Stack:** FastAPI, Python pipeline artifacts, React/Vite/TypeScript, existing `CanvasView`, existing pipeline job artifact APIs.

---

### Task 1: Add Backend Review Workspace Schema Helpers

**Files:**
- Create: `backend/garnet/review_workspace.py`
- Test: `backend/tests/test_review_workspace.py`

**Step 1: Write failing tests**

Create tests for:

- default workspace state contains `objects`, `equipment`, `manual_ports`, `deleted_entities`, `line_association_overrides`, and `trace_overrides`.
- building workspace state from existing Stage 3/4/5/5b/6 artifacts preserves IDs and bbox fields.
- deriving Stage 3 equipment artifact from workspace equipment.
- deriving Stage 4 objects artifact from workspace objects, excluding rejected entities.

Use `unittest` to match current backend tests.

**Step 2: Run tests and verify failure**

Run from `backend/`:

```bash
../.venv/bin/python3 -m unittest discover -s tests -p 'test_review_workspace.py' -v
```

Expected: fails because `garnet.review_workspace` does not exist.

**Step 3: Implement minimal helper module**

Implement in `backend/garnet/review_workspace.py`:

- `empty_review_workspace(job_id: str | None = None) -> dict[str, Any]`
- `load_review_workspace(job_dir: str | Path) -> dict[str, Any]`
- `save_review_workspace(job_dir: str | Path, state: dict[str, Any]) -> dict[str, Any]`
- `build_workspace_from_artifacts(job_dir: str | Path) -> dict[str, Any]`
- `workspace_to_stage3_equipment(state: dict[str, Any]) -> dict[str, Any]`
- `workspace_to_stage4_objects(state: dict[str, Any], image_id: str | None = None) -> dict[str, Any]`

Keep schema permissive and JSON-native. Do not add pydantic models yet.

**Step 4: Run tests and verify pass**

Run:

```bash
../.venv/bin/python3 -m unittest discover -s tests -p 'test_review_workspace.py' -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/garnet/review_workspace.py backend/tests/test_review_workspace.py
git commit -m "feat(pipeline): add review workspace state helpers"
```

---

### Task 2: Add Backend Review Workspace API Endpoints

**Files:**
- Modify: `backend/api.py`
- Test: `backend/tests/test_pipeline_api.py`

**Step 1: Write failing API tests**

Add tests for:

- `GET /api/pipeline/jobs/{job_id}/review-workspace` returns existing `review_workspace_state.json` or initializes from artifacts.
- `PUT /api/pipeline/jobs/{job_id}/review-workspace` persists draft state.
- invalid job id returns 404.

**Step 2: Run test and verify failure**

Run:

```bash
../.venv/bin/python3 -m unittest discover -s tests -p 'test_pipeline_api.py' -v
```

Expected: new endpoint tests fail with 404.

**Step 3: Implement endpoints**

Add to `backend/api.py`:

- `GET /api/pipeline/jobs/{job_id}/review-workspace`
- `PUT /api/pipeline/jobs/{job_id}/review-workspace`

Use `review_workspace.py` helpers. Return JSON with:

```json
{
  "job_id": "...",
  "workspace": {},
  "artifact": {"name": "review_workspace_state.json", "url": "..."}
}
```

Do not run recompute in this task.

**Step 4: Run tests and verify pass**

Run:

```bash
../.venv/bin/python3 -m unittest discover -s tests -p 'test_pipeline_api.py' -v
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_pipeline_api.py
git commit -m "feat(api): expose pipeline review workspace state"
```

---

### Task 3: Add Backend Stage 5-to-6 Recompute Endpoint

**Files:**
- Modify: `backend/api.py`
- Modify: `backend/garnet/pid_extractor.py` if stage runner helpers are needed
- Test: `backend/tests/test_pipeline_api.py`

**Step 1: Write failing API test**

Create a temp pipeline job with enough minimal artifacts to exercise artifact writing/invalidation. Test:

- `POST /api/pipeline/jobs/{job_id}/review-workspace/recompute` saves workspace state.
- It writes reviewed `stage3_equipment_bboxes.json` and `stage4_objects.json` from workspace.
- It marks or reruns Stage 5/5b/6 depending on availability.
- It returns `workspace`, `stages`, and `layers` keys.

For unit scope, mock the heavy pipeline execution function so the test does not require YOLO/OCR/tracing.

**Step 2: Run test and verify failure**

Run:

```bash
../.venv/bin/python3 -m unittest discover -s tests -p 'test_pipeline_api.py' -v
```

Expected: endpoint missing.

**Step 3: Implement recompute endpoint**

Add:

`POST /api/pipeline/jobs/{job_id}/review-workspace/recompute`

Behavior:

1. Save incoming workspace state to `review_workspace_state.json`.
2. Write derived `stage3_equipment_bboxes.json` and `stage4_objects.json`.
3. Refresh Stage 4 summary/topology marker artifacts using existing helper.
4. Run/resume from `stage5_pipe_mask` or `stage5b_pipe_trace` depending on chosen scope.
5. Return current artifacts needed for frontend layers:
   - `stage5_connection_ports.json`
   - `stage5b_trace_results.json`
   - `stage5b_branch_trace_results.json`
   - `stage6_trace_associations.json`
   - `stage6_line_number_review.json`

Implementation can initially call the existing resume mechanism synchronously for Stage 5 to Stage 6 if the pipeline job runner supports stop-after preservation. If that is too coupled, add a small internal function that instantiates `PIDExtractor` with the job config and runs stage methods in order.

**Step 4: Run tests and compile**

Run:

```bash
../.venv/bin/python3 -m unittest discover -s tests -p 'test_pipeline_api.py' -v
../.venv/bin/python3 -m py_compile api.py garnet/*.py garnet/utils/*.py garnet/path_tracer/*.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/api.py backend/garnet/pid_extractor.py backend/tests/test_pipeline_api.py
git commit -m "feat(api): recompute reviewed pipeline layers"
```

---

### Task 4: Add Frontend API Client and Types

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/lib/api.ts`

**Step 1: Add TypeScript types**

Add:

- `PipelineReviewWorkspaceState`
- `PipelineReviewWorkspaceResponse`
- `PipelineReviewRecomputeResponse`
- layer payload types as permissive `Record<string, unknown>` for first slice.

**Step 2: Add API client functions**

Add:

- `getPipelineReviewWorkspace(jobId)`
- `putPipelineReviewWorkspace(jobId, workspace)`
- `recomputePipelineReviewWorkspace(jobId, workspace, scope)`

**Step 3: Run frontend lint/build**

Run from `frontend/`:

```bash
bun run lint
bun run build
```

Expected: PASS.

**Step 4: Commit**

```bash
git add frontend/src/types.ts frontend/src/lib/api.ts
git commit -m "feat(frontend): add review workspace API client"
```

---

### Task 5: Create Unified Pipeline Review Workspace View

**Files:**
- Create: `frontend/src/components/PipelineReviewWorkspaceView.tsx`
- Modify: `frontend/src/components/PipelineResultsView.tsx`

**Step 1: Create component skeleton**

Build `PipelineReviewWorkspaceView` with:

- left/main canvas area
- top toolbar with layer toggles and recompute status
- right inspector panel placeholder
- `Recompute now` button
- `Back to artifacts` or `Summary` button

Reuse current base image selection behavior from `PipelineHitlReviewView`.

**Step 2: Wire entry point**

Modify `PipelineResultsView` so completed pipeline jobs default to the workspace view, while the current summary page remains accessible as an artifact/details mode.

**Step 3: Run frontend checks**

Run:

```bash
bun run lint
bun run build
```

Expected: PASS.

**Step 4: Commit**

```bash
git add frontend/src/components/PipelineReviewWorkspaceView.tsx frontend/src/components/PipelineResultsView.tsx
git commit -m "feat(frontend): add unified pipeline review workspace"
```

---

### Task 6: Render Review Layers on One Canvas

**Files:**
- Modify: `frontend/src/components/PipelineReviewWorkspaceView.tsx`
- Optionally create: `frontend/src/components/ReviewCanvasLayers.tsx`

**Step 1: Build layer normalization helpers**

Normalize artifacts into layer entities:

- object/equipment boxes
- port points
- trace polylines
- branch trace polylines
- line number labels

Use permissive parsing so missing artifacts do not crash the UI.

**Step 2: Render layers**

Add SVG/canvas overlay rendering above the image. Use color-coded layers and toggles.

**Step 3: Selection behavior**

Clicking an entity selects it and shows its properties in the right inspector.

**Step 4: Run frontend checks**

Run:

```bash
bun run lint
bun run build
```

Expected: PASS.

**Step 5: Commit**

```bash
git add frontend/src/components/PipelineReviewWorkspaceView.tsx frontend/src/components/ReviewCanvasLayers.tsx
git commit -m "feat(frontend): render unified review layers"
```

---

### Task 7: Add Edit/Add/Delete for Equipment and Objects

**Files:**
- Modify: `frontend/src/components/PipelineReviewWorkspaceView.tsx`
- Reuse or modify: `frontend/src/components/CanvasView.tsx` only if required

**Step 1: Add editing operations**

Support:

- move/resize equipment boxes
- move/resize Stage 4 object boxes
- add equipment box
- add object box
- delete/reject selected entity

Update local `PipelineReviewWorkspaceState` only. Do not call backend on every mouse event.

**Step 2: Add dirty tracking**

Set workspace dirty after any edit. Show `Unsaved changes` and `Trace layers stale` indicators.

**Step 3: Run frontend checks**

Run:

```bash
bun run lint
bun run build
```

Expected: PASS.

**Step 4: Commit**

```bash
git add frontend/src/components/PipelineReviewWorkspaceView.tsx frontend/src/components/CanvasView.tsx
git commit -m "feat(frontend): edit pipeline review entities"
```

---

### Task 8: Add Debounced Recompute and Manual Recompute Button

**Files:**
- Modify: `frontend/src/components/PipelineReviewWorkspaceView.tsx`

**Step 1: Add debounce**

When relevant workspace state changes, schedule recompute after 500 ms. Cancel and reschedule while edits continue.

**Step 2: Add request state**

Track:

- `idle`
- `scheduled`
- `running`
- `succeeded`
- `failed`

Show status in toolbar.

**Step 3: Wire `Recompute now`**

Button cancels pending debounce and calls recompute immediately.

**Step 4: Merge response**

On success, update returned layer payloads and stage statuses without discarding current workspace state.

**Step 5: Run frontend checks**

Run:

```bash
bun run lint
bun run build
```

Expected: PASS.

**Step 6: Commit**

```bash
git add frontend/src/components/PipelineReviewWorkspaceView.tsx
git commit -m "feat(frontend): debounce pipeline review recompute"
```

---

### Task 9: Add Commit Reviewed Workspace Action

**Files:**
- Modify: `backend/api.py`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/components/PipelineReviewWorkspaceView.tsx`
- Test: `backend/tests/test_pipeline_api.py`

**Step 1: Backend test**

Test:

- `POST /api/pipeline/jobs/{job_id}/review-workspace/commit` writes canonical artifacts.
- It marks Stage 7+ stale.
- It returns updated stage statuses.

**Step 2: Implement backend endpoint**

Persist reviewed artifacts as canonical and mark Stage 7+ stale.

**Step 3: Add frontend client and button**

Add `Commit reviewed workspace` button. Disable while recompute is running.

**Step 4: Run checks**

Run:

```bash
cd backend && ../.venv/bin/python3 -m unittest discover -s tests -p 'test_pipeline_api.py' -v
cd backend && ../.venv/bin/python3 -m py_compile api.py garnet/*.py garnet/utils/*.py garnet/path_tracer/*.py
cd frontend && bun run lint
cd frontend && bun run build
```

Expected: PASS.

**Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_pipeline_api.py frontend/src/lib/api.ts frontend/src/components/PipelineReviewWorkspaceView.tsx
git commit -m "feat(pipeline): commit unified review workspace"
```

---

### Task 10: Final Verification on Test Images

**Files:**
- No source edits unless defects are found.

**Step 1: Run backend and frontend checks**

Run:

```bash
cd backend && ../.venv/bin/python3 -m unittest discover -s tests -p 'test*.py' -v
cd backend && ../.venv/bin/python3 -m py_compile api.py garnet/*.py garnet/utils/*.py garnet/path_tracer/*.py
cd frontend && bun run lint
cd frontend && bun run build
```

**Step 2: Run pipeline smoke test**

Run the current pipeline command/script for one representative test image, then inspect that the review workspace loads and recompute updates Stage 5/5b/6 layers.

**Step 3: Document any remaining issues**

If any known limitations remain, append them to `punch_list.md` or the relevant plan doc.

**Step 4: Commit final fixes**

```bash
git add <changed source files only>
git commit -m "test: verify unified pipeline review workspace"
```
