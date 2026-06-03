# Stage 6 Line Association HITL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a canvas-based Stage 6 HITL workflow so reviewers can click traced paths, assign/confirm line numbers, and then resume downstream graph/export stages.

**Architecture:** Reuse existing pipeline artifacts and update only the Stage 6 review artifact. The UI will render the Stage 6 association overlay plus SVG trace polylines on top for interactive selection, then write reviewed line-number assignments to `stage6_line_number_review.json`. The backend artifact invalidation will mark Stage 7+ stale after Stage 6 review changes.

**Tech Stack:** FastAPI artifact update endpoint, React/Vite frontend, existing pipeline job artifact API, SVG overlay for trace path hit targets.

---

### Task 1: Backend Invalidation For Stage 6 Review

**Files:**
- Modify: `backend/api.py`

**Steps:**
1. Add `stage6_line_number_review.json` to `ARTIFACT_INVALIDATION_START_STAGE`, mapped to `stage7_geometric_graph_assembly`.
2. Add downstream Stage 7-11 artifacts to `STALE_ARTIFACTS_BY_SOURCE` for `stage6_line_number_review.json` so stale downstream outputs are removed or marked.
3. Run backend py_compile.

**Verification:**
- `cd backend && ../.venv/bin/python3 -m py_compile api.py garnet/*.py garnet/utils/*.py garnet/path_tracer/*.py`

### Task 2: Stage 6 Review Data Types And Builders

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/components/PipelineResultsView.tsx`

**Steps:**
1. Add `stage6_line_association` to `PipelineReviewBucket`.
2. Load `stage6_trace_associations.json` and `stage6_line_number_review.json` in PipelineResultsView detail artifacts.
3. Build per-trace review items from `trace_edges`, including current line-number attachment if present and a missing state when absent.
4. Include Stage 6 counts in the HITL summary.

**Verification:**
- `cd frontend && bun run build`

### Task 3: Canvas-Based Stage 6 Trace Editor

**Files:**
- Create: `frontend/src/components/Stage6LineAssociationReview.tsx`
- Modify: `frontend/src/components/PipelineResultsView.tsx`

**Steps:**
1. Create a review component that accepts the Stage 6 payload, existing review payload, image artifact URL, current decisions, and save callback.
2. Render the `stage6_trace_association_overlay.png` image.
3. Render SVG polylines from Stage 6 `trace_edges` over the image.
4. Use thick transparent SVG strokes for click targets.
5. On click, select a trace and show editable line-number field, existing attachments, trace metadata, and accept/reject/defer buttons.
6. Keep unassigned traces visibly highlighted.

**Verification:**
- Manual browser/UI review or frontend build if browser is not launched.

### Task 4: Save Stage 6 Line Review Artifact

**Files:**
- Modify: `frontend/src/components/PipelineResultsView.tsx`
- Modify: `frontend/src/components/Stage6LineAssociationReview.tsx`

**Steps:**
1. Convert edited trace line numbers into `stage6_line_number_review.json` shape: `accepted`, `needs_review`, and `traces_without_line_number`.
2. Preserve original reviewed fields when possible.
3. Call `putPipelineArtifact(jobId, 'stage6_line_number_review.json', payload)`.
4. Refresh job and stage status.
5. Show “Resume from Stage 7” when Stage 7+ is stale.

**Verification:**
- Backend artifact endpoint returns stages with Stage 7+ stale.
- `cd frontend && bun run build`.

### Task 5: Final Checks

**Files:**
- All touched files

**Steps:**
1. Run backend compile.
2. Run backend CLI tests.
3. Run frontend build.
4. Review git diff and avoid generated artifacts.

**Verification:**
- `cd backend && ../.venv/bin/python3 -m py_compile api.py garnet/*.py garnet/utils/*.py garnet/path_tracer/*.py`
- `cd backend && ../.venv/bin/python3 -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v`
- `cd frontend && bun run build`
