# Stage 4 Object Detection HITL Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add HITL review/edit/save support for Stage 4 object detections so reviewed object boxes become the source of truth for downstream stages.

**Architecture:** Reuse the existing full pipeline review canvas for bounding-box editing. Convert edited canvas objects back to the existing `stage4_objects.json` schema, save it through the existing artifact update endpoint, and mark Stage 4 fusion plus Stage 5+ stale. Backend review-state validation will accept a new `stage4_object` bucket.

**Tech Stack:** React/Vite frontend, existing pipeline artifact API, FastAPI artifact invalidation, Python unittest.

---

### Task 1: Backend Review-State Bucket

**Files:**
- Modify: `backend/garnet/review_state.py`
- Test: `backend/tests/test_review_state.py`

**Steps:**
1. Add a failing test that `stage4_object` appears in default review-state buckets and saves/loads.
2. Run `../.venv/bin/python3 -m unittest discover -s tests -p 'test_review_state.py' -v` and confirm failure.
3. Add `stage4_object` to `VALID_BUCKETS` and `empty_review_state`.
4. Rerun the targeted test and confirm pass.

### Task 2: Stage 4 Artifact Invalidation

**Files:**
- Modify: `backend/api.py`
- Test: `backend/tests/test_pipeline_api.py`

**Steps:**
1. Add a failing API test that updating `stage4_objects.json` marks `stage4_line_number_fusion`, `stage4_instrument_tag_fusion`, Stage 5, Stage 5b, and Stage 6 stale.
2. Add `stage4_objects.json` to `ARTIFACT_INVALIDATION_START_STAGE` starting at `stage4_line_number_fusion`.
3. Add downstream artifacts to `STALE_ARTIFACTS_BY_SOURCE['stage4_objects.json']`.
4. Rerun targeted API tests.

### Task 3: Frontend Stage 4 Object Review Bucket

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/components/PipelineResultsView.tsx`
- Modify: `frontend/src/components/PipelineHitlReviewView.tsx`

**Steps:**
1. Add `stage4_object` to `PipelineReviewBucket`.
2. Build review items from `stage4_objects.json`.
3. Seed the existing full review workspace from Stage 4 objects.
4. Add a “Stage 4 Objects” review-flow tile.
5. Add save handler that converts edited canvas objects to `stage4_objects.json` shape and PUTs the artifact.
6. After save, refresh job/status and show stale downstream stages.

### Task 4: Verification

**Commands:**
- `cd backend && ../.venv/bin/python3 -m py_compile api.py garnet/*.py garnet/utils/*.py garnet/path_tracer/*.py`
- `cd backend && ../.venv/bin/python3 -m unittest discover -s tests -p 'test_review_state.py' -v`
- `cd backend && ../.venv/bin/python3 -m unittest discover -s tests -p 'test_pipeline_api.py' -v`
- `cd frontend && bun run lint`
- `cd frontend && bun run build`
