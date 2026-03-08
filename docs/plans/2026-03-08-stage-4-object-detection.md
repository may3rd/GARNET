# Stage 4 Object Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a fixed-baseline Stage 4 Ultralytics + SAHI object-detection slice to the rebuilt image-only pipeline and expose it through the existing pipeline API/frontend review flow.

**Architecture:** Keep `backend/garnet/pid_extractor.py` as the orchestrator, add one small detection helper for pipeline-native object artifacts, and extend the pipeline job/API/frontend flow to support sparse stage numbering with `stop_after=4`. Reuse the existing detection baseline and keep all user-facing detector controls out of Pipeline mode for this first slice.

**Tech Stack:** Python, FastAPI, Ultralytics, SAHI, NumPy, OpenCV, React, Zustand, TypeScript

---

### Task 1: Lock the sparse Stage 4 contract in tests

**Files:**
- Modify: `backend/tests/test_pid_extractor_cli.py`
- Modify: `backend/tests/test_pipeline_api.py`

**Step 1: Write the failing tests**

Add tests that require:
- `_stage_definitions()` to include `stage4_object_detection`
- `run(stop_after=4)` to execute Stage 1, Stage 2, and Stage 4
- Stage 4 artifacts to exist in the pipeline job bundle

**Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v`

Expected: FAIL because the runner only supports up to Stage 2.

**Step 3: Write minimal implementation**

Do not implement detection yet. Only update tests.

**Step 4: Run test to verify it fails for the right reason**

Run the same command and confirm the failure is about missing Stage 4 behavior, not a syntax error.

**Step 5: Commit**

```bash
git add backend/tests/test_pid_extractor_cli.py backend/tests/test_pipeline_api.py
git commit -m "test: lock stage 4 pipeline contract"
```

### Task 2: Add the pipeline object-detection helper

**Files:**
- Create: `backend/garnet/object_detection_sahi.py`
- Test: `backend/tests/test_object_detection_sahi.py`

**Step 1: Write the failing test**

Add a helper-level test that requires:
- fixed-baseline summary fields
- object JSON shape with `id`, `class_name`, `confidence`, `bbox`, `source_model`, `source_weight`
- overlay image generation

**Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_object_detection_sahi.py' -v`

Expected: FAIL because the helper file does not exist yet.

**Step 3: Write minimal implementation**

Implement a helper that:
- loads the raw image
- initializes `AutoDetectionModel.from_pretrained(...)`
- calls `get_sliced_prediction(...)`
- converts results into:
  - `objects_payload`
  - `summary`
  - `overlay_image`

**Step 4: Run test to verify it passes**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add backend/garnet/object_detection_sahi.py backend/tests/test_object_detection_sahi.py
git commit -m "feat: add stage 4 object detection helper"
```

### Task 3: Wire Stage 4 into the pipeline runner

**Files:**
- Modify: `backend/garnet/pid_extractor.py`
- Test: `backend/tests/test_pid_extractor_cli.py`

**Step 1: Write the failing test**

Add or extend tests to require:
- `stage4_object_detection()` writes:
  - `stage4_objects.json`
  - `stage4_objects_summary.json`
  - `stage4_objects_overlay.png`
- Stage 4 summary contains the fixed baseline weight path

**Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v`

Expected: FAIL because `stage4_object_detection` does not exist yet.

**Step 3: Write minimal implementation**

Update the runner to:
- include `stage4_object_detection`
- validate `stop_after` against the highest implemented stage number
- execute all stages whose numeric id is `<= stop_after`
- write the Stage 4 artifact bundle using the new helper

**Step 4: Run test to verify it passes**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add backend/garnet/pid_extractor.py backend/tests/test_pid_extractor_cli.py
git commit -m "feat: add stage 4 object detection to pipeline"
```

### Task 4: Extend the pipeline API contract to Stage 4

**Files:**
- Modify: `backend/api.py`
- Test: `backend/tests/test_pipeline_api.py`

**Step 1: Write the failing test**

Add a test for:
- `POST /api/pipeline/jobs` with `stop_after=4`
- completed job includes Stage 4 artifacts

**Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_pipeline_api.py' -v`

Expected: FAIL because the API currently rejects `stop_after=4`.

**Step 3: Write minimal implementation**

Update the API to:
- allow `stop_after=4`
- stop deriving `current_stage` from `stop_after - 1`
- keep current stage updates driven by callbacks and manifest state

**Step 4: Run test to verify it passes**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_pipeline_api.py
git commit -m "feat: expose stage 4 through pipeline job api"
```

### Task 5: Update the frontend pipeline review flow

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/stores/appStore.ts`
- Modify: `frontend/src/components/DetectionSetup.tsx`
- Modify: `frontend/src/components/PipelineResultsView.tsx`

**Step 1: Write the failing test or review assertion**

If there are no frontend tests, use the build as the red/green guard:
- change the UI copy/logic first
- expect build or type errors until all references are updated

**Step 2: Run build to verify the incomplete state fails or is inconsistent**

Run: `cd frontend && bun run build`

Expected: either build failure while wiring or manual mismatch in the review copy/stopAfter constant.

**Step 3: Write minimal implementation**

Update the frontend to:
- default pipeline `stopAfter` to `4`
- progress based on the manifest stage list length when available
- update review copy from “Stage 1 and Stage 2” to include Stage 4 object detection

**Step 4: Run build to verify it passes**

Run: `cd frontend && bun run build`

Expected: PASS

**Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/stores/appStore.ts frontend/src/components/DetectionSetup.tsx frontend/src/components/PipelineResultsView.tsx
git commit -m "feat: show stage 4 object detection in pipeline ui"
```

### Task 6: Update docs and run end-to-end verification

**Files:**
- Modify: `IMPLEMENTATION_TRACKER.md`
- Modify: `backend/garnet/AGENTS.md`
- Modify: `SLICE_2_PROGRESS.md` or add a new Stage 4 progress log if preferred

**Step 1: Update docs**

Record:
- fixed baseline weight
- Stage 4 artifacts
- sparse stage numbering rule

**Step 2: Run backend verification**

Run:

```bash
cd backend
../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py
../.venv/bin/python -m unittest discover -s tests -p 'test_object_detection_sahi.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_pipeline_api.py' -v
```

Expected: PASS

**Step 3: Run frontend verification**

Run:

```bash
cd frontend
bun run build
```

Expected: PASS

**Step 4: Run real sample**

Run:

```bash
cd backend
XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/stage4_sample --ocr-route easyocr --stop-after 4
```

Expected:
- Stage 4 completes
- output includes `stage4_objects.json`
- output includes `stage4_objects_overlay.png`
- output includes `stage4_objects_summary.json`

**Step 5: Commit**

```bash
git add IMPLEMENTATION_TRACKER.md backend/garnet/AGENTS.md SLICE_2_PROGRESS.md docs/plans/2026-03-08-stage-4-object-detection-design.md docs/plans/2026-03-08-stage-4-object-detection.md
git commit -m "docs: record stage 4 object detection slice"
```
