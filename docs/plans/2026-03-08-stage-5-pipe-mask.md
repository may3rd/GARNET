# Stage 5 Pipe Mask Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a conservative Stage 5 pipe-mask generation slice that converts the current image and evidence bundle into a reviewable provisional pipe mask with visible artifacts and metrics.

**Architecture:** Keep `backend/garnet/pid_extractor.py` as the orchestrator, add a focused `pipe_mask.py` helper for mask construction and suppression, and extend the pipeline runner/API/frontend flow to support sparse `stop_after=5`. Stage 5 intentionally stops at mask generation and defers morphology sealing and skeletonization to later stages.

**Tech Stack:** Python, NumPy, OpenCV, FastAPI, React, Zustand, TypeScript

---

### Task 1: Lock the Stage 5 contract in tests

**Files:**
- Modify: `backend/tests/test_pid_extractor_cli.py`
- Modify: `backend/tests/test_pipeline_api.py`

**Step 1: Write the failing tests**

Add tests that require:
- `_stage_definitions()` to include `stage5_pipe_mask`
- `run(stop_after=5)` to execute Stage 1, Stage 2, Stage 4, and Stage 5
- pipeline job artifact bundle to include Stage 5 outputs

**Step 2: Run test to verify it fails**

Run:

```bash
cd backend
../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v
../.venv/bin/python -m unittest discover -s tests -p 'test_pipeline_api.py' -v
```

Expected: FAIL because Stage 5 does not exist yet.

**Step 3: Write minimal implementation**

Do not implement Stage 5 code yet. Only update tests.

**Step 4: Run test to verify it fails for the right reason**

Confirm failures are about missing Stage 5 behavior, not syntax errors.

**Step 5: Commit**

```bash
git add backend/tests/test_pid_extractor_cli.py backend/tests/test_pipeline_api.py
git commit -m "test: lock stage 5 pipe mask contract"
```

### Task 2: Add the pipe-mask helper

**Files:**
- Create: `backend/garnet/pipe_mask.py`
- Create: `backend/tests/test_pipe_mask.py`

**Step 1: Write the failing test**

Add helper tests for:
- OCR suppression removes masked regions from a candidate mask
- tiny-component filtering removes obvious specks
- overlay rendering works

**Step 2: Run test to verify it fails**

Run:

```bash
cd backend
../.venv/bin/python -m unittest discover -s tests -p 'test_pipe_mask.py' -v
```

Expected: FAIL because `pipe_mask.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement a helper that:
- accepts current artifacts and config values
- builds a candidate mask
- suppresses OCR/object regions conservatively
- filters tiny components
- returns:
  - `mask_image`
  - `overlay_image`
  - `summary`

**Step 4: Run test to verify it passes**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add backend/garnet/pipe_mask.py backend/tests/test_pipe_mask.py
git commit -m "feat: add stage 5 pipe mask helper"
```

### Task 3: Wire Stage 5 into the pipeline runner

**Files:**
- Modify: `backend/garnet/pid_extractor.py`
- Modify: `backend/tests/test_pid_extractor_cli.py`

**Step 1: Write the failing test**

Add or extend tests to require:
- `stage5_pipe_mask()` writes:
  - `stage5_pipe_mask.png`
  - `stage5_pipe_mask_overlay.png`
  - `stage5_pipe_mask_summary.json`
- `stop_after=5` is accepted

**Step 2: Run test to verify it fails**

Run:

```bash
cd backend
../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v
```

Expected: FAIL because the runner does not yet expose Stage 5.

**Step 3: Write minimal implementation**

Update `pid_extractor.py` to:
- add `stage5_pipe_mask`
- read Stage 2 OCR JSON and Stage 4 object JSON as suppression inputs
- call the helper
- save Stage 5 artifacts
- extend sparse-stage validation to include `5`

**Step 4: Run test to verify it passes**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add backend/garnet/pid_extractor.py backend/tests/test_pid_extractor_cli.py
git commit -m "feat: add stage 5 pipe mask to pipeline"
```

### Task 4: Extend the pipeline API to Stage 5

**Files:**
- Modify: `backend/api.py`
- Modify: `backend/tests/test_pipeline_api.py`

**Step 1: Write the failing test**

Add an API test for:
- `POST /api/pipeline/jobs` with `stop_after=5`
- returned artifacts include Stage 5 outputs

**Step 2: Run test to verify it fails**

Run:

```bash
cd backend
../.venv/bin/python -m unittest discover -s tests -p 'test_pipeline_api.py' -v
```

Expected: FAIL because the API currently stops before Stage 5.

**Step 3: Write minimal implementation**

Update the API to:
- allow `stop_after=5`
- keep current stage reporting driven by callbacks/manifest

**Step 4: Run test to verify it passes**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add backend/api.py backend/tests/test_pipeline_api.py
git commit -m "feat: expose stage 5 pipe mask in pipeline api"
```

### Task 5: Update the frontend review flow

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/stores/appStore.ts`
- Modify: `frontend/src/components/DetectionSetup.tsx`
- Modify: `frontend/src/components/PipelineResultsView.tsx`

**Step 1: Use build as the red/green guard**

There may not be dedicated frontend tests, so use the build and the copy/logic diff as the guardrail.

**Step 2: Run build against the incomplete wiring**

Run:

```bash
cd frontend
bun run build
```

Expected: either mismatch in review copy or temporary wiring inconsistency.

**Step 3: Write minimal implementation**

Update the frontend to:
- default pipeline `stopAfter` to `5`
- update pipeline review copy to include Stage 5 pipe mask
- keep artifact display generic so Stage 5 images and JSON appear automatically

**Step 4: Run build to verify it passes**

Run the same command and confirm it passes.

**Step 5: Commit**

```bash
git add frontend/src/lib/api.ts frontend/src/stores/appStore.ts frontend/src/components/DetectionSetup.tsx frontend/src/components/PipelineResultsView.tsx
git commit -m "feat: show stage 5 pipe mask in pipeline ui"
```

### Task 6: Document the accepted Stage 5 baseline and verify end to end

**Files:**
- Modify: `IMPLEMENTATION_TRACKER.md`
- Modify: `backend/garnet/AGENTS.md`
- Add or modify: `SLICE_5_PROGRESS.md`

**Step 1: Update docs**

Record:
- Stage 5 purpose
- suppression-only methodology
- Stage 5 artifacts
- explicit boundary between Stage 5 and later morphology/skeleton stages

**Step 2: Run backend verification**

Run:

```bash
cd backend
../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py
../.venv/bin/python -m unittest discover -s tests -p 'test_pipe_mask.py' -v
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

**Step 4: Run a real sample**

Run:

```bash
cd backend
XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/stage5_sample --ocr-route easyocr --stop-after 5
```

Expected:
- Stage 5 completes
- output includes `stage5_pipe_mask.png`
- output includes `stage5_pipe_mask_overlay.png`
- output includes `stage5_pipe_mask_summary.json`

**Step 5: Commit**

```bash
git add IMPLEMENTATION_TRACKER.md backend/garnet/AGENTS.md SLICE_5_PROGRESS.md docs/plans/2026-03-08-stage-5-pipe-mask-design.md docs/plans/2026-03-08-stage-5-pipe-mask.md
git commit -m "docs: record stage 5 pipe mask slice"
```
