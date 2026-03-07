# Slice 1 Stage 1 Pipeline And Frontend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current multi-stage pipeline entrypoint with a Stage 1-only image normalization pipeline, expose it through a job-style backend API, and add a frontend Pipeline mode that shows real stage progress and Stage 1 artifacts.

**Architecture:** Keep the new pipeline small: raw image in, Stage 1 artifact bundle out, manifest always written. Add a background in-memory job runner in the backend so the frontend can poll real progress, then reuse the existing upload, processing, and results surfaces by branching on a new processing mode instead of building a separate UI.

**Tech Stack:** Python 3.14, FastAPI, in-memory job store, existing frontend React + Zustand + Vite, `unittest`, TypeScript build validation.

---

### Task 1: Lock the Stage 1-only backend contract in tests

**Files:**
- Modify: `backend/tests/test_pid_extractor_cli.py`
- Create: `backend/tests/test_pipeline_api.py`

**Step 1: Write the failing test**
- Assert the pipeline exposes exactly one stage: `stage1_input_normalization`.
- Assert the job API returns a job id and reports stage progress and artifacts.

**Step 2: Run test to verify it fails**
- Run: `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v`

**Step 3: Write minimal implementation**
- Replace the current stage registry and add a minimal pipeline job API.

**Step 4: Run test to verify it passes**
- Re-run the same test suite.

### Task 2: Replace `pid_extractor.py` with Stage 1-only logic

**Files:**
- Modify: `backend/garnet/pid_extractor.py`

**Step 1: Write the failing test**
- Expect Stage 1-only manifest and normalization artifact outputs.

**Step 2: Run test to verify it fails**
- Re-run the targeted backend tests.

**Step 3: Write minimal implementation**
- Keep only:
  - image input loading
  - grayscale/binary/contrast-normalized outputs
  - stage manifest support
  - `--stop-after` limited to Stage 1

**Step 4: Run test to verify it passes**
- Re-run the targeted backend tests and sample pipeline command.

### Task 3: Add the backend pipeline job endpoints

**Files:**
- Modify: `backend/api.py`

**Step 1: Write the failing test**
- Add API tests for:
  - `POST /api/pipeline/jobs`
  - `GET /api/pipeline/jobs/{job_id}`

**Step 2: Run test to verify it fails**
- Re-run targeted backend tests.

**Step 3: Write minimal implementation**
- Add:
  - in-memory job store
  - background thread runner
  - Stage 1 result serialization and artifact URLs

**Step 4: Run test to verify it passes**
- Re-run the backend tests.

### Task 4: Add frontend Pipeline mode and progress/results UI

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/stores/appStore.ts`
- Modify: `frontend/src/components/DetectionSetup.tsx`
- Modify: `frontend/src/components/ProcessingView.tsx`
- Modify: `frontend/src/components/ResultsView.tsx`
- Create: `frontend/src/components/PipelineResultsView.tsx`

**Step 1: Write the failing test or validation target**
- Use TypeScript build as the validation gate for the new frontend state shape and API calls.

**Step 2: Run validation to verify failure**
- Run: `cd frontend && npm run build`

**Step 3: Write minimal implementation**
- Add:
  - processing mode toggle (`Detection` / `Pipeline`)
  - pipeline job start/poll API helpers
  - Zustand state for pipeline job progress/result
  - progress screen driven by real stage status
  - results screen showing Stage 1 artifacts and manifest metadata

**Step 4: Run validation to verify it passes**
- Re-run `npm run build`.

### Task 5: Verify the full Slice 1 vertical slice

**Files:**
- Modify: `IMPLEMENTATION_TRACKER.md`
- Modify: `SPRINT_0_PROGRESS.md` only if a continuity note is needed

**Step 1: Run backend compile verification**
- `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py`

**Step 2: Run backend tests**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v`

**Step 3: Run frontend build**
- `cd frontend && npm run build`

**Step 4: Run a real Stage 1 sample**
- Start backend and use the frontend Pipeline mode or call the API directly against `sample.png`.

**Step 5: Record actual status**
- Update tracker/docs with the Slice 1 outcome and remaining gaps for Slice 2.
