# Slice 2 OCR SAHI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add the EasyOCR Stage 2 route to the rebuilt pipeline using SAHI-style tiling, while keeping the OCR artifact contract compatible with the later selectable Gemini route.

**Architecture:** Keep the current Stage 1-only runner intact and extend it to a second reviewable stage. Stage 2 introduces a dedicated EasyOCR tiling helper that emits a canonical sheet-level OCR schema, summary artifacts, and an exception queue. The later Gemini route is planned separately, so this slice focuses on making EasyOCR a complete, reviewable route while preserving a shared OCR contract.

**Tech Stack:** Python 3.14, EasyOCR, OpenCV, NumPy, existing FastAPI pipeline job API, frontend React + Zustand + Vite, `unittest`.

---

### Task 1: Lock the Stage 2 OCR contract in tests

**Files:**
- Modify: `backend/tests/test_pid_extractor_cli.py`
- Modify: `backend/tests/test_pipeline_api.py`

**Step 1: Write the failing test**
- Assert the pipeline exposes two stages in order:
  - `stage1_input_normalization`
  - `stage2_ocr_discovery`
- Assert Stage 2 writes:
  - `stage2_ocr_regions.json`
  - `stage2_ocr_summary.json`
  - `stage2_ocr_exception_candidates.json`

**Step 2: Run test to verify it fails**
- Run: `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v`

**Step 3: Write minimal implementation**
- Extend the stage registry and manifest expectations only enough to satisfy the new Stage 2 test contract.

**Step 4: Run test to verify it passes**
- Re-run the same test suite.

### Task 2: Add the EasyOCR SAHI helper

**Files:**
- Create: `backend/garnet/easyocr_sahi.py`
- Modify: `backend/garnet/text_ocr.py`

**Step 1: Write the failing test**
- Add a focused test for:
  - tile generation with overlap
  - coordinate shifting from tile-local to sheet-local
  - duplicate merge across overlapping tiles

**Step 2: Run test to verify it fails**
- Run the targeted backend test module.

**Step 3: Write minimal implementation**
- Add:
  - EasyOCR reader initialization wrapper
  - tiling helper with overlap configuration
  - tile OCR execution
  - merge logic for overlapping text regions
  - canonical Stage 2 OCR JSON serializer using `pass_type: "sheet"`

**Step 4: Run test to verify it passes**
- Re-run the targeted test module.

### Task 3: Implement Stage 2 in the pipeline runner

**Files:**
- Modify: `backend/garnet/pid_extractor.py`

**Step 1: Write the failing test**
- Expect Stage 2 to consume Stage 1 outputs and emit OCR artifacts plus a manifest entry.

**Step 2: Run test to verify it fails**
- Re-run the targeted backend tests.

**Step 3: Write minimal implementation**
- Add `stage2_ocr_discovery` that:
  - uses Stage 1 output as input
  - runs the EasyOCR SAHI helper
  - writes:
    - `stage2_ocr_regions.json`
    - `stage2_ocr_overlay.png`
    - `stage2_ocr_summary.json`
    - `stage2_ocr_exception_candidates.json`
- Keep `--stop-after` aligned to two stages.

**Step 4: Run test to verify it passes**
- Re-run backend tests and a sample pipeline command through Stage 2.

### Task 4: Prepare a stable shared OCR contract without implementing the Gemini route yet

**Files:**
- Modify: `backend/garnet/pid_extractor.py`
- Modify: `backend/garnet/text_ocr.py`
- Reference: `backend/gemini_detector/gemini_sahi.py`

**Step 1: Write the failing test**
- Assert Stage 2 exception candidates include enough data for later route-specific analysis:
  - source bbox
  - source coordinates
  - reason codes
  - original EasyOCR record id

**Step 2: Run test to verify it fails**
- Re-run the targeted backend tests.

**Step 3: Write minimal implementation**
- Add a stable exception queue schema that later OCR route work can consume.
- Do not call Gemini in this slice.

**Step 4: Run test to verify it passes**
- Re-run the targeted backend tests.

### Task 5: Surface Stage 2 artifacts through the API and frontend

**Files:**
- Modify: `backend/api.py`
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/stores/appStore.ts`
- Modify: `frontend/src/components/ProcessingView.tsx`
- Modify: `frontend/src/components/PipelineResultsView.tsx`

**Step 1: Write the failing validation target**
- The frontend should build with a Stage 2-aware pipeline job shape and display OCR artifacts.

**Step 2: Run validation to verify it fails**
- Run: `cd frontend && bun run build`

**Step 3: Write minimal implementation**
- Keep the same pipeline job endpoints.
- Update the frontend to:
  - show Stage 2 progress
  - preview OCR overlay output
  - list OCR summary and exception candidate artifacts

**Step 4: Run validation to verify it passes**
- Re-run `cd frontend && bun run build`.

### Task 6: Verify the full Slice 2 vertical slice

**Files:**
- Modify: `IMPLEMENTATION_TRACKER.md`
- Modify: `SLICE_1_PROGRESS.md`
- Add or update: `SLICE_2_PROGRESS.md` if needed when implementation starts

**Step 1: Run backend compile verification**
- `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py`

**Step 2: Run backend tests**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v`

**Step 3: Run frontend build**
- `cd frontend && bun run build`

**Step 4: Run a real Stage 2 sample**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/slice2_ocr --stop-after 2`

**Step 5: Record actual status**
- Update tracker/docs with the Slice 2 outcome and the shared OCR contract needed by the later selectable Gemini route.
