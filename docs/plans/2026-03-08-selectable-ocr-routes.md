# Selectable OCR Routes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let the user choose one OCR route per pipeline run, using either the current EasyOCR route or a Gemini/OpenRouter route that uses `1024x1024` full-page patches with crop fallback only for misses or confidence below `0.3`.

**Architecture:** Keep Stage 1 normalization shared. Keep a single Stage 2 OCR stage, but make it dispatch by `ocr_route`. EasyOCR stays as the current local tiled OCR path. Gemini becomes an alternative Stage 2 route that uses full-page prompts on `1024x1024` patches and route-local crop fallback only when needed, while still emitting the same shared Stage 2 OCR artifact contract for the API and frontend.

**Tech Stack:** Python 3.14, FastAPI, OpenCV, NumPy, EasyOCR, existing Gemini/OpenRouter integration, React, Zustand, Vite, `unittest`.

---

### Task 1: Lock the route-selection contract in tests

**Files:**
- Modify: `backend/tests/test_pid_extractor_cli.py`
- Modify: `backend/tests/test_pipeline_api.py`
- Modify: `frontend/src/types.ts`

**Step 1: Write the failing test**
- Add backend tests that assert:
  - `PipelineConfig` supports `ocr_route`
  - `PIDPipeline.run()` still exposes two stages only
  - Stage 2 dispatches by route
- Add API coverage that asserts `/api/pipeline/jobs` rejects missing or invalid `ocr_route`.

**Step 2: Run test to verify it fails**
- Run: `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v`

**Step 3: Write minimal implementation**
- Add only enough API and pipeline route validation to make the tests pass.

**Step 4: Run test to verify it passes**
- Re-run the same backend test suite.

### Task 2: Add route selection to the pipeline runner

**Files:**
- Modify: `backend/garnet/pid_extractor.py`

**Step 1: Write the failing test**
- Assert `stage2_ocr_discovery` calls:
  - `run_easyocr_sahi(...)` when `ocr_route="easyocr"`
  - Gemini route helper when `ocr_route="gemini"`

**Step 2: Run test to verify it fails**
- Run the targeted pipeline test module.

**Step 3: Write minimal implementation**
- Add `ocr_route` to `PipelineConfig`.
- Route Stage 2 by `ocr_route`.
- Persist the selected route in the stage manifest summary.

**Step 4: Run test to verify it passes**
- Re-run the targeted pipeline test module.

### Task 3: Build the Gemini OCR route helper

**Files:**
- Create: `backend/garnet/gemini_ocr_sahi.py`
- Reference: `backend/garnet/OCR_prompts/*.md`
- Reference: `backend/gemini_detector/gemini_sahi.py`

**Step 1: Write the failing test**
- Add focused tests for:
  - prompt file loading
  - `1024x1024` patch generation
  - crop fallback trigger when confidence is below `0.3`
  - mapping patch/crop-local boxes back to sheet coordinates
  - returning the same top-level OCR result shape as EasyOCR route

**Step 2: Run test to verify it fails**
- Run the targeted Gemini OCR route test module.

**Step 3: Write minimal implementation**
- Implement:
  - prompt loading from `backend/garnet/OCR_prompts`
  - patch generation at `1024x1024`
  - Gemini/OpenRouter patch inference using the full-page prompt pair
  - crop-pass fallback only for misses or confidence below `0.3`
  - canonical Stage 2 OCR payload plus Gemini raw audit artifacts

**Step 4: Run test to verify it passes**
- Re-run the targeted Gemini OCR route tests.

### Task 4: Extend the pipeline API with `ocr_route`

**Files:**
- Modify: `backend/api.py`
- Modify: `backend/tests/test_pipeline_api.py`

**Step 1: Write the failing test**
- Assert pipeline jobs:
  - require `ocr_route`
  - store the route in job payloads
  - return manifest metadata showing the selected route

**Step 2: Run test to verify it fails**
- Run the targeted API test module.

**Step 3: Write minimal implementation**
- Update `POST /api/pipeline/jobs` to accept `ocr_route`.
- Pass `ocr_route` into `PIDPipeline`.
- Include route metadata in the in-memory job state and manifest response.

**Step 4: Run test to verify it passes**
- Re-run the targeted API test module.

### Task 5: Add route selection to the frontend Pipeline mode

**Files:**
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/stores/appStore.ts`
- Modify: `frontend/src/components/DetectionSetup.tsx`
- Modify: `frontend/src/components/PipelineResultsView.tsx`
- Modify: `frontend/src/types.ts`

**Step 1: Write the failing validation target**
- Frontend build should fail until the new `ocrRoute` field is wired through pipeline start, polling state, and results display.

**Step 2: Run validation to verify it fails**
- Run: `cd frontend && bun run build`

**Step 3: Write minimal implementation**
- Add Pipeline-mode route selection UI.
- Default to one explicit route value in store state.
- Send `ocrRoute` with pipeline job creation.
- Show the selected route in the review UI.

**Step 4: Run validation to verify it passes**
- Re-run `cd frontend && bun run build`.

### Task 6: Keep artifact review route-neutral

**Files:**
- Modify: `frontend/src/components/PipelineResultsView.tsx`
- Modify: `backend/garnet/pid_extractor.py`

**Step 1: Write the failing test or validation target**
- Ensure both routes still expose the common Stage 2 artifact bundle:
  - `stage2_ocr_regions.json`
  - `stage2_ocr_summary.json`
  - `stage2_ocr_exception_candidates.json`
  - `stage2_ocr_overlay.png`

**Step 2: Run verification to confirm the gap**
- Use targeted backend tests and frontend build.

**Step 3: Write minimal implementation**
- Normalize Gemini route outputs into the same Stage 2 artifact names.
- Add optional display of Gemini-specific raw audit artifacts without making them required.

**Step 4: Run verification to verify it passes**
- Re-run the same backend/frontend checks.

### Task 7: Update docs and tracker after implementation

**Files:**
- Modify: `IMPLEMENTATION_TRACKER.md`
- Modify: `backend/garnet/AGENTS.md`
- Modify: `SLICE_2_PROGRESS.md`
- Add: `SLICE_3_PROGRESS.md`

**Step 1: Record the selected architecture**
- Replace any remaining text that describes Slice 3 as a mandatory Stage 3 after EasyOCR.

**Step 2: Record production route details**
- Document:
  - `easyocr` route baseline
  - `gemini` route patch size `1024x1024`
  - crop fallback confidence threshold `0.3`

**Step 3: Verify doc consistency**
- Run: `rg -n "Stage 3|fallback|ocr_route|easyocr|gemini" IMPLEMENTATION_TRACKER.md backend/garnet/AGENTS.md docs/plans SLICE_*.md`

### Task 8: Verify the full selectable-route vertical slice

**Files:**
- No new source files required in this task.

**Step 1: Run backend compile verification**
- `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py`

**Step 2: Run backend tests**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v`

**Step 3: Run frontend build**
- `cd frontend && bun run build`

**Step 4: Run a real EasyOCR-route sample**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/slice3_easyocr --stop-after 2`

**Step 5: Run a real Gemini-route sample**
- `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/slice3_gemini --stop-after 2`

**Step 6: Record actual evidence**
- Capture artifact paths, runtime observations, and any Gemini config prerequisites in the progress log.
