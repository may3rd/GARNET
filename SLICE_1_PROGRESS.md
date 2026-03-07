# Slice 1 Progress Log

## Purpose
- Working log for the Stage 1 rebuild slice.
- Track the first clean vertical slice after the hard reset: raw P&ID image input, Stage 1 normalization outputs, backend job API, and frontend progress/results review.

## Entry format
- Timestamp
- Task
- Action
- Evidence
- Verification
- Next step / blocker

## Entries

### 2026-03-07 21:43 ICT
- Task: `Slice 1 / Backend contract`
- Action: Locked the new Stage 1-only pipeline contract in backend tests and removed the expectation of any OCR/COCO side inputs.
- Evidence:
  - [`backend/tests/test_pid_extractor_cli.py`](/Users/maetee/Code/GARNET/backend/tests/test_pid_extractor_cli.py)
  - [`backend/tests/test_pipeline_api.py`](/Users/maetee/Code/GARNET/backend/tests/test_pipeline_api.py)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v` -> pass
- Next step / blocker:
  - Replace the pipeline implementation itself so the tests represent the new architecture, not the old runner.

### 2026-03-07 21:48 ICT
- Task: `Slice 1 / Stage 1 backend`
- Action: Replaced [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) with a Stage 1-only orchestrator that accepts only an image path and writes normalization artifacts plus a stage manifest.
- Evidence:
  - [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - sample output folder [`backend/output/slice1_stage1`](/Users/maetee/Code/GARNET/backend/output/slice1_stage1)
- Verification:
  - `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py` -> pass
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/slice1_stage1` -> pass
- Next step / blocker:
  - Expose the new runner through a backend job API so the frontend can poll real stage progress.

### 2026-03-07 21:55 ICT
- Task: `Slice 1 / Backend API`
- Action: Added a minimal pipeline job API with in-memory job tracking, background execution, job-status serialization, and artifact serving.
- Evidence:
  - [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py)
  - generated job folders under [`backend/output/pipeline_jobs`](/Users/maetee/Code/GARNET/backend/output/pipeline_jobs)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v` -> pass
- Next step / blocker:
  - Add a frontend mode that starts the job and presents stage-level status instead of the legacy detection flow.

### 2026-03-07 22:06 ICT
- Task: `Slice 1 / Frontend integration`
- Action: Added a new Pipeline mode in the frontend, reused the existing setup/processing/results UX, and introduced a dedicated pipeline results view for Stage 1 artifacts and manifest data.
- Evidence:
  - [`frontend/src/components/DetectionSetup.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/DetectionSetup.tsx)
  - [`frontend/src/components/ProcessingView.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/ProcessingView.tsx)
  - [`frontend/src/components/PipelineResultsView.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/PipelineResultsView.tsx)
  - [`frontend/src/stores/appStore.ts`](/Users/maetee/Code/GARNET/frontend/src/stores/appStore.ts)
  - [`frontend/src/lib/api.ts`](/Users/maetee/Code/GARNET/frontend/src/lib/api.ts)
- Verification:
  - `cd frontend && npm run build` -> pass
- Next step / blocker:
  - Slice 1 is complete. Next build slice is OCR discovery as Stage 2, using the same job API and stage-progress UI.
