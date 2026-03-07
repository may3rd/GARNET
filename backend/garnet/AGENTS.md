# Agent instructions (scope: this directory and subdirectories)

## Scope and source of truth
- This AGENTS.md applies to `backend/garnet/` and below.
- Use `/Users/maetee/Code/GARNET/MASTER_PLAN.md` as the roadmap for pipeline design and sequencing.
- Keep the pipeline aligned with the plan's governing rule: early detections, OCR, and geometry are provisional evidence until they survive topology and graph QA.

## What this module owns
- P&ID digitizing orchestration: [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
- Detection batch helpers used by the API path: [`backend/garnet/predict_images.py`](/Users/maetee/Code/GARNET/backend/garnet/predict_images.py)
- OCR utilities and text post-processing: [`backend/garnet/text_ocr.py`](/Users/maetee/Code/GARNET/backend/garnet/text_ocr.py)
- Shared utilities: [`backend/garnet/utils/utils.py`](/Users/maetee/Code/GARNET/backend/garnet/utils/utils.py)
- Static config and class metadata: [`backend/garnet/Settings.py`](/Users/maetee/Code/GARNET/backend/garnet/Settings.py)

## Pipeline architecture rules
- Preserve the phase order from `MASTER_PLAN.md` as future work, but keep the live code honest about what exists today.
- The current active rebuild is Stage 1-first: raw image input -> normalization artifacts -> manifest -> API/frontend review.
- Favor geometry first, semantics second. Do not promote OCR text or object detections directly into graph truth without geometric/topological support.
- Keep task-specific masks and derived views separate from the original raster. Never destroy source evidence early.
- Later stages should be added one at a time with visible artifacts and manifest entries. Do not reintroduce a large opaque pipeline.
- Keep stage outputs inspectable and small enough to review in code and UI at each slice.

## Phase-to-file map
- Stage 1 normalization and manifest writing: `pid_extractor.py`
- OCR support for future slices: `text_ocr.py`
- Detection-serving helpers for the existing `/api/detect` path: `predict_images.py`, `Settings.py`
- Shared image utilities: `utils/utils.py`

## Working conventions
- Add new thresholds and toggles to `PipelineConfig` instead of scattering magic numbers through stage code.
- Keep stage outputs inspectable. If you add debug artifacts, write them under `/Users/maetee/Code/GARNET/backend/output`, `/Users/maetee/Code/GARNET/backend/runs`, or `/Users/maetee/Code/GARNET/backend/temp`.
- Prefer additive stage methods or focused helper functions over enlarging already-long methods with unrelated side effects.
- When future slices add OCR or symbol classes, update role mapping and review artifacts in the same change.
- Keep heavyweight model loading and external-service calls configurable. Never hardcode secrets, API keys, or machine-specific absolute paths.

## Runtime and verification
- Run backend commands from `/Users/maetee/Code/GARNET/backend` so relative paths for weights, outputs, and datasets resolve consistently.
- Install dependencies with `pip install -r requirements.txt` inside the backend environment.
- Start the API with `uvicorn api:app --reload --port 8001`.
- Run the pipeline entrypoint with `python -m garnet.pid_extractor`.
- Use `python -m py_compile garnet/*.py garnet/utils/*.py api.py` as the minimum non-destructive verification after edits.
- Prefer `python -m unittest discover -s tests -p 'test*.py' -v` for backend regression checks.

## Common pitfalls
- Many scripts assume local model weights, OCR dependencies, and sample files already exist. Check inputs before treating a failure as a code regression.
- `pid_extractor.py` is the orchestration center; keep it small and staged while the rebuild is in progress.
- `api.py` still serves the legacy `/api/detect` path and the new pipeline job path. Keep those concerns separate.
- Generated artifacts and caches can get large. Avoid committing anything under `backend/output`, `backend/runs`, `.ultralytics_runs`, or temporary experiment folders.

## Do not
- Do not bypass the staged rebuild model by coupling OCR, graph construction, and export into one opaque step.
- Do not replace configurable thresholds with hidden constants.
- Do not introduce irreversible preprocessing on the only copy of an image.
- Do not commit secrets, weights, notebooks, cached predictions, or debug images as part of routine pipeline work.
