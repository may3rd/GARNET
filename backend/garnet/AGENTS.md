# Agent instructions (scope: this directory and subdirectories)

## Scope and source of truth
- This AGENTS.md applies to `backend/garnet/` and below.
- Use `/Users/maetee/Code/GARNET/MASTER_PLAN.md` as the roadmap for pipeline design and sequencing.
- Keep the pipeline aligned with the plan's governing rule: early detections, OCR, and geometry are provisional evidence until they survive topology and graph QA.

## What this module owns
- P&ID digitizing orchestration: [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
- Connectivity and graph construction: [`backend/garnet/connectivity_graph.py`](/Users/maetee/Code/GARNET/backend/garnet/connectivity_graph.py)
- OCR utilities and text post-processing: [`backend/garnet/text_ocr.py`](/Users/maetee/Code/GARNET/backend/garnet/text_ocr.py)
- Object + text detection helpers: [`backend/garnet/object_and_text_detect.py`](/Users/maetee/Code/GARNET/backend/garnet/object_and_text_detect.py)
- DEXPI export: [`backend/garnet/dexpi_exporter.py`](/Users/maetee/Code/GARNET/backend/garnet/dexpi_exporter.py)
- DeepLSD helpers and shared utilities: [`backend/garnet/utils/deeplsd_utils.py`](/Users/maetee/Code/GARNET/backend/garnet/utils/deeplsd_utils.py), [`backend/garnet/utils/utils.py`](/Users/maetee/Code/GARNET/backend/garnet/utils/utils.py)

## Pipeline architecture rules
- Preserve the phase order from `MASTER_PLAN.md`: evidence extraction -> geometry engine -> attachment/association -> graph engine -> QA/recovery -> export.
- Favor geometry first, semantics second. Do not promote OCR text or object detections directly into graph truth without geometric/topological support.
- Keep task-specific masks and derived views separate from the original raster. Never destroy source evidence early.
- Treat skeletonization as fragile. If you change line extraction, keep or improve the pre-skeleton morphology and cleanup stages.
- Flow arrows are part of the pipeline, not optional noise. If directionality changes, update arrow detection, association, and directed-edge assignment together.
- Graph QA is part of extraction. New topology logic should add or preserve NetworkX-based validation, anomaly detection, and confidence/provenance tracking.
- Simplify edge polylines before export. Do not push raw pixel-by-pixel paths into DEXPI or downstream graph outputs.

## Phase-to-file map
- Evidence extraction: `object_and_text_detect.py`, `text_ocr.py`
- Geometry and staged orchestration: `pid_extractor.py`
- Graph assembly and snapping: `connectivity_graph.py`
- Export and interchange: `dexpi_exporter.py`
- Line-model support: `utils/deeplsd_utils.py`

## Working conventions
- Add new thresholds and toggles to `PipelineConfig` instead of scattering magic numbers through stage code.
- Keep stage outputs inspectable. If you add debug artifacts, write them under `/Users/maetee/Code/GARNET/backend/output`, `/Users/maetee/Code/GARNET/backend/runs`, or `/Users/maetee/Code/GARNET/backend/temp`.
- Preserve node/edge provenance, confidence, and review-ready metadata when extending graph attributes.
- Prefer additive stage methods or focused helper functions over enlarging already-long methods with unrelated side effects.
- When adding new symbol classes, update role mapping, link behavior, and any edge-direction assumptions in the same change.
- Keep heavyweight model loading and external-service calls configurable. Never hardcode secrets, API keys, or machine-specific absolute paths.

## Runtime and verification
- Run backend commands from `/Users/maetee/Code/GARNET/backend` so relative paths for weights, outputs, and datasets resolve consistently.
- Install dependencies with `pip install -r requirements.txt` inside the backend environment.
- Start the API with `uvicorn api:app --reload --port 8001`.
- Run the pipeline entrypoint with `python -m garnet.pid_extractor`.
- Use `python -m py_compile garnet/*.py garnet/utils/*.py api.py` as the minimum non-destructive verification after edits.
- Treat `garnet/deeplsd_test.py` and `garnet/test_paddleocr.py` as manual smoke scripts, not reliable automated tests.

## Common pitfalls
- Many scripts assume local model weights, OCR dependencies, and sample files already exist. Check inputs before treating a failure as a code regression.
- `pid_extractor.py` is the orchestration center; changes there can silently break OCR, graph assembly, and export in later stages.
- `connectivity_graph.py` currently favors orthogonal geometry. If you broaden line support, review simplification and snapping behavior together.
- Generated artifacts and caches can get large. Avoid committing anything under `backend/output`, `backend/runs`, `.ultralytics_runs`, or temporary experiment folders.

## Do not
- Do not bypass the stage model from `MASTER_PLAN.md` by coupling detection, graph construction, and export into one opaque step.
- Do not replace configurable thresholds with hidden constants.
- Do not introduce irreversible preprocessing on the only copy of an image.
- Do not commit secrets, weights, notebooks, cached predictions, or debug images as part of routine pipeline work.
