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
- Stage 1 normalization, Stage 2 OCR routing, Stage 4 object detection, and Stage 5 pipe mask orchestration: `pid_extractor.py`
- Stage 2 EasyOCR route: `easyocr_sahi.py`
- Stage 2 Gemini/OpenRouter route: `gemini_ocr_sahi.py`
- Stage 2 PaddleOCR route: `paddle_ocr_sahi.py`
- Stage 4 fixed-baseline detection: `object_detection_sahi.py`
- Stage 5 provisional pipe-mask generation: `pipe_mask.py`
- Detection-serving helpers for the existing `/api/detect` path: `predict_images.py`, `Settings.py`
- Shared image utilities: `utils/utils.py`

## Working conventions
- Add new thresholds and toggles to `PipelineConfig` instead of scattering magic numbers through stage code.
- Keep stage outputs inspectable. If you add debug artifacts, write them under `/Users/maetee/Code/GARNET/backend/output`, `/Users/maetee/Code/GARNET/backend/runs`, or `/Users/maetee/Code/GARNET/backend/temp`.
- Prefer additive stage methods or focused helper functions over enlarging already-long methods with unrelated side effects.
- When future slices add OCR or symbol classes, update role mapping and review artifacts in the same change.
- Keep heavyweight model loading and external-service calls configurable. Never hardcode secrets, API keys, or machine-specific absolute paths.
- Current Stage 2 OCR baseline:
  - use `stage1_gray.png` as the primary OCR input
  - keep `stage1_gray_equalized.png` available for comparison runs, not as the default
  - EasyOCR route now uses real SAHI `get_sliced_prediction(...)`, not a custom tile loop
  - EasyOCR route uses `postprocess_match_metric = IOS`
  - EasyOCR route uses `postprocess_match_threshold = 0.1`
  - keep rotated OCR enabled for Stage 2 tile passes
  - keep same-line merge enabled so nearby words on one engineering line become one text region when the baseline and spacing support it
- Current pipeline OCR routing baseline:
  - every pipeline run must choose exactly one OCR route
  - current supported routes are `easyocr`, `gemini`, and `paddleocr`
  - keep route handling extensible because more OCR routes will be added later
  - `easyocr` remains the local tiled OCR route
  - `gemini` is a separate Stage 2 route, not a downstream Stage 3 after EasyOCR
  - `paddleocr` is a separate Stage 2 route and must emit the same Stage 2 artifact bundle shape as the other routes
  - Gemini route uses real SAHI slicing with `1024x1024` patches
  - Gemini route uses `postprocess_match_metric = IOS`
  - Gemini route uses `postprocess_match_threshold = 0.1`
  - Gemini route uses crop fallback only when the slice result is empty or best confidence is below `0.3`
- Current Stage 4 object-detection baseline:
  - stage name is `stage4_object_detection`
  - keep sparse stage numbering honest; do not add a fake Stage 3 placeholder
  - fixed baseline weight is `yolo_weights/yolo26n_PPCL_640_20260227.pt`
  - detection path uses Ultralytics + SAHI
  - detection baseline uses `image_size = 640`
  - detection baseline uses `overlap_ratio = 0.2`
  - detection baseline uses `postprocess_type = GREEDYNMM`
  - detection baseline uses `postprocess_match_metric = IOS`
  - detection baseline uses `postprocess_match_threshold = 0.1`
  - Stage 4 artifacts are `stage4_objects.json`, `stage4_objects_summary.json`, and `stage4_objects_overlay.png`
- Current Stage 5 pipe-mask baseline:
  - stage name is `stage5_pipe_mask`
  - Stage 5 is provisional geometry evidence only; do not add morphology, skeletonization, or graph logic here
  - candidate mask starts from `stage1_binary_adaptive.png` OR `stage1_binary_otsu.png`
  - OCR suppression uses Stage 2 `text_regions` boxes with conservative padding
  - object suppression uses Stage 4 object boxes with conservative interior suppression
  - keep the output reviewable rather than aggressively repaired
  - Stage 5 artifacts are `stage5_pipe_mask.png`, `stage5_pipe_mask_overlay.png`, and `stage5_pipe_mask_summary.json`
- If you tune OCR parameters, record the accepted values in `docs/plans/2026-03-08-slice-2-ocr-sahi-design.md` and log the reason in `SLICE_2_PROGRESS.md`.
- If you tune Stage 5 suppression parameters, update `PipelineConfig`, log the reason in `SLICE_5_PROGRESS.md`, and keep the Stage 5 design docs aligned with the accepted baseline.

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
