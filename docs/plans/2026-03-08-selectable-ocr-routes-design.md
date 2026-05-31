# Selectable OCR Routes Design

## Goal
- Replace the planned EasyOCR -> Gemini chained OCR flow with a user-selected OCR route per pipeline run.
- Keep the rebuild simple: Stage 1 remains shared, then Stage 2 runs exactly one OCR route.
- Preserve a common OCR artifact contract so the API and frontend can review either route without route-specific screens.

## Approved direction
- Pipeline mode must require the user to choose one OCR route per run:
  - `easyocr`
  - `gemini`
- There is no `auto` mode and no automatic chaining between routes.
- The selected route must be recorded in the pipeline manifest and exposed in the API/frontend job payload.

## Shared stage model

### Stage 1: input normalization
- Shared by both routes.
- Inputs:
  - raw P&ID image
- Outputs:
  - `stage1_gray.png`
  - existing normalization artifacts
  - manifest entry

### Stage 2: OCR discovery
- Stage name remains `stage2_ocr_discovery`.
- Internals branch by `ocr_route`.
- Outputs must stay contract-compatible across routes:
  - `stage2_ocr_regions.json`
  - `stage2_ocr_summary.json`
  - `stage2_ocr_exception_candidates.json`
  - `stage2_ocr_overlay.png`

## Route 1: EasyOCR

### Purpose
- Keep the current production baseline as the fast local OCR route.

### Flow
- Use `stage1_gray.png` as the OCR input.
- Run SAHI-style tiled OCR with the current EasyOCR tuning baseline.
- Keep rotated OCR and same-line merge enabled.

### Route-specific behavior
- Exception candidates remain heuristic outputs for later review or future targeted reruns.
- No Gemini fallback is invoked inside the EasyOCR route.

## Route 2: Gemini

### Purpose
- Provide an alternative OCR route driven by Gemini/OpenRouter instead of EasyOCR.

### Flow
- Use Stage 1 output as the source raster.
- Use SAHI slicing with `1024x1024` patches.
- Run the full-page prompt pair from [`backend/garnet/OCR_prompts/`](/Users/maetee/Code/GARNET/backend/garnet/OCR_prompts) on the patches.
- If a patch still misses text or the best confidence is below `0.3`, use the crop-pass prompt as the last fallback for that patch/item.
- Baseline SAHI merge settings:
  - `postprocess_type = GREEDYNMM`
  - `postprocess_match_metric = IOS`
  - `postprocess_match_threshold = 0.1`

### Important constraint
- The crop-pass prompt is a route-local fallback, not a separate pipeline stage.
- Gemini route is still one selected route for the run.

### Prompt sources
- Full-page system prompt: [`backend/garnet/OCR_prompts/full_page_pass_system_prompt.md`](/Users/maetee/Code/GARNET/backend/garnet/OCR_prompts/full_page_pass_system_prompt.md)
- Full-page user prompt: [`backend/garnet/OCR_prompts/full_page_pass_user_prompt.md`](/Users/maetee/Code/GARNET/backend/garnet/OCR_prompts/full_page_pass_user_prompt.md)
- Crop system prompt: [`backend/garnet/OCR_prompts/crop_pass_system_prompt.md`](/Users/maetee/Code/GARNET/backend/garnet/OCR_prompts/crop_pass_system_prompt.md)
- Crop user prompt: [`backend/garnet/OCR_prompts/crop_pass_user_prompt.md`](/Users/maetee/Code/GARNET/backend/garnet/OCR_prompts/crop_pass_user_prompt.md)

## Canonical OCR contract
- Both routes must emit the same sheet-level `text_regions` structure for Stage 2 review.
- The canonical artifact remains sheet-local and route-neutral.
- Route-specific audit artifacts are allowed in addition to the shared Stage 2 bundle.

### Gemini route audit artifacts
- `stage2_gemini_patch_requests.json`
- `stage2_gemini_patch_raw.json`
- `stage2_gemini_crop_raw.json`

## Backend design

### Pipeline config
- Add `ocr_route` to `PipelineConfig`.
- Allowed values:
  - `easyocr`
  - `gemini`

### Pipeline runner
- Keep Stage 1 unchanged.
- Make `stage2_ocr_discovery` a dispatcher:
  - `easyocr` -> current `run_easyocr_sahi(...)`
  - `gemini` -> new Gemini OCR route helper

### New helper module
- Add [`backend/garnet/gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/gemini_ocr_sahi.py) to:
  - load OCR prompt files
  - use SAHI `get_sliced_prediction(...)` over `1024x1024` slices
  - call the Gemini/OpenRouter path through a SAHI-compatible detection model
  - apply crop fallback only when needed
  - select text/class payload from overlapping candidates by highest confidence after SAHI postprocessing
  - emit the shared OCR contract plus route-specific audit artifacts

## API design
- `POST /api/pipeline/jobs` should accept a required `ocr_route` form field.
- Validation:
  - allow only `easyocr` or `gemini`
- Job payload should include:
  - selected OCR route
  - manifest with route metadata
- The pipeline artifact endpoint remains unchanged.

## Frontend design

### Setup
- In Pipeline mode, add a route selector:
  - `EasyOCR`
  - `Gemini`
- The user must choose one route before starting the run.

### Processing
- Reuse the existing processing screen.
- Show the selected route in the pipeline progress view or metadata summary.

### Results
- Reuse the existing pipeline results screen.
- Show shared Stage 2 artifacts for both routes.
- If the Gemini route is used, also show any route-specific raw response artifacts.

## Why this design is better than chained Stage 2 -> Stage 3
- It matches the user workflow: choose one OCR route per run.
- It avoids mixing route selection with fallback stage sequencing.
- It keeps the frontend simple because both routes still land on one Stage 2 review contract.
- It avoids paying Gemini cost on runs where the user only wants EasyOCR.

## Risks and guardrails
- Gemini route depends on prompt files and `OPENROUTER_API_KEY`; missing configuration must fail clearly.
- Gemini route will be slower and more expensive than EasyOCR.
- The shared OCR contract must stay stable so the frontend does not branch on route-specific schemas.
- Existing tests assume Stage 2 is always EasyOCR; they need route-aware updates.

## Definition of done
- Pipeline jobs require the user to choose `easyocr` or `gemini`.
- Stage 1 stays shared.
- Stage 2 can execute either route and still write the common OCR artifact bundle.
- Gemini route uses `1024x1024` full-page patches first and crop fallback only for misses or confidence below `0.3`.
- API and frontend show which route was used for the run.
