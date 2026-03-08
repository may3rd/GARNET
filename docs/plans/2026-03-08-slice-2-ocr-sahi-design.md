# Slice 2 OCR SAHI Design

## Goal
- Define the Slice 2 EasyOCR route for the rebuilt P&ID pipeline.
- Keep the rebuild image-only.
- Use EasyOCR as one selectable OCR route.
- Keep the Stage 2 artifact contract compatible with the later Gemini route.

## Approved direction
- Stage 2 uses real SAHI slicing over the normalized P&ID image and runs EasyOCR on each patch.
- EasyOCR is a selectable OCR route, not a mandatory first pass before Gemini.
- The later Gemini route must reuse the same Stage 2 artifact contract so the API and frontend stay route-neutral.
- VLM is not part of the EasyOCR route in this slice.

## Why SAHI patching is required
- Full-sheet OCR on dense P&ID drawings under-detects small text and mixed-orientation labels.
- Tiling improves local scale and recall without forcing global upscaling of the entire sheet.
- The repo already has a proven SAHI-style pattern in [`backend/gemini_detector/gemini_sahi.py`](/Users/maetee/Code/GARNET/backend/gemini_detector/gemini_sahi.py).
- Reusing the same SAHI slicing/postprocess backbone keeps the EasyOCR route and Gemini route structurally consistent.

## Stage boundaries

### Stage 2: EasyOCR discovery
- Stage name: `stage2_ocr_discovery`
- Input:
  - Stage 1 normalized image outputs from [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - Primary working image should be the most OCR-friendly normalized view, with room to compare additional views later if needed.
- Core flow:
  - let SAHI cut the sheet into overlapping tiles
  - run EasyOCR on each tile, including rotated passes when enabled
  - let SAHI merge overlapping detections across tiles
  - run a post-SAHI same-line merge so adjacent engineering words can become one text region
  - classify obvious text categories conservatively
  - emit exception candidates for review and for any later route-specific targeted analysis
- Non-goals:
  - no VLM in Stage 2
  - no graph reasoning
  - no topology or semantic attachment

### Later Gemini route
- Gemini is no longer defined here as Stage 3 after EasyOCR.
- The route-selection design now lives in [`docs/plans/2026-03-08-selectable-ocr-routes-design.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-selectable-ocr-routes-design.md).
- The only contract this Slice 2 design must preserve is:
  - Stage 2 artifacts stay stable and route-neutral
  - exception candidates remain machine-readable
  - downstream review can compare route outputs without schema branching

## Stage 2 module design
- Add a new helper module under [`backend/garnet/`](/Users/maetee/Code/GARNET/backend/garnet), likely `easyocr_sahi.py`.
- The module should follow the same broad adapter pattern as [`backend/gemini_detector/gemini_sahi.py`](/Users/maetee/Code/GARNET/backend/gemini_detector/gemini_sahi.py), but for OCR tiles instead of VLM object detections.
- Responsibilities:
  - SAHI slice generation and overlap settings
  - EasyOCR reader initialization and reuse
  - tile-local OCR execution
  - conversion from EasyOCR tile detections into SAHI predictions
  - SAHI postprocess merge across overlaps
  - final same-line phrase merge on the merged sheet-level OCR regions
  - conservative exception flagging

## Canonical OCR schema

### Shared record shape
- Both OCR routes should use the same `text_regions[]` record shape so downstream code and frontend review do not need format branching.
- Required fields per region:
  - `id`
  - `text`
  - `normalized_text`
  - `class`
  - `confidence`
  - `bbox`
  - `rotation`
  - `reading_direction`
  - `legibility`

### Stage 2 output contract
- Stage 2 should emit sheet-level OCR JSON with this structure:

```json
{
  "image_id": "<string if provided, else empty string>",
  "pass_type": "sheet",
  "text_regions": [
    {
      "id": "ocr_000001",
      "text": "6\"-P-1001-A1",
      "normalized_text": "6\"-P-1001-A1",
      "class": "line_number",
      "confidence": 0.93,
      "bbox": {
        "x_min": 100,
        "y_min": 200,
        "x_max": 220,
        "y_max": 238
      },
      "rotation": 0,
      "reading_direction": "horizontal",
      "legibility": "clear"
    }
  ]
}
```

- This is the same schema as the approved crop format, except `pass_type` is `sheet` and coordinates are sheet-local.

### Gemini route compatibility
- The later Gemini route should normalize its accepted OCR output into the same sheet-level Stage 2 contract.
- Raw Gemini patch/crop responses may use route-specific formats and artifacts, but the common Stage 2 review bundle must remain stable.

## Classification policy
- Stage 2 classification is conservative and local-only.
- Default to `unknown` unless the crop strongly supports another allowed class.
- EasyOCR is responsible for text localization and transcription within this route.

## Proposed Stage 2 artifacts
- `stage2_ocr_regions.json`
- `stage2_ocr_overlay.png`
- `stage2_ocr_summary.json`
- `stage2_ocr_exception_candidates.json`
- optional debug folder for tile outputs when enabled

## Current production baseline
- The current accepted Stage 2 baseline uses the plain grayscale artifact from Stage 1 as the OCR input:
  - `stage1_gray.png`
- Do not use `stage1_gray_equalized.png` as the primary OCR input unless a later comparison run proves it is better on the target drawing set.
- The current Stage 2 tuning values for production-oriented runs are:
  - `slice_height = 1600`
  - `slice_width = 1600`
  - `overlap_height_ratio = 0.2`
  - `overlap_width_ratio = 0.2`
  - `postprocess_type = GREEDYNMM`
  - `postprocess_match_metric = IOS`
  - `postprocess_match_threshold = 0.1`
  - `min_score = 0.2`
  - `min_text_len = 2`
  - `text_threshold = 0.7`
  - `low_text = 0.3`
  - `link_threshold = 0.7`
  - `line_merge_gap_px = 24`
  - `line_merge_y_tolerance_px = 10`
  - `enable_rotated_ocr = true`
  - `paragraph = false`
- The current intent of these values:
  - let SAHI handle tile overlap merging with a permissive OCR-oriented IOS threshold
  - keep same-line engineering text together as one text region when spacing and baseline support it
  - recover more rotated or vertical text by running OCR per tile in multiple orientations
  - stay conservative enough that the shared artifact contract remains usable by a later Gemini route
- Operational note:
  - this baseline improves OCR quality on the current sample, but it increases CPU runtime because each tile is processed in multiple orientations
  - treat this as the current quality-first baseline, not the final speed-optimized baseline

## Frontend review impact
- Pipeline mode should show Stage 2 as OCR discovery over tiles, not as a full-sheet single pass.
- Results should surface:
  - discovered text region count
  - exception candidate count
- The review UI should treat the EasyOCR route as one complete Stage 2 run, not as a half-finished chain waiting for Stage 3.

## Risks and guardrails
- Tile overlap can produce duplicate detections if merge logic is weak.
- EasyOCR class assignment can become noisy if classification is too aggressive.
- The pipeline must preserve provenance so future debugging can answer:
  - what EasyOCR found
  - which OCR route produced the result
  - what remained uncertain

## Immediate implementation consequences
- Slice 2 remains the EasyOCR route implementation: real SAHI-backed OCR discovery from the Stage 1 image bundle.
- The selectable-route architecture is now defined separately in [`docs/plans/2026-03-08-selectable-ocr-routes-design.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-selectable-ocr-routes-design.md).
- Do not turn this Slice 2 plan back into a mandatory EasyOCR-then-Gemini chain.
