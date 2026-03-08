# Slice 2 OCR SAHI Design

## Goal
- Define Slice 2 and Slice 3 OCR architecture for the rebuilt P&ID pipeline.
- Keep the rebuild image-only.
- Use EasyOCR as the primary OCR detector.
- Use Gemini/OpenRouter as a fallback refiner only for exception cases.

## Approved direction
- Stage 2 uses SAHI-style patching over the normalized P&ID image and runs EasyOCR on each patch.
- Stage 3 reuses the Gemini/OpenRouter path from [`backend/gemini_detector/gemini_sahi.py`](/Users/maetee/Code/GARNET/backend/gemini_detector/gemini_sahi.py) for exception handling only.
- EasyOCR remains the first-pass source of OCR detections.
- VLM is not allowed to replace full-sheet OCR in this slice.

## Why SAHI-style patching is required
- Full-sheet OCR on dense P&ID drawings under-detects small text and mixed-orientation labels.
- Tiling improves local scale and recall without forcing global upscaling of the entire sheet.
- The repo already has a proven SAHI-style pattern in [`backend/gemini_detector/gemini_sahi.py`](/Users/maetee/Code/GARNET/backend/gemini_detector/gemini_sahi.py).
- Reusing the same tiling mental model keeps Stage 2 and Stage 3 consistent.

## Stage boundaries

### Stage 2: EasyOCR discovery
- Stage name: `stage2_ocr_discovery`
- Input:
  - Stage 1 normalized image outputs from [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - Primary working image should be the most OCR-friendly normalized view, with room to compare additional views later if needed.
- Core flow:
  - cut the sheet into overlapping tiles
  - run EasyOCR on each tile
  - convert tile-local detections to sheet-local coordinates
  - merge duplicate detections across overlapping tiles
  - classify obvious text categories conservatively
  - emit exception candidates for Stage 3
- Non-goals:
  - no VLM in Stage 2
  - no graph reasoning
  - no topology or semantic attachment

### Stage 3: Gemini fallback / refiner
- Stage name: `stage3_ocr_refinement`
- Input:
  - Stage 2 OCR output
  - exception candidate crops from Stage 2
- Core flow:
  - crop only flagged OCR regions
  - send each crop to Gemini/OpenRouter using the existing Gemini path
  - refine text, class, bbox, rotation, reading direction, and legibility when the crop supports it
  - merge refinement results back into the canonical OCR table
  - emit unresolved cases when the fallback cannot safely improve the record
- Non-goals:
  - no full-sheet VLM OCR
  - no replacing valid EasyOCR records without evidence from the crop

## Stage 2 module design
- Add a new helper module under [`backend/garnet/`](/Users/maetee/Code/GARNET/backend/garnet), likely `easyocr_sahi.py`.
- The module should follow the same broad adapter pattern as [`backend/gemini_detector/gemini_sahi.py`](/Users/maetee/Code/GARNET/backend/gemini_detector/gemini_sahi.py), but for OCR tiles instead of VLM object detections.
- Responsibilities:
  - tile generation and overlap settings
  - EasyOCR reader initialization and reuse
  - tile-local OCR execution
  - coordinate shifting from tile to sheet
  - duplicate merging across overlaps
  - conservative exception flagging

## Canonical OCR schema

### Shared record shape
- Both Stage 2 and Stage 3 outputs should use the same `text_regions[]` record shape so downstream code and frontend review do not need format branching.
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

### Stage 3 output contract
- Stage 3 Gemini refinement must use the approved crop contract exactly for each crop response:
  - `pass_type` must be `"crop"`
  - coordinates must be crop-local
  - `text_regions` records must follow the exact field set and allowed values specified by the user
- The merged Stage 3 artifact can add provenance fields outside the Gemini raw response, but the stored raw Gemini output should remain faithful to that crop schema.

## Classification policy
- Stage 2 classification is conservative and local-only.
- Default to `unknown` unless the crop strongly supports another allowed class.
- EasyOCR is primarily responsible for text localization and transcription.
- Gemini can refine class labels only for exception crops.

## Exception policy for Stage 3
- Send a region to Stage 3 only if one or more of these conditions apply:
  - low confidence
  - partial or degraded legibility
  - rotated or vertical reading direction
  - merged multi-string region suspected
  - duplicate conflict after cross-tile merge
  - unsafe normalization
  - uncertain class assignment

## Proposed Stage 2 artifacts
- `stage2_ocr_regions.json`
- `stage2_ocr_overlay.png`
- `stage2_ocr_summary.json`
- `stage2_ocr_exception_candidates.json`
- optional debug folder for tile outputs when enabled

## Proposed Stage 3 artifacts
- `stage3_ocr_refined.json`
- `stage3_ocr_comparison.json`
- `stage3_ocr_unresolved.json`
- optional raw crop-response folder when debug is enabled

## Frontend review impact
- Pipeline mode should show Stage 2 as OCR discovery over tiles, not as a full-sheet single pass.
- Results should surface:
  - discovered text region count
  - exception candidate count
  - refined count
  - unresolved count
- The review UI should keep Stage 2 and Stage 3 artifacts distinct so EasyOCR output and Gemini refinements remain auditable.

## Risks and guardrails
- Tile overlap can produce duplicate detections if merge logic is weak.
- EasyOCR class assignment can become noisy if classification is too aggressive.
- Gemini fallback can become expensive if exception thresholds are too loose.
- The pipeline must preserve provenance so future debugging can answer:
  - what EasyOCR found
  - what Gemini changed
  - what remained unresolved

## Immediate implementation consequences
- Slice 2 should implement Stage 2 only: SAHI-style EasyOCR discovery from the Stage 1 image bundle.
- Slice 3 should implement Gemini crop fallback on top of the Stage 2 exception queue.
- Do not combine Stages 2 and 3 into one release if the result hides which engine produced which output.
