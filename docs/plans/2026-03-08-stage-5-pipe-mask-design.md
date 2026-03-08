# Stage 5 Pipe Mask Design

## Goal
- Add the first geometry stage to the rebuilt pipeline.
- Produce a reviewable binary pipe mask from the raster drawing without mixing in skeletonization, topology, or semantic attachment.
- Keep the stage small, conservative, and traceable.

## Approved direction
- Add a new pipeline stage named `stage5_pipe_mask`.
- Stage 5 is only pipe-mask generation.
- Do not combine Stage 5 with:
  - morphology sealing
  - skeletonization
  - node detection
  - graph construction
- Use current evidence layers only as suppression aids:
  - Stage 1 normalized views
  - Stage 2 OCR boxes
  - Stage 4 object boxes

## Why this stage exists now
- The master plan is geometry-first.
- Stage 1, Stage 2, and Stage 4 already provide reviewable evidence.
- The next missing backbone is provisional pipe geometry.
- If pipe-mask extraction is not isolated, later failures become hard to debug because mask generation, morphology, and skeleton artifacts blur together.

## Scope

### In scope
- Generate a provisional binary pipe mask from the raw image and current evidence.
- Suppress obvious text and object contamination conservatively.
- Filter small obvious non-pipe blobs.
- Emit visible and machine-readable Stage 5 artifacts.

### Out of scope
- Morphological sealing and gap repair
- Skeletonization
- Endpoint or junction detection
- OCR-object attachment
- Topology or graph reasoning

## Stage numbering rule
- Keep the stage named `stage5_pipe_mask`.
- Continue using sparse stage numbering honestly:
  - `stage1_input_normalization`
  - `stage2_ocr_discovery`
  - `stage4_object_detection`
  - `stage5_pipe_mask`
- Do not add fake placeholder stages just to keep numbering contiguous.

## Methodology

### 1. Build a line-faithful working view
- Start from the raw image and Stage 1 outputs.
- Use `stage1_gray.png` as the primary intensity view.
- Use `stage1_binary_adaptive.png` and `stage1_binary_otsu.png` as supporting binary evidence.
- Treat these as candidate views, not truth.

### 2. Generate a candidate line mask
- Detect dark line-like content from the grayscale and binary views.
- Preserve thin pipe strokes.
- Prefer conservative operations:
  - threshold combination
  - light blur or denoise only if needed
  - optional line-emphasis filtering if it stays local and reviewable
- Do not repair broken pipes in Stage 5.

### 3. Suppress text contamination
- Use Stage 2 OCR boxes to remove text-heavy regions from the candidate mask.
- Suppression should be box-based and conservative.
- Track how much mask area is removed by OCR suppression.

### 4. Suppress object interiors conservatively
- Use Stage 4 object boxes to reduce non-pipe interiors.
- Do not erase aggressively around box borders because many pipes terminate at or pass through symbols.
- A safe first step is to suppress object interiors while preserving a configurable border margin.

### 5. Filter obvious non-pipe connected components
- Remove very small blobs and specks.
- Keep elongated connected content even if imperfect.
- Do not make topological decisions yet.

### 6. Save the mask as provisional geometry evidence
- The output is a raw pipe mask for later refinement.
- Stage 6 will handle morphology sealing before skeletonization.

## Stage 5 artifacts
- `stage5_pipe_mask.png`
- `stage5_pipe_mask_overlay.png`
- `stage5_pipe_mask_summary.json`

## `stage5_pipe_mask_summary.json`
- Include:
  - `image_id`
  - `pass_type`
  - `mask_pixel_count`
  - `connected_component_count`
  - `small_component_removals`
  - `ocr_suppression_pixel_count`
  - `object_suppression_pixel_count`
  - `source_artifacts`
  - threshold and filter settings used

## Architecture

### Pipeline runner
- [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) remains the orchestrator.
- Add:
  - `stage5_pipe_mask`
- It should read:
  - Stage 1 artifacts from the run directory
  - Stage 2 OCR JSON
  - Stage 4 object JSON

### Helper module
- Add a focused helper:
  - [`backend/garnet/pipe_mask.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_mask.py)
- Responsibilities:
  - build candidate mask
  - apply OCR/object suppression
  - remove tiny components
  - render overlay
  - emit summary metrics

## Configuration
- Add Stage 5 thresholds to `PipelineConfig` rather than hardcoding them in helpers.
- Initial config should stay small:
  - OCR suppression padding
  - object suppression inset or border preserve margin
  - minimum component area
  - candidate threshold strategy selector if needed

## Frontend impact
- Keep the current Pipeline review screen.
- No new controls yet.
- Stage 5 should appear as another completed stage in the manifest/progress flow.
- Results view should show:
  - pipe-mask image
  - overlay image
  - summary JSON

## Verification
- Add helper tests for:
  - OCR suppression removes masked text regions
  - tiny-component filtering removes obvious specks
  - overlay rendering works
- Add pipeline runner tests for:
  - Stage 5 artifact writing
  - sparse `stop_after=5`
- Add API test for:
  - pipeline job with `stop_after=5`
  - returned artifact list includes Stage 5 outputs
- Run one real sample on `sample.png`.

## Risks
- Over-suppressing object boxes can erase nearby pipe segments.
- Under-suppressing OCR boxes can leave text strokes inside the pipe mask.
- If Stage 5 does too much repair work, Stage 6 loses its purpose and debugging becomes harder.

## Success criteria
- The output mask visibly highlights likely pipe geometry more than text or symbol interiors.
- The stage is reviewable on its own from code, overlay, and summary metrics.
- Later morphology and skeleton stages can build on it without redoing evidence suppression.
