# Stage 4 Object Detection Design

## Goal
- Add the first object-detection slice to the rebuilt image-only pipeline.
- Reuse the existing Ultralytics + SAHI detection approach that already powers `/api/detect`.
- Keep the slice small, reviewable, and traceable.

## Approved direction
- Add a new pipeline stage named `stage4_object_detection`.
- Start with one fixed production baseline for review:
  - weight: [`backend/yolo_weights/yolo11n_PPCL_640_20250204.pt`](/Users/maetee/Code/GARNET/backend/yolo_weights/yolo11n_PPCL_640_20250204.pt)
  - model family: Ultralytics + SAHI
  - image size: `640`
  - overlap ratio: `0.2`
  - postprocess type: `GREEDYNMM`
  - postprocess match metric: `IOS`
  - postprocess match threshold: `0.1`
- Do not expose model or weight selection in Pipeline mode yet.

## Why this slice exists now
- Stage 1 normalization is already stable.
- Stage 2 OCR is already reviewable and route-selectable.
- The next missing evidence layer from the master plan is visual object evidence from the raw drawing.
- The repo already has a solid detection path, so this slice should reuse that approach rather than inventing a second detector stack.

## Scope

### In scope
- Pipeline Stage 4 object detection from raw image input.
- Fixed baseline detector configuration for initial review.
- Backend artifact writing and stage manifest integration.
- Pipeline API updates so the job system can run through Stage 4.
- Frontend Pipeline review updates to display the new stage and artifacts.

### Out of scope
- User-facing detector model/weight selection in Pipeline mode.
- Attachment logic between OCR and detected objects.
- Graph or topology reasoning.
- OCR-object reconciliation.

## Stage numbering rule
- Keep the stage named `stage4_object_detection`.
- Do not add a fake Stage 3 placeholder.
- The runner must support sparse implemented stage numbers.
- `stop_after` must be validated against the highest implemented stage number, not the count of implemented stages.
- Progress UI should use the manifest stage list, not assume contiguous stage numbering.

## Architecture

### Pipeline runner
- [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py) remains the orchestrator.
- Stage definitions become:
  - `stage1_input_normalization`
  - `stage2_ocr_discovery`
  - `stage4_object_detection`
- Stage 4 reads the original image path, not OCR artifacts.

### Detection helper
- Add a dedicated helper module for the pipeline detector route:
  - [`backend/garnet/object_detection_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/object_detection_sahi.py)
- It should follow the same Ultralytics + SAHI mechanics already used in [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py), but return pipeline-native artifacts instead of the legacy detection response shape.
- Keep this helper narrow:
  - load cached or local detection model for the fixed baseline
  - run `get_sliced_prediction(...)`
  - convert detections into a stable pipeline JSON contract
  - render an overlay image for review

## Stage 4 artifact contract

### Files
- `stage4_objects.json`
- `stage4_objects_overlay.png`
- `stage4_objects_summary.json`

### `stage4_objects.json`
- Sheet-level object evidence table.
- One record per detected object with:
  - `id`
  - `class_name`
  - `confidence`
  - `bbox`
  - `source_model`
  - `source_weight`

Example shape:

```json
{
  "image_id": "sample.png",
  "pass_type": "sheet",
  "objects": [
    {
      "id": "obj_000001",
      "class_name": "gate_valve",
      "confidence": 0.91,
      "bbox": {
        "x_min": 120,
        "y_min": 340,
        "x_max": 188,
        "y_max": 402
      },
      "source_model": "ultralytics",
      "source_weight": "yolo_weights/yolo11n_PPCL_640_20250204.pt"
    }
  ]
}
```

### `stage4_objects_summary.json`
- Include:
  - `image_id`
  - `pass_type`
  - `route`
  - `object_count`
  - `class_counts`
  - `image_size`
  - `overlap_ratio`
  - `postprocess_type`
  - `postprocess_match_metric`
  - `postprocess_match_threshold`
  - `source_model`
  - `source_weight`

## API impact
- `POST /api/pipeline/jobs` must allow `stop_after=4`.
- Job state should still store the current stage name from stage callbacks, not by indexing into the stage list with `stop_after - 1`.
- Serialization stays route-neutral; Stage 4 just adds more artifacts to the job bundle.

## Frontend impact
- Keep Pipeline mode simple.
- Do not add new controls yet.
- Change the pipeline run target to Stage 4 for the default vertical slice.
- Update pipeline review copy so it no longer says the pipeline ends at OCR.
- Show Stage 4 artifacts in the existing review layout:
  - object overlay image
  - objects JSON
  - objects summary JSON

## Verification
- Add backend tests for:
  - sparse stage numbering and `stop_after=4`
  - Stage 4 artifact generation
  - Stage 4 summary contains the fixed baseline weight
- Add API test for:
  - `POST /api/pipeline/jobs` with `stop_after=4`
  - artifact bundle includes Stage 4 outputs
- Run one real sample:
  - `sample.png`
  - `ocr_route=easyocr`
  - `stop_after=4`

## Risks
- Detection code now exists in two shapes unless later unified with `/api/detect`.
- Sparse stage numbering affects progress calculations if any caller assumes contiguous numbers.
- Stage 4 can be slower than OCR on CPU with SAHI enabled, so artifact-based review matters.
