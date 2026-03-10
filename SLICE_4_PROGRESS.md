# Slice 4 Progress Log

## Purpose
- Working log for the Stage 4 object-detection slice.
- Track the first pipeline-native detection stage after Stage 1 and Stage 2.

## Entry format
- Timestamp
- Task
- Action
- Evidence
- Verification
- Next step / blocker

## Entries

### 2026-03-08 21:00 ICT
- Task: `Slice 4 / Design`
- Action: Fixed the first Stage 4 baseline to one production review path using Ultralytics + SAHI with [`backend/yolo_weights/yolo11n_PPCL_640_20250204.pt`](/Users/maetee/Code/GARNET/backend/yolo_weights/yolo11n_PPCL_640_20250204.pt).
- Evidence:
  - [`docs/plans/2026-03-08-stage-4-object-detection-design.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-stage-4-object-detection-design.md)
  - [`docs/plans/2026-03-08-stage-4-object-detection.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-stage-4-object-detection.md)
- Verification:
  - design and implementation docs saved
- Next step / blocker:
  - implement sparse stage numbering and the Stage 4 artifact bundle.

### 2026-03-08 21:06 ICT
- Task: `Slice 4 / Implementation`
- Action: Added the pipeline-native detection helper, wired `stage4_object_detection` into the runner, updated the API to allow `stop_after=4`, and updated the frontend pipeline flow to review Stage 4 artifacts.
- Evidence:
  - [`backend/garnet/object_detection_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/object_detection_sahi.py)
  - [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py)
  - [`frontend/src/lib/api.ts`](/Users/maetee/Code/GARNET/frontend/src/lib/api.ts)
  - [`frontend/src/stores/appStore.ts`](/Users/maetee/Code/GARNET/frontend/src/stores/appStore.ts)
  - [`frontend/src/components/DetectionSetup.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/DetectionSetup.tsx)
  - [`frontend/src/components/PipelineResultsView.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/PipelineResultsView.tsx)
- Verification:
  - `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py` -> pass
  - `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_object_detection_sahi.py' -v` -> pass
  - `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v` -> pass
  - `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_pipeline_api.py' -v` -> pass
  - `cd frontend && bun run build` -> pass
- Next step / blocker:
  - wait for the real `sample.png` run at `backend/output/stage4_sample` to finish and confirm the Stage 4 artifact bundle.

### 2026-03-08 21:10 ICT
- Task: `Slice 4 / Real sample`
- Action: Ran the full Stage 1 -> Stage 2 -> Stage 4 pipeline on [`backend/sample.png`](/Users/maetee/Code/GARNET/backend/sample.png) with the fixed Stage 4 baseline weight.
- Evidence:
  - [`backend/output/stage4_sample/stage_manifest.json`](/Users/maetee/Code/GARNET/backend/output/stage4_sample/stage_manifest.json)
  - [`backend/output/stage4_sample/stage4_objects.json`](/Users/maetee/Code/GARNET/backend/output/stage4_sample/stage4_objects.json)
  - [`backend/output/stage4_sample/stage4_objects_summary.json`](/Users/maetee/Code/GARNET/backend/output/stage4_sample/stage4_objects_summary.json)
  - [`backend/output/stage4_sample/stage4_objects_overlay.png`](/Users/maetee/Code/GARNET/backend/output/stage4_sample/stage4_objects_overlay.png)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/stage4_sample --ocr-route easyocr --stop-after 4` -> pass
  - manifest shows:
    - Stage 2 completed in `229.404717s`
    - Stage 4 completed in `3.473007s`
  - Stage 4 summary reports `object_count = 165`
- Next step / blocker:
  - the next slice should turn generic Stage 4 object evidence into more explicit category tables or begin the next geometry stage, depending on whether you want to keep building evidence first or move into pipe-mask extraction.

### 2026-03-09 09:40 ICT
- Task: `Slice 4 / Line number fusion`
- Action: Added a dedicated `stage4_line_number_fusion` substage that uses Stage 4 `line number` detections as anchors, fuses nearby Stage 2 OCR fragments into a curated line-number list, and writes a review overlay with accepted boxes in blue and rejected boxes in red.
- Evidence:
  - [`backend/garnet/line_number_fusion.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/line_number_fusion.py)
  - [`backend/output/pid_extractor_stage13_latest_full/stage4_line_numbers.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/pid_extractor_stage13_latest_full/stage4_line_numbers.json)
  - [`backend/output/pid_extractor_stage13_latest_full/stage4_line_number_summary.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/pid_extractor_stage13_latest_full/stage4_line_number_summary.json)
  - [`backend/output/pid_extractor_stage13_latest_full/stage4_line_number_overlay.png`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/pid_extractor_stage13_latest_full/stage4_line_number_overlay.png)
- Verification:
  - `cd backend && python -m unittest discover -s tests -p 'test*.py' -v` -> pass with API tests skipped when `pdf2image` is unavailable
  - `cd backend && python -m garnet.pid_extractor --image sample.png --out output/pid_extractor_stage13_latest_full --ocr-route easyocr --stop-after 13` -> pass
  - Stage 4 line-number summary reports:
    - `line_number_object_count = 23`
    - `matched_line_number_count = 21`
    - `rejected_line_number_count = 2`
- Next step / blocker:
  - expose the fused line-number list more directly in the frontend review flow and keep dedicated equipment detection separate as planned `Stage 4.1`.

### 2026-03-10 08:25 ICT
- Task: `Slice 4 / Validated semantic refinement`
- Action: Tightened Stage 4 semantic fusion so line numbers prefer fuller crop OCR over partial sheet OCR, reject bogus OD-only page-border/title artifacts, and instrumentation semantics use tighter balloon crops plus crop provenance. This turned the remaining residual line-number and instrument OCR issues into either real geometry cases or explicit rejects.
- Evidence:
  - [`backend/garnet/line_number_fusion.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/line_number_fusion.py)
  - [`backend/garnet/instrument_tag_fusion.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/instrument_tag_fusion.py)
  - [`backend/output/validation3x_Test-00009/stage4_line_number_summary.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/validation3x_Test-00009/stage4_line_number_summary.json)
  - [`backend/output/validation3x_Test-00001/stage4_line_number_summary.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/validation3x_Test-00001/stage4_line_number_summary.json)
  - [`backend/output/validation3x_Test-00003/stage4_line_number_summary.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/validation3x_Test-00003/stage4_line_number_summary.json)
  - [`backend/output/validation3x_Test-00001/stage4_instrument_tag_summary.json`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/validation3x_Test-00001/stage4_instrument_tag_summary.json)
- Verification:
  - `cd backend && python -m unittest discover -s tests -p 'test_pipe_text_attachment.py' -v` -> pass
  - Random PPCL validation reruns:
    - `Test-00009`: line numbers `27/27` OCR-confirmed, `0` OD-only; instrumentation `14/14` OCR-confirmed
    - `Test-00001`: line numbers `22/23` OCR-confirmed with `1` explicit reject and `0` OD-only; instrumentation `33/33` OCR-confirmed
    - `Test-00003`: line numbers `11/13` OCR-confirmed with `2` explicit rejects and `0` OD-only; instrumentation `12/12` OCR-confirmed
- Next step / blocker:
  - dedicated equipment detection is still on hold as planned `Stage 4.1`
  - remaining Stage 4 line-number non-matches are now mostly true rejects rather than ambiguous OD-only cases
