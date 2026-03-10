# Slice 2 Progress Log

## Purpose
- Working log for the Stage 2 OCR rebuild slice.
- Track the first OCR-capable vertical slice after Stage 1 normalization.

## Entry format
- Timestamp
- Task
- Action
- Evidence
- Verification
- Next step / blocker

## Entries

### 2026-03-08 13:58 ICT
- Task: `Slice 2 / Contract`
- Action: Updated the backend tests to require a second pipeline stage named `stage2_ocr_discovery`, with Stage 2 OCR artifacts exposed through the pipeline job API.
- Evidence:
  - [`backend/tests/test_pid_extractor_cli.py`](/Users/maetee/Code/GARNET/backend/tests/test_pid_extractor_cli.py)
  - [`backend/tests/test_pipeline_api.py`](/Users/maetee/Code/GARNET/backend/tests/test_pipeline_api.py)
- Verification:
  - red run failed for the expected reason: the runner still exposed only Stage 1 and the API rejected `stop_after=2`
- Next step / blocker:
  - Implement the minimal EasyOCR tiled OCR path and allow Stage 2 in the job API.

### 2026-03-08 14:01 ICT
- Task: `Slice 2 / Backend implementation`
- Action: Added a new SAHI-style EasyOCR helper and wired Stage 2 OCR discovery into the pipeline runner and job API.
- Evidence:
  - [`backend/garnet/easyocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/easyocr_sahi.py)
  - [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py)
- Verification:
  - `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py` -> pass
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v` -> pass
- Next step / blocker:
  - Confirm the real sample run produces the new Stage 2 artifact bundle and then align the frontend wording/default stage target.

### 2026-03-08 14:04 ICT
- Task: `Slice 2 / Frontend integration`
- Action: Updated the frontend pipeline mode to target Stage 2 by default and changed the UI copy from Slice 1-only messaging to Stage 1 + Stage 2 OCR messaging.
- Evidence:
  - [`frontend/src/stores/appStore.ts`](/Users/maetee/Code/GARNET/frontend/src/stores/appStore.ts)
  - [`frontend/src/components/DetectionSetup.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/DetectionSetup.tsx)
  - [`frontend/src/components/PipelineResultsView.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/PipelineResultsView.tsx)
  - [`frontend/src/lib/api.ts`](/Users/maetee/Code/GARNET/frontend/src/lib/api.ts)
- Verification:
  - `cd frontend && bun run build` -> pass
- Next step / blocker:
  - Record the real Stage 2 sample outputs and decide whether Stage 2 runtime needs tuning before Slice 3.

### 2026-03-08 14:05 ICT
- Task: `Slice 2 / Real sample`
- Action: Ran a real Stage 2 sample on [`backend/sample.png`](/Users/maetee/Code/GARNET/backend/sample.png) and confirmed the Stage 2 OCR artifact bundle is written.
- Evidence:
  - [`backend/output/slice2_ocr/stage_manifest.json`](/Users/maetee/Code/GARNET/backend/output/slice2_ocr/stage_manifest.json)
  - [`backend/output/slice2_ocr/stage2_ocr_regions.json`](/Users/maetee/Code/GARNET/backend/output/slice2_ocr/stage2_ocr_regions.json)
  - [`backend/output/slice2_ocr/stage2_ocr_summary.json`](/Users/maetee/Code/GARNET/backend/output/slice2_ocr/stage2_ocr_summary.json)
  - [`backend/output/slice2_ocr/stage2_ocr_exception_candidates.json`](/Users/maetee/Code/GARNET/backend/output/slice2_ocr/stage2_ocr_exception_candidates.json)
  - [`backend/output/slice2_ocr/stage2_ocr_overlay.png`](/Users/maetee/Code/GARNET/backend/output/slice2_ocr/stage2_ocr_overlay.png)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/slice2_ocr --stop-after 2` -> completed with Stage 2 duration recorded in the manifest
- Next step / blocker:
  - Stage 2 is functionally complete, but CPU runtime is still high on the sample image. The next slice should keep Stage 3 fallback separate and may also tune Stage 2 tile sizing and exception thresholds.

### 2026-03-08 14:10 ICT
- Task: `Slice 2 / OCR tuning`
- Action: Switched the Stage 2 OCR input from the equalized grayscale artifact to the plain grayscale artifact because the equalized view was producing poor OCR quality.
- Evidence:
  - [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - regression coverage in [`backend/tests/test_pid_extractor_cli.py`](/Users/maetee/Code/GARNET/backend/tests/test_pid_extractor_cli.py)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v` -> pass
  - fresh sample rerun started at [`backend/output/slice2_ocr_gray`](/Users/maetee/Code/GARNET/backend/output/slice2_ocr_gray)
- Next step / blocker:
  - Compare the gray-input OCR output against the previous equalized-input run and decide whether Stage 2 also needs tile-size or overlap tuning.

### 2026-03-08 14:20 ICT
- Task: `Slice 2 / Production baseline`
- Action: Recorded the accepted Stage 2 production tuning baseline after the gray-image switch, rotated OCR pass, and same-line merge tuning.
- Evidence:
  - [`docs/plans/2026-03-08-slice-2-ocr-sahi-design.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-slice-2-ocr-sahi-design.md)
  - [`backend/garnet/AGENTS.md`](/Users/maetee/Code/GARNET/backend/garnet/AGENTS.md)
- Verification:
  - documentation readback confirmed the accepted values and rationale are present
- Next step / blocker:
  - keep Gemini fallback separate and treat Stage 2 runtime tuning as a follow-up optimization pass.

### 2026-03-08 20:28 ICT
- Task: `Slice 2 / Real SAHI integration`
- Action: Replaced the EasyOCR custom tile loop with a real SAHI `DetectionModel` + `get_sliced_prediction(...)` path, keeping rotated OCR and the post-SAHI same-line merge on top.
- Evidence:
  - [`backend/garnet/easyocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/easyocr_sahi.py)
  - [`backend/tests/test_easyocr_sahi.py`](/Users/maetee/Code/GARNET/backend/tests/test_easyocr_sahi.py)
- Verification:
  - `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_easyocr_sahi.py' -v` -> pass
- Next step / blocker:
  - run the broader backend regression suite and, if needed, re-benchmark the EasyOCR sample run with the real SAHI merge path.
- Task: `Slice 2 / OCR tuning`
- Action: Tuned Stage 2 OCR for two concrete failure modes:
  - merge adjacent same-line text boxes into one text region
  - run orientation-aware OCR per tile to recover more rotated text
- Evidence:
  - [`backend/garnet/easyocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/easyocr_sahi.py)
  - helper regression coverage in [`backend/tests/test_easyocr_sahi.py`](/Users/maetee/Code/GARNET/backend/tests/test_easyocr_sahi.py)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test_easyocr_sahi.py' -v` -> pass
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v` -> pass
- Next step / blocker:
  - Review the new tuned sample output under [`backend/output/slice2_ocr_tuned`](/Users/maetee/Code/GARNET/backend/output/slice2_ocr_tuned) after the CPU OCR run completes.

### 2026-03-08 14:30 ICT
- Task: `Slice 2 / Production baseline`
- Action: Captured the currently accepted Stage 2 OCR settings as the quality-first baseline for production-oriented runs.
- Evidence:
  - [`docs/plans/2026-03-08-slice-2-ocr-sahi-design.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-slice-2-ocr-sahi-design.md)
  - [`backend/garnet/AGENTS.md`](/Users/maetee/Code/GARNET/backend/garnet/AGENTS.md)
- Verification:
  - documented baseline includes the gray-image input rule and the current accepted OCR parameters
- Next step / blocker:
  - future tuning should compare quality and runtime against this baseline, not against the older equalized-image configuration.

### 2026-03-08 15:10 ICT
- Task: `Slice 3 / Selectable OCR routes`
- Action: Replaced the planned mandatory EasyOCR -> Gemini chain with user-selected OCR routing per pipeline run. Added backend route selection, API validation, a first Gemini OCR helper with `1024x1024` patching and low-confidence crop fallback, and frontend route selection in Pipeline mode.
- Evidence:
  - [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - [`backend/garnet/gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/gemini_ocr_sahi.py)
  - [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py)
  - [`backend/tests/test_gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/tests/test_gemini_ocr_sahi.py)
  - [`frontend/src/components/DetectionSetup.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/DetectionSetup.tsx)
  - [`frontend/src/stores/appStore.ts`](/Users/maetee/Code/GARNET/frontend/src/stores/appStore.ts)
  - [`frontend/src/lib/api.ts`](/Users/maetee/Code/GARNET/frontend/src/lib/api.ts)
  - [`frontend/src/components/PipelineResultsView.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/PipelineResultsView.tsx)
- Verification:
  - `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py` -> pass
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v` -> pass
  - `cd frontend && bun run build` -> pass
- Next step / blocker:
  - run a real Gemini-route sample once `OPENROUTER_API_KEY` is available locally and review the raw patch/crop artifacts before calling the route production-ready.

### 2026-03-09 13:20 ICT
- Task: `Slice 2 / OCRMac route`
- Action: Added a new macOS-only `ocrmac` Stage 2 route using the Vision framework with tiled sheet processing and the shared Stage 2 OCR artifact contract.
- Evidence:
  - [`backend/garnet/ocrmac_sahi.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/ocrmac_sahi.py)
  - [`backend/garnet/pid_extractor.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - [`backend/api.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/api.py)
  - [`frontend/src/components/DetectionSetup.tsx`](/Volumes/Ginnungagap/maetee/Code/GARNET/frontend/src/components/DetectionSetup.tsx)
- Verification:
  - `cd backend && python -m unittest discover -s tests -p 'test*.py' -v` -> pass with API tests skipped when `pdf2image` is unavailable
  - `cd backend && python -m garnet.pid_extractor --image sample.png --out output/pid_extractor_ocrmac_sample --ocr-route ocrmac --stop-after 2` -> pass
  - OCRMac sample summary reports:
    - `tile_count = 9`
    - `raw_detection_count = 593`
    - `merged_region_count = 413`
    - `exception_candidate_count = 315`
    - Stage 2 duration `3.333389s`
- Next step / blocker:
  - compare OCRMac quality against the EasyOCR baseline on the same sample and decide whether OCRMac should become a preferred macOS route or stay as an optional alternative.

### 2026-03-11 06:35 ICT
- Task: `Slice 2 / OCRMac rotated line numbers`
- Action: Extended the `ocrmac` Stage 2 route to run 90-degree tile passes, restore rotated OCR boxes back into sheet coordinates, and carry rotation metadata through the shared Stage 2 contract so vertical line numbers survive into Stage 4 fusion. Re-ran the sample sheet and a 9-image PPCL batch to validate the change on real drawings.
- Evidence:
  - [`backend/garnet/ocrmac_sahi.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/ocrmac_sahi.py)
  - [`backend/garnet/pid_extractor.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - [`backend/tests/test_ocrmac_sahi.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/tests/test_ocrmac_sahi.py)
  - [`backend/tests/test_pid_extractor_cli.py`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/tests/test_pid_extractor_cli.py)
  - sample rerun: [`backend/output/ocrmac_sample_rerun_20260311`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/ocrmac_sample_rerun_20260311)
  - PPCL reruns: [`backend/output/ppcl_batch_20260311`](/Volumes/Ginnungagap/maetee/Code/GARNET/backend/output/ppcl_batch_20260311)
- Verification:
  - `cd backend && python -m py_compile api.py garnet/*.py garnet/utils/*.py` -> pass
  - `cd backend && python -m unittest discover -s tests -p 'test_ocrmac_sahi.py' -v` -> pass
  - `cd backend && python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v` -> pass
  - `cd backend && python -m garnet.pid_extractor --image sample.png --ocr-route ocrmac --stop-after 4 --out output/ocrmac_sample_rerun_20260311` -> pass
  - `cd backend && for img in test/ppcl/Test-*.jpg; do python -m garnet.pid_extractor --image "$img" --ocr-route ocrmac --stop-after 4 --out "output/ppcl_batch_20260311/$(basename "${img%.jpg}")"; done` -> pass
  - Sample rerun summary:
    - `merged_region_count = 443`
    - `rotated_regions = 63`
    - `line_number_object_count = 23`
    - `matched_line_number_count = 23`
    - `ocr_confirmed_line_number_count = 23`
  - PPCL batch summary:
    - all 9 images completed through Stage 4
    - per-sheet rotated OCR region counts ranged from `39` to `82`
    - two sheets (`Test-00001`, `Test-00009`) reached `0` rejected line numbers
- Next step / blocker:
  - keep Stage 4 line-number fusion tuning separate; the current Slice 2 result is good enough to move forward without expecting 100 percent recall yet.
