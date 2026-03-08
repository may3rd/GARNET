# Slice 5 Progress Log

## Purpose
- Working log for the Stage 5 pipe-mask slice.
- Track the first geometry-focused stage after Stage 1 normalization, Stage 2 OCR, and Stage 4 object detection.

## Entry format
- Timestamp
- Task
- Action
- Evidence
- Verification
- Next step / blocker

## Entries

### 2026-03-08 21:20 ICT
- Task: `Slice 5 / Design`
- Action: Locked the Stage 5 methodology to a conservative provisional pipe-mask stage only. Stage 5 starts from Stage 1 binary views, suppresses OCR and object evidence conservatively, removes tiny blobs, and stops before morphology, skeletonization, or graph work.
- Evidence:
  - [`docs/plans/2026-03-08-stage-5-pipe-mask-design.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-stage-5-pipe-mask-design.md)
  - [`docs/plans/2026-03-08-stage-5-pipe-mask.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-stage-5-pipe-mask.md)
- Verification:
  - design and implementation docs saved
- Next step / blocker:
  - implement the helper module and wire Stage 5 into the sparse stage runner.

### 2026-03-08 21:28 ICT
- Task: `Slice 5 / Implementation`
- Action: Added the Stage 5 helper module, wired `stage5_pipe_mask` into the pipeline runner, updated API/frontend `stop_after=5` flow, and kept OCR route handling extensible so `easyocr`, `gemini`, and `paddleocr` all feed the same Stage 5 inputs.
- Evidence:
  - [`backend/garnet/pipe_mask.py`](/Users/maetee/Code/GARNET/backend/garnet/pipe_mask.py)
  - [`backend/garnet/pid_extractor.py`](/Users/maetee/Code/GARNET/backend/garnet/pid_extractor.py)
  - [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py)
  - [`frontend/src/lib/api.ts`](/Users/maetee/Code/GARNET/frontend/src/lib/api.ts)
  - [`frontend/src/stores/appStore.ts`](/Users/maetee/Code/GARNET/frontend/src/stores/appStore.ts)
  - [`frontend/src/components/DetectionSetup.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/DetectionSetup.tsx)
  - [`frontend/src/components/PipelineResultsView.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/PipelineResultsView.tsx)
- Verification:
  - `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_pipe_mask.py' -v` -> pass
  - `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_pid_extractor_cli.py' -v` -> pass
  - `cd backend && ../.venv/bin/python -m unittest discover -s tests -p 'test_pipeline_api.py' -v` -> pass
  - `cd frontend && bun run build` -> pass
- Next step / blocker:
  - run a real `sample.png` Stage 5 pipeline and record the artifact bundle.

### 2026-03-08 21:41 ICT
- Task: `Slice 5 / Real sample`
- Action: Ran the full Stage 1 -> Stage 2 -> Stage 4 -> Stage 5 pipeline on [`backend/sample.png`](/Users/maetee/Code/GARNET/backend/sample.png) using the `easyocr` Stage 2 route and recorded the first real Stage 5 artifact bundle.
- Evidence:
  - [`backend/output/stage5_sample/stage_manifest.json`](/Users/maetee/Code/GARNET/backend/output/stage5_sample/stage_manifest.json)
  - [`backend/output/stage5_sample/stage5_pipe_mask.png`](/Users/maetee/Code/GARNET/backend/output/stage5_sample/stage5_pipe_mask.png)
  - [`backend/output/stage5_sample/stage5_pipe_mask_overlay.png`](/Users/maetee/Code/GARNET/backend/output/stage5_sample/stage5_pipe_mask_overlay.png)
  - [`backend/output/stage5_sample/stage5_pipe_mask_summary.json`](/Users/maetee/Code/GARNET/backend/output/stage5_sample/stage5_pipe_mask_summary.json)
- Verification:
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m garnet.pid_extractor --image sample.png --out output/stage5_sample --ocr-route easyocr --stop-after 5` -> pass
  - manifest shows:
    - Stage 2 completed in `175.457322s`
    - Stage 4 completed in `2.817066s`
    - Stage 5 completed in `2.330667s`
  - Stage 5 summary reports:
    - `mask_pixel_count = 223570`
    - `connected_component_count = 452`
    - `small_component_removals = 38`
    - `ocr_suppression_pixel_count = 427068`
    - `object_suppression_pixel_count = 82649`
- Next step / blocker:
  - Stage 5 is complete as a provisional mask stage. The next slice should be Stage 6 morphology/sealing or a review pass on the accepted Stage 5 suppression parameters.
