# Slice 3 Progress Log

## Purpose
- Working log for the selectable OCR routes slice.
- Track the shift from a chained fallback design to one OCR route per pipeline run.

## Entry format
- Timestamp
- Task
- Action
- Evidence
- Verification
- Next step / blocker

## Entries

### 2026-03-08 15:10 ICT
- Task: `Slice 3 / Route selection`
- Action: Implemented the first selectable OCR route vertical slice:
  - required `ocr_route` in the pipeline job API
  - added `easyocr` and `gemini` Stage 2 dispatch in the pipeline
  - added a first Gemini OCR helper with prompt loading, `1024x1024` patch extraction, patch-grid execution, and crop fallback for empty or `<0.3` confidence patch results
  - added frontend route selection and route-aware pipeline review metadata
- Evidence:
  - [`docs/plans/2026-03-08-selectable-ocr-routes-design.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-selectable-ocr-routes-design.md)
  - [`docs/plans/2026-03-08-selectable-ocr-routes.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-selectable-ocr-routes.md)
  - [`backend/garnet/gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/gemini_ocr_sahi.py)
  - [`backend/tests/test_gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/tests/test_gemini_ocr_sahi.py)
  - [`backend/api.py`](/Users/maetee/Code/GARNET/backend/api.py)
  - [`frontend/src/components/DetectionSetup.tsx`](/Users/maetee/Code/GARNET/frontend/src/components/DetectionSetup.tsx)
- Verification:
  - `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py` -> pass
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v` -> pass
  - `cd frontend && bun run build` -> pass
- Next step / blocker:
  - the Gemini route has automated coverage and API/frontend wiring, but it still needs a real keyed sample run and output review before the OCR quality can be judged.

### 2026-03-08 19:59 ICT
- Task: `Slice 3 / Real SAHI integration`
- Action: Reworked [`backend/garnet/gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/gemini_ocr_sahi.py) to use real SAHI slicing and postprocessing instead of a custom patch loop. Added the Gemini OCR route baseline:
  - `postprocess_match_metric = IOS`
  - `postprocess_match_threshold = 0.1`
  - text/class payload selection by highest confidence among overlapping candidates
- Evidence:
  - [`backend/garnet/gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/garnet/gemini_ocr_sahi.py)
  - [`backend/tests/test_gemini_ocr_sahi.py`](/Users/maetee/Code/GARNET/backend/tests/test_gemini_ocr_sahi.py)
  - [`docs/plans/2026-03-08-selectable-ocr-routes-design.md`](/Users/maetee/Code/GARNET/docs/plans/2026-03-08-selectable-ocr-routes-design.md)
- Verification:
  - `cd backend && ../.venv/bin/python -m py_compile api.py garnet/*.py garnet/utils/*.py tests/*.py` -> pass
  - `cd backend && XDG_CACHE_HOME=../.tmp-cache MPLCONFIGDIR=../.tmp-mpl ../.venv/bin/python -m unittest discover -s tests -p 'test*.py' -v` -> pass
- Next step / blocker:
  - the route is now SAHI-backed, but the live OpenRouter run is still blocked by credential rejection rather than code behavior.
