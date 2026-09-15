# Slice 2 progress log

## 2026-08-25 — ocrmac recognition level `accurate` -> `fast`

**Change:** all ocrmac (macOS Vision) `recognition_level` call sites changed
from `accurate` to `fast`:
- `OcrMacSahiConfig.recognition_level` default — `garnet/ocrmac_sahi.py`
- `PipelineConfig.ocrmac_recognition_level` — `garnet/pid_extractor.py`
- crop OCR in `garnet/instrument_tag_fusion.py`
- crop OCR in `garnet/line_number_fusion.py`

**Reason:** macOS Vision's `accurate` recognizer returns zero detections in
this environment even on large, clear control text, while `fast` detects text
reliably. Under the `accurate` default the whole OCR text layer was empty
(0 `text_regions` on every sheet), which removed line numbers, page references,
and instrument-tag text. Downstream this left `connector_key` and
`target_sheet_reference` empty on every off-page connector, so the strict
multi-sheet merge produced `missing_connector_key` for all of them and zero
cross-sheet resolutions.

**Verification:** with `fast`, the route now emits ~340 text regions on
`Test-00009` (was 0), including connector-adjacent line numbers and drawing
references. Accepted values recorded in
`docs/plans/2026-03-08-slice-2-ocr-sahi-design.md`.
