# Slice 2 — ocrmac (macOS Vision) SAHI OCR design

## Accepted OCR baseline

- Primary OCR input: `stage1_gray.png`
- Route: ocrmac (`garnet/ocrmac_sahi.py`, macOS Vision framework)
- Recognition level: **`fast`** (see decision note below)
- Framework: `vision`
- Language preference: `en-US`
- Tiling: 1600x1600 slices, 0.2 overlap, rotated OCR enabled
- `postprocess_match_metric = IOS`
- `postprocess_match_threshold = 0.4`

## Decision note — recognition level changed `accurate` -> `fast`

macOS Vision's `accurate` recognizer returns **zero detections** in this
environment even for large, clear control text (verified: a control image with
"HELLO WORLD 12345 26-0003" yields 0 detections at `accurate`, 4 at `fast`).
Because every call site defaulted to `accurate`, the whole OCR text layer was
empty, which cascaded downstream: no line numbers, no instrument-tag text, no
page references, and therefore empty `connector_key` / `target_sheet_reference`
on every off-page connector (the strict multi-sheet merge then surfaced
`missing_connector_key` for all of them).

All call sites now use `fast`:
- `OcrMacSahiConfig.recognition_level` (default) — `garnet/ocrmac_sahi.py`
- `PipelineConfig.ocrmac_recognition_level` — `garnet/pid_extractor.py`
- crop-based OCR in `garnet/instrument_tag_fusion.py`
- crop-based OCR in `garnet/line_number_fusion.py`

Switching to `fast` restores OCR output: on a real sheet (`Test-00009`) the
route now emits ~340 text regions (was 0), including line numbers and drawing
references near connectors that populate connector keys.

Recorded per `backend/garnet/AGENTS.md` OCR tuning guidance; reason logged in
`SLICE_2_PROGRESS.md`.
