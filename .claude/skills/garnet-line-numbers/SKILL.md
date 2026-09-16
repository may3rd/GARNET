---
name: garnet-line-numbers
description: Extract piping line-number text labels from a GARNET P&ID sheet as Contract C JSON, for import into the pipeline with --ai-line-numbers. Use for phrasings like "read the line numbers off this P&ID", "extract pipe line number text", "get line-number bounding boxes for this sheet", or "prep line-number JSON for GARNET". The default weight has no line-number class and YOLO never reads text — this is the source for it.
---

# garnet-line-numbers

Extracts every piping line-number label (e.g. `3"-CUL-25-002007-B1A2-NI`) from one P&ID
sheet, as text with a bounding box, as Contract C JSON for `--ai-line-numbers`.

**Follow the prompt at `backend/garnet/OCR_prompts/pid_line_number_extraction_prompt.md` step
by step.** It defines what counts as a line number (vs. instrument tags, equipment tags,
page-connector numbers, size callouts — all out of scope) and the transcription rules. Do not
re-derive them here.

Full contract reference (Contract C, frame-reconciliation rules): `.claude/skills/garnet-digitize/reference/contracts.md`.

## GARNET-specific constraints

1. **Frame must match the job image.** `image_width`/`image_height` in your output are
   compared exactly against the job's real pixel size; a mismatch is a hard error unless you
   pass `--ai-align` (translation-only fit, needs the job's own `stage4_line_numbers.json` to
   already exist and ≥4 confident text matches — see the reference doc). Extract from the exact
   raster you will feed to `--image`.
2. **Empty `Text` is skipped**, not imported as a placeholder. Don't emit a box with no
   transcription — either read it or leave it out.
3. Imported entries get id `line_number_ai_%06d` and `review_state: "ocr_confirmed"` — they are
   trusted text, so only emit a string you are confident is printed exactly as transcribed.

## Output shape

```json
{"id": "<sheet-id>", "objects": [
  {"Index": 1, "Object": "line number", "CategoryID": 1, "ObjectID": 1,
   "Left": 0, "Top": 0, "Width": 0, "Height": 0, "Score": 1.0, "Text": "..."}],
 "image_url": "", "image_width": 0, "image_height": 0, "count": 0}
```

Pixel `xywh`, source raster frame (per the prompt's coordinate rules).

## Feeding it back in

```bash
python -m garnet.pid_extractor --image <sheet> --out <dir> --ai-line-numbers <path.json> --stop-after 11
```

## Multi-sheet work

For more than one sheet, delegate each sheet to the `garnet-line-number-extractor` agent in
parallel rather than doing them serially in this conversation.
