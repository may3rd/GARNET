---
name: garnet-line-numbers
description: Extract piping line-number text labels from a GARNET P&ID sheet as Contract C JSON, using a vision LLM. Use for phrasings like "read the line numbers off this P&ID", "extract pipe line number text", "get line-number bounding boxes for this sheet", or "prep line-number JSON for GARNET". The YOLO weight has no line-number class and never reads text — this is the source for it.
---

# garnet-line-numbers

Every piping line-number label (e.g. `3"-CUL-25-002007-B1A2-NI`) on one P&ID sheet, as text
with a bounding box, in Contract C JSON.

**This is LLM work, and there is no script for it.** Unlike equipment
(`garnet/detect_equipment_bboxes.py`), line numbers have no automated extractor in the repo —
you do it inline here, or delegate to the `garnet-line-number-extractor` agent.

**Follow `backend/garnet/OCR_prompts/pid_line_number_extraction_prompt.md` step by step.** It
defines what counts as a line number (vs. instrument tags, equipment tags, page-connector
numbers, size callouts — all out of scope) and the transcription rules. Do not re-derive them.

Full contract reference (Contract C, fixture staging, frame rules):
`.claude/skills/garnet-digitize/reference/contracts.md`.

## Output shape

```json
{"id": "<sheet-id>", "objects": [
  {"Index": 1, "Object": "line number", "CategoryID": 1, "ObjectID": 1,
   "Left": 0, "Top": 0, "Width": 0, "Height": 0, "Score": 1.0, "Text": "..."}],
 "image_url": "", "image_width": 0, "image_height": 0, "count": 0}
```

Pixel `xywh`, source raster frame (per the prompt's coordinate rules).

## Constraints

1. **Frame must match the raster you trace.** `image_width`/`image_height` are compared
   against the real pixel size; on the backend route a mismatch is a hard error unless you pass
   `--ai-align` (translation-only fit, needs `stage4_line_numbers.json` to exist and ≥4
   confident text matches). Extract from the exact image you feed downstream.
2. **Empty `Text` is skipped**, not imported as a placeholder. Either read it or leave it out.
3. Backend-imported entries get id `line_number_ai_%06d` and `review_state: "ocr_confirmed"` —
   they are treated as trusted text, so only emit a string you are confident is printed exactly
   as transcribed.
4. Line numbers matter twice: they suppress text from the pipe mask (so the walker doesn't
   follow lettering) **and** they are the point of Stage 6 association. A missed vertical label
   costs you both.

## Feeding it downstream

**`garnet/` route** — stage as a fixture under either accepted name, then trace:

```bash
cp <your.json> <indir>/<stem>_line_numbers.json     # or <stem>_line_number_boxes.json
cd /Users/maetee/Code/GARNET
.venv/bin/python -m garnet.stage5b.run --stem <stem> --indir <indir> --out output/<stem>
.venv/bin/python -m garnet.stage5b.run_associations --stem <stem> --indir <indir>
```

Then check the association visually:
`.venv/bin/python -m garnet.stage5b.inspect_line_numbers --stem <stem>` → open
`inspection.html`. A label attaches to the path it is written **along**, not the nearest one,
so the tile shows both the winner (green) and the closer loser (blue).

**backend route:**

```bash
cd backend && ../.venv/bin/python -m garnet.pid_extractor \
  --image <sheet> --out <dir> --ai-line-numbers <path.json> --stop-after 11
```

## Multi-sheet work

Delegate each sheet to the `garnet-line-number-extractor` agent in parallel rather than doing
them serially in one conversation — each agent handles exactly one sheet and views its own
overlay.
