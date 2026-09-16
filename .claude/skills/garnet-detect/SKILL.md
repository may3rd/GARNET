---
name: garnet-detect
description: Run YOLO+SAHI symbol detection on a GARNET P&ID raster to get the object list (valves, instrument tags, connections, arrows, nodes, etc). Use for phrasings like "detect objects on this P&ID", "run YOLO on this drawing", "get the symbol/object list for this sheet", "stop after stage 4", or "what symbols are on this P&ID". Does not read equipment tags, nozzles, or line-number text — see garnet-equipment-bbox / garnet-line-numbers for those.
---

# garnet-detect

Runs GARNET's pipeline through Stage 4 (object detection + topology markers + line-number/
instrument-tag fusion) and stops. No LLM, no tracing — pure YOLO+SAHI.

Full contract reference: `.claude/skills/garnet-digitize/reference/contracts.md`.

## Command

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m garnet.pid_extractor \
  --image <path-to-sheet-raster> \
  --out <output-dir> \
  --stop-after 4
```

Optional: `--weight-file yolo_weights/<name>.pt` (default `yolo26n_PPCL_640_20260227.pt`),
`--ocr-route easyocr|gemini|paddleocr|ocrmac` (default `ocrmac`).

## Why sheet-wide OCR runs even at `--stop-after 4`

Stage 2 runs before Stage 4 and cannot be skipped. Its **primary** job is not reading text —
it is producing text bounding boxes so `stage5_pipe_mask` can erase text from the raster
before tracing (`pid_extractor.py` loads `stage2_ocr_regions.text_regions` and passes them to
`run_pipe_mask_stage(ocr_regions=...)`, suppressed with 1px padding). Without it every note,
dimension, legend entry and title-block string stays black in the pipe mask and the Stage 5b
walker treats letter strokes as pipe. The `pipe_mask_min_component_area=16` filter does not
save you — glyphs are larger than that.

Its secondary job is transcription, feeding Stage 4's line-number and instrument-tag fusion.
That half *is* partly redundant when you supply `--ai-line-numbers`, since `ai_import` replaces
`stage4_line_numbers.json` wholesale. Instrument tags still depend on it.

So: OCR coverage must stay sheet-wide; OCR *quality* only has to be good enough to localize
text. Do not "optimize" this by cropping to detected objects only — the mask needs boxes for
text that has no detected object near it.

### Detect-only Stage 2

Because the mask only needs boxes, `--ocr-detect-only` (requires `--ocr-route easyocr`) runs
CRAFT detection without the per-box CRNN recognition pass — same boxes, no transcription.
All three orientation passes are kept, because a missed vertical line-number box means the
mask fails to erase it. Every region comes back `text: ""`, `class: "unknown"`,
`legibility: "illegible"`, which suppresses correctly (`pipe_mask_preserve_ocr_classes` is
empty by default) and is only cosmetic in the Stage 2 exception report.

Pair it with `--ai-line-numbers`, which replaces `stage4_line_numbers.json` wholesale.
Instrument tags then fall through to `line_number_fusion`/`instrument_tag_fusion`'s
per-box crop-OCR fallback (`_confirm_with_crop_ocr`, ocrmac, macOS-only).

## Output

`<output-dir>/stage4_objects.json` — Contract A (see reference). Also written:
`stage4_objects_summary.json`, `stage4_objects_overlay.png` (view this to sanity-check),
`stage4_topology_markers.json` (arrow/node → flow_marker/junction_marker), `stage4_line_numbers.json`,
`stage4_instrument_tags.json`.

Detection classes come from the weight file, not a YAML — the default weight currently produces:
`arrow, check valve, connection, control valve, gate valve, globe valve, instrument dcs,
instrument logic, instrument tag, node, page connection, pressure relief valve, reducer,
sampling point, spectacle blind, strainer, utility connection`.

## Failure modes

- **Model not found**: check `backend/yolo_weights/` for the `.pt`/`.onnx` file you named.
- **CUDA OOM**: pass a smaller image or run from an environment with `device="cpu"` (pipeline
  default device handling; no separate flag here).
- Detected objects do **not** include major equipment or read line-number text — that requires
  the `garnet-equipment-bbox` / `garnet-line-numbers` skills and `--ai-equipment` /
  `--ai-line-numbers`.

## Do not confuse with `/api/detect`

`POST /api/detect` (legacy HTTP path) returns a different flat schema (`CategoryID, ObjectID,
Left, Top, Width, Height, Score, Text`, xywh). This skill's output is the pipeline schema
(Contract A, xyxy) only — do not mix the two when writing downstream code or JSON.
