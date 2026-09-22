---
name: garnet-detect
description: Run YOLO+SAHI symbol detection on a GARNET P&ID raster to get the object list (valves, instrument tags, connections, arrows, nodes, etc). Use for phrasings like "detect objects on this P&ID", "run YOLO on this drawing", "get the symbol/object list for this sheet", "stop after stage 4", or "what symbols are on this P&ID". Does not read equipment tags, nozzles, or line-number text — see garnet-equipment-bbox / garnet-line-numbers for those.
---

# garnet-detect

YOLO+SAHI symbol detection on one P&ID raster. No LLM, no tracing.

Full contract reference: `.claude/skills/garnet-digitize/reference/contracts.md` — read its
"Two trees" section first if you don't know which route you're on.

## Default route — `garnet/` standalone script

Run from the repo root. Standalone: no pipeline, no OCR, no stage manifest.

```bash
cd /Users/maetee/Code/GARNET
.venv/bin/python -m garnet.detect_objects --image <path-to-sheet-raster>
```

Writes `output/<stem>_objects.json` (Contract A) and `output/<stem>_objects_overlay.png` —
**view the overlay** to sanity-check. `--out <path>` to place it elsewhere; stage it as
`<indir>/<stem>_objects.json` if it feeds `garnet.stage5b.run`.

Useful flags: `--weight <file>` (default `backend/yolo_weights/yolo26n_PPCL_640_20260227.pt`),
`--image-size` (640), `--overlap-ratio` (0.2), `--conf-th` (0.8), `--device cpu|mps|cuda:0`,
`--no-overlay`. `--help` for the rest.

## Alternative route — backend pipeline, Stage 4

Use only when you need a backend job output dir (stage manifest, tag fusion, then stages 5–11).

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m garnet.pid_extractor \
  --image <path-to-sheet-raster> --out <output-dir> --stop-after 4
```

Optional: `--weight-file yolo_weights/<name>.pt`, `--ocr-route easyocr|gemini|paddleocr|ocrmac`
(default `ocrmac`).

Extra outputs this route gives you that the script does not: `stage4_objects_summary.json`,
`stage4_topology_markers.json` (arrow/node → flow_marker/junction_marker),
`stage4_line_numbers.json`, `stage4_instrument_tags.json`.

### Why sheet-wide OCR runs even at `--stop-after 4` (backend route only)

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

`--ocr-detect-only` (requires `--ocr-route easyocr`) runs CRAFT detection without the per-box
CRNN recognition pass — same boxes, no transcription. All three orientation passes are kept,
because a missed vertical line-number box means the mask fails to erase it. Every region comes
back `text: ""`, `class: "unknown"`, `legibility: "illegible"`, which suppresses correctly
(`pipe_mask_preserve_ocr_classes` is empty by default). Pair it with `--ai-line-numbers`.

The `garnet/` tracer solves the same problem differently: `garnet.stage5b.run --ocr` runs
macOS Vision OCR on demand, and without it the staged line-number fixture supplies the boxes.

## Detection classes

From the weight file, not a YAML. The default weight currently produces: `arrow, check valve,
connection, control valve, gate valve, globe valve, instrument dcs, instrument logic,
instrument tag, node, page connection, pressure relief valve, reducer, sampling point,
spectacle blind, strainer, utility connection`.

## Failure modes

- **Model not found**: check `backend/yolo_weights/` for the `.pt`/`.onnx` file you named.
- **CUDA OOM**: `--device cpu` (script route) or a smaller image.
- Detected objects do **not** include major equipment or read line-number text — that needs
  `garnet-equipment-bbox` and `garnet-line-numbers`.

## Do not confuse with `/api/detect`

`POST /api/detect` (legacy HTTP path) returns a different flat schema (`CategoryID, ObjectID,
Left, Top, Width, Height, Score, Text`, xywh). This skill's output is Contract A (xyxy) —
do not mix the two when writing downstream code or JSON.
