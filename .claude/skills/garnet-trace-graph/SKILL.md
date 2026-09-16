---
name: garnet-trace-graph
description: Run pipe tracing (Stage 5/5b) and graph assembly (Stage 6-11) on a GARNET P&ID job that already has Stage 4 object detection, producing the connectivity graph and line list. Use for phrasings like "trace the pipes on this P&ID", "build the connectivity graph", "re-run tracing with new equipment/line-number JSON", "iterate on Stage 5b", or "why did the trace stop / jump to the wrong pipe". Assumes stage4_objects.json already exists in the job's output dir.
---

# garnet-trace-graph

Runs pipe-mask generation, geometric tracing, and graph assembly against an output dir that
already has Stage 4 artifacts (from `garnet-detect`), optionally folding in equipment/
line-number JSON from `garnet-equipment-bbox` / `garnet-line-numbers`.

Full contract reference (tracing algorithm, tunables, graph schema): `.claude/skills/garnet-digitize/reference/contracts.md`.

## Full run, from the image, with AI imports

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m garnet.pid_extractor \
  --image <sheet> --out <output-dir> \
  --ai-equipment <equipment.json> --ai-line-numbers <line_numbers.json> \
  --stop-after 11
```

Add `--ai-align` only if the AI JSON's declared frame doesn't match the job image and you can't
re-extract from the same raster (see the reference doc for what that actually fits and requires).

## Iterating on tracing only (no re-detection)

Once `stage4_objects.json`, `stage4_instrument_tags.json`, and `stage5_pipe_mask.png` already
exist in the output dir, re-running Stage 5b alone (after editing `PipelineConfig` tunables, or
after equipment/line-number JSON changed the mask inputs) is much faster than a full pipeline
run:

```bash
cd /Users/maetee/Code/GARNET/backend
PY_BIN=/Users/maetee/Code/GARNET/.venv/bin/python ./run_stage5b_only.sh <output-dir>
```

This re-runs **only** `stage5b_pipe_trace()` (not mask generation, not stage 6+) and writes
`stage5b_trace_results.json` in place. It reads `image_path` from the output dir's
`stage_manifest.json`, so the dir must already have been produced by a real pipeline run.

## When tracing goes wrong

| Symptom | Likely knob (`PipelineConfig`, in `backend/garnet/pid_extractor.py`) |
|---|---|
| Trace stops short at a text label | raise `trace_raycast_max_px` |
| Walker hops onto a parallel pipe | lower `trace_raycast_max_snap_shift_px` |
| Tee legs missed | raise `trace_branch_max_iterations`, or lower `trace_branch_min_run_px` |
| A symbol swallows the pipe (trace can't get through it) | check `pipe_mask_inline_object_inset` |

There is no CLI flag for these — set them by constructing `PipelineConfig(...)` in a short
script (as `run_stage5b_only.sh` does) or editing the dataclass default.

## Output

`stage7_graph.json` (internal trace graph), `stage7b_graph_v1.json` (public export, node bbox
is **xywh** unlike stage4/stage7's xyxy), `stage10_line_list.json` (line number → edges),
`stage10_equipment_connectivity.json`. Full field lists in the reference doc.
