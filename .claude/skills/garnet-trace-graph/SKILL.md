---
name: garnet-trace-graph
description: Run pipe tracing and Stage 6 evidence association on a GARNET P&ID sheet that already has object/equipment/line-number JSON, producing traced paths, line-number attachments, and (backend route only) the connectivity graph and line list. Use for phrasings like "trace the pipes on this P&ID", "build the connectivity graph", "re-run tracing with new equipment/line-number JSON", "iterate on Stage 5b", or "why did the trace stop / jump to the wrong pipe".
---

# garnet-trace-graph

Pipe tracing and what comes after it. Two routes with **different reach**:

| | `garnet/` (repo root) | `backend/` |
|---|---|---|
| Tracing (5/5b) | yes | yes |
| Review gate over traces | yes (`review_gate`) | no |
| Stage 6 associations | yes | yes |
| Graph assembly + line list | **no** | yes (`stage7b_graph_v1.json`, `stage10_line_list.json`) |

If the ask is "the connectivity graph" or "the line list", you need the backend route. If it is
"trace the pipes" or "why did this trace go wrong", use `garnet/` — it is the tree under active
development and its tracer carries fixes the backend copy does not have.

Full contract reference: `.claude/skills/garnet-digitize/reference/contracts.md`.

## `garnet/` route — fixture-driven

**Inputs are files in a directory, not `--image`.** Stage them first (see "fixture staging" in
the reference doc): `<indir>/<stem>.jpg` plus `<stem>_objects.json`,
`<stem>_equipment_bboxes.json`|`<stem>_equipment.json`,
`<stem>_line_number_boxes.json`|`<stem>_line_numbers.json`. `garnet/tests/input/` is the
worked example.

```bash
cd /Users/maetee/Code/GARNET

# 1. trace  -> stage5b_trace_overlay.png, stage5b_branch_trace_overlay.png + JSON
.venv/bin/python -m garnet.stage5b.run --stem <stem> --indir <indir> --out output/<stem>

# 2. clean up the traced network (dashed followers, duplicates, runs into empty space)
.venv/bin/python -m garnet.stage5b.review_gate --stem <stem> --dir output/<stem>

# 3. attach line numbers, valves, tags, arrows -> stage6_* + overlay
.venv/bin/python -m garnet.stage5b.run_associations --stem <stem> --indir <indir> --dir output/<stem>

# 4. eyeball the line-number assignments
.venv/bin/python -m garnet.stage5b.inspect_line_numbers --stem <stem> --dir output/<stem>
```

`garnet.trace_pipes --stem <stem>` is a leaner self-contained alternative to step 1 (same
walking algorithm, no host/mixin layer) when you want to iterate on the walker itself.

Notes:
- `--ocr` on `stage5b.run` runs macOS Vision OCR and merges its text boxes into the mask.
  Without it, only the staged line-number boxes suppress text. `trace_pipes` has the inverse
  flag, `--no-ocr`.
- `review_gate` **calls a vision model** on the residual R1 cases. `--no-llm` for deterministic
  rules only, `--dry-run` to inspect without an API call.
- `bash garnet/run_test.sh` re-runs step 1 across every fixture sheet into a cleared `output/`.
- `garnet.stage5b.coverage_audit --out-dir output/<stem>` after a tracing change: walk counts
  and terminal histograms can look identical while whole branches go unvisited.

### Iteration knobs (`garnet/` route — real CLI flags)

| Symptom | Flag |
|---|---|
| Tee legs missed | `--branch-iterations` ↑, `--branch-min-run-px` ↓ |
| Too many spurious branches | `--max-branches` ↓, `--branch-min-run-px` ↑ |
| Branch starts skipped along a run | `--branch-sample-px` ↓ (default 16) |

The raycast/centreline tunables are still constants in the vendored tracer — edit
`garnet/stage5b/cv_pipe_tracer.py` (see the divergence warning in root `AGENTS.md`: fixes flow
out of `garnet/`, never in).

## backend route — full pipeline through the graph

```bash
cd /Users/maetee/Code/GARNET/backend
/Users/maetee/Code/GARNET/.venv/bin/python -m garnet.pid_extractor \
  --image <sheet> --out <output-dir> \
  --ai-equipment <equipment.json> --ai-line-numbers <line_numbers.json> \
  --stop-after 11
```

Add `--ai-align` only if the AI JSON's frame doesn't match and you can't re-extract from the
same raster.

Re-run tracing alone (much faster than a full run) once `stage4_objects.json`,
`stage4_instrument_tags.json` and `stage5_pipe_mask.png` exist:

```bash
cd /Users/maetee/Code/GARNET/backend
PY_BIN=/Users/maetee/Code/GARNET/.venv/bin/python ./run_stage5b_only.sh <output-dir>
```

It reads `image_path` from the dir's `stage_manifest.json`, so the dir must come from a real
pipeline run.

### Iteration knobs (backend route — no CLI flags)

Set them by constructing `PipelineConfig(...)` in a short script (as `run_stage5b_only.sh`
does) or by editing the dataclass default in `backend/garnet/pid_extractor.py`.

| Symptom | Knob |
|---|---|
| Trace stops short at a text label | raise `trace_raycast_max_px` |
| Walker hops onto a parallel pipe | lower `trace_raycast_max_snap_shift_px` |
| Tee legs missed | raise `trace_branch_max_iterations`, lower `trace_branch_min_run_px` |
| A symbol swallows the pipe | check `pipe_mask_inline_object_inset` |

### Outputs

`stage7_graph.json` (internal), `stage7b_graph_v1.json` (public export — node bbox is **xywh**,
unlike stage4/stage7's xyxy), `stage10_line_list.json`, `stage10_equipment_connectivity.json`.
Field lists in the reference doc.
