# GARNET pipeline contracts — shared reference

Single source of truth for all `garnet-*` skills. Every other SKILL.md links here instead of
repeating this. Verified against `backend/garnet/pid_extractor.py` and `backend/garnet/ai_import.py`.

Runner: `/Users/maetee/Code/GARNET/.venv/bin/python`, cwd = `/Users/maetee/Code/GARNET/backend`.

## CLI

```bash
python -m garnet.pid_extractor --image <path> --out <dir> \
  [--ocr-route easyocr|gemini|paddleocr|ocrmac] [--ocr-detect-only]   # default ocrmac
  [--weight-file yolo_weights/<name>.pt]          # default yolo26n_PPCL_640_20260227.pt
  [--stop-after 1|2|4|5|6|7|8|9|10|11]            # default 11 (full pipeline)
  [--debug-artifacts]                             # save heavy per-stage diagnostics
  [--ai-equipment FILE] [--ai-line-numbers FILE]  # fold in Contract B / Contract C JSON
  [--ai-align]                                    # allow a frame-mismatched AI JSON (see below)
```

Resume is automatic: re-running with the same `--out` and a matching input-image/config hash
picks up after the last completed stage (`PIDPipeline.run(resume=True)`).

## Stage table

| # | Stage | Key output artifacts |
|---|---|---|
| 1 | stage1_input_normalization | — |
| 2 | stage2_ocr_discovery | `stage2_ocr_*` |
| 3 | *external HITL* (not run by the CLI — this is where `--ai-equipment`/`--ai-line-numbers` get folded in, ahead of stage 5) | — |
| 4 | stage4_object_detection, stage4_line_number_fusion, stage4_instrument_tag_fusion | `stage4_objects.json`, `stage4_objects_summary.json`, `stage4_objects_overlay.png`, `stage4_topology_markers.json`, `stage4_line_numbers.json`, `stage4_instrument_tags.json` |
| 5 | stage5_pipe_mask, stage5b_pipe_trace | `stage5_pipe_mask.png`, `stage5_connection_ports.json`, `stage5b_trace_results.json`, `stage5b_branch_candidates.json`, `stage5b_branch_trace_results.json` |
| 6 | stage6_trace_associations | `stage6_trace_associations.json` |
| 7 | stage7_geometric_graph_assembly, stage7c_page_connector_labeling, stage7b_graph_export | `stage7_graph.json`, `stage7b_graph_v1.json` |
| 8 | stage8_graph_qa | `stage13_review_package.json`-style QA output |
| 9 | stage9_apply_review_decisions | — |
| 10 | stage10_process_exports | `stage10_line_list.json`, `stage10_equipment_connectivity.json`, `stage10_final_export.json` |
| 11 | stage11_connection_overlay | `stage16_connection_overlay.png`-style overlay |

Every stage also updates `stage_manifest.json` in `--out`.

## Contract A — `stage4_objects.json` (YOLO detection, pipeline schema)

```json
{"image_id": "input.png", "pass_type": "sheet", "objects": [
  {"id": "obj_000001", "class_name": "instrument tag", "confidence": 0.9766,
   "bbox": {"x_min": 3094, "y_min": 1821, "x_max": 3195, "y_max": 1917},
   "source_model": "ultralytics", "source_weight": "yolo_weights/yolo26n_PPCL_640_20260227.pt"}
]}
```

- Absolute pixel, **xyxy**, origin top-left, integers. `id` = `obj_%06d` in detection order.
- After HITL review, `source_model`/`source_weight` is replaced by `review_state` + `source: "hitl"`.
- SAHI defaults (`DetectionSahiConfig`): 640px tiles, 0.2 overlap, `GREEDYNMM`/`IOS`/0.1, confidence threshold 0.8.
- Default weight: `yolo_weights/yolo26n_PPCL_640_20260227.pt`.
- Classes observed from the default weight: `arrow, check valve, connection, control valve, gate valve, globe valve, instrument dcs, instrument logic, instrument tag, node, page connection, pressure relief valve, reducer, sampling point, spectacle blind, strainer, utility connection`. No YAML in the repo lists exactly this set — **the weight file is the authority**.
- `arrow` and `node` feed `stage4_topology_markers.json` (roles `flow_marker` / `junction_marker`) and are never suppressed from the pipe mask.
- YOLO does **not** detect major equipment and does **not** read line-number text — that's what the two AI-extraction skills supply via `--ai-equipment` / `--ai-line-numbers`.

**Warning:** `POST /api/detect` (the legacy single-image endpoint) returns a **different, flat**
schema: `Index, Object, CategoryID, ObjectID, Left, Top, Width, Height, Score, Text` (xywh).
Pipeline skills use Contract A only — never mix the two schemas.

## Contract B — equipment bbox JSON (input to `--ai-equipment`)

```json
{"source_drawing": "...",
 "coordinate_frame": {"width_px": 0, "height_px": 0, "dpi": 0, "page_pt": [0,0], "page_rotate_deg": 0, "note": ""},
 "objects": [{"Index": 1, "Object": "", "Tag": "", "Equipment_type": "", "Service": "", "Size": "",
   "Evidence": "", "Left": 0, "Top": 0, "Width": 0, "Height": 0,
   "Bounding_box_px": {"x_min":0,"y_min":0,"x_max":0,"y_max":0},
   "Bounding_box_norm": [0,0,0,0], "Score": 1.0,
   "Ports": [{"mark": "AV", "size": null, "line_number": null, "side": "top|bottom|left|right",
              "point_px": [0,0], "point_norm": [0,0]}]}],
 "unresolved": [{"what":"","why":"","location_hint_px":[0,0]}],
 "verification": "overlay rendered and visually checked: yes/no"}
```

- `ai_import.py` reads only pixel fields (`Bounding_box_px`, or `Left/Top/Width/Height` as
  fallback, and `point_px`). `Bounding_box_norm` / `point_norm` are ignored by the importer —
  keep them for human/resolution-independent checking only.
- A port needs **both** a usable `side` and a 2-element `point_px`, or it is silently skipped.
- Imported equipment gets id `equip_<slug-of-tag>`, `confidence: 1.0`, `source_model: "ai_import"`.
- **Hard vocabulary gate**: `Equipment_type` must normalize into `EQUIPMENT_LABELS`
  (`backend/garnet/pid_extractor.py`): `vessel, column, pump, compressor, blower, heat
  exchanger, tank, reactor, mixer, pot, knockout drum, filter, cooler, heater, injection pump`
  (underscore variants accepted). `AI_CLASS_MAP` (`backend/garnet/ai_import.py`) maps synonyms:
  static/inline mixer→mixer, ko drum→knockout drum, shell and tube exchanger / exchanger→heat
  exchanger, air cooler→cooler, drum/separator/accumulator→vessel. Anything else is **reported
  as skipped, not imported**.
- Importing equipment **replaces the entire equipment bucket** in `stage4_objects.json` — every
  existing object whose `class_name` is in `EQUIPMENT_LABELS` is dropped first.

## Contract C — line-number JSON (input to `--ai-line-numbers`)

```json
{"id": "Test-00002", "objects": [
  {"Index": 1, "Object": "line number", "CategoryID": 1, "ObjectID": 1,
   "Left": 491, "Top": 652, "Width": 347, "Height": 18, "Score": 1.0,
   "Text": "3\"-CUL-25-002007-B1A2-NI"}],
 "image_url": "", "image_width": 4963, "image_height": 3509, "count": 16}
```

- Pixel xywh, source raster frame. Empty `Text` → skipped.
- Imported ids are `line_number_ai_%06d`, `review_state: "ocr_confirmed"`.

## Frame reconciliation (`ai_import.fit_transform`) — the #1 failure mode

Declared frame = `coordinate_frame.{width_px,height_px}` (equipment) or `image_width`/
`image_height` (line numbers), compared against the job image's real pixel size.

- **Mismatch without `--ai-align`** → hard error.
- **With `--ai-align`**: fits a **translation only** (scale pinned to 1.0) by fuzzy-matching AI
  line-number text against the job's own `stage4_line_numbers.json` (must already exist).
  Needs ≥4 unique text matches at `SequenceMatcher` ratio ≥0.85, takes median dx/dy with MAD
  outlier rejection, and requires median residual ≤6px or it raises.
- **Practical guidance**: extract the LLM JSON from the *same* raster you feed the pipeline and
  `--ai-align` is never needed. Treat needing `--ai-align` as a signal something upstream drifted.

## Tracing (stage 5 + 5b) — pure CV, no LLM

- **Pipe mask**: adaptive|Otsu binary, with OCR text boxes and non-inline object boxes
  suppressed; inline symbols inset 12px so the pipe stub survives; `arrow`/`node` never
  suppressed; components <16px dropped. Morphologically closed 3×3 and extended up to 80px into
  equipment/instrument boxes so the walker can reach terminals. No skeletonization — the walker
  walks the thick raster and self-centres.
- **Ports**: CV edge-scan on each connection/equipment bbox (≥3px dark run, tracked 60px
  outward), snapped to the pipe centreline. AI-supplied nozzle points override CV for the
  equipment they cover.
- **Walk**: per-step centreline snap → loop guard → inline-symbol jump to the far edge of the
  overlapping group (pressure-relief valves exit perpendicular) → straight-ahead line-of-sight →
  terminal check → junction/tee check → turn resolution → straight raycast 20–50px to bridge
  gaps from text or crossing pipes (max lateral snap 4px so it cannot hop onto a parallel pipe).
  Terminates as `equipment` / `page_connection` / `instrument_tag` / `tee_junction` /
  `sheet_edge` / `dead_end` / `max_steps`. `visited` mask is shared across all traces — first
  trace to claim a segment wins; equipment ports traced before page connections for determinism.
- **Branches**: up to 5 fixed-point iterations discovering perpendicular side-runs ≥25px off
  known traces, each traced as `branch_%06d`, successful ones seeding the next iteration.

### Tunables (`PipelineConfig`)

| Field | Default |
|---|---|
| `trace_max_steps` | 5000 |
| `trace_lookahead_px` | 30 |
| `trace_raycast_start_px` | 20 |
| `trace_raycast_max_px` | 50 |
| `trace_raycast_max_snap_shift_px` | 4 |
| `trace_centerline_radius_px` | 8 |
| `trace_branch_min_run_px` | 25 |
| `trace_branch_max_iterations` | 5 |
| `pipe_mask_inline_object_inset` | 12 |
| `pipe_mask_min_component_area` | 16 |
| `trace_association_text_max_distance_px` | 100.0 |
| `trace_association_instrument_max_distance_px` | 90.0 |
| `line_number_fusion_max_distance_px` | 80.0 |

### When tracing goes wrong

| Symptom | Likely knob |
|---|---|
| Trace stops short at a text label | raise `trace_raycast_max_px` |
| Walker hops onto a parallel pipe | lower `trace_raycast_max_snap_shift_px` |
| Tee legs missed | raise `trace_branch_max_iterations`, or lower `trace_branch_min_run_px` |
| A symbol swallows the pipe (trace can't get through) | check `pipe_mask_inline_object_inset` |

## Graph output

`stage7_graph.json` — `schema_version: "stage7_trace_graph_v1"`. Keys: `nodes, edges, equipment,
ports, line_groups, review_queue, metadata`. Node types: `equipment_port, page_connection,
utility_connection, connection, branch_start, tee_junction, junction, branch, instrument_tag,
equipment, dead_end, terminal, source`. Node ids are stable where possible
(`equipment::<drawing>::<id>::port::NN`, `connection::<id>`, `junction::xy::<x>::<y>`). Edge
type is `pipe_trace`; edges carry `source`/`target` (canonical) plus `legacy_source`/
`legacy_target`, `polyline`, `segments`, `turns`, `hits`, `attachments`, `line_style`,
`review_state`, and may be split into `::part_001` etc with `original_trace_id` preserved.

`stage7b_graph_v1.json` — the public export: `document, nodes, edges, equipment,
equipment_ports, lines, line_to_edges, inline_objects, instruments, relationships,
constraints`. **Node bbox here is `xywh`** (`bbox_format: "xywh"`) — unlike stage4/stage7 which
are `xyxy`.

`stage10_line_list.json` — line number → edge ids, canonical line id, total length, flow
direction. `stage10_equipment_connectivity.json` — equipment-level connectivity.
