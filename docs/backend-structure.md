# Backend Structure

The GARNET backend is a FastAPI service. The `garnet/` package is intentionally **flat** — most pipeline stages are methods on a single `PIDPipeline` class inside `pid_extractor.py`, with a small number of subpackages (`path_tracer/`, `utils/`, `OCR_prompts/`, `visual_primitives/`) for cohesive functionality.

## Top Level

```
backend/
├── api.py                       # Canonical FastAPI application (uvicorn api:app)
├── main.py                      # 10-line compatibility shim that re-exports api:app
├── requirements.txt             # Python dependencies
├── .env.example                 # Backend environment template
├── run_debug.sh                 # Debug-mode pipeline runner
├── run_stage5b_only.sh          # Stage 5b-only smoke test
├── garnet/                      # Core pipeline package (see below)
├── datasets/                    # YOLO dataset configs and class definitions
│   ├── yaml/                    #   data.yaml, balanced.yaml, iso.yaml, pttep.yaml, ...
│   ├── classes.txt
│   ├── predefined_classes.txt
│   └── settings_labels.json
├── yolo_weights/                # YOLO model weight files (.pt)
├── static/
│   └── images/predictions/      # Generated prediction overlay images
├── scripts/                     # Operator scripts (batch testing, debug, demos)
│   ├── batch_pipeline_test.py
│   ├── batch_timing.py
│   ├── compare_ab.py
│   ├── debug_timing.py
│   ├── demo_line_inpaint.py
│   ├── phase3_visual_spotcheck.py
│   └── test_single_fix.py
├── tools/                       # Standalone CLI tools
│   ├── merge_predictions.py
│   ├── merge_sheets.py
│   └── patchify.py
├── tests/                       # pytest suite
├── output/                      # Runtime output directory (per-job artifacts)
├── runs/                        # YOLO training run outputs
├── temp/                        # Transient working files
├── json/                        # Cached COCO image indexes
├── schema/                      # JSON schemas for API contracts
├── gemini_detector/             # Gemini-specific detector assets
├── output_debug/                # Debug-mode pipeline outputs
└── sample.png                   # Sample/test image
```

Note: Excel export is implemented inline in `api.py` (`/api/export/excel` uses `pandas` + `openpyxl`); there is no separate `export_to_excel.py` module. Sample data files (`coco_annotations.json`, `coco_arrows.json`, `ocr_results.json`) live outside the repo in your local working copy and are not part of the tree.

## `garnet/` Package — Flat Layout

```
backend/garnet/
├── __init__.py                  # Package docstring only
├── AGENTS.md                    # Package-level operating instructions
│
├── pid_extractor.py             # ⚙️ Orchestrator — PIDPipeline class with all stage methods
│                                #   (stage1_input_normalization, stage2_ocr_discovery,
│                                #    stage4_object_detection, stage4_line_number_fusion,
│                                #    stage4_instrument_tag_fusion, stage5_pipe_mask,
│                                #    stage5b_pipe_trace, stage6_trace_associations,
│                                #    stage7_geometric_graph_assembly, stage7b_graph_export,
│                                #    stage7c_page_connector_labeling, stage8_graph_qa,
│                                #    stage9_apply_review_decisions, stage10_process_exports,
│                                #    stage11_connection_overlay) + CLI entrypoint
│
├── Settings.py                  # Settings class (paths, model types)
├── model_defaults.py            # Model weight file discovery
│
├── object_detection_sahi.py     # SAHI-based object detection (YOLOv8/YOLOv11)
│
├── easyocr_sahi.py              # OCR engine: EasyOCR
├── gemini_ocr_sahi.py           # OCR engine: Gemini (vision LLM)
├── paddle_ocr_sahi.py           # OCR engine: PaddleOCR
├── ocrmac_sahi.py               # OCR engine: OCRMac (macOS native)
│
├── line_number_fusion.py        # Line number text→symbol association
├── instrument_tag_fusion.py     # Instrument tag text→symbol association
├── topology_markers.py          # Junction topology marker detection
│
├── pipe_mask.py                 # Binary pipe mask generation
├── pipe_sheet_merge.py          # Multi-sheet pipe network merge
├── page_connector.py            # Cross-sheet page connector handling
│
├── trace_associations.py        # Trace paths ↔ objects/ports/instruments
├── trace_graph_builder.py       # Build geometric graph from traces
├── trace_graph_qa.py            # Graph quality assurance
├── pipe_graph_qa.py             # Pipe-network-level QA checks
│
├── review_state.py              # Per-job review state container
├── review_workspace.py          # Review workspace (junctions, lines, edits)
├── reviewed_outputs.py          # Apply review decisions → final outputs
│
├── graph_export_adapter.py      # Graph export format adapter
├── render_connection_pipeline_overlay.py  # Connection overlay image renderer
│
├── stage8_review_package.py     # Stage 8 — graph HITL review package builder
├── stage9_review_decisions.py   # Stage 9 — apply human review decisions
├── stage10_process_exports.py   # Stage 10 — line lists / MTO / equipment index
│
├── path_tracer/                 # CV pipe-tracing subpackage
│   ├── __init__.py
│   ├── cv_pipe_tracer.py        # Core CV tracer (terminals, junctions, passthrough)
│   └── stage5b_pipeline.py      # Stage5bPipelineMixin (mixed into PIDPipeline)
│
├── utils/
│   ├── __init__.py              # re-exports rotate_image
│   └── utils.py                 # rotate_image, shared helpers
│
├── visual_primitives/           # Visual primitives subpackage (compiled)
│
└── OCR_prompts/                 # Gemini OCR prompt templates (.md)
    ├── crop_pass_system_prompt.md
    ├── crop_pass_user_prompt.md
    ├── full_page_pass_system_prompt.md
    └── full_page_pass_user_prompt.md
```

## Architecture Notes

- **Stages are methods, not modules.** The `PIDPipeline` class in `pid_extractor.py` defines one method per pipeline stage. Most stage bodies are inline logic; for three stages (8 review package, 9 review decisions, 10 process exports) the heavy lifting lives in standalone modules (`stage8_review_package.py`, `stage9_review_decisions.py`, `stage10_process_exports.py`) and the corresponding orchestrator methods are thin wrappers that import and call them.
- **`path_tracer/`** is a focused subpackage for the CV-based pipe-following logic. `Stage5bPipelineMixin` is mixed into `PIDPipeline` so the orchestrator can call into CV tracer code without owning it.
- **OCR engines are pluggable.** Each engine has its own `*_sahi.py` file exposing a common interface, selected via the `model_type` setting.
- **No `pipeline/`, `review/`, or `tracing/` subdirectories.** The `garnet/` package is flat by design — see the tree above for actual layout.

## Entry Points

| Command | What runs |
|---|---|
| `uvicorn api:app --reload --port 8001` | FastAPI service (canonical) |
| `uvicorn main:app --port 8001` | Same service via compatibility shim |
| `python garnet/pid_extractor.py --image <png> --out <dir>` | CLI pipeline runner |
| `bash run_stage5b_only.sh` | Stage 5b-only smoke test |
| `bash run_debug.sh` | Debug-mode pipeline runner |
