# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GARNET (GCME AI-Recognition Network for Engineering Technology) is an AI-powered tool for automating symbol detection, classification, and connectivity analysis in Piping and Instrumentation Diagrams (P&IDs). It combines YOLOv11 object detection with graph-based analytics to transform P&ID workflows.

**Tech Stack:**
- **Backend**: FastAPI + Python (YOLOv11, SAHI, EasyOCR/PaddleOCR/Gemini OCR, NetworkX, OpenCV)
- **Frontend**: React 18 + TypeScript + Vite + Zustand (state management) + Radix UI + Tailwind CSS + Konva (canvas) + **Bun (package manager)**
- **AI Models**: Ultralytics YOLOv11, EasyOCR, PaddleOCR (RapidOCR), Gemini via OpenRouter, DeepLSD (line detection)

## Repository Layout

The backend and garnet module live under `backend/`. Always run backend commands from the `backend/` directory.

```
/GARNET
├── backend/                        # Python backend (run commands from here)
│   ├── api.py                      # FastAPI app (21+ endpoints)
│   ├── main.py                     # Legacy Streamlit shim (deprecated)
│   ├── requirements.txt            # Python dependencies
│   ├── garnet/                     # Core pipeline module (60+ files)
│   │   ├── pid_extractor.py        # Stage-by-stage pipeline orchestrator
│   │   ├── Settings.py             # Global config (paths, symbol types, text classes)
│   │   ├── pipe_mask.py            # Pipe segmentation mask generation
│   │   ├── pipe_skeleton.py        # Skeleton extraction
│   │   ├── pipe_nodes.py           # Node detection on skeleton
│   │   ├── pipe_edges.py           # Edge tracing along skeleton paths
│   │   ├── pipe_junctions.py       # Junction detection
│   │   ├── pipe_terminals.py       # Terminal classification
│   │   ├── pipe_crossings.py       # Crossing vs. junction resolution
│   │   ├── pipe_node_clusters.py   # DBSCAN node clustering
│   │   ├── pipe_text_attachment.py # Attach OCR text to pipe edges
│   │   ├── pipe_equipment_attachment.py # Equipment-to-pipe connections
│   │   ├── pipe_edge_connectivity.py    # Edge connectivity analysis
│   │   ├── pipe_continuity_checker.py   # Continuity validation
│   │   ├── pipe_continuity_helpers.py   # Continuity check utilities
│   │   ├── pipe_graph.py           # NetworkX graph construction
│   │   ├── pipe_graph_qa.py        # Graph quality assurance
│   │   ├── pipe_seal.py            # Morphological sealing
│   │   ├── pipe_sheet_merge.py     # Multi-sheet merge via connectors
│   │   ├── geometric_graph_builder.py  # Geometry-based graph assembly
│   │   ├── trace_graph_builder.py  # Trace-based graph builder (stages 10-11)
│   │   ├── trace_graph_qa.py       # Trace graph QA
│   │   ├── edge_direction.py       # Edge direction classification
│   │   ├── edge_split.py           # Edge splitting at junctions
│   │   ├── polyline_simplify.py    # Polyline simplification
│   │   ├── graph_export_adapter.py # GraphML/JSON export
│   │   ├── page_connector.py       # Off-page connector handling
│   │   ├── symbol_aware_splitter.py # Symbol-based line splitting
│   │   ├── line_detection_inpaint.py  # Inpaint-based line detection
│   │   ├── topology_markers.py     # Topology markers from detections
│   │   ├── continuity_aware_connections.py # Continuity-aware graph connections
│   │   ├── equipment_pipe_association.py  # Equipment-pipe KDTree association
│   │   ├── easyocr_sahi.py         # EasyOCR with SAHI tiling (Stage 2)
│   │   ├── gemini_ocr_sahi.py      # Gemini/OpenRouter OCR route (Stage 2)
│   │   ├── paddle_ocr_sahi.py      # PaddleOCR route (Stage 2)
│   │   ├── ocrmac_sahi.py          # OCRMac route (Stage 2)
│   │   ├── object_detection_sahi.py # YOLO+SAHI object detection (Stage 4)
│   │   ├── predict_images.py       # Batch detection helpers
│   │   ├── text_ocr.py             # Text extraction utilities
│   │   ├── text_classify.py        # Text classification
│   │   ├── equipment_tag_fusion.py # Equipment tag fusion
│   │   ├── instrument_tag_fusion.py # Instrument tag fusion
│   │   ├── line_number_fusion.py   # Line number fusion
│   │   ├── model_defaults.py       # Model weight file discovery
│   │   ├── review_state.py         # Review state persistence
│   │   ├── reviewed_outputs.py     # Review-corrected output generation
│   │   ├── recovery_loop.py        # Error recovery loop
│   │   ├── render_connection_pipeline_overlay.py # Connection overlay rendering
│   │   ├── topology_pipeline.py    # Topology pipeline orchestration
│   │   ├── stage13_review_package.py    # Stage 13: review package generation
│   │   ├── stage14_review_decisions.py  # Stage 14: apply review decisions
│   │   ├── stage15_process_exports.py   # Stage 15: process exports
│   │   ├── run_continuity_checker_stage.py # Continuity checker stage runner
│   │   ├── path_tracer/            # Path tracing submodule
│   │   ├── OCR_prompts/            # Gemini OCR prompt templates
│   │   └── utils/                  # Image utilities (rotation, morphology, line removal)
│   ├── tests/                      # 44+ unittest test files
│   ├── scripts/                    # Batch and debugging scripts
│   │   ├── batch_pipeline_test.py
│   │   ├── batch_timing.py
│   │   ├── compare_ab.py
│   │   ├── debug_timing.py
│   │   ├── demo_line_inpaint.py
│   │   ├── phase3_visual_spotcheck.py
│   │   └── test_single_fix.py
│   ├── gemini_detector/            # Alternative Gemini-based SAHI detector
│   ├── tools/                      # Utility tools (merge_predictions.py, patchify.py, etc.)
│   ├── schema/                     # Data schema definitions
│   ├── docs/                       # Backend-specific documentation
│   ├── datasets/                   # YOLO config files (yaml/)
│   ├── output/                     # Pipeline job artifacts
│   ├── output_debug/               # Debug artifacts
│   ├── runs/                       # Detection run outputs
│   ├── static/                     # Static files (prediction images)
│   ├── yolo_weights/               # Model weight files (.pt/.onnx)
│   ├── run_debug.sh                # Run pipeline with debug flags
│   ├── run_stage5b_only.sh         # Run Stage 5b with recovery loop
│   └── pid_extractor.sh            # Basic pipeline invocation
├── frontend/                       # React app
│   ├── src/
│   │   ├── App.tsx                 # Root component with view routing
│   │   ├── components/
│   │   │   ├── ui/                 # Radix UI primitives (button, dialog, select, etc.)
│   │   │   ├── UploadZone.tsx
│   │   │   ├── DetectionSetup.tsx
│   │   │   ├── ProcessingView.tsx
│   │   │   ├── ResultsView.tsx     # Detection results editor
│   │   │   ├── CanvasView.tsx      # Interactive Konva canvas (zoom/pan/edit)
│   │   │   ├── ObjectSidebar.tsx
│   │   │   ├── BatchResultsView.tsx
│   │   │   ├── PipelineResultsView.tsx  # Pipeline job results
│   │   │   ├── PipelineArtifactCanvas.tsx # Pipeline artifact display
│   │   │   ├── PipelineHitlReviewView.tsx # HITL review interface
│   │   │   ├── PdfPageSelector.tsx
│   │   │   ├── Header.tsx
│   │   │   ├── ZoomControls.tsx
│   │   │   └── ErrorBoundary.tsx
│   │   ├── stores/
│   │   │   ├── appStore.ts         # Main app state (Zustand)
│   │   │   └── historyStore.ts     # Undo/redo history
│   │   ├── hooks/                  # Custom React hooks
│   │   ├── lib/                    # API client, export utilities, helpers
│   │   └── types.ts                # TypeScript type definitions
│   ├── package.json                # Bun dependencies
│   ├── vite.config.ts              # Vite config (proxies /api and /runs to :8001)
│   └── tailwind.config.ts
├── DeepLSD/                        # Line detection submodule
├── design/                         # Design references and assets
├── docs/                           # Project plans and documentation
├── AGENTS.md                       # Root agent instructions (cross-module guidance)
├── README.md                       # Project README
├── MASTER_PLAN.md                  # P&ID digitizing architecture roadmap
├── GEMINI.md                       # Gemini integration notes
├── .env.example                    # Root environment template
└── punch_list.md                   # Development punch list
```

## Development Commands

All backend commands must be run from the `backend/` directory.

### Backend

```bash
cd backend

# Start FastAPI server on port 8001
uvicorn api:app --reload --port 8001

# Install Python dependencies
pip install -r requirements.txt

# Run the P&ID pipeline on an image
python -m garnet.pid_extractor --image <path> --out <output_dir> --ocr-route easyocr

# Run pipeline with debug flags
bash run_debug.sh

# Run Stage 5b only with recovery loop
bash run_stage5b_only.sh

# Compile-check all backend Python files (minimum verification after edits)
python -m py_compile api.py garnet/*.py garnet/utils/*.py

# Run all backend tests
python -m unittest discover -s tests -p "test*.py" -v

# Run a single test file
python -m unittest tests.test_pipeline_api -v

# Run a single test method
python -m unittest tests.test_pipeline_api.TestPipelineAPI.test_pipeline_job_runs_stage2_and_reports_artifacts -v
```

### Frontend

**IMPORTANT**: The frontend **MUST use Bun** as the package manager, **NOT npm or yarn**.

```bash
cd frontend

# Install dependencies
bun install

# Start dev server on port 5173 (proxies /api and /runs to localhost:8001)
bun run dev

# Build for production
bun run build

# Preview production build
bun run preview

# Lint
bun run lint

# Add/remove dependencies (use bun, not npm)
bun add <package-name>
bun remove <package-name>
```

### Batch Processing

```bash
# Run batch inference on multiple P&ID images
python backend/garnet/predict_images.py \
    --image_path path/to/pids_folder \
    --model_type yolov8 \
    --model_path path/to/model_weights.pt \
    --output_path results/
```

## Environment Variables

Copy `.env.example` (root) to `.env` and configure:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | Environment mode |
| `HOST` / `PORT` | `localhost` / `8001` | Server bind address |
| `ALLOWED_ORIGINS` | `http://localhost:5173,...` | CORS origins (comma-separated) |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `DEFAULT_CONF_THRESHOLD` | `0.8` | Default detection confidence |
| `DEFAULT_IMAGE_SIZE` | `640` | Default SAHI inference size |
| `DEFAULT_OVERLAP_RATIO` | `0.2` | Default SAHI tile overlap |
| `OPENROUTER_API_KEY` | — | Required for Gemini OCR/detection routes |
| `OPENROUTER_MODEL` | `google/gemini-3-flash-preview` | Gemini model via OpenRouter |
| `OCR_CACHE_ENABLED` | `true` | Enable OCR result caching |
| `OCR_LANGUAGES` | `en` | EasyOCR languages |
| `OCR_GPU` | `true` | Use GPU for EasyOCR |
| `API_KEY_ENABLED` | `false` | Enable API key auth |
| `API_KEY` | — | API key value |
| `RATE_LIMIT_ENABLED` | `false` | Enable rate limiting |
| `RATE_LIMIT_REQUESTS` / `RATE_LIMIT_WINDOW` | `100` / `60` | Rate limit config |
| `LOG_LEVEL` / `LOG_FILE` / `LOG_ROTATION` | `INFO` / `garnet.log` / `10 MB` | Logging config |

## Architecture

### System Architecture

```
┌──────────────────┐         ┌───────────────────┐         ┌──────────────────┐
│  React Frontend  │ ──HTTP─→│  FastAPI Backend  │ ──→───  │  garnet/ Module  │
│  (Port 5173)     │ ←──JSON─│  (Port 8001)      │ ←───   │  (Pipeline Core) │
└──────────────────┘         └───────────────────┘         └──────────────────┘
        │                             │                              │
        │ Zustand State               │ Model Cache                  │ 60+ Python files
        │ History Store               │ OCR Cache                    │ Stage-by-stage
        │ Konva Canvas                │ RESULTS_STORE                │ YOLO + SAHI
        │ Radix UI + Tailwind         │ Pipeline Job Store           │ OCR routes
        └─ Bun (package manager)      │ Review State Store           │ NetworkX graphs
                                      └─ Static Files                │ OpenCV geometry
```

### Two API Paths

The backend serves two distinct API paths:

1. **Legacy `/api/detect`** — Single-image YOLO detection with SAHI + optional OCR. In-memory result storage with CRUD endpoints for object editing.

2. **Pipeline Job API** (`/api/pipeline/*`) — Stage-by-stage P&ID rebuild. Jobs are created, polled for status, and produce inspectable artifacts at each stage. Supports review state persistence and graph export.

### Frontend View States

```
empty → preview → processing → results (detection)
  ↓              ↓
batch ←─────────┘

Pipeline flow: upload → pipeline setup → pipeline results → HITL review
```

**Key components beyond basic detection:**
- `PipelineResultsView.tsx`: Displays pipeline job progress, artifacts, and stage outputs
- `PipelineArtifactCanvas.tsx`: Renders pipeline artifact overlays (masks, skeletons, graphs)
- `PipelineHitlReviewView.tsx`: Human-in-the-loop review entrypoint for object and traced-path gates
- `PipelineReviewWorkspaceView.tsx`: Dedicated review workspace for object boxes, ports, traces, and branches
- `ReviewCanvasLayers.tsx`: SVG overlay for equipment/object boxes, ports, trace paths, and branch paths
- `CanvasView.tsx`: Interactive canvas with zoom/pan, bbox editing, cursor guide, minimap, and image overlays
- `PdfPageSelector.tsx`: Page selection for multi-page PDF uploads

### Backend API Endpoints

**Detection (legacy path):**
- `POST /api/detect` — Run YOLO detection with SAHI + optional OCR
- `GET /api/results/{result_id}` — Get detection result
- `PATCH /api/results/{result_id}/objects/{obj_id}` — Update detected object
- `POST /api/results/{result_id}/objects` — Create new object
- `DELETE /api/results/{result_id}/objects/{obj_id}` — Delete object

**Pipeline jobs:**
- `POST /api/pipeline/jobs` — Start a pipeline job (staged: normalize → OCR → detect → mask → skeleton → edges → graph → QA)
- `GET /api/pipeline/jobs/{job_id}` — Get job status and progress
- `POST /api/pipeline/merge` — Merge multi-sheet pipeline results
- `GET /api/pipeline/jobs/{job_id}/review-state` — Get review state
- `PUT /api/pipeline/jobs/{job_id}/review-state` — Update review state
- `GET /api/pipeline/jobs/{job_id}/reviewed-graph` — Get review-corrected graph
- `GET /api/pipeline/jobs/{job_id}/reviewed-qa` — Get review-corrected QA
- `GET /api/pipeline/jobs/{job_id}/artifacts/{artifact_name}` — Download stage artifact

**Model discovery:**
- `GET /api/health` — Health check with model status and memory usage
- `GET /api/model-types` — List available detection model types
- `GET /api/models` — List available model configurations
- `GET /api/weight-files` — Scan for .pt/.onnx weight files
- `GET /api/config-files` — Scan for YOLO config files

**Export:**
- `POST /api/export/excel` — Export detection results to Excel
- `POST /api/pdf-extract` — Extract images from PDF uploads

### P&ID Pipeline Stages

The pipeline in `garnet/pid_extractor.py` orchestrates a multi-stage rebuild:

| Stage | Name | What it does | Key output artifacts |
|-------|------|-------------|---------------------|
| 1 | Normalization | Grayscale, histogram equalization, adaptive/Otsu binary | `stage1_gray.png`, `stage1_binary_adaptive.png`, `stage1_binary_otsu.png` |
| 2 | OCR Discovery | Tiled OCR via EasyOCR/Gemini/PaddleOCR route | `stage2_ocr_results.json`, text regions |
| 4 | Object Detection | YOLOv11 + SAHI symbol detection | `stage4_objects.json`, `stage4_objects_overlay.png`, topology markers |
| 5 | Pipe Mask | Provisional pipe segmentation from binary + suppression | `stage5_pipe_mask.png`, `stage5_pipe_mask_overlay.png` |
| 5b | Geometric Lines | DeepLSD line detection or inpaint-based line extraction | Line geometry for topology |
| 6 | Morphological Seal | Seal gaps in pipe mask | `stage6_pipe_seal.png` |
| 7 | Skeleton | 1-pixel centerline skeleton from pipe mask | `stage7_pipe_skeleton.png` |
| 8 | Node Detection | Endpoint/junction detection on skeleton (8-neighbor degree) | `stage8_pipe_nodes.json` |
| 9 | Node Clustering | DBSCAN clustering of pixel-dense skeleton nodes | `stage9_node_clusters.json` |
| 10 | Edge Tracing | Depth-first skeleton traversal, crossing resolution | `stage10_pipe_edges.json`, `stage10_crossing_resolution.json` |
| 11 | Trace Associations | Text/equipment attachment to edges, terminal classification | `stage11_trace_associations.json` |
| 12 | Graph Assembly | NetworkX graph construction + edge topology | `stage12_graph.graphml`, `stage12_edge_terminals.json` |
| 13 | Graph QA | Anomaly detection, crossing verification, review package | `stage13_review_package.json` |
| 14 | Apply Reviews | Merge review decisions into graph corrections | `stage14_reviewed_graph.graphml` |
| 15 | Process Exports | Final export generation (GraphML, JSON, connection overlays) | Final graph exports |
| 16 | Connection Overlay | Visual overlay of connections on original | `stage16_connection_overlay.png` |

**Pipeline config** is controlled via `PipelineConfig` dataclass (thresholds, device, OCR route, stage stop points).

## Important Conventions

### Backend

- **Run from `backend/`**: All backend commands expect `backend/` as the working directory so relative paths for weights, outputs, and datasets resolve correctly.
- **Settings import**: Always use `import garnet.Settings as Settings` (module-level import, not `from garnet.Settings import Settings`).
- **Pipeline config**: Add new thresholds and toggles to `PipelineConfig` instead of scattering magic numbers through stage code.
- **Stage outputs**: Keep them inspectable — every stage writes a manifest entry and artifact files.
- **Geometry first, semantics second**: Do not promote OCR text or detections directly into graph truth without geometric/topological support.
- **Never destroy source**: Keep task-specific masks and derived views separate from the original raster.
- **Model caching**: Use the cached model infrastructure in `api.py`; never load models directly.
- **Logging**: Use the configured `logger` (writes to `garnet.log`).
- **Nested AGENTS.md**: Follow `backend/garnet/AGENTS.md` for pipeline-specific conventions and `AGENTS.md` for cross-module rules.

### Frontend

- **Bun only**: Use `bun` for all package management. Never use `npm` or `yarn`.
- **Zustand state**: Use `set()` with immutable updates; access current state with `get()`.
- **History actions**: Record undo/redo via `useHistoryStore.getState().addAction()`.
- **Object keys**: Use `objectKey(obj)` (CategoryID + ObjectID) for unique identification.
- **Confidence filtering**: Filter objects by `confidenceFilter` in components using `useMemo`.
- **Pipeline review gates**: The HITL review flow has distinct object review and traced-path review modes. Keep UI copy, class suggestions, and editing controls scoped to the active review type.
- **Add-box class input**: The new-box class control must support both suggested classes and manual text entry. Suggested classes should come from the current review bucket or current workspace collection, not a hard-coded global-only list.
- **YOLO label alignment**: For object/equipment review, keep `Object` aligned with the detection class and keep detected text/tag values in the text/label field. Do not concatenate class and object ID into the class field.
- **Canvas guide lines**: Cursor guide lines are screen-space overlays and must not scale with zoom. Compute guide position from image coordinates, but render the vertical/horizontal lines in the canvas viewport layer.
- **Trace overlay selection**: Trace and branch overlays in `ReviewCanvasLayers.tsx` must remain image-coordinate aligned with the source raster. Use a wide transparent SVG stroke for hit-testing and keep empty overlay areas pass-through so normal pan/zoom still works.
- **Trace removal**: In traced-path review, selecting a trace/branch should expose the remove action and persist the removal as a `trace_overrides` rejection. Do not delete raw stage artifacts from the frontend.

### Cross-Cutting

- **No secrets in code**: API keys, model paths, and secrets are configured via environment variables.
- **Generated artifacts stay out of git**: `backend/output/`, `backend/runs/`, `backend/output_debug/`, `.ultralytics_runs/`.
- **Keep module boundaries**: Backend API ↔ garnet pipeline ↔ frontend each have clear ownership.
- **Two API paths are separate**: The legacy `/api/detect` path and the pipeline job API serve different purposes — don't mix their concerns.

## Testing

Tests use Python's `unittest` framework and live in `backend/tests/` (44+ test files).

```bash
cd backend

# Run all tests
python -m unittest discover -s tests -p "test*.py" -v

# Run a specific test file
python -m unittest tests.test_pipe_graph -v

# Run a specific test method
python -m unittest tests.test_pipeline_api.TestPipelineAPI.test_pipeline_job_runs_stage2_and_reports_artifacts -v
```

Key test files:
- `test_pipeline_api.py` — FastAPI TestClient integration tests for pipeline job endpoints (62KB, most comprehensive)
- `test_pid_extractor_cli.py` — CLI-level pipeline integration tests (45KB)
- `test_pipe_edge_connectivity.py` — Edge connectivity and topology tests
- `test_trace_graph_builder.py` — Trace-based graph construction tests
- `test_pipe_text_attachment.py` — Text-to-pipe attachment tests
- Stage-specific tests: `test_stage5b_branch_terminal.py`, `test_stage13_review_package.py`, `test_stage14_review_decisions.py`, `test_stage15_process_exports.py`

Frontend has no test framework configured — verify with `bun run lint` and `bun run build`.

## Debugging

- FastAPI auto-docs: `http://localhost:8001/docs`
- Backend logs: `garnet.log` in the backend working directory
- React DevTools for component tree and Zustand store inspection
- Pipeline debug: use `run_debug.sh` for verbose stage output
- Compile-check backend: `python -m py_compile api.py garnet/*.py garnet/utils/*.py`

**Common issues:**
- **Model not found**: Check `backend/yolo_weights/` for .pt/.onnx files
- **OCR fails**: Ensure EasyOCR is installed and cached reader initializes (check `OCR_GPU` setting)
- **Frontend proxy error**: Backend must be running on port 8001
- **CUDA OOM**: Reduce `image_size` or set `device="cpu"`
- **Import errors**: Always run from `backend/` directory
- **Gemini OCR fails**: Verify `OPENROUTER_API_KEY` is set in `.env`

## Key Design Decisions

1. **Two API paths**: Legacy `/api/detect` for simple detection + pipeline job API for staged P&ID rebuild. Keep concerns separate.
2. **Staged pipeline with artifacts**: Every stage writes inspectable outputs + manifest. Enables progressive review and debugging.
3. **SAHI for large images**: Slicing Aided Hyper Inference handles P&IDs that can exceed 30k×20k pixels.
4. **Multiple OCR routes**: EasyOCR (local), Gemini (cloud via OpenRouter), PaddleOCR (RapidOCR) — runtime-selectable per pipeline job.
5. **Geometry-first topology**: OCR text and detections are provisional evidence; geometric consistency (skeleton, crossings, terminals) is the primary signal for graph construction.
6. **Review/HITL workflow**: Graph QA results feed a human-in-the-loop review interface. Review decisions persist and produce corrected graph exports.
7. **Multi-sheet merge**: Off-page connectors allow merging graphs across multiple P&ID sheets.
8. **Client-side canvas editing**: All bbox/text edits happen in frontend via Konva; backend stores results in memory for the session.
9. **Bun package manager**: Frontend uses Bun exclusively for faster installs and runtime.
10. **Model caching**: YOLO and OCR models cached in memory to avoid reload overhead.
