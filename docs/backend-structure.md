# Backend Structure

```
backend/
├── api.py                   # Canonical FastAPI application entry point
├── main.py                  # Compatibility shim that re-exports api:app
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
├── garnet/                  # Core logic package
│   ├── __init__.py          # Package initializer
│   ├── Settings.py          # Settings management
│   ├── model_defaults.py    # Model weight file discovery
│   ├── object_detection_sahi.py # SAHI-based object detection
│   ├── easyocr_sahi.py      # EasyOCR with SAHI
│   ├── gemini_ocr_sahi.py   # Gemini OCR with SAHI
│   ├── paddle_ocr_sahi.py   # PaddleOCR with SAHI
│   ├── ocrmac_sahi.py       # OCRMac with SAHI
│   ├── pipeline/            # Pipeline processing stages
│   │   ├── pid_extractor.py # Main pipeline orchestrator
│   │   ├── stage1_input_normalization.py
│   │   ├── stage2_ocr_discovery.py
│   │   ├── stage4_object_detection.py
│   │   ├── stage4_line_number_fusion.py
│   │   ├── stage4_instrument_tag_fusion.py
│   │   ├── stage5_pipe_mask.py
│   │   ├── stage5b_pipe_trace.py
│   │   ├── stage6_trace_associations.py
│   │   ├── stage7_geometric_graph_assembly.py
│   │   ├── stage7c_page_connector_labeling.py
│   │   ├── stage7b_graph_export.py
│   │   ├── stage8_graph_qa.py
│   │   ├── stage9_apply_review_decisions.py
│   │   ├── stage10_process_exports.py
│   │   └── stage11_connection_overlay.py
│   ├── review/              # Review and HITL functionality
│   │   ├── review_state.py  # Review state management
│   │   ├── review_workspace.py # Review workspace handling
│   │   ├── stage8_review_package.py
│   │   ├── stage9_review_decisions.py
│   │   └── stage10_review_package.py
│   ├── tracing/             # Path tracing and graph construction
│   │   ├── path_tracer/
│   │   │   ├── cv_pipe_tracer.py
│   │   │   └── stage5b_pipeline.py
│   │   ├── trace_associations.py
│   │   ├── trace_graph_builder.py
│   │   ├── trace_graph_qa.py
│   │   └── topology_markers.py
│   ├── utils/               # Utility functions
│   │   ├── utils.py         # Common utilities
│   │   └__init__.py
│   ├── graph_export_adapter.py # Graph export functionality
│   ├── pipe_mask.py         # Pipe mask generation
│   ├── pipe_sheet_merge.py  # Pipe sheet merging
│   ├── reviewed_outputs.py  # Reviewed output generation
│   ├── topology_markers.py  # Topology detection
│   ├── line_number_fusion.py # Line number fusion
│   ├── instrument_tag_fusion.py # Instrument tag fusion
│   ├── page_connector.py    # Page connector handling
│   └── render_connection_pipeline_overlay.py # Connection overlay rendering
├── datasets/                # Dataset configuration files
├── yolo_weights/            # Model weights
└── static/                  # Static assets
    └── images/predictions/  # Generated prediction images
```