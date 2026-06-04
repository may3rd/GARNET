# Data Flow

## Detection Mode

```
1. Upload P&ID Image (Frontend)
   ├─ Drag & drop or file selection
   └─ Preview image with metadata
         │
         ▼
2. Configure Detection Parameters (Frontend)
   ├─ Select model type (ultralytics or gemini)
   ├─ Choose weight file (for ultralytics)
   ├─ Set confidence threshold (0.0 - 1.0)
   ├─ Configure image size (128 - 1280)
   ├─ Set overlap ratio (0.0 - 0.5)
   ├─ Configure post-processing options
   ├─ Enable/disable OCR
   └─ Select processing mode: Detection or Pipeline
         │
         ▼
3. Frontend → POST /api/detect (Detection Mode)
   ├─ FormData with image and parameters
   └─ AbortController for cancellation
         │
         ▼
4. Backend: Validation & Processing
   ├─ Validate file extension and size
   ├─ Load cached model or create new one
   ├─ Decode image with OpenCV
   └─ Store result in memory cache
         │
         ▼
5. Backend: SAHI Slicing (if applicable)
   ├─ Slice large image into tiles (configurable size)
   ├─ Run YOLO inference on each tile
   └─ Merge overlapping detections (NMM postprocessing)
         │
         ▼
6. Backend: OCR (optional, if enabled)
   ├─ Extract text from symbol regions
   ├─ Rotate vertical text objects
   ├─ Apply image preprocessing
   └─ Use EasyOCR/Gemini/PaddleOCR/OCRMac with wordbeamsearch decoder
         │
         ▼
7. Backend: Response
   ├─ JSON with detections + image URL
   ├─ Store in RESULTS_STORE with TTL
   └─ Return result_id for future operations
         │
         ▼
8. Frontend: ResultsView
   ├─ Interactive canvas with pan/zoom/minimap
   ├─ Color-coded bounding boxes by category
   ├─ Object list with editing capabilities
   ├─ Accept/reject workflow with status indicators
   ├─ Undo/redo support for all operations
   └─ Export options (JSON, YOLO, COCO, LabelMe, PDF)
```

## Pipeline Mode

```
1. Upload P&ID Image (Frontend)
   ├─ Drag & drop or file selection
   └─ Preview image with metadata
         │
         ▼
2. Configure Detection Parameters (Frontend)
   ├─ Select model type (ultralytics or gemini)
   ├─ Choose weight file (for ultralytics)
   ├─ Set confidence threshold (0.0 - 1.0)
   ├─ Configure image size (128 - 1280)
   ├─ Set overlap ratio (0.0 - 0.5)
   ├─ Configure post-processing options
   ├─ Enable/disable OCR
   └─ Select processing mode: Detection or Pipeline
         │
         ▼
3. Frontend → POST /api/pipeline/jobs (Pipeline Mode)
   ├─ FormData with image and parameters
   └─ AbortController for cancellation
         │
         ▼
4. Backend: Validation & Processing
   ├─ Validate file extension and size
   ├─ Load cached model or create new one
   ├─ Decode image with OpenCV
   ├─ Store job info and start background processing
   └─ Store result in PIPELINE_JOBS
         │
         ▼
5. Backend: Pipeline Processing (Background)
   ├─ Stage 1: Input Normalization
   │   ├─ Load P&ID image (BGR format)
   │   ├─ Apply normalization (deskew, contrast enhancement)
   │   └─ Generate binary mask for pipe detection
   ├─ Stage 2: OCR Discovery
   │   ├─ Run selected OCR route (EasyOCR, Gemini, PaddleOCR, or OCRMac)
   │   ├─ Detect text regions and extract text content
   │   └─ Output: OCR results with bounding boxes and text
   ├─ Stage 3: HITL Review (Manual - outside pipeline)
   │   ├─ Reserved for human-in-the-loop review
   │   └─ Major equipment bounding boxes supplied through LabelMe/frontend review
   ├─ Stage 4: Object Detection & Fusion
   │   ├─ Run SAHI-based object detection (YOLOv8/YOLOv11)
   │   ├─ Process line number fusion and instrument tag fusion
   │   ├─ Apply topology-marker routing for junction detection
   │   └─ Output: Detected objects with bounding boxes, class labels, and confidence scores
   ├─ Stage 5: Pipe Mask Generation
   │   ├─ Create provisional pipe mask from binary image
   │   ├─ Suppress OCR and object regions to focus on pipe regions
   │   └─ Output: Pipe mask image and summary statistics
   ├─ Stage 5b: Pipe Tracing
   │   ├─ Trace pipe centerlines from mask using computer vision algorithms
   │   ├─ Start from detected ports and follow pipe paths
   │   ├─ Handle inline objects (valves, reducers) by jumping over them
   │   ├─ Detect terminals (equipment, tags, junctions, sheet edges)
   │   └─ Output: Traced pipe paths, junctions, and endpoints
   ├─ Stage 6: Trace Associations
   │   ├─ Associate traced paths with ports, inline objects, line numbers, instruments
   │   ├─ Connect detected objects to the pipe network
   │   ├─ Fill missing line numbers with simulated-HITL placeholders
   │   └─ Output: Object-to-pipe associations and connection data
   ├─ Stage 7: Geometric Graph Assembly
   │   ├─ Assemble and normalize the geometric trace graph
   │   ├─ Label page connectors for multi-document support
   │   ├─ Run graph quality assurance checks
   │   └─ Export initial graph payload (v1)
   ├─ Stage 8: HITL Review Package
   │   ├─ Build graph/line-number human-in-the-loop review package
   │   └─ Prepare items for validation (junctions, connections, line numbers)
   ├─ Stage 9: Apply Review Decisions
   │   ├─ Apply human review decisions to correct the graph
   │   └─ Pass through unchanged when no decisions exist
   ├─ Stage 10: Process Exports
   │   ├─ Generate line lists, equipment connectivity, inline MTO
   │   ├─ Create inline observations and instrument index
   │   └─ Output: Process exports in JSON/CSV formats
   └─ Stage 11: Connection Overlay
       ├─ Render connection-pipeline overlay for visual review
       └─ Create visual representation of connections on original image
         │
         ▼
6. Frontend: ProcessingView/ResultsView
   ├─ Progress tracking and stage-wise results
   ├─ Real-time stage status updates
   ├─ Access to intermediate artifacts
   └─ Review workflow for HITL stages
```