# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GARNET (GCME AI-Recognition Network for Engineering Technology) is an AI-powered tool for automating symbol detection, classification, and connectivity analysis in Piping and Instrumentation Diagrams (P&IDs). It combines YOLOv11 object detection with graph-based analytics to transform P&ID workflows.

**Tech Stack:**
- **Backend**: FastAPI + Python (YOLOv11, SAHI, EasyOCR, NetworkX)
- **Frontend**: React 18 + TypeScript + Vite + Zustand (state management) + Radix UI + Tailwind CSS + **Bun (package manager)**
- **AI Models**: Ultralytics YOLOv11, EasyOCR/PaddleOCR, DeepLSD (line detection)

## Development Commands

### Backend Development

```bash
# Start API backend (FastAPI) on port 8001
uvicorn api:app --reload --port 8001

# Install Python dependencies
pip install -r requirements.txt

# Install DeepLSD (required for line extraction)
git clone --recurse-submodules https://github.com/cvg/DeepLSD.git
cd DeepLSD
bash quickstart_install.sh
cd ..
```

### Frontend Development

**IMPORTANT**: The frontend **MUST use Bun** as the package manager, **NOT npm or yarn**. All commands should use `bun` instead of `npm`.

```bash
# Install frontend dependencies
cd frontend
bun install

# Start React dev server on port 5173 (proxies /api and /static to port 8001)
bun run dev

# Build for production
bun run build

# Preview production build
bun run preview

# Lint frontend code
bun run lint

# Add a new dependency (use bun add, NOT npm install)
bun add <package-name>

# Remove a dependency (use bun remove, NOT npm uninstall)
bun remove <package-name>
```

**Note**: Vite dev server automatically proxies `/api/*` and `/static/*` requests to the backend at `http://localhost:8001`.

### Batch Processing

```bash
# Run batch inference on multiple P&ID images
python predict_images.py \
    --image_path path/to/pids_folder \
    --model_type yolov8 \
    --model_path path/to/model_weights.pt \
    --output_path results/
```

### Model Training

```bash
# Train custom YOLO models for P&ID symbols
yolo train \
    data=data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
```

## Architecture Overview

### System Architecture

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│  React Frontend │ ──HTTP─→│  FastAPI Backend │ ──→──  │  garnet/ Module │
│  (Port 5173)    │ ←──JSON─│  (Port 8001)     │ ←──    │  (Core Logic)   │
└─────────────────┘         └──────────────────┘         └─────────────────┘
        │                            │                            │
        │ Zustand State              │ Model Cache               │
        │ History Store              │ OCR Cache                 │
        └─ Radix UI                  │ RESULTS_STORE             │
           TailwindCSS               └─ Static Files             │
                                        (prediction images)      │
                                                                  │
                                                          YOLOv11, SAHI
                                                          EasyOCR, NetworkX
                                                          DeepLSD, OpenCV
```

### Frontend Architecture (frontend/src/)

**State Management (Zustand)**:
- `stores/appStore.ts`: Main app state (image, detection result, batch processing, view state, UI settings)
- `stores/historyStore.ts`: Undo/redo history for object edits (review status, updates, delete, create)

**Key Components**:
- `App.tsx`: Root component with view routing (empty/preview/processing/results/batch)
- `components/UploadZone.tsx`: Drag-and-drop file upload
- `components/DetectionSetup.tsx`: Model configuration panel (weight file, confidence, image size, overlap, OCR toggle)
- `components/ProcessingView.tsx`: Real-time progress indicator with step visualization
- `components/ResultsView.tsx`: Main results editor (canvas + sidebar + toolbar)
- `components/CanvasView.tsx`: **Interactive canvas** with zoom/pan, bbox editing, object selection, minimap
- `components/ObjectSidebar.tsx`: Detected objects list with filter/sort
- `components/BatchResultsView.tsx`: Batch job management UI

**View States Flow**:
```
empty → preview → processing → results
  ↓              ↓
batch ←─────────┘
```

**Key Features**:
- **Undo/Redo**: Stack-based history for all edits (Ctrl/Cmd+Z, Shift+Z)
- **Confidence Filter**: Hide/show objects below threshold (slider in header)
- **Review Status**: Accept/reject objects with visual overlay (✓/✗)
- **Batch Processing**: Queue-based with pause/resume/retry
- **Export Formats**: JSON, YOLO, COCO, LabelMe, PDF report

### Backend Architecture (api.py)

**Main Endpoints**:
- `POST /api/detect`: Run YOLO detection with SAHI (sliced inference) + optional OCR
- `GET /api/models`: List available model types
- `GET /api/weight-files`: Scan `yolo_weights/` for .pt/.onnx files
- `GET /api/config-files`: Scan `datasets/yaml/` for YOLO config files
- `PATCH /api/results/{id}/objects/{obj_id}`: Update detected object
- `POST /api/results/{id}/objects`: Create new object
- `DELETE /api/results/{id}/objects/{obj_id}`: Delete object

**Key Patterns**:
- **Model Caching**: Detection models cached in `MODEL_CACHE` dict to avoid reload overhead
- **OCR Caching**: EasyOCR reader cached globally (expensive to initialize)
- **In-Memory Storage**: Detection results stored in `RESULTS_STORE` dict (keyed by UUID)
- **Static Files**: Prediction images saved to `static/images/` and served via FastAPI

**Detection Workflow**:
1. Receive FormData with image file + options (model, weights, confidence, size, overlap, OCR flag)
2. Get or create cached SAHI-wrapped YOLO model
3. Run `get_sliced_prediction()` for large image handling (tiles with overlap)
4. Process detections into table format (Index, Object, CategoryID, ObjectID, bbox, Score)
5. If OCR enabled: crop symbols, preprocess, run EasyOCR on text-bearing classes
6. Store result in memory, save visualization, return JSON

### Core Module (garnet/)

**Key Modules**:

| Module | Purpose |
|--------|---------|
| `Settings.py` | Global configuration (paths, symbol types, text classes) |
| `predict_images.py` | Batch inference function for multiple P&ID images |
| `object_and_text_detect.py` | YOLO11 + PaddleOCR detection pipeline with masking |
| `text_ocr.py` | Text extraction utilities (rotation, preprocessing) |
| `connectivity_graph.py` | Graph-based connectivity analysis (NetworkX) |
| `pid_extractor.py` | **Complete P&ID digitization pipeline** (7+ stages) |
| `dexpi_exporter.py` | DEXPI XML export for P&ID interchange |
| `utils/utils.py` | Image utilities (rotation, line removal, morphology) |
| `utils/deeplsd_utils.py` | DeepLSD line detection wrapper |

**P&ID Processing Pipeline (pid_extractor.py)**:

The most complex module (98KB) implementing end-to-end P&ID digitization:

1. **Ingest**: Load image, COCO detections, OCR JSON
2. **Preprocess**: Grayscale, adaptive binarization, optional deskew (perspective correction)
3. **Symbols/Text**: Import detections, normalize labels, overlay on image
4. **Linework**: Binarize → Skeletonize → Extract endpoints/junctions
5. **Graph**: Build nodes/edges; attach symbols/text; merge nearby nodes
6. **From/To**: Emit endpoint pairs per segment (connectivity table)
7. **Export**: GraphML, CSV/JSON, DEXPI XML

**Configuration Dataclass**:
```python
@dataclass
class PipelineConfig:
    device: str = "auto"                    # "cuda", "cpu", "mps", or "auto"
    canny_low, canny_high: int              # Edge detection thresholds
    binarize_block, binarize_c: int         # Adaptive thresholding params
    min_segment_len: int                    # Line segment filtering
    merge_node_dist: int                    # Node clustering distance
    close_hv: bool                          # Close horizontal/vertical gaps
    deskew: bool                            # Perspective correction
    default_conf_thresh: float = 0.25       # Detection confidence threshold
    class_conf_thresh: Dict                 # Per-class confidence overrides
    class_role_map: Dict                    # Class → NodeType mapping
```

### Data Flow: Upload to Results

```
User uploads image
  ↓
Frontend: appStore.setImageFile() → currentView='preview'
  ↓
User configures detection (model, weights, confidence, size, overlap, OCR)
  ↓
User clicks "Run Detection" → appStore.runDetection()
  ↓
Frontend: POST /api/detect with FormData
  ↓
Backend: Load image → Get cached model → Run SAHI sliced inference
         → Process detections → Optional OCR → Store result → Return JSON
  ↓
Frontend: setResult(response) → currentView='results'
  ↓
User views results in CanvasView (zoom/pan/edit)
  ↓
User edits objects (bbox, text, review status) → History actions
  ↓
User exports results (JSON/YOLO/COCO/LabelMe/PDF)
```

## Important Conventions

### Frontend Code Patterns

**State Updates (Zustand)**:
```typescript
// Always use set() with immutable updates
set(state => ({
  result: { ...state.result, objects: [...updatedObjects] }
}))

// Access current state with get()
const currentState = get()
```

**History Actions**:
```typescript
// Add history action when editing
useHistoryStore.getState().addAction({
  type: 'update',
  prev: originalObject,
  next: updatedObject
})
```

**Object Keys**:
```typescript
// Use objectKey() for unique object identification
import { objectKey } from '@/lib/objectKey'
const key = objectKey(obj)  // CategoryID + ObjectID
```

**Confidence Filtering**:
```typescript
// Always filter objects by confidenceFilter in components
const visibleObjects = useMemo(() =>
  result?.objects.filter(obj => obj.Score >= confidenceFilter) ?? [],
  [result, confidenceFilter]
)
```

### Backend Code Patterns

**Model Caching**:
```python
# Use get_cached_detection_model() for all model loading
model = get_cached_detection_model(
    model_type="ultralytics",
    model_path=weight_file,
    config_path=config_file,
    conf_th=conf_th,
    image_size=image_size
)
```

**OCR Text Extraction**:
```python
# Only run OCR on symbol classes in settings.SYMBOL_WITH_TEXT
if text_OCR and symbol.type in settings.SYMBOL_WITH_TEXT:
    text = extract_text_from_image(cropped_img, is_vertical)
```

**Logging**:
```python
# Use the configured logger (writes to garnet.log)
logger.info(f"Processing {len(detections)} detections")
logger.error(f"Failed to load model: {e}")
```

### Python Module Patterns

**Settings Import**:
```python
# Always import settings as module
import garnet.Settings as Settings
settings = Settings.Settings()

# Access configuration
OUTPUT_PATH = settings.OUTPUT_PATH
SYMBOL_WITH_TEXT = settings.SYMBOL_WITH_TEXT
```

**Pipeline Configuration**:
```python
# Use PipelineConfig dataclass for pid_extractor
from garnet.pid_extractor import PipelineConfig, run_pipeline

config = PipelineConfig(
    device="cuda",
    deskew=True,
    merge_node_dist=10,
    class_conf_thresh={"valve": 0.8, "pump": 0.9}
)
result = run_pipeline(image_path, config)
```

## Testing & Debugging

**Frontend Debugging**:
- React DevTools: Inspect component tree and Zustand store state
- Console logs in browser for state changes
- Check Network tab for API requests/responses

**Backend Debugging**:
- Check `garnet.log` for detailed logs
- Use `logger.info()` / `logger.error()` for debugging
- FastAPI auto-docs at `http://localhost:8001/docs`

**Common Issues**:
- **Model not found**: Check `yolo_weights/` directory has .pt/.onnx files
- **OCR fails**: Ensure EasyOCR reader is initialized (check cache)
- **Frontend proxy error**: Ensure backend is running on port 8001
- **CUDA out of memory**: Reduce `image_size` or use CPU (`device="cpu"`)
- **Frontend dependency issues**: Always use `bun install` and `bun add`, NEVER use `npm install` or `yarn`

## Configuration Files

**Backend Configuration**:
- `garnet/Settings.py`: Global paths, symbol types, text classes
- `datasets/yaml/*.yaml`: YOLO model configs (class names, dataset paths)

**Frontend Configuration**:
- `frontend/vite.config.ts`: Dev server proxy to backend
- `frontend/src/lib/api.ts`: Default detection options

**YOLO Dataset Format**:
```
dataset/
├── train/
│   ├── images/  # P&ID images (.jpg, .png)
│   └── labels/  # YOLO-format labels (.txt)
├── val/
│   ├── images/
│   └── labels/
└── data.yaml     # Class names, paths
```

## File Structure Reference

```
/GARNET
├── api.py                         # FastAPI backend (616 lines)
├── main.py                        # Legacy Streamlit app (not used)
├── export_to_excel.py             # Utility for Excel export
├── requirements.txt               # Python dependencies
├── garnet.log                     # Backend logs
├── static/images/                 # Prediction visualizations
├── yolo_weights/                  # .pt/.onnx model weights
├── datasets/yaml/                 # YOLO config files
├── output/                        # Generated artifacts
│   ├── cropped object detected/
│   └── text detected/
├── frontend/                      # React app
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/            # UI components
│   │   ├── stores/                # Zustand state
│   │   ├── hooks/                 # Custom hooks
│   │   ├── lib/                   # API client, exports, utils
│   │   └── types.ts               # TypeScript types
│   ├── package.json               # Frontend deps (Bun)
│   ├── vite.config.ts             # Vite config (proxy)
│   └── tailwind.config.ts         # Tailwind CSS
├── garnet/                        # Core Python module
│   ├── __init__.py
│   ├── Settings.py                # Configuration
│   ├── predict_images.py          # Batch inference
│   ├── object_and_text_detect.py  # YOLO + OCR pipeline
│   ├── text_ocr.py                # OCR utilities
│   ├── connectivity_graph.py      # Graph analysis
│   ├── pid_extractor.py           # Full P&ID pipeline (2800+ lines)
│   ├── dexpi_exporter.py          # DEXPI XML export
│   └── utils/
│       ├── utils.py               # Image utilities
│       └── deeplsd_utils.py       # Line detection
├── DeepLSD/                       # Line extraction submodule
└── test/                          # Test datasets
```

## Key Design Decisions

1. **Separation of Concerns**: Frontend (React) and backend (FastAPI) run independently. Vite dev server proxies API requests.

2. **Bun Package Manager**: Frontend uses Bun exclusively for faster installs and runtime performance. Do NOT use npm or yarn.

3. **SAHI for Large Images**: Use Slicing Aided Hyper Inference (SAHI) to handle large P&ID images by tiling with overlap.

4. **Model Caching**: Cache models in memory to avoid reload overhead during batch processing.

5. **OCR Optimization**: Only run OCR on symbol classes that typically contain text (defined in `settings.SYMBOL_WITH_TEXT`).

6. **Client-Side Editing**: All bbox/text edits happen in frontend; backend stores results in memory for session.

7. **History Stack**: Undo/redo implemented with separate history store to keep appStore clean.

8. **Confidence Filtering**: Non-destructive filter (objects below threshold hidden but not deleted).

9. **Batch Processing**: Queue-based with pause/resume/retry to handle large datasets.

10. **Export Flexibility**: Multiple export formats (YOLO, COCO, LabelMe, PDF) for different workflows.

11. **Graph-Based Analysis**: P&ID connectivity modeled as NetworkX graph for path analysis and system dependencies.
