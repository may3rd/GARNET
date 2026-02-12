# 🛠️ GARNET: AI-Driven P&ID Symbol Detection and Analysis

**G**CME **A**I-**R**ecognition **N**etwork for **E**ngineering **T**echnology  
_Precision in Every Connection_

[![YOLOv11](https://img.shields.io/badge/YOLOv11-💻-brightgreen)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-🖼️-orange)](https://opencv.org/)
[![NetworkX](https://img.shields.io/badge/NetworkX-📊-blue)](https://networkx.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-🔤-yellow)](https://github.com/JaidedAI/EasyOCR)
[![FastAPI](https://img.shields.io/badge/FastAPI-⚡-teal)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18.3-61DAFB)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7-3178C6)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF)](https://vitejs.dev/)

GARNET is an AI-powered tool designed to **automate symbol detection, classification, and connectivity analysis** in Piping and Instrumentation Diagrams (P&IDs). Built for engineers and maintenance teams, it combines state-of-the-art object detection (YOLOv11/YOLOv8) with graph-based analytics to transform P&ID workflows.

---

## 🚀 Features

### Frontend Features

- **Interactive Canvas**: Pan, zoom, and navigate through large P&ID images with minimap support
- **Object Detection Visualization**: Color-coded bounding boxes with confidence scores
- **Object Editing**: Create, update, and delete detected objects directly on the canvas
- **Review Workflow**: Accept/reject objects with visual status indicators
- **Batch Processing**: Process multiple images with queue management, pause/resume, and progress tracking
- **Undo/Redo**: Full history support for all editing operations
- **Keyboard Shortcuts**: Efficient navigation and editing with keyboard shortcuts
- **Export Formats**: Export to JSON, YOLO, COCO, LabelMe, or PDF
- **Dark Mode**: Toggle between light and dark themes
- **Confidence Filtering**: Filter objects by confidence threshold
- **Class Visibility**: Toggle visibility of specific object classes

### Backend Features

- **Symbol Detection**: Identify valves (gate, globe, check), pumps, tanks, and more using YOLOv11/YOLOv8.
- **SAHI Integration**: Slicing Aided Hyper Inference for accurate detection on large images.
- **Automated Counting**: Generate counts for each symbol type in a P&ID.
- **Text Recognition (OCR)**: Extract text annotations from symbols using EasyOCR with support for vertical text.
- **Model Caching**: Cache loaded models for faster subsequent detections.
- **Results Caching**: In-memory cache with TTL for detection results.
- **Automatic Cleanup**: Periodic cleanup of old prediction images and expired cache entries.
- **Health Monitoring**: Health check endpoint with model loading status and memory usage.
- **Environment Configuration**: Configurable via environment variables for development and production.

---

## 📐 Architecture

### System Overview

GARNET follows a modern client-server architecture with a clear separation between the interactive frontend and the AI-powered backend:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GARNET Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────┐          ┌─────────────────────────────┐ │
│  │      React Frontend          │          │      FastAPI Backend        │ │
│  │  ┌────────────────────────┐  │          │  ┌───────────────────────┐  │ │
│  │  │  React 18 + TypeScript │  │◄────────►│  │  FastAPI              │  │ │
│  │  │  Zustand (State)       │  │   HTTP   │  │  SAHI (Slicing)       │  │ │
│  │  │  Tailwind CSS + Radix  │  │          │  │  Ultralytics (YOLO)   │  │ │
│  │  │  Vite (Build Tool)     │  │          │  │  EasyOCR (Text)       │  │ │
│  │  └────────────────────────┘  │          │  │  OpenCV (Image Proc)  │  │ │
│  │                              │          │  └───────────────────────┘  │ │
│  │  Port: 5173 (dev) / 80/443   │          │  Port: 8001                 │ │
│  frontend/                   │          │  backend/api.py             │ │
│  └──────────────────────────────┘          └─────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Frontend Stack

| Technology       | Version | Purpose                                         |
| ---------------- | ------- | ----------------------------------------------- |
| **React**        | 18.3.1  | UI library with hooks and functional components |
| **TypeScript**   | 5.7.3   | Type-safe development                           |
| **Vite**         | 6.0.7   | Fast development server and optimized builds    |
| **Zustand**      | 5.0.3   | Lightweight state management                    |
| **Tailwind CSS** | 3.4.17  | Utility-first CSS framework                     |
| **Radix UI**     | Latest  | Accessible, unstyled UI primitives              |
| **Lucide React** | Latest  | Modern icon library                             |
| **jsPDF**        | 4.0.0   | PDF export functionality                        |

#### Frontend Structure

```
frontend/
├── src/
│   ├── components/           # React components
│   │   ├── ui/              # Reusable UI primitives (Radix-based)
│   │   ├── App.tsx          # Main application component
│   │   ├── UploadZone.tsx   # Image upload with drag-and-drop
│   │   ├── DetectionSetup.tsx  # Detection parameter configuration
│   │   ├── ResultsView.tsx  # Detection results with interactive canvas
│   │   ├── BatchResultsView.tsx  # Batch processing interface
│   │   └── ...
│   ├── stores/
│   │   ├── appStore.ts      # Main application state (Zustand)
│   │   └── historyStore.ts  # Undo/redo action history
│   ├── lib/
│   │   ├── api.ts           # API client functions
│   │   ├── pdfExport.ts     # PDF generation utilities
│   │   └── exportFormats.ts # CSV/JSON export utilities
│   ├── hooks/               # Custom React hooks
│   ├── types.ts             # TypeScript type definitions
│   └── styles/              # Global styles and CSS variables
├── package.json
├── vite.config.ts           # Vite configuration with API proxy
└── tsconfig.json
```

### Backend Stack

| Technology           | Purpose                                        |
| -------------------- | ---------------------------------------------- |
| **FastAPI**          | High-performance Python web framework          |
| **SAHI**             | Slicing Aided Hyper Inference for large images |
| **Ultralytics**      | YOLOv11/YOLOv8 object detection models         |
| **EasyOCR**          | Text recognition from detected symbols         |
| **OpenCV**           | Image processing and manipulation              |
| **NetworkX**         | Graph construction and connectivity analysis   |
| **Pydantic**         | Data validation and settings management        |
| **python-multipart** | File upload handling                           |

#### Backend Structure

```
backend/
├── api.py                   # Canonical FastAPI application entry point
├── main.py                  # Compatibility shim that re-exports api:app
├── requirements.txt         # Python dependencies
├── .env                     # Environment configuration
├── garnet/                  # Core logic package
├── datasets/                # Dataset configuration files
├── yolo_weights/            # Model weights
└── static/                  # Static assets
    └── images/predictions/  # Generated prediction images
```

### API Endpoints

| Method | Endpoint                                    | Description                                     |
| ------ | ------------------------------------------- | ----------------------------------------------- |
| GET    | `/`                                         | API root with service info                      |
| GET    | `/api/health`                               | Health check with model status and memory usage |
| GET    | `/api/model-types`                          | Get available model types                       |
| GET    | `/api/models`                               | Get model type values                           |
| GET    | `/api/weight-files`                         | Get available model weight files                |
| GET    | `/api/config-files`                         | Get available dataset config files              |
| POST   | `/api/pdf-extract`                          | Extract PDF pages to PNG images                 |
| POST   | `/api/detect`                               | Run object detection on uploaded image          |
| GET    | `/api/results/{result_id}`                  | Fetch a previously detected result              |
| PATCH  | `/api/results/{result_id}/objects/{obj_id}` | Update a detected object                        |
| POST   | `/api/results/{result_id}/objects`          | Create a new object                             |
| DELETE | `/api/results/{result_id}/objects/{obj_id}` | Delete a detected object                        |

### Data Flow

```
1. Upload P&ID Image (Frontend)
   ├─ Drag & drop or file selection
   └─ Preview image with metadata
         │
         ▼
2. Configure Detection Parameters (Frontend)
   ├─ Select model type (ultralytics)
   ├─ Choose weight file
   ├─ Set confidence threshold (0.0 - 1.0)
   ├─ Configure image size (320 - 2048)
   ├─ Set overlap ratio (0.0 - 0.95)
   └─ Enable/disable OCR
         │
         ▼
3. Frontend → POST /api/detect
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
5. Backend: SAHI Slicing
   ├─ Slice large image into tiles (configurable size)
   ├─ Run YOLO inference on each tile
   └─ Merge overlapping detections (NMM postprocessing)
         │
         ▼
6. Backend: OCR (optional)
   ├─ Extract text from symbol regions
   ├─ Rotate vertical text objects
   ├─ Apply image preprocessing
   └─ Use EasyOCR with wordbeamsearch decoder
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

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- Node.js 18+ (for frontend)
- Bun or npm (package manager)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/may3rd/GARNET.git
cd GARNET
```

### 2. Install DeepLSD (Optional - for line extraction)

```bash
git clone --recurse-submodules https://github.com/cvg/DeepLSD.git
cd DeepLSD
bash quickstart_install.sh
cd ..
```

### 3. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4. Install Frontend Dependencies

```bash
cd frontend
bun install
# or: npm install
cd ..
```

---

## 🖥️ Usage

### 1. React Frontend + API Backend (Recommended)

This is the primary mode for interactive P&ID analysis. The React frontend provides a modern UI for uploading images, configuring detection parameters, reviewing results, and exporting data.

#### Tech Stack Summary

| Component  | Technology         | Version |
| ---------- | ------------------ | ------- |
| Frontend   | React + TypeScript | 18.3.1  |
| Build Tool | Vite               | 6.0.7   |
| Styling    | Tailwind CSS       | 3.4.17  |
| State      | Zustand            | 5.0.3   |
| Backend    | FastAPI            | Latest  |
| AI Engine  | SAHI + Ultralytics | Latest  |

#### Quick Start

**Terminal 1 - Start the API backend:**

```bash
# Copy environment file and configure
cp .env.example .env

# Start FastAPI server
cd backend
uvicorn api:app --reload --port 8001
```

Backend runs at `http://localhost:8001` with auto-reload enabled for development.

**Terminal 2 - Start the React frontend:**

```bash
cd frontend

# Copy environment file and configure
cp .env.example .env.local

# Start development server
bun run dev
# or: npm run dev
```

Frontend runs at `http://localhost:5173` with hot module replacement.

#### Environment Variables

**Backend Configuration (`.env`):**

```bash
# Environment (development, production)
ENV=development
DEBUG=true

# Server Configuration
HOST=localhost
PORT=8001

# CORS - Comma-separated list of allowed origins
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:4173

# File Upload Limits
MAX_FILE_SIZE_MB=50
ALLOWED_IMAGE_EXTENSIONS=.jpg,.jpeg,.png,.webp,.bmp,.tiff

# Model Defaults
DEFAULT_CONF_THRESHOLD=0.8
DEFAULT_IMAGE_SIZE=640
DEFAULT_OVERLAP_RATIO=0.2

# Cache Configuration
RESULTS_CACHE_MAX_SIZE=100
RESULTS_CACHE_TTL=3600
MODEL_CACHE_MAX_SIZE=10

# Cleanup Configuration
PREDICTION_IMAGE_TTL_HOURS=24
CLEANUP_INTERVAL_MINUTES=60

# OCR Configuration
OCR_CACHE_ENABLED=true
OCR_LANGUAGES=en
OCR_GPU=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=garnet.log

# Paths
PREDICTIONS_DIR=static/images/predictions
```

**Frontend Configuration (`frontend/.env.local`):**

```bash
# API Configuration
VITE_API_URL=http://localhost:8001

# Development Server
VITE_PORT=5173
VITE_HOST=localhost

# Build Configuration
VITE_SOURCEMAP=false
VITE_OUT_DIR=dist
```

#### Configuration Options

The Detection Setup panel in the frontend provides the following options:

| Parameter                | Description                              | Default                   | Range         |
| ------------------------ | ---------------------------------------- | ------------------------- | ------------- |
| **Model**                | Detection model type                     | `ultralytics`             | `ultralytics` |
| **Weight File**          | Path to model weights (`.pt` or `.onnx`) | Auto-selected             | -             |
| **Config File**          | YAML dataset configuration               | `datasets/yaml/data.yaml` | -             |
| **Confidence Threshold** | Minimum detection confidence             | `0.8`                     | `0.0 - 1.0`   |
| **Image Size**           | Input size for model inference           | `640`                     | `320 - 2048`  |
| **Overlap Ratio**        | SAHI slice overlap for large images      | `0.2`                     | `0.0 - 0.95`  |
| **Text OCR**             | Enable text extraction from symbols      | `false`                   | `true/false`  |

**Frontend Features:**

| Feature                | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| **Minimap**            | Navigate large images with a minimap showing viewport position           |
| **Zoom Controls**      | Zoom in/out, reset to 100%, fit to screen                                |
| **Keyboard Shortcuts** | Arrow keys for navigation, A/R for accept/reject, Ctrl+Z/Y for undo/redo |
| **Object Editing**     | Click to select, drag to move, resize handles to adjust bounding box     |
| **Create Object**      | Draw new bounding boxes on canvas to add custom objects                  |
| **Delete Object**      | Remove objects with confirmation                                         |
| **Review Status**      | Mark objects as accepted (green) or rejected (red/dashed)                |
| **Export**             | Download results in JSON, YOLO, COCO, LabelMe, or PDF format             |
| **Batch Mode**         | Queue multiple images, pause/resume processing, navigate between results |

#### Production Deployment

**Build the frontend for production:**

```bash
cd frontend
bun run build
# or: npm run build
```

This creates an optimized build in the `frontend/dist/` directory.

**Production deployment options:**

1. **Separate Services (Recommended)**
    - Serve frontend via nginx/Apache or CDN
    - Run backend with production ASGI server (Uvicorn + Gunicorn)

    ```bash
    # Backend production start
    cd backend
    gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001
    ```

2. **Combined Serving**
    - Mount the built frontend static files in FastAPI:

    ```python
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
    ```

**Production Environment Variables:**

```bash
# Backend
ENV=production
DEBUG=false
HOST=0.0.0.0
PORT=8001
CORS_ORIGINS=https://yourdomain.com

# Frontend (build time)
VITE_API_URL=https://api.yourdomain.com
VITE_SOURCEMAP=false
```

---

### 2. Batch Inference Script

Run inference on multiple P&IDs in a folder using `garnet/predict_images.py`:

**Command-Line Arguments:**

```bash
python garnet/predict_images.py \
    --image_path path/to/pids_folder \
    --model_type yolov8 \
    --model_path path/to/model_weights.pt \
    --output_path results/
```

**Output:**

- Annotated images (saved in output_path).
- CSV file with symbol counts (`output_path/symbol_counts.csv`).

**Example code snippet:**

```python
from garnet.predict_images import predict_images

predict_images(
    image_path="path/to/pids_folder",
    model_type="yolov8",
    model_path="path/to/model_weights.pt",
    output_path="results/"
)
```

---

### 3. Pipeline: End-to-End P&ID Digitization and Connectivity Analysis

A staged, runnable pipeline with CLI is available in `garnet/pid_extractor.py`. This pipeline performs comprehensive P&ID analysis with the following stages:

**Usage:**

```bash
python garnet/pid_extractor.py \
    --image path/to/pid_image.png \
    --coco path/to/coco_annotations.json \
    --ocr path/to/ocr_results.json \
    --arrow-coco path/to/coco_arrows.json \
    --out output/ \
    --stop-after 7
```

**Pipeline Stages:**

1. **Stage 1: Ingest**
    - Load P&ID image (BGR format)
    - Load COCO annotations (symbols, categories)
    - Load OCR results (text annotations)
    - Optional: Load arrow COCO annotations
    - Output: Summary JSON with counts

2. **Stage 2: Preprocess**
    - Convert to grayscale
    - Optional de-skew using Hough line detection
    - Adaptive thresholding for binarization
    - Morphological cleanup (horizontal/vertical closing)
    - Connected component filtering to remove noise
    - Output: Gray image, binary image, deskewed image

3. **Stage 3: Symbols/Text**
    - Import detections from COCO with confidence filtering
    - Shrink bounding boxes for text classes
    - Refine bounding boxes using binary mask
    - Associate nearest text to symbols
    - Adjust center for reducer symbols
    - Output: Overlay image with symbols and text

4. **Stage 4: Linework (Skeleton)**
    - Mask symbol and text regions
    - Skeletonize binary image
    - Output: Skeleton image, masked binary

5. **Stage 5: Graph Construction**
    - DeepLSD line detection integration
    - Process arrow symbols with port detection
    - Process other symbols (valves, reducers, connections)
    - Detect line crossings on bbox borders
    - Validate ports with raycasting
    - Output: Ports overlay, graph nodes/edges

6. **Stage 6: Line Graph**
    - Build connectivity graph using ConnectivityEngine
    - Merge nearby nodes
    - Snap to skeleton endpoints
    - Create pipeline edges
    - Output: Final graph overlay, GraphML export

7. **Stage 7: Export**
    - Export to GraphML format
    - Export to CSV/JSON
    - DEXPI XML export (via dexpi_exporter.py)
    - Output: Export files in output directory

**Configurable Parameters (PipelineConfig):**

- Image processing: DPI, Canny thresholds, binarization settings
- Detection: Confidence thresholds per class
- Text association: Multiplier for bbox diagonal
- Graph: Connection radius, port counts, angle separation
- Valve linking: Directional strategy, edge offset, raycast step
- Template matching: For valve orientation detection
- Cleanup: Bridge max distance, angle tolerance

**Note:** This pipeline is designed to be modular, so each step can be run independently or as part of the full digitization workflow. Use `--stop-after N` to run only specific stages.

---

### 4. Model Training (Optional)

To train custom YOLO models for P&ID symbols using Ultralytics:

```bash
yolo train \
    data=datasets/yaml/data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    imgsz=640 \
    batch=16
```

**Available dataset configurations:**

- `datasets/yaml/data.yaml` - Default dataset configuration
- `datasets/yaml/balanced.yaml` - Balanced class distribution
- `datasets/yaml/iso.yaml` - ISO standard symbols
- `datasets/yaml/pttep.yaml` - PTEP-specific symbols

**Training tips:**

- Use balanced datasets for better model performance
- Adjust `imgsz` based on your P&ID image resolution
- Increase epochs for better convergence (100-300 typical)
- Use data augmentation for improved generalization

## ⌨️ Keyboard Shortcuts

The frontend supports the following keyboard shortcuts for efficient navigation and editing:

| Shortcut                        | Action                           |
| ------------------------------- | -------------------------------- |
| `←` / `→`                       | Navigate to previous/next object |
| `Enter`                         | Accept selected object           |
| `Delete` / `Backspace`          | Reject selected object           |
| `Ctrl + Z`                      | Undo last action                 |
| `Ctrl + Y` / `Ctrl + Shift + Z` | Redo last action                 |
| `F`                             | Fit image to screen              |
| `0`                             | Reset zoom to 100%               |
| `+` / `-`                       | Zoom in/out                      |
| `Esc`                           | Deselect object / Cancel editing |

## 📂 Dataset

GARNET uses the YOLOv8 dataset format. Example structure:

```
dataset/
├── train/
│   ├── images/  # P&ID images (.jpg, .png)
│   └── labels/  # YOLO-format labels (.txt)
├── val/
│   ├── images/
│   └── labels/
└── data.yaml     # Dataset config (class names, paths)
```

**Available dataset configurations:**

- `datasets/yaml/data.yaml` - Default dataset configuration
- `datasets/yaml/balanced.yaml` - Balanced class distribution
- `datasets/yaml/iso.yaml` - ISO standard symbols
- `datasets/yaml/pttep.yaml` - PTEP-specific symbols

**Class definitions:**

- `datasets/classes.txt` - List of all class names
- `datasets/predefined_classes.txt` - Predefined class mappings
- `datasets/settings_labels.json` - Label settings configuration

Example `data.yaml`:

```yaml
train: dataset/train/images
val: dataset/val/images

nc: 6 # Number of classes
names: ["valve", "gate_valve", "globe_valve", "check_valve", "pump", "tank"]
```

---

## 📊 Results

| **Detection**                                     | **Graph Analysis**                               |
| ------------------------------------------------- | ------------------------------------------------ |
| ![Detected Symbols](assets/detection_example.jpg) | ![Graph Visualization](assets/graph_example.png) |

_Example output: Symbol counts and connectivity graph for a P&ID._

---

## 📈 Future Outcomes

Additional planned outcomes from the GARNET project include:

- **MTO for Valves**: Automated generation of material take-off lists for all detected valve types, including specifications and quantities.
- **Line List**: Extraction and tabulation of pipeline data, including line tags, sizes, service, and connected equipment.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.  
For major changes, open an issue first to discuss your ideas.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📧 Contact

For questions or collaborations, contact:

- **Your Name** - [may3rd@gmail.com](mailto:may3rd@gmail.com)
- **GCME (GC Maintenance and Engineering Co., Ltd.)** - [www.gcme.com](https://www.gcme.com)
